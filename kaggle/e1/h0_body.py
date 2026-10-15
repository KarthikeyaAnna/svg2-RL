# ==== H0 pre-check: does RL have headroom? greedy vs 8 samples per training video, scored by the a-priori graph reward
T = {"arm": ARM}
def slug_root(p, slug):
    parts = p.split("/"); return "/".join(parts[:parts.index(slug) + 1]) if slug in parts else os.path.dirname(p)
FRAMEDIRS = collections.defaultdict(list)
for dp, dn, fn in os.walk("/kaggle/input" if not globals().get("FAKE_FRAMES") else R):
    if any(f.endswith(".jpg") for f in fn): FRAMEDIRS[os.path.basename(dp)].append(dp)
def frames_of(v, root):
    ds = [d for d in FRAMEDIRS.get(v, []) if d.startswith(root)]
    return sorted(glob.glob(f"{ds[0]}/*.jpg")) if ds else []
def load_rows(path): return {str(r["video_id"]): r for r in pq.read_table(path).to_pylist()}
def make_cpu_fn(b, max_obj=40):
    def cpu_fn(v):
        r = b["masks"][v]; H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
        files = frames_of(v, b["root"]); total = min(len(files), nf)
        md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(nf)][:total]
        lf = lambda idx: np.stack([np.array(Image.open(files[i]).convert("RGB").resize((W, H))) for i in idx])
        return dict(load_frames=lf, total=total, fps=b["fps"](v), mask_data=md, obj_ids=list(range(no))[:max_obj], H=H, W=W)
    return cpu_fn
fo = open(f"{OUT}/h0_rollouts.jsonl", "w"); summ = {}
for src, d in TRAIN_DIRS.items():
    tm = load_rows(f"{d}/masks_train.parquet"); tl = load_rows(f"{d}/labels_train.parquet"); me = json.load(open(f"{d}/meta_train.json"))
    vids = [v for v in sorted(tm) if v in tl and frames_of(v, d)]; random.Random(1).shuffle(vids); vids = vids[:N_PER_SOURCE]
    cpu_fn = make_cpu_fn(dict(masks=tm, root=d, fps=(lambda me: lambda v: me[v]["fps_eff"])(me)))
    t0 = time.time(); prepared, keep = eng.prepare(vids, cpu_fn, 6, "scene_graph"); t_prep = time.time() - t0
    items = [(v, prepared[v], token_cap(prepared[v]["n_obj"])) for v in vids]
    t0 = time.time(); greedy = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run(items); t_g = time.time() - t0
    t0 = time.time()
    samp = SlotDecoder(model, ftok, 64, 40.0, 64, True, log=lambda s: None, temperature=TEMP, top_p=TOP_P, seed=0).run(
        [(f"{v}#{g}", P, c) for v, P, c in items for g in range(G)]); t_s = time.time() - t0
    rows = []
    for v in vids:
        lab = tl[v]; ob = json.loads(lab["objects"]); rl = json.loads(lab["relationships"]); kept = prepared[v]["kept"]
        rg, dg = reward(greedy[v]["text"], kept, ob, rl)
        rs = [reward(samp[f"{v}#{g}"]["text"], kept, ob, rl) for g in range(G)]
        rows.append(dict(src=src, video_id=v, kept=kept, greedy=dict(R=rg, **dg, stop=greedy[v]["stop"], n_tokens=greedy[v]["n_tokens"]),
                         samples=[dict(R=r, **d_, stop=samp[f"{v}#{g}"]["stop"], n_tokens=samp[f"{v}#{g}"]["n_tokens"]) for g, (r, d_) in enumerate(rs)],
                         greedy_text=greedy[v]["text"], sample_texts=[samp[f"{v}#{g}"]["text"] for g in range(G)]))
        fo.write(json.dumps(rows[-1]) + "\n")
    g_ = np.array([r["greedy"]["R"] for r in rows]); S = np.array([[s["R"] for s in r["samples"]] for r in rows])
    summ[src] = dict(n=len(rows), greedy=float(g_.mean()), sample_mean=float(S.mean()), best_of_G=float(S.max(1).mean()),
                     best_of_G_minus_greedy=float((S.max(1) - g_).mean()), frac_best_gt_greedy=float((S.max(1) > g_ + 1e-6).mean()),
                     group_std_mean=float(S.std(1).mean()), frac_groups_std_gt_0_02=float((S.std(1) > 0.02).mean()),
                     greedy_f1=float(np.mean([r["greedy"]["f1"] for r in rows])), seconds=dict(prep=round(t_prep, 1), greedy=round(t_g, 1), samples=round(t_s, 1)))
    print("H0", src, json.dumps(summ[src]), flush=True)
    del prepared, keep; gc.collect(); torch.cuda.empty_cache()
fo.close(); json.dump(summ, open(f"{OUT}/h0_summary.json", "w")); print("H0 DONE", json.dumps(summ), flush=True)
