# ==== E1: optional LoRA SFT arm, then unified evaluation on SVG2test-VIPSeg, VidOR-val, PVSG-val (fast engine)
T = {"arm": ARM}
def slug_root(p, slug):
    parts = p.split("/"); return "/".join(parts[:parts.index(slug) + 1]) if slug in parts else os.path.dirname(p)
FRAMEDIRS = collections.defaultdict(list)
for dp, dn, fn in os.walk("/kaggle/input"):
    if any(f.endswith(".jpg") for f in fn): FRAMEDIRS[os.path.basename(dp)].append(dp)
def frames_of(v, root):
    ds = [d for d in FRAMEDIRS.get(v, []) if d.startswith(root)]
    if ds: return sorted(glob.glob(f"{ds[0]}/*.jpg"))
    return ["FAKE"] * 400 if globals().get("FAKE_FRAMES") else []
def load_rows(path): return {str(r["video_id"]): r for r in pq.read_table(path).to_pylist()}

# ---------------- benchmarks: (masks rows, gt rows, frame root, mask stride, fps fn)
SVG2_ROOT = slug_root(SVG2_MASKS, "svg2test-vipseg")
B = {
  "svg2": dict(masks=load_rows(SVG2_MASKS), gt=load_rows(SVG2_GT), root=SVG2_ROOT, stride=3, fps=lambda v: 2.0),
  "vidor": dict(masks=load_rows(f"{VIDOR_DIR}/masks_vidor.parquet"), gt=load_rows(f"{VIDOR_DIR}/gt_vidor.parquet"), root=VIDOR_DIR, stride=1,
                fps=(lambda m: (lambda v: m[v]["fps_eff"]))(json.load(open(f"{VIDOR_DIR}/vidor_meta.json")))),
  "pvsg": dict(masks=load_rows(f"{PVSG_DIR}/masks.parquet"), gt=load_rows(f"{PVSG_DIR}/gt.parquet"), root=PVSG_DIR, stride=1,
               fps=(lambda m: (lambda v: m[v]["fps_eff"]))(json.load(open(f"{PVSG_DIR}/meta.json")))),
}
def make_cpu_fn(b, max_obj=40):
    def cpu_fn(v):
        r = b["masks"][v]; H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
        files = frames_of(v, b["root"]); s = b["stride"]
        total = min(len(files), len(range(0, nf, s)))
        md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, s)][:total]
        if files[0] == "FAKE":
            fr = np.random.default_rng(0).integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8); lf = lambda idx: fr[idx]
        else: lf = lambda idx: np.stack([np.array(Image.open(files[i]).convert("RGB").resize((W, H))) for i in idx])
        return dict(load_frames=lf, total=total, fps=b["fps"](v), mask_data=md, obj_ids=list(range(no))[:max_obj], H=H, W=W)
    return cpu_fn

# ---------------- training (A1 control: original prompt; A2: style-tagged prompts)
def train_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV)            # same dev hold-out as the GRPO arm
    rng0 = random.Random(0); items = []
    for name in sorted(tr):
        vv = list(tr[name]); rng0.shuffle(vv); items += [(v, name) for v in vv[:MAX_PER_SOURCE]]
    rng0.shuffle(items)
    model.eval(); d0 = dev_reward(srcs, dv); print("SFT dev step 0", json.dumps(d0), flush=True)
    def cpu_job(v, src): return eng.cpu_stage(**srcs[src]["cpu_fn"](v))
    nl = apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    params = lora_params(model); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = math.ceil(len(items) / ACCUM); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(); print(f"TRAIN arm={ARM} samples={len(items)} lora_modules={nl} steps={steps}", flush=True)
    t0 = time.time(); losses = []; ntok = 0; skipped = 0
    import concurrent.futures as cf
    ex = cf.ThreadPoolExecutor(4); futs = collections.deque(ex.submit(cpu_job, v, s_) for v, s_ in items[:8])
    for i, (v, src) in enumerate(items):
        C = futs.popleft().result()
        if i + 8 < len(items): futs.append(ex.submit(cpu_job, *items[i + 8]))
        prompt = STYLE_PROMPT_TRAIN[src]
        with torch.no_grad(): P = eng.gpu_stage(C, prompt=prompt)
        lab = srcs[src]["lab"][v]; objs = json.loads(lab["objects"]); rels = json.loads(lab["relationships"])
        kr = None
        if TARGET_VIEW[src] == "salient":
            touch = {frozenset(x) for x in json.loads(lab["touching"])}; kr = lambda s_, o_: frozenset((s_, o_)) in touch
        tgt = build_target(objs, rels, P["kept"], keep_rel=kr)
        if tgt is None: skipped += 1; continue
        loss, Tn = sample_loss(eng, P, tgt); (loss / ACCUM).backward(); losses.append(float(loss)); ntok += Tn
        if (i + 1) % ACCUM == 0 or i + 1 == len(items):
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if i % 50 == 0: print(f"train {i}/{len(items)} loss {np.mean(losses[-50:]):.4f} tok {ntok} skipped {skipped} {time.time()-t0:.0f}s", flush=True)
        del C, P
    model.eval(); model.model.language_model.gradient_checkpointing_disable()
    torch.save(lora_state_dict(model.model.language_model), f"{OUT}/lora_{ARM}.pt"); merge_lora(model.model.language_model)
    T.update(train_seconds=round(time.time() - t0, 1), train_samples=len(items), train_skipped=skipped, train_tokens=ntok,
             loss_first50=round(float(np.mean(losses[:50])), 4), loss_last50=round(float(np.mean(losses[-50:])), 4))
    json.dump(losses, open(f"{OUT}/losses_{ARM}.json", "w")); print("TRAIN DONE", T, flush=True)
    d1 = dev_reward(srcs, dv); T.update(dev_start=d0, dev_final=d1); print("SFT dev final", json.dumps(d1), flush=True)
    torch.cuda.empty_cache()
# GRPO_PART
if ARM in ("A1", "A2"): train_arm()
elif ARM == "G1": grpo_arm()

# ---------------- evaluation
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
def rel_ids(text):
    out = []
    for r in parse_rels(text):
        try: out.append((int(r[0]), int(r[2])))
        except Exception: pass
    return out
def select_pairs(res_v, k_sal, k_rnd, rng):
    C = res_v["C"]; kept = C["kept"]; om = C["om"][:, :, ::4, ::4].numpy() > 0; have = set()
    for a, b in rel_ids(res_v["text"]):
        if a >= 1 and b >= 1 and a <= len(kept) and b <= len(kept): have.add(frozenset((a - 1, b - 1)))
    D = [[om[o, j] if om[o, j].any() else None for j in range(om.shape[1])] for o in range(len(kept))]
    DL = [[dil(m) if m is not None else None for m in row] for row in D]
    sc, unc = [], []
    for a, b in itertools.combinations(range(len(kept)), 2):
        if frozenset((a, b)) in have: continue
        unc.append((a, b))
        c = [bool((DL[a][j] & D[b][j]).any()) for j in range(om.shape[1]) if D[a][j] is not None and D[b][j] is not None]
        if c and np.mean(c) > 0: sc.append((np.mean(c), a, b))
    sal = [(a, b) for _, a, b in sorted(sc, reverse=True)[:k_sal]]
    rnd = rng.sample(unc, min(k_rnd, len(unc)))
    return sal, rnd
CHUNKS = {"svg2": 96, "vidor": 96, "pvsg": 16}                  # PVSG videos are long (host RAM)
for bn in BENCHES:
    b = B[bn]; vids = sorted(v for v in b["masks"] if v in b["gt"] and frames_of(v, b["root"]))[:LIMIT_VIDS]; cpu_fn = make_cpu_fn(b)
    prompt = STYLE_PROMPT_TEST[bn]; tb = time.time(); rng = random.Random(0); stops = collections.Counter(); ntok = 0
    fo = {n: open(f"{OUT}/{bn}_{n}.jsonl", "w") for n in ("full", "pairs_sal", "pairs_rnd")}
    t_full = t_pairs = 0.0
    CHUNK = CHUNKS[bn]
    for c0 in range(0, len(vids), CHUNK):
        ch = vids[c0:c0 + CHUNK]; t1 = time.time()
        res = eng.run_slots(ch, cpu_fn, max_slots=32, kv_budget_gb=40.0, loop_stop=True, log=lambda s: None, prompt=prompt)
        gpu_sync(); t_full += time.time() - t1; t2 = time.time()
        for v in ch:
            r = res[v]; stops[r["stop"]] += 1; ntok += r["n_tokens"]
            fo["full"].write(json.dumps({"video_id": v, "kept_obj_ids": r["kept"], "text": r["text"], "n_tokens": r["n_tokens"], "stop": r["stop"]}) + "\n")
        jobs = []
        for v in ch:
            sal, rnd = select_pairs(res[v], K_SAL, K_RND, rng)
            jobs += [("pairs_sal", v, a, b_) for a, b_ in sal] + [("pairs_rnd", v, a, b_) for a, b_ in rnd]
        Pp = [eng.gpu_stage(res[v]["C"], obj_subset=[a, b_], prompt=prompt) for _, v, a, b_ in jobs]
        if Pp:
            pr = SlotDecoder(model, ftok, max_slots=64, kv_budget_gb=40.0, every=64, loop_stop=False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
            for i, ((n, v, a, b_), p) in enumerate(zip(jobs, Pp)):
                fo[n].write(json.dumps({"video_id": v, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": pr[i]["text"]}) + "\n")
        gpu_sync(); t_pairs += time.time() - t2
        del res, Pp; gc.collect(); torch.cuda.empty_cache()
        print(f"{bn} {min(c0 + CHUNK, len(vids))}/{len(vids)} full {t_full:.0f}s pairs {t_pairs:.0f}s", flush=True)
    for f in fo.values(): f.close()
    T[bn] = dict(videos=len(vids), full_seconds=round(t_full, 1), pair_seconds=round(t_pairs, 1), stops=dict(stops), full_tokens=ntok, prompt=prompt)
    json.dump(T, open(f"{OUT}/timing_{ARM}.json", "w")); print("BENCH DONE", bn, T[bn], flush=True)

# ---------------- judge new (GT, predicted) label pairs (Qwen3-8B, SVG2 categories)
if JUDGE_DIR:
    del model; gc.collect(); torch.cuda.empty_cache()
    pairs = set()
    for bn in BENCHES:
        gt = B[bn]["gt"]
        for n in ("full", "pairs_sal", "pairs_rnd"):
            for l in open(f"{OUT}/{bn}_{n}.jsonl"):
                o = json.loads(l); kept = o["kept_obj_ids"]; g = gt[o["video_id"]]
                P_ = collections.defaultdict(list)
                for r in parse_rels(o["text"]):
                    try: P_[(kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1)].append(str(r[1]))
                    except Exception: pass
                for s, p, ob, *_ in json.loads(g["relationships"]):
                    for q in P_.get((s, ob), []):
                        if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))
    pairs = sorted(pairs); print("pairs to judge", len(pairs), flush=True)
    run_judge(pairs, JUDGE_DIR, f"{OUT}/judge_{ARM}.json")
print("E1 DONE", ARM, json.dumps(T), flush=True)
