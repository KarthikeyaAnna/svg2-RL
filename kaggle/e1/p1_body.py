# ==== P1: released TraSeR decodes learned-scorer pair lists (no training). Compared offline against A0's contact-ranked and random pairs.
T = {"arm": ARM}
def slug_root(p, slug):
    parts = p.split("/"); return "/".join(parts[:parts.index(slug) + 1]) if slug in parts else os.path.dirname(p)
FRAMEDIRS = collections.defaultdict(list)
for dp, dn, fn in os.walk("/kaggle/input" if not globals().get("FAKE_FRAMES") else R):
    if any(f.endswith(".jpg") for f in fn): FRAMEDIRS[os.path.basename(dp)].append(dp)
def frames_of(v, root):
    ds = [d for d in FRAMEDIRS.get(v, []) if d.startswith(root)]
    if ds: return sorted(glob.glob(f"{ds[0]}/*.jpg"))
    return ["FAKE"] * 400 if globals().get("FAKE_FRAMES") else []
def load_rows(path): return {str(r["video_id"]): r for r in pq.read_table(path).to_pylist()}
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
        files = frames_of(v, b["root"]); s = b["stride"]; total = min(len(files), len(range(0, nf, s)))
        md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, s)][:total]
        if files[0] == "FAKE":
            fr = np.random.default_rng(0).integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8); lf = lambda idx: fr[idx]
        else: lf = lambda idx: np.stack([np.array(Image.open(files[i]).convert("RGB").resize((W, H))) for i in idx])
        return dict(load_frames=lf, total=total, fps=b["fps"](v), mask_data=md, obj_ids=list(range(no))[:max_obj], H=H, W=W)
    return cpu_fn
PL = json.load(open(PAIRS_FILE))
CHUNKS = {"svg2": 96, "vidor": 96, "pvsg": 16}
for bn in BENCHES:
    b = B[bn]; vids = sorted(v for v in PL[bn] if v in b["masks"] and frames_of(v, b["root"]) and PL[bn][v])[:LIMIT_VIDS]
    cpu_fn = make_cpu_fn(b); t0 = time.time(); n = 0
    with open(f"{OUT}/{bn}_pairs_learned.jsonl", "w") as fo:
        for c0 in range(0, len(vids), CHUNKS[bn]):
            ch = vids[c0:c0 + CHUNKS[bn]]
            prepared, keep = eng.prepare(ch, cpu_fn, 6, "scene_graph")
            jobs = []
            for v in ch:
                kept = keep[v]["kept"]; pos = {oid: i for i, oid in enumerate(kept)}
                for rk, e in enumerate(PL[bn][v]):                       # entries [a, b] or [a, b, prob]
                    a, b_ = int(e[0]), int(e[1])
                    if a in pos and b_ in pos: jobs.append((v, rk, [pos[a], pos[b_]]))
            Pp = [eng.gpu_stage(keep[v], obj_subset=sub, prompt="scene_graph") for v, _, sub in jobs]
            if Pp:
                res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
                for i, ((v, rk, sub), p) in enumerate(zip(jobs, Pp)):
                    fo.write(json.dumps({"video_id": v, "rank": rk, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"]}) + "\n"); n += 1
            del prepared, keep, Pp; gc.collect(); torch.cuda.empty_cache()
            print(f"{bn} {min(c0 + CHUNKS[bn], len(vids))}/{len(vids)} pairs {n} {time.time()-t0:.0f}s", flush=True)
    T[bn] = dict(videos=len(vids), pairs=n, seconds=round(time.time() - t0, 1)); print("BENCH DONE", bn, T[bn], flush=True)
    json.dump(T, open(f"{OUT}/timing_{ARM}.json", "w"))
if JUDGE_DIR:
    del model; gc.collect(); torch.cuda.empty_cache(); pairs = set()
    for bn in BENCHES:
        gt = B[bn]["gt"]
        for l in open(f"{OUT}/{bn}_pairs_learned.jsonl"):
            o = json.loads(l); kept = o["kept_obj_ids"]; g = gt[o["video_id"]]; P_ = collections.defaultdict(list)
            for r in parse_rels(o["text"]):
                try: P_[(kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1)].append(str(r[1]))
                except Exception: pass
            for s, p, ob, *_ in json.loads(g["relationships"]):
                for q in P_.get((s, ob), []):
                    if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))
    pairs = sorted(pairs); print("pairs to judge", len(pairs), flush=True); run_judge(pairs, JUDGE_DIR, f"{OUT}/judge_{ARM}.json")
print("P1 DONE", json.dumps(T), flush=True)
