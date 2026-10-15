# ---- SVG2test (VIPSeg part): full graphs + batched salient-pair decodes with the fast engine
import itertools, collections, pyarrow.parquet as pq
from pycocotools import mask as Mk
masks = {r["video_id"]: r for r in pq.read_table(MASKS).to_pylist()}
vids = sorted(masks)[:NVID]
def frame_files(v): return sorted(glob.glob(f"{FRAMES}/**/{v}/*.jpg", recursive=True))
def cpu_fn(v):
    r = masks[v]; H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
    files = frame_files(v); total = len(files) if files else (nf + 2) // 3
    md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
    if files: lf = lambda idx: np.stack([np.array(Image.open(files[i]).convert("RGB").resize((W, H))) for i in idx])
    else:
        rng = np.random.default_rng(0); fr = rng.integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8); lf = lambda idx: fr[idx]
    return dict(load_frames=lf, total=total, fps=2.0, mask_data=md, obj_ids=list(range(no))[:40], H=H, W=W)
T = {}
t0 = time.time()
res = eng.run_slots(vids, cpu_fn, max_slots=MAXB, kv_budget_gb=KVGB, loop_stop=True, log=lambda s: print(s, flush=True))
T["full_seconds"] = round(time.time() - t0, 1)
with open(f"{OUT}/fast6_full.jsonl", "w") as f:
    for v in vids: f.write(json.dumps({"video_id": v, "kept_obj_ids": res[v]["kept"], "text": res[v]["text"], "n_tokens": res[v]["n_tokens"], "stop": res[v]["stop"]}) + "\n")
T["full_stops"] = dict(collections.Counter(res[v]["stop"] for v in vids)); T["full_tokens"] = sum(res[v]["n_tokens"] for v in vids)
# ---- salient pairs (contact at ~1 Hz on the sampled frames), batched across videos
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
t1 = time.time(); pair_jobs = []
for v in vids:
    C = res[v]["C"]; kept = C["kept"]; om = C["om"][:, :, ::4, ::4].numpy() > 0   # reuse prepared masks, 4x downsampled
    have = set()
    for rr in parse_rel_ids(res[v]["text"]):
        if rr[0] < 1 or rr[1] < 1: continue
        try: have.add(frozenset((kept[rr[0] - 1], kept[rr[1] - 1])))
        except Exception: pass
    D = [[om[o, j] if om[o, j].any() else None for j in range(om.shape[1])] for o in range(len(kept))]
    DL = [[dil(m) if m is not None else None for m in row] for row in D]
    sc = []
    for a, b in itertools.combinations(range(len(kept)), 2):
        if frozenset((kept[a], kept[b])) in have: continue
        c = [bool((DL[a][j] & D[b][j]).any()) for j in range(om.shape[1]) if D[a][j] is not None and D[b][j] is not None]
        if c and np.mean(c) > 0: sc.append((np.mean(c), a, b))
    for _, a, b in sorted(sc, reverse=True)[:KPAIRS]: pair_jobs.append((v, a, b, C))
T["pair_select_seconds"] = round(time.time() - t1, 1); t2 = time.time()
Pp = [(v, eng.gpu_stage(C, obj_subset=[a, b])) for v, a, b, C in pair_jobs]
T["pair_prep_seconds"] = round(time.time() - t2, 1)
pdec = SlotDecoder(eng.m, eng.ftok, max_slots=MAXB_PAIRS, kv_budget_gb=KVGB, every=64, loop_stop=False, log=lambda s: print(s, flush=True))
pres = pdec.run([(i, p, 1024) for i, (_, p) in enumerate(Pp)])
with open(f"{OUT}/fast6_pairs.jsonl", "w") as f:
    for i, (v, p) in enumerate(Pp):
        f.write(json.dumps({"video_id": v, "pair": [p["kept"][0], p["kept"][1]], "kept_obj_ids": p["kept"], "text": pres[i]["text"], "n_tokens": pres[i]["n_tokens"]}) + "\n")
T["pair_decode_seconds"] = round(time.time() - t2, 1); T["n_pairs"] = len(Pp); T["n_videos"] = len(vids)
json.dump(T, open(f"{OUT}/fast6_timing.json", "w")); print("TIMING", T, flush=True); print("FAST6 DONE", flush=True)
