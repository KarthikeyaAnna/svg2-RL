"""EDA of a prepared training subset vs its full SVG2 split. Usage: subset_dir full_labels.parquet name"""
import sys, json, collections, numpy as np, pyarrow.parquet as pq
from transformers import AutoTokenizer
sys.path.insert(0, "/home/karthik/svg2-killtest/engine"); from sft import build_target
D, FULL, NAME = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained("/home/karthik/svg2-killtest/traser_cfg", use_fast=True)
lab = pq.read_table(f"{D}/labels_train.parquet").to_pylist(); meta = json.load(open(f"{D}/meta_train.json"))
mk = {r["video_id"]: r for r in pq.read_table(f"{D}/masks_train.parquet", columns=["video_id", "num_objects", "mask_height", "mask_width", "num_frames"]).to_pylist()}
def stats(rows):
    o, r, rp, bid, pred = [], [], [], [], collections.Counter()
    for x in rows:
        ob = json.loads(x["objects"]); R = json.loads(x["relationships"]); o.append(len(ob)); r.append(len(R))
        by = collections.defaultdict(int); und = collections.defaultdict(set)
        for q in R: by[(int(q[0]), int(q[2]))] += 1; pred[q[1].lower()] += 1
        for k in by: und[frozenset(k)].add(k)
        for k, v in und.items(): rp.append(sum(by[d] for d in v)); bid.append(len(v) == 2 and len(k) == 2)
    return dict(n=len(rows), obj_mean=np.mean(o), obj_p95=np.percentile(o, 95), frac_obj_gt40=np.mean(np.array(o) > 40), rel_mean=np.mean(r),
                rel_per_pair=np.mean(rp), bidir=np.mean(bid), top_pred=[p for p, _ in pred.most_common(6)])
full = pq.read_table(FULL).to_pylist()
import random; random.Random(0).shuffle(full)
S, F = stats(lab), stats(full[:5000])
print(f"== {NAME}"); [print(f"  {k:14s} subset {S[k] if isinstance(S[k], (list,int)) else round(float(S[k]),3)}   full {F[k] if isinstance(F[k], (list,int)) else round(float(F[k]),3)}") for k in S]
dur = [m["duration"] for m in meta.values()]; fr = [m["sampled"] for m in meta.values()]
print(f"  duration s p50/p95/max {np.percentile(dur,[50,95,100]).round(1)}  sampled frames p50/max {np.percentile(fr,[50,100])}  frames==128 cap {np.mean(np.array(fr)==128):.3f}")
# visual tokens estimate: per 2 frames, (H*W scaled to <= 768*28*28 px) / (28*28)
vt = []
for v, m in mk.items():
    H, W = m["mask_height"], m["mask_width"]; s = min(1.0, (768 * 28 * 28 / (H * W)) ** 0.5)
    vt.append((m["num_frames"] // 2) * round(H * s / 28) * round(W * s / 28))
print(f"  video tokens (pre token-selection) p50/p95/max {np.percentile(vt,[50,95,100]).astype(int)}")
# target token lengths (all objects kept, original vs salient view)
L0, L1, keep = [], [], []
for x in lab[:400]:
    ob = json.loads(x["objects"]); R = json.loads(x["relationships"]); kept = list(range(len(ob)))[:40]
    t0 = build_target(ob, R, kept)
    if t0 is None: continue
    L0.append(len(tok(t0)["input_ids"]))
    if "touching" in x:
        tch = {frozenset(p) for p in json.loads(x["touching"])}
        t1 = build_target(ob, R, kept, keep_rel=lambda s, o: frozenset((s, o)) in tch); L1.append(len(tok(t1)["input_ids"]))
        keep.append(len(json.loads(t1)["relationships"]) / max(1, len(json.loads(t0)["relationships"])))
print(f"  target tokens p50/p95/max {np.percentile(L0,[50,95,100]).astype(int)}" + (f"  salient view {np.percentile(L1,[50,95,100]).astype(int)}  relations kept {np.mean(keep):.2f} (videos with 0 kept {np.mean(np.array(keep)==0):.2f})" if L1 else ""))
