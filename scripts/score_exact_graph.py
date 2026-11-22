"""Exact-normalised relation P/R/F1 of GRAPH-ONLY outputs under the qtm_score.py protocol (dedup, per-span tIoU >= .5, lowercase,
'_' -> ' ', pooled counts), for SVG2test-100 / VidOR / PVSG. No judge, no TraSeR outputs needed. Usage: score_exact_graph.py DIR TAG"""
import sys, os, json, subprocess, collections, numpy as np, pyarrow.parquet as pq
W = "/home/karthik/svg2-killtest"; D, TAG = sys.argv[1], sys.argv[2]; sys.path[:0] = [f"{W}/scripts"]; from relparse import parse_rels
S = f"{W}/out/score_exact_{TAG}"; os.makedirs(S, exist_ok=True)
nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())
def iou1(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else float(a == b)
tiou = lambda x, y: max((iou1(a, b) for a in x for b in y if len(a) == 2 and len(b) == 2), default=0)
def counts(path, gt):
    C = {}
    for o in map(json.loads, open(path)):
        v = o["video_id"]
        if "error" in o or v not in gt: continue
        k = o["kept_obj_ids"]; M = collections.OrderedDict()
        for r in parse_rels(o["text"]):
            try: M.setdefault((k[int(r[0]) - 1], nrm(r[1]), k[int(r[2]) - 1]), []).extend([[float(x), float(y)] for x, y in r[3]])
            except Exception: pass
        G = gt[v]; tp = 0; hit = set()
        for (s, p, ob), sp in M.items():
            m = [i for i, g in enumerate(G) if g[0] == s and g[2] == ob and nrm(g[1]) == p and tiou(sp, g[3]) >= .5]
            tp += bool(m); hit.update(m)
        C[v] = (tp, len(M), len(hit), len(G))
    for v in gt: C.setdefault(v, (0, 0, 0, len(gt[v])))
    return C
def prf(x): P = x[:, 0].sum() / max(x[:, 1].sum(), 1); R = x[:, 2].sum() / max(x[:, 3].sum(), 1); return np.array([P, R, 2 * P * R / (P + R + 1e-9)]) * 100
gtp = {"svg2": f"{W}/data/gt/vipseg/gt_vipseg.parquet", "sav": f"{W}/data/gt/sav/gt_sav.parquet", "vidor": f"{W}/data/gt/vidor/gt_vidor.parquet", "pvsg": f"{W}/data/gt/pvsg/gt.parquet"}
GT = {b: {str(r["video_id"]): json.loads(r["relationships"]) for r in pq.read_table(p).to_pylist()} for b, p in gtp.items()}
GT["svg2_100"] = {**GT["svg2"], **GT["sav"]}
def prep(bn):
    out = f"{S}/{bn}.dd"; subprocess.run([sys.executable, f"{W}/scripts/dedup.py", f"{D}/{bn}_graph_{TAG}.jsonl", out], check=True); return out
files = {b: prep(b) for b in ("svg2", "sav", "vidor", "pvsg")}
open(f"{S}/svg2_100.dd", "w").write(open(files["svg2"]).read() + open(files["sav"]).read()); files["svg2_100"] = f"{S}/svg2_100.dd"
rng = np.random.default_rng(0)
for bn in ("svg2_100", "svg2", "sav", "vidor", "pvsg"):
    if bn in ("svg2", "sav"): f = files[bn]
    else: f = files[bn]
    C = counts(f, GT[bn]); x = np.array([C[v] for v in sorted(GT[bn])], float); m = prf(x)
    bs = np.array([prf(x[rng.integers(0, len(x), len(x))])[2] for _ in range(2000)]); lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{bn:9s} exact P/R/F1 {m[0]:.1f}/{m[1]:.1f}/{m[2]:.1f}  F1 95% CI [{lo:.1f}, {hi:.1f}]   (videos {len(GT[bn])}, GT rels {int(x[:,3].sum())}, preds {int(x[:,1].sum())})")
