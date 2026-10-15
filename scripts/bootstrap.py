"""Paired per-video bootstrap of P/R/F1 differences between two systems. Usage: A.jsonl B.jsonl gt.parquet judge.json [clean_ids.json]"""
import sys, os, json, collections, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from relparse import parse_rels
A, B, G, J = sys.argv[1:5]; clean = set(json.load(open(sys.argv[5]))) if len(sys.argv) > 5 else None
gt = {r["video_id"]: r for r in pq.read_table(G).to_pylist()}; J = json.load(open(J))
def ok(g, p):
    g, p = g.lower().strip(), p.lower().strip(); c = J.get(f"relation predicate|||{g}|||{p}"); return g == p or (c is not None and not c.startswith("mismatch"))
def iou1(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else float(a == b)
def tiou(x, y): return max((iou1(a, b) for a in x for b in y if len(a) == 2 and len(b) == 2), default=0)
def counts(path):
    C = {}
    for o in map(json.loads, open(path)):
        v = o["video_id"]
        if "error" in o or v not in gt or (clean and v not in clean): continue
        k = o["kept_obj_ids"]; M = collections.OrderedDict()
        for r in parse_rels(o["text"]):
            try: M.setdefault((k[int(r[0]) - 1], r[1].lower().strip(), k[int(r[2]) - 1]), []).extend([[float(x), float(y)] for x, y in r[3]])
            except Exception: pass
        Gr = json.loads(gt[v]["relationships"]); tp = 0; hit = set()
        for (s, p, ob), sp in M.items():
            m = [i for i, g in enumerate(Gr) if g[0] == s and g[2] == ob and ok(g[1], p) and tiou(sp, g[3]) >= .5]
            tp += bool(m); hit.update(m)
        C[v] = (tp, len(M), len(hit), len(Gr))
    return C
ca, cb = counts(A), counts(B); vids = sorted(set(ca) & set(cb)); a = np.array([ca[v] for v in vids], float); b = np.array([cb[v] for v in vids], float)
def prf(x): P = x[:, 0].sum() / max(x[:, 1].sum(), 1); R = x[:, 2].sum() / max(x[:, 3].sum(), 1); return P, R, 2 * P * R / (P + R + 1e-9)
rng = np.random.default_rng(0); D = []
for _ in range(2000):
    i = rng.integers(0, len(vids), len(vids)); D.append(np.array(prf(b[i])) - np.array(prf(a[i])))
D = np.array(D) * 100; pa, pb = np.array(prf(a)) * 100, np.array(prf(b)) * 100
win = np.mean((b[:, 2] / np.maximum(b[:, 3], 1)) > (a[:, 2] / np.maximum(a[:, 3], 1))); lose = np.mean((b[:, 2] / np.maximum(b[:, 3], 1)) < (a[:, 2] / np.maximum(a[:, 3], 1)))
for j, n in enumerate("PRF"):
    lo, hi = np.percentile(D[:, j], [2.5, 97.5]); sig = "*" if lo > 0 or hi < 0 else " "
    print(f"  {n}: {pa[j]:5.1f} -> {pb[j]:5.1f}  delta {pb[j]-pa[j]:+5.1f}  95% CI [{lo:+5.1f}, {hi:+5.1f}] {sig}", end="")
print(f"   | videos {len(vids)}: recall better {100*win:.0f}% / worse {100*lose:.0f}%")
