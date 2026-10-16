import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Paired per-video bootstrap of relation recall, triplet recall and object accuracy (SVG2-style, judge, per-span tIoU >= .5).
Each system = merged graph + label sources (object names: first source that names a GT object wins, as in triplet.py).
Object accuracy = share of GT objects in the system's input whose predicted name is judged equal (labels from the first source only,
i.e. the full graph). Reports unjudged non-identical (GT, pred) name pairs so judge coverage is visible.
Usage: boot_triplet.py gt.parquet judge.json A_merged.jsonl A_lab1[,A_lab2] B_merged.jsonl B_lab1[,B_lab2] [clean_ids.json]"""
import sys, json, re, collections, numpy as np, pyarrow.parquet as pq
G, J, A, AL, B, BL = sys.argv[1:7]; clean = set(json.load(open(sys.argv[7]))) if len(sys.argv) > 7 else None
gt = {r["video_id"]: r for r in pq.read_table(G).to_pylist()}; J = json.load(open(J)); unj = collections.Counter()
def ok(kind, g, p):
    g, p = g.lower().strip(), p.lower().strip()
    if g == p: return True
    c = J.get(f"{kind}|||{g}|||{p}")
    if c is None: unj[kind] += 1
    return bool(c) and not c.startswith("mismatch")
def iou1(a, b):
    i = max(0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0]))); u = max(float(a[1]), float(b[1])) - min(float(a[0]), float(b[0]))
    return i / u if u > 0 else float(list(a) == list(b))
def tiou(a, b): return max((iou1(x, y) for x in a for y in b if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)) and len(x) == 2 and len(y) == 2), default=0)
def labels(srcs):
    LAB, FIRST = collections.defaultdict(dict), collections.defaultdict(dict)
    for i, src in enumerate(srcs):
        for l in open(src):
            o = json.loads(l)
            if "error" in o: continue
            k = o["kept_obj_ids"]
            try: objs = json.loads(o["text"]).get("objects", [])
            except Exception: continue
            for ob in objs:
                if not isinstance(ob, dict): continue
                for kk, v in ob.items():
                    m = re.match(r"object[ _](\d+)", kk)
                    if m and int(m.group(1)) - 1 < len(k):
                        nm = re.sub(r"\s*\(uncertain\)\s*$", "", str(v).lower().strip()); LAB[o["video_id"]].setdefault(k[int(m.group(1)) - 1], nm)
                        if i == 0: FIRST[o["video_id"]].setdefault(k[int(m.group(1)) - 1], nm)
            if i == 0: FIRST[o["video_id"]].setdefault("__kept__", k)
    return LAB, FIRST
def counts(merged, srcs):
    LAB, FIRST = labels(srcs); C = {}
    for o in map(json.loads, open(merged)):
        v = o["video_id"]
        if "error" in o or v not in gt or (clean and v not in clean): continue
        k = o["kept_obj_ids"]; g = gt[v]
        gl = {int(kk.split("_")[1]): str(x) for ob in json.loads(g["objects"]) for kk, x in ob.items() if kk.startswith("object_")}
        P = collections.defaultdict(list)
        for r in parse_rels(o["text"]):
            try: P[(k[int(r[0]) - 1], k[int(r[2]) - 1])].append((str(r[1]), r[3] if isinstance(r[3], (list, tuple)) else []))
            except Exception: pass
        ng = rh = th = 0; L = LAB[v]
        for s, p, ob, sp, *_ in json.loads(g["relationships"]):
            ng += 1
            if any(ok("relation predicate", p, q) and tiou(qs, sp) >= .5 for q, qs in P.get((s, ob), [])):
                rh += 1
                if s in L and ob in L and ok("object", gl.get(s, ""), L[s]) and ok("object", gl.get(ob, ""), L[ob]): th += 1
        F = FIRST[v]; kept = F.get("__kept__", []); no = sum(1 for x in kept if x in gl); oc = sum(1 for x in kept if x in gl and x in F and ok("object", gl[x], F[x]))
        C[v] = (rh, th, ng, oc, no)
    return C
ca = counts(A, AL.split(",")); ub = dict(unj); unj.clear(); cb = counts(B, BL.split(","))
vids = sorted(set(ca) & set(cb)); a = np.array([ca[v] for v in vids], float); b = np.array([cb[v] for v in vids], float)
def m(x): return np.array([x[:, 0].sum() / max(x[:, 2].sum(), 1), x[:, 1].sum() / max(x[:, 2].sum(), 1), x[:, 3].sum() / max(x[:, 4].sum(), 1)])
rng = np.random.default_rng(0); D = np.array([m(b[i]) - m(a[i]) for i in (rng.integers(0, len(vids), len(vids)) for _ in range(2000))]) * 100
pa, pb = m(a) * 100, m(b) * 100
for j, n in enumerate(("rel-recall", "triplet-recall", "object-acc")):
    lo, hi = np.percentile(D[:, j], [2.5, 97.5]); print(f"  {n}: {pa[j]:5.1f} -> {pb[j]:5.1f}  delta {pb[j]-pa[j]:+5.1f}  95% CI [{lo:+5.1f}, {hi:+5.1f}] {'*' if lo > 0 or hi < 0 else ' '}", end="")
print(f"  | videos {len(vids)} | unjudged name pairs A {ub.get('object', 0)} B {dict(unj).get('object', 0)}")
