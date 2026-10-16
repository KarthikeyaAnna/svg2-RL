import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Where relations are lost (GT side) and where false positives come from (prediction side). Same matching as bootstrap.py
(ordered pair, judge-equivalent predicate, per-span tIoU >= .5; exact duplicates merged upstream by dedup.py).
Usage: error_breakdown.py merged.dd.jsonl gt.parquet judge.json [base_full.jsonl]
  base_full: optional base graph; predicted relations on pairs the base graph does not mention are reported as "added"."""
import sys, json, collections, pyarrow.parquet as pq
M_, G_, J_ = sys.argv[1:4]; base = sys.argv[4] if len(sys.argv) > 4 else None
gt = {r["video_id"]: r for r in pq.read_table(G_).to_pylist()}; J = json.load(open(J_))
def ok(g, p):
    g, p = g.lower().strip(), p.lower().strip(); c = J.get(f"relation predicate|||{g}|||{p}"); return g == p or (c is not None and not c.startswith("mismatch"))
def iou1(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else float(a == b)
def tiou(x, y): return max((iou1(a, b) for a in x for b in y if len(a) == 2 and len(b) == 2), default=0)
def rels(o):
    k = o["kept_obj_ids"]; out = []
    for r in parse_rels(o["text"]):
        try:
            sp = [[float(a), float(b)] for a, b in r[3]] if isinstance(r[3], (list, tuple)) else []
            out.append((k[int(r[0]) - 1], str(r[1]).lower().strip(), k[int(r[2]) - 1], sp))
        except Exception: pass
    return out
basepairs = {}
if base:
    for o in map(json.loads, open(base)): basepairs[o["video_id"]] = {frozenset((s, ob)) for s, _, ob, _ in rels(o)}
miss, fp = collections.Counter(), collections.Counter(); fp_src = collections.defaultdict(collections.Counter); ng = npred = 0
for o in map(json.loads, open(M_)):
    v = o["video_id"]
    if "error" in o or v not in gt: continue
    P = rels(o); kept = set(o["kept_obj_ids"]); Gr = json.loads(gt[v]["relationships"]); ng += len(Gr); npred += len(P)
    byo = collections.defaultdict(list)
    for s, p, ob, sp in P: byo[(s, ob)].append((p, sp))
    for s, p, ob, sp, *_ in Gr:
        if s not in kept or ob not in kept: miss["object not in input"] += 1
        elif any(ok(p, q) and tiou(qs, sp) >= .5 for q, qs in byo.get((s, ob), [])): miss["found"] += 1
        elif any(ok(p, q) for q, qs in byo.get((s, ob), [])): miss["right relation, wrong time"] += 1
        elif byo.get((s, ob)): miss["right pair, wrong predicate"] += 1
        elif byo.get((ob, s)): miss["pair stated only in other direction"] += 1
        else: miss["pair never mentioned"] += 1
    go = collections.defaultdict(list)
    for s, p, ob, sp, *_ in Gr: go[(s, ob)].append((p, sp))
    for s, p, ob, sp in P:
        src = "added" if base and frozenset((s, ob)) not in basepairs.get(v, set()) else "base"
        if any(ok(g, p) and tiou(sp, gs) >= .5 for g, gs in go.get((s, ob), [])): c = "correct"
        elif any(ok(g, p) for g, gs in go.get((s, ob), [])): c = "right relation, wrong time"
        elif go.get((s, ob)): c = "labelled pair, wrong predicate"
        elif go.get((ob, s)): c = "pair labelled only in other direction"
        else: c = "pair has no GT relation"
        fp[c] += 1; fp_src[src][c] += 1
print(f"GT relations {ng}: " + " | ".join(f"{k} {100*v/ng:.1f}" for k, v in sorted(miss.items(), key=lambda x: -x[1])))
print(f"predicted {npred}: " + " | ".join(f"{k} {100*v/npred:.1f}" for k, v in sorted(fp.items(), key=lambda x: -x[1])))
for src, C in sorted(fp_src.items()):
    n = sum(C.values()); print(f"  {src:5s} {n:6d} preds: " + " | ".join(f"{k} {100*v/n:.1f}" for k, v in sorted(C.items(), key=lambda x: -x[1])))
