import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Judge-free kill-test analysis: density + miss breakdown (ID-matched pairs). Optional judge cache for lenient matching."""
import sys, json, re, collections
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, sys.argv[3]); from geo_verify import canon
outs = [json.loads(l) for l in open(sys.argv[1])]
gt = {r["video_id"]: r for r in pq.read_table(sys.argv[2]).to_pylist()}
J = json.load(open(sys.argv[4])) if len(sys.argv) > 4 else None
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
def norm(p):
    p = p.lower().strip(); p = re.sub(r"^(is|are)\s+", "", p)
    return {"wears":"wearing","wear":"wearing","holds":"holding","hold":"holding","approaches":"approaching","carries":"carrying","ride":"riding","rides":"riding"}.get(p, p)
def match(g, p):
    if norm(g) == norm(p): return "strict"
    if J is not None:
        c = J.get(f"relation predicate|||{g.lower().strip()}|||{p.lower().strip()}")
        return "lenient" if c and not c.startswith("mismatch") else None
    cg, cp = canon(g), canon(p); return "lenient" if cg and cg == cp else None
def secs(spans):
    s = set()
    for ab in spans:
        try: a, b = float(ab[0]), float(ab[1])
        except Exception: continue
        s.update(np.round(np.arange(a, b + 1e-6, 0.5), 1).tolist())
    return s
def tiou(a, b):
    A, B = secs(a), secs(b); return len(A & B) / len(A | B) if (A | B) else 0.0
tot = collections.Counter(); dens = []; npred = 0; pf = 0
for o in outs:
    if "error" in o: tot["videos_error"] += 1; continue
    g = gt[o["video_id"]]; kept = o["kept_obj_ids"]; txt = o.get("text", "")
    rels = parse_rels(txt)
    P = collections.defaultdict(list)
    for r in rels:
        try:
            s, p, ob, sp = int(r[0]), r[1], int(r[2]), r[3]
            P[(kept[s-1] if s > 0 else s, kept[ob-1] if ob > 0 else ob)].append((p, sp))
        except Exception: pass
    npred += len(rels); G = json.loads(g["relationships"]); dens.append((len(rels), len(G))); ks = set(kept)
    for s, p, ob, spans, *_ in G:
        tot["gt"] += 1
        if s not in ks or ob not in ks: tot["unreachable"] += 1; continue
        c = P.get((s, ob), [])
        if not c: tot["pair_missing" + ("_reverse_only" if P.get((ob, s)) else "")] += 1; continue
        m = [(match(p, q), tiou(sp, spans)) for q, sp in c]
        if any(k and t >= .5 for k, t in m): tot["hit"] += 1; tot["hit_strict"] += any(k == "strict" and t >= .5 for k, t in m)
        elif any(k for k, t in m): tot["time_wrong"] += 1
        else: tot["predicate_wrong"] += 1
n = tot["gt"]; d = np.array(dens)
print(f"videos {len(dens)} (errors {tot['videos_error']}, parse-fallback {pf}) | pred rels/video {d[:,0].mean():.1f} vs GT {d[:,1].mean():.1f} | pair-level precision proxy {100*tot['hit']/max(npred,1):.1f}%")
for k in ["hit","predicate_wrong","time_wrong","pair_missing","pair_missing_reverse_only","unreachable"]: print(f"  {k:28s} {tot[k]:5d} {100*tot[k]/n:5.1f}%")
print(f"RELATION RECALL ({'judge' if J else 'canon-synonym'} lenient, tIoU>=.5): {100*tot['hit']/n:.1f}%  | strict {100*tot['hit_strict']/n:.1f}%")
