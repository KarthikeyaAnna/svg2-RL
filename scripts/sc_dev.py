"""Pre-registered SC dev gate (RESEARCH_LOG §63): each aggregator vs the same model's greedy graph on held-out training videos,
relation F1 (engine/reward.py). Gate: mean dF1 >= 0 on both sources and pooled 95% CI above 0. Also the two §63 diagnostics.
Usage: sc_dev.py dev_samples.jsonl"""
import sys, json, collections
import numpy as np, pyarrow.parquet as pq
W = "/home/karthik/svg2-killtest"; sys.path[:0] = [f"{W}/scripts", f"{W}/engine"]
from sc_agg import load, aggregate, rels
from reward import reward, tiou
S = load(sys.argv[1])
LAB = {s: {str(r["video_id"]): r for r in pq.read_table(f"{W}/data/ds_train_e1/train_{s}/labels_train.parquet").to_pylist()} for s in ("vidor", "pvd")}
def lab(src, v): L = LAB[src][v]; return json.loads(L["objects"]), json.loads(L["relationships"])
F = {}
for mode in ("greedy", "vote3", "vote5", "vote7", "mbr"):
    F[mode] = {}
    for o in aggregate(S, mode):
        ob, rl = lab(o["source"], o["video_id"]); _, d = reward(o["text"], o["kept_obj_ids"], ob, rl)
        F[mode][(o["source"], o["video_id"])] = (d["f1"], d["p"], d["r"], d["n_pred"])
keys = sorted(F["greedy"]); rng = np.random.default_rng(0); res = {}
print(f"videos {len(keys)} | samples per video {collections.Counter(len(x) for x in S.values())}")
for mode in F:
    line = f"{mode:7s}"
    dl = np.array([F[mode][k][0] - F["greedy"][k][0] for k in keys]); ok = True
    for src in ("pvd", "vidor"):
        kk = [k for k in keys if k[0] == src]; x = np.array([F[mode][k] for k in kk]).mean(0); d = np.mean([F[mode][k][0] - F["greedy"][k][0] for k in kk])
        line += f" | {src} F1 {100*x[0]:5.1f} P {100*x[1]:5.1f} R {100*x[2]:5.1f} rels {x[3]:5.1f} dF1 {100*d:+5.1f}"; ok &= d >= 0
    bs = [dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(2000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
    ok &= lo > 0; res[mode] = (dl.mean(), ok)
    print(line + f" | pooled dF1 {100*dl.mean():+5.2f} [{100*lo:+5.2f}, {100*hi:+5.2f}] {'PASS' if ok and mode != 'greedy' else ''}")
win = [m for m, (d, ok) in res.items() if ok and m != "greedy"]
print("WINNER:", max(win, key=lambda m: res[m][0]) if win else "none (SC not adopted)")
# diagnostics: (a) union over all graphs of label-relation recall (exact predicate, tIoU >= .5); (b) vote count of greedy relations by outcome
U = collections.Counter(); C = collections.defaultdict(list)
for (src, v), Sm in S.items():
    kept = Sm[0]["kept_obj_ids"]; n = len(kept); ob, rl = lab(src, v); idx = {o: i + 1 for i, o in enumerate(kept)}
    GL = [(idx[int(r[0])], str(r[1]).lower().strip(), idx[int(r[2])], r[3]) for r in rl if int(r[0]) in idx and int(r[2]) in idx]
    G = {k: rels(Sm[k]["text"], n) for k in Sm}; lp = {(s, o) for s, _, o, _ in GL}
    for s, p, o, sp in GL:
        U[src, "label"] += 1
        U[src, "greedy"] += any(q == p and (a, b) == (s, o) and tiou(qs, sp) >= .5 for a, q, b, qs in G[0])
        U[src, "union"] += any(q == p and (a, b) == (s, o) and tiou(qs, sp) >= .5 for k in G for a, q, b, qs in G[k])
    cnt = collections.Counter(key for k in G for key in {(s, p, o) for s, p, o, _ in G[k]})
    for s, p, o, sp in {(s, p, o): (s, p, o, sp) for s, p, o, sp in G[0]}.values():
        hit = any(q == p and (a, b) == (s, o) and tiou(sp, qs) >= .5 for a, q, b, qs in GL)
        C[src, "correct" if hit else ("labelled pair" if (s, o) in lp else "no-label pair")].append(cnt[(s, p, o)])
for src in ("pvd", "vidor"):
    print(f"{src}: label relations {U[src,'label']} | greedy recall {100*U[src,'greedy']/max(1,U[src,'label']):.1f} | union-of-graphs recall {100*U[src,'union']/max(1,U[src,'label']):.1f}")
    print("   greedy relations, mean vote count (of", len(next(iter(S.values()))), "graphs):", " | ".join(f"{c} {np.mean(C[src, c]):.2f} (n={len(C[src, c])})" for c in ("correct", "labelled pair", "no-label pair")))
