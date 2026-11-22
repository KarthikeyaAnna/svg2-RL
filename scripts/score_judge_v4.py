"""TraSeR vs QS v4 (graph, graph + pair answers) on SVG2test-100 / VidOR / PVSG, exact AND Gemini-judge mode, same protocol as
score_exact_graph.py (dedup, per-span tIoU >= .5, normalised predicates, pooled counts) + paired per-video bootstrap of dF1 vs TraSeR.
Judge mode: match = normalised equal OR Gemini verdict (out/judge_gemini/judge_gemini.json) not "mismatch". ONE judge for all systems
(no older Qwen3-8B verdicts mixed in). Usage: score_judge_v4.py"""
import sys, os, json, subprocess, collections, numpy as np, pyarrow.parquet as pq
W = "/home/karthik/svg2-killtest"; sys.path[:0] = [f"{W}/scripts"]; from relparse import parse_rels
S = f"{W}/out/score_judge_v4"; os.makedirs(S, exist_ok=True); D = f"{W}/out/qtm_v4c2_eval"
J = json.load(open(f"{W}/out/judge_gemini/judge_gemini.json")) if os.path.exists(f"{W}/out/judge_gemini/judge_gemini.json") else {}
nrm = lambda s: " ".join(str(s).lower().replace("_", " ").split())
def ok(g, p, judge):
    g = nrm(g)
    if g == p: return True
    if not judge: return False
    c = J.get(f"relation predicate|||{g}|||{p}")
    if c is None: return False
    return c in ("identical", "synonym", "hypernym/hyponym") if judge == "strict" else not c.startswith("mismatch")
def iou1(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0])); u = max(a[1], b[1]) - min(a[0], b[0]); return i / u if u > 0 else float(a == b)
tiou = lambda x, y: max((iou1(a, b) for a in x for b in y if len(a) == 2 and len(b) == 2), default=0)
def counts(path, gt, judge):
    C = {}
    for o in map(json.loads, open(path)):
        v = str(o["video_id"])
        if "error" in o or v not in gt: continue
        k = o["kept_obj_ids"]; M = collections.OrderedDict()
        for r in parse_rels(o["text"]):
            try: M.setdefault((k[int(r[0]) - 1], nrm(r[1]), k[int(r[2]) - 1]), []).extend([[float(x), float(y)] for x, y in r[3]])
            except Exception: pass
        G = gt[v]; tp = 0; hit = set()
        for (s, p, ob), sp in M.items():
            m = [i for i, g in enumerate(G) if g[0] == s and g[2] == ob and ok(g[1], p, judge) and tiou(sp, g[3]) >= .5]
            tp += bool(m); hit.update(m)
        C[v] = (tp, len(M), len(hit), len(G))
    for v in gt: C.setdefault(v, (0, 0, 0, len(gt[v])))
    return C
def prf(x): P = x[:, 0].sum() / max(x[:, 1].sum(), 1); R = x[:, 2].sum() / max(x[:, 3].sum(), 1); return np.array([P, R, 2 * P * R / (P + R + 1e-9)]) * 100
gtp = {"svg2": f"{W}/data/gt/vipseg/gt_vipseg.parquet", "sav": f"{W}/data/gt/sav/gt_sav.parquet", "vidor": f"{W}/data/gt/vidor/gt_vidor.parquet", "pvsg": f"{W}/data/gt/pvsg/gt.parquet"}
GT = {b: {str(r["video_id"]): json.loads(r["relationships"]) for r in pq.read_table(p).to_pylist()} for b, p in gtp.items()}
GT["svg2_100"] = {**GT["svg2"], **GT["sav"]}
A0 = {"svg2": f"{W}/out/e1_a0/svg2_full.jsonl", "sav": f"{W}/out/e1_sv/svg2sav_full_A0.jsonl", "vidor": f"{W}/out/e1_a0/vidor_full.jsonl", "pvsg": f"{W}/out/e1_a0p/pvsg_full.jsonl"}
py = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True)
F = collections.defaultdict(dict)
for bn in ("svg2", "sav", "vidor", "pvsg"):
    py(f"{W}/scripts/dedup.py", A0[bn], f"{S}/{bn}_t.dd"); F[bn]["TraSeR"] = f"{S}/{bn}_t.dd"
    py(f"{W}/scripts/dedup.py", f"{D}/{bn}_graph_v4c2.jsonl", f"{S}/{bn}_g.dd"); F[bn]["v4 graph"] = f"{S}/{bn}_g.dd"
    py(f"{W}/scripts/merge_pairs.py", f"{D}/{bn}_graph_v4c2.jsonl", f"{D}/{bn}_pairs_v4c2.jsonl", "40", f"{S}/{bn}_gp.raw")
    py(f"{W}/scripts/dedup.py", f"{S}/{bn}_gp.raw", f"{S}/{bn}_gp.dd"); F[bn]["v4 graph+pairs"] = f"{S}/{bn}_gp.dd"
for name in F["svg2"]:
    out = f"{S}/100_{name.replace(' ', '_').replace('+', 'p')}.dd"; open(out, "w").write(open(F["svg2"][name]).read() + open(F["sav"][name]).read()); F["svg2_100"][name] = out
rng = np.random.default_rng(0)
for judge in (False, "lenient", "strict"):
    title = {False: "EXACT (normalised)", "lenient": "GEMINI JUDGE, lenient (old rule: all but Mismatch)", "strict": "GEMINI JUDGE, strict (identical/synonym/hypernym only)"}[judge]
    print(f"\n===== {title}: P / R / F1 ; dF1 vs TraSeR [95% paired bootstrap CI]")
    for bn in ("svg2_100", "vidor", "pvsg"):
        vids = sorted(GT[bn]); X = {n: np.array([v for _, v in sorted(counts(p, GT[bn], judge).items()) if _ in GT[bn]], float) for n, p in F[bn].items()}
        idx = [rng.integers(0, len(vids), len(vids)) for _ in range(2000)]
        for n, x in X.items():
            m = prf(x); s = f"{bn:9s} {n:15s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.1f}"
            if n != "TraSeR":
                d = np.array([prf(x[i])[2] - prf(X["TraSeR"][i])[2] for i in idx]); lo, hi = np.percentile(d, [2.5, 97.5])
                s += f"   dF1 {m[2] - prf(X['TraSeR'])[2]:+5.1f} [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"
            print(s)
