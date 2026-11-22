"""TraSeR vs QS v4 / v5 / v6 on SVG2test-100, VidOR-val, PVSG-val: exact / Gemini-judge lenient / strict, graph and graph+pairs.
Same protocol and verdict cache as score_judge_v4.py (§104). Paired per-video bootstrap of dF1 vs TraSeR.
Usage: score_judge_all.py [need OUT.json]  (with 'need': only list uncached (GT, pred) predicate pairs)"""
import sys, os, json, subprocess, collections, numpy as np
W = "/home/karthik/svg2-killtest"; NEED = len(sys.argv) > 1 and sys.argv[1] == "need"
src = open(f"{W}/scripts/score_judge_v4.py").read(); exec(src[:src.index("A0 = ")])
S = f"{W}/out/score_judge_all"; os.makedirs(S, exist_ok=True)
py = lambda *a: subprocess.run([sys.executable, *a], check=True, capture_output=True)
A0 = {"svg2": f"{W}/out/e1_a0/svg2_full.jsonl", "sav": f"{W}/out/e1_sv/svg2sav_full_A0.jsonl", "vidor": f"{W}/out/e1_a0/vidor_full.jsonl", "pvsg": f"{W}/out/e1_a0p/pvsg_full.jsonl"}
M = {"v4": (f"{W}/out/qtm_v4c2_eval", "v4c2"), "v5": (f"{W}/out/qe_v5_par", "v5"), "v6": (f"{W}/out/qe_v6_par", "v6")}
F = collections.defaultdict(dict)
for bn in ("svg2", "sav", "vidor", "pvsg"):
    py(f"{W}/scripts/dedup.py", A0[bn], f"{S}/{bn}_t.dd"); F[bn]["TraSeR"] = f"{S}/{bn}_t.dd"
    for n, (d, t) in M.items():
        py(f"{W}/scripts/dedup.py", f"{d}/{bn}_graph_{t}.jsonl", f"{S}/{bn}_{n}_g.dd"); F[bn][f"{n} graph"] = f"{S}/{bn}_{n}_g.dd"
        py(f"{W}/scripts/merge_pairs.py", f"{d}/{bn}_graph_{t}.jsonl", f"{d}/{bn}_pairs_{t}.jsonl", "40", f"{S}/{bn}_{n}_gp.raw")
        py(f"{W}/scripts/dedup.py", f"{S}/{bn}_{n}_gp.raw", f"{S}/{bn}_{n}_gp.dd"); F[bn][f"{n} graph+pairs"] = f"{S}/{bn}_{n}_gp.dd"
for name in F["svg2"]:
    out = f"{S}/100_{name.replace(' ', '_').replace('+', 'p')}.dd"; open(out, "w").write(open(F["svg2"][name]).read() + open(F["sav"][name]).read()); F["svg2_100"][name] = out
if NEED:
    need = set()
    for bn in ("svg2_100", "vidor", "pvsg"):
        G = GT[bn]
        for p in F[bn].values():
            for o in map(json.loads, open(p)):
                v = str(o["video_id"])
                if "error" in o or v not in G: continue
                k = o["kept_obj_ids"]
                for r in parse_rels(o["text"]):
                    try: s, pp, ob, sp = k[int(r[0]) - 1], nrm(r[1]), k[int(r[2]) - 1], [[float(x), float(y)] for x, y in r[3]]
                    except Exception: continue
                    for g in G[v]:
                        if g[0] == s and g[2] == ob and nrm(g[1]) != pp and tiou(sp, g[3]) >= .5 and f"relation predicate|||{nrm(g[1])}|||{pp}" not in J:
                            need.add((nrm(g[1]), pp))
    json.dump(sorted(need), open(sys.argv[2], "w")); print("uncached pairs", len(need)); sys.exit()
rng = np.random.default_rng(0)
for judge in (False, "lenient", "strict"):
    print(f"\n===== {'EXACT' if not judge else 'GEMINI JUDGE ' + judge}: P / R / F1 ; dF1 vs TraSeR [95% paired bootstrap CI]")
    for bn in ("svg2_100", "vidor", "pvsg"):
        vids = sorted(GT[bn]); X = {n: np.array([c for _, c in sorted(counts(p, GT[bn], judge).items()) if _ in GT[bn]], float) for n, p in F[bn].items()}
        idx = [rng.integers(0, len(vids), len(vids)) for _ in range(1000)]
        for n, x in X.items():
            m = prf(x); s = f"{bn:9s} {n:15s} {m[0]:5.1f} / {m[1]:5.1f} / {m[2]:5.1f}"
            if n != "TraSeR":
                d = np.array([prf(x[i])[2] - prf(X["TraSeR"][i])[2] for i in idx]); lo, hi = np.percentile(d, [2.5, 97.5])
                s += f"   dF1 {m[2] - prf(X['TraSeR'])[2]:+5.1f} [{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"
            print(s, flush=True)
