"""Paired per-video bootstrap of exact graph F1 (score_exact_graph.py protocol) between model outputs.
Usage: score_paired.py BENCH NAME=DIR:TAG [NAME=DIR:TAG ...]   (BENCH in svg2_100, svg2, sav, vidor, pvsg). Deltas vs the first model."""
import sys, os, json, subprocess, numpy as np, pyarrow.parquet as pq
W = "/home/karthik/svg2-killtest"; sys.argv = [sys.argv[0], "x", "x"] + sys.argv[1:]
src = open(f"{W}/scripts/score_exact_graph.py").read(); exec(src[:src.index("gtp = ")])     # nrm, tiou, counts, prf
gtp = {"svg2": f"{W}/data/gt/vipseg/gt_vipseg.parquet", "sav": f"{W}/data/gt/sav/gt_sav.parquet", "vidor": f"{W}/data/gt/vidor/gt_vidor.parquet", "pvsg": f"{W}/data/gt/pvsg/gt.parquet"}
bench = sys.argv[3]; parts = ["svg2", "sav"] if bench == "svg2_100" else [bench]
GT = {}
GTP = {b: {str(r["video_id"]): json.loads(r["relationships"]) for r in pq.read_table(gtp[b]).to_pylist()} for b in parts}
for b in parts: GT.update(GTP[b])
vids = sorted(GT); X = {}
for spec in sys.argv[4:]:
    name, rest = spec.split("="); d, tag = rest.split(":"); dd = f"{W}/out/score_paired_tmp"; os.makedirs(dd, exist_ok=True)
    C = {}
    for b in parts:
        out = f"{dd}/{name}_{b}.dd"; subprocess.run([sys.executable, f"{W}/scripts/dedup.py", f"{d}/{b}_graph_{tag}.jsonl", out], check=True, stdout=subprocess.DEVNULL)
        C.update(counts(out, GTP[b]))
    X[name] = np.array([C[v] for v in vids], float)
rng = np.random.default_rng(0); idx = [rng.integers(0, len(vids), len(vids)) for _ in range(2000)]
names = list(X); base = names[0]
for n in names:
    m = prf(X[n]); line = f"{bench:9s} {n:8s} P/R/F1 {m[0]:5.1f}/{m[1]:5.1f}/{m[2]:5.1f}  preds {int(X[n][:,1].sum()):6d}  TP {int(X[n][:,0].sum()):5d}"
    if n != base:
        dl = np.array([prf(X[n][i])[2] - prf(X[base][i])[2] for i in idx]); lo, hi = np.percentile(dl, [2.5, 97.5])
        better = int(((X[n][:, 0] - X[base][:, 0]) > 0).sum()); worse = int(((X[n][:, 0] - X[base][:, 0]) < 0).sum())
        line += f"  dF1 vs {base} {m[2]-prf(X[base])[2]:+.1f} [{lo:+.1f}, {hi:+.1f}]  more TP on {better}, fewer on {worse}"
    print(line)
