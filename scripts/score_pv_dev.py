"""Pre-registered PV dev gate (RESEARCH_LOG §51): BASE vs VERIFY vs REPLACE on held-out training videos, relation F1 (engine/reward.py).
Usage: score_pv_dev.py PV_OUT_DIR"""
import sys, os, json, collections
import numpy as np, pyarrow.parquet as pq
W = "/home/karthik/svg2-killtest"; sys.path[:0] = [f"{W}/scripts", f"{W}/engine"]
from pv_apply import apply
from reward import reward
d = sys.argv[1]
full = [json.loads(l) for l in open(f"{d}/dev_full.jsonl")]; pairs = [json.loads(l) for l in open(f"{d}/dev_pairs_mentioned.jsonl")]
LAB = {s: {str(r["video_id"]): r for r in pq.read_table(f"{W}/data/ds_train_e1/train_{s}/labels_train.parquet").to_pylist()} for s in ("vidor", "pvd")}
rows = {"base": full, "verify": apply("verify", full, pairs)[0], "replace": apply("replace", full, pairs)[0]}
F = collections.defaultdict(lambda: collections.defaultdict(list))
for mode, R in rows.items():
    for o in R:
        L = LAB[o["source"]][o["video_id"]]
        _, dd = reward(o["text"], o["kept_obj_ids"], json.loads(L["objects"]), json.loads(L["relationships"]))
        for k in ("f1", "p", "r", "n_pred"): F[(mode, o["source"])][k].append(dd[k])
rng = np.random.default_rng(0)
for src in ("vidor", "pvd"):
    b = np.array(F[("base", src)]["f1"])
    for mode in ("base", "verify", "replace"):
        x = {k: np.mean(v) for k, v in F[(mode, src)].items()}; dlt = np.array(F[(mode, src)]["f1"]) - b
        bs = [dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(2000)]
        print(f"{src:5s} {mode:7s} n={len(b)} F1 {100*x['f1']:5.1f}  P {100*x['p']:5.1f}  R {100*x['r']:5.1f}  rels {x['n_pred']:5.1f}  "
              f"dF1 {100*dlt.mean():+5.1f} [{100*np.percentile(bs, 2.5):+5.1f}, {100*np.percentile(bs, 97.5):+5.1f}]")
