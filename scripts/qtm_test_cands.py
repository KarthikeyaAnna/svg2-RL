"""Test-set candidate pairs for the Qwen Token-Marks pair mode: the P1B geometry ranker (HistGBM, trained on all PVD +
VidOR-train pairs, exactly as P1B) scores EVERY co-visible object pair of each test video; top 60 per video with p.
No test labels are used (pair_feats.py computes Y but it is ignored here). Usage: qtm_test_cands.py"""
import json, collections, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
W = "/home/karthik/svg2-killtest"
tr = [np.load(f"{W}/out/feats/{f}.npz", allow_pickle=True) for f in ("feats_pvd_all", "feats_vidortrain_all")]
m = HGB(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, random_state=0).fit(
    np.concatenate([d["X"] for d in tr]), np.concatenate([d["Y"] for d in tr]).astype(int))
out = {}
for bn in ("svg2", "sav", "vidor", "pvsg"):
    D = np.load(f"{W}/out/feats/feats_test_{bn}.npz", allow_pickle=True); p = m.predict_proba(D["X"])[:, 1]
    C = collections.defaultdict(list)
    for v, (a, b), q in zip(D["V"], D["P"], p): C[str(v)].append([int(a), int(b), round(float(q), 5)])
    out[bn] = {v: sorted(L, key=lambda e: -e[2])[:60] for v, L in C.items()}
    print(bn, "videos", len(out[bn]), "mean pairs p>=.5:", round(np.mean([sum(e[2] >= .5 for e in L) for L in out[bn].values()]), 1))
json.dump(out, open(f"{W}/data/ds_qtm/qtm_test_cands.json", "w"))
