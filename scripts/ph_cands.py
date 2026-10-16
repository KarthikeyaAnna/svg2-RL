"""Out-of-fold geometry-ranker probabilities for every pair in TraSeR's training videos (PVD + VidOR-train), 5 folds by video, same
ranker as P1B (HistGBM on pair_feats.py features). Writes {source: {video: [[a, b, p], ...]}} sorted by p, top TOP per video.
Usage: ph_cands.py feats_pvd_all.npz feats_vidortrain_all.npz out.json"""
import sys, json, collections, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
fp, fv, out = sys.argv[1:4]; TOP = 60
D = {s: np.load(f, allow_pickle=True) for s, f in (("pvd", fp), ("vidor", fv))}
X = np.concatenate([D[s]["X"] for s in D]); Y = np.concatenate([D[s]["Y"].astype(int) for s in D])
V = np.concatenate([[f"{s}:{v}" for v in D[s]["V"]] for s in D]); P = np.concatenate([D[s]["P"] for s in D])
uv = np.array(sorted(set(V))); fold = dict(zip(uv, np.random.default_rng(0).integers(0, 5, len(uv)))); F = np.array([fold[v] for v in V])
p = np.zeros(len(Y))
for k in range(5):
    tr, te = F != k, F == k
    p[te] = HGB(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, random_state=0).fit(X[tr], Y[tr]).predict_proba(X[te])[:, 1]
C = collections.defaultdict(lambda: collections.defaultdict(list))
for v, (a, b), q in zip(V, P, p):
    s, vid = v.split(":", 1); C[s][vid].append([int(a), int(b), round(float(q), 5)])
C = {s: {v: sorted(L, key=lambda e: -e[2])[:TOP] for v, L in d.items()} for s, d in C.items()}
json.dump(C, open(out, "w")); print({s: (len(d), np.mean([len(L) for L in d.values()])) for s, d in C.items()})
