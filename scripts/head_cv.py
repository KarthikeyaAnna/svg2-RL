"""Pre-registered head comparison (RESEARCH_LOG §58): GEO (HistGBM on geometry) vs LAT (MLP on TraSeR latents) vs LAT+GEO (MLP on both).
5-fold CV by video on training videos (PVD + VidOR). Primary: recall of positive pairs at K = GEO's tau-.5 count per video. Secondary: AP.
Latents: per layer standardize + PCA(128) fitted on training-fold objects only; pair = [a+b, a*b, |a-b|] per layer + video context.
Usage: head_cv.py FEATS_DIR LAT_DIR"""
import sys, collections, numpy as np, torch
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
FD, LD = sys.argv[1:3]; torch.manual_seed(0); torch.set_num_threads(8)
D = {s: np.load(f"{FD}/feats_{f}.npz", allow_pickle=True) for s, f in (("pvd", "pvd_all"), ("vidor", "vidortrain_all"))}
LAT = {s: torch.load(f"{LD}/lat_train_{s}.pt")["data"] for s in ("pvd", "vidor")}
rows = []                                   # (src, vid, a, b, geo, y)
for s in D:
    X, Y, V, P = D[s]["X"], D[s]["Y"].astype(int), D[s]["V"], D[s]["P"]
    for x, y, v, p in zip(X, Y, V, P):
        L = LAT[s].get(str(v))
        if L is None: continue
        pos = {o: i for i, o in enumerate(L["kept"])}; a, b = int(p[0]), int(p[1])
        if a in pos and b in pos: rows.append((s, str(v), pos[a], pos[b], x, y))
print("pairs with latents", len(rows), "of", sum(len(D[s]["Y"]) for s in D))
src = np.array([r[0] for r in rows]); vid = np.array([f"{r[0]}:{r[1]}" for r in rows]); Y = np.array([r[5] for r in rows])
GEO = np.array([r[4] for r in rows], float)
uv = np.array(sorted(set(vid))); fold_of = dict(zip(uv, np.random.default_rng(0).integers(0, 5, len(uv)))); F = np.array([fold_of[v] for v in vid])
nl = next(iter(LAT["pvd"].values()))["obj"].shape[1]
def lat_feats(tr_mask):
    trv = set(vid[tr_mask]); objs = [[] for _ in range(nl)]
    for s in LAT:
        for v, L in LAT[s].items():
            if f"{s}:{v}" in trv:
                for l in range(nl): objs[l].append(L["obj"][:, l].float().numpy())
    tf = []
    for l in range(nl):
        A = np.concatenate(objs[l]); mu, sd = A.mean(0), A.std(0) + 1e-6
        pca = PCA(128, random_state=0, svd_solver="randomized").fit((A - mu) / sd); tf.append((mu, sd, pca))
    def emb(t, l): mu, sd, pca = tf[l]; return pca.transform((t.float().numpy() - mu) / sd)
    cache = {}
    out = np.zeros((len(rows), nl * 4 * 128), np.float32)
    for i, (s, v, a, b, _, _) in enumerate(rows):
        k = (s, v)
        if k not in cache:
            L = LAT[s][v]; cache = {k: ([emb(L["obj"][:, l], l) for l in range(nl)], [emb(L["ctx"][l][None], l)[0] for l in range(nl)])}
        O, C = cache[k]; f = []
        for l in range(nl): f += [O[l][a] + O[l][b], O[l][a] * O[l][b], np.abs(O[l][a] - O[l][b]), C[l]]
        out[i] = np.concatenate(f)
    return out
def mlp(Xtr, ytr, Xte, epochs=12):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); Xte = torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    m = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(), torch.nn.Dropout(0.3), torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=1e-2); yt = torch.tensor(ytr, dtype=torch.float32)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]; loss = torch.nn.functional.binary_cross_entropy_with_logits(m(Xtr[j]).squeeze(1), yt[j]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad(): return torch.sigmoid(m(Xte).squeeze(1)).numpy()
pred = {k: np.zeros(len(rows)) for k in ("GEO", "LAT", "LAT+GEO")}
for k in range(5):
    tr, te = F != k, F == k
    pred["GEO"][te] = HGB(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, random_state=0).fit(GEO[tr], Y[tr]).predict_proba(GEO[te])[:, 1]
    LX = lat_feats(tr)
    pred["LAT"][te] = mlp(LX[tr], Y[tr], LX[te]); pred["LAT+GEO"][te] = mlp(np.hstack([LX[tr], GEO[tr]]), Y[tr], np.hstack([LX[te], GEO[te]]))
    print("fold", k, "done", flush=True); del LX
np.savez(f"{LD}/head_cv_preds.npz", vid=vid, Y=Y, src=src, **{k.replace("+", "_"): v for k, v in pred.items()})
idx = collections.defaultdict(list)
for i, v in enumerate(vid): idx[v].append(i)
for s in ("pvd", "vidor"):
    R = collections.defaultdict(list); m_ = src == s
    for v, ii in idx.items():
        if not v.startswith(s + ":"): continue
        ii = np.array(ii); y = Y[ii]
        if y.sum() == 0: continue
        K = int((pred["GEO"][ii] >= 0.5).sum())
        for name, p in pred.items(): R[name].append(y[np.argsort(-p[ii])[:K]].sum() / y.sum())
        R["oracle"].append(min(K, y.sum()) / y.sum())
    print(s, f"videos {len(R['GEO'])} | recall@K_GEO: " + " | ".join(f"{k} {100*np.mean(x):.1f}" for k, x in R.items()) +
          " | AP: " + " | ".join(f"{k} {100*average_precision_score(Y[m_], p[m_]):.1f}" for k, p in pred.items()))
