"""Geometry-only pair features from GT masks at 1 Hz. Usage: masks.parquet gt.parquet fps out.npz"""
import sys, json, itertools, numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
mp, gp, fps, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
gt = {r["video_id"]: r for r in pq.read_table(gp).to_pylist()}
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
t = pq.read_table(mp); X = []; Y = []; V = []; P = []
for i in range(t.num_rows):
    row = t.slice(i, 1).to_pylist()[0]; vid = row["video_id"]
    if vid not in gt: continue
    H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
    step = max(1, int(round(fps))); fr = list(range(0, nf, step)); ds = 4 if W <= 1000 else 8
    D, DL, BB, AR, CY = {}, {}, {}, {}, {}
    for ob in range(no):
        D[ob], DL[ob], BB[ob], AR[ob], CY[ob] = [], [], [], [], []
        for f in fr:
            rle = {"size": [H, W], "counts": row["counts"][f*no + ob]}
            if M.area(rle) == 0: D[ob].append(None); DL[ob].append(None); BB[ob].append(None); continue
            m = M.decode(rle)[::ds, ::ds].astype(bool); ys, xs = np.nonzero(m)
            if len(ys) == 0: D[ob].append(None); DL[ob].append(None); BB[ob].append(None); continue
            D[ob].append(m); DL[ob].append(dil(m)); BB[ob].append((xs.min(), ys.min(), xs.max(), ys.max())); AR[ob].append(len(ys)); CY[ob].append(ys.mean())
    img = (H // ds) * (W // ds)
    G = {frozenset((s, o)) for s, p, o, *_ in json.loads(gt[vid]["relationships"]) if s != o}
    pairs = list(itertools.combinations(range(no), 2)); feats = {}
    for a, b in pairs:
        co = [j for j in range(len(fr)) if D[a][j] is not None and D[b][j] is not None]
        if not co: feats[(a, b)] = None; continue
        contact = np.mean([bool((DL[a][j] & D[b][j]).any()) for j in co])
        def inb(x, y, j):
            bx = BB[y][j]; ys, xs = np.nonzero(D[x][j]); return ((xs >= bx[0]) & (xs <= bx[2]) & (ys >= bx[1]) & (ys <= bx[3])).mean()
        cont = max(np.mean([inb(a, b, j) for j in co]), np.mean([inb(b, a, j) for j in co]))
        Aa, Ab = np.mean(AR[a]), np.mean(AR[b]); ratio = np.log(min(Aa, Ab) / max(Aa, Ab))
        size = np.log(np.sqrt(Aa * Ab) / img + 1e-6)
        dist = np.mean([np.hypot(np.mean(np.nonzero(D[a][j])[1]) - np.mean(np.nonzero(D[b][j])[1]), np.mean(np.nonzero(D[a][j])[0]) - np.mean(np.nonzero(D[b][j])[0])) for j in co]) / np.sqrt(Aa + Ab)
        dy = abs(np.mean(CY[a]) - np.mean(CY[b])) / np.sqrt(img)
        def biou(j):
            x, y = BB[a][j], BB[b][j]; iw = max(0, min(x[2], y[2]) - max(x[0], y[0])); ih = max(0, min(x[3], y[3]) - max(x[1], y[1]))
            u = (x[2]-x[0])*(x[3]-x[1]) + (y[2]-y[0])*(y[3]-y[1]) - iw*ih; return iw*ih / u if u > 0 else 0
        feats[(a, b)] = [contact, len(co) / len(fr), cont, ratio, size, dist, dy, np.mean([biou(j) for j in co]), np.log(no)]
    # partner rank by contact (per object), min over the two endpoints
    rank = {}
    for ob in range(no):
        part = sorted([(feats[tuple(sorted((ob, x)))][0], x) for x in range(no) if x != ob and feats.get(tuple(sorted((ob, x)))) is not None], reverse=True)
        for r_, (c, x) in enumerate(part): rank[(ob, x)] = r_ if c > 0 else 99
    for (a, b), f in feats.items():
        if f is None: continue
        X.append(f + [min(rank.get((a, b), 99), rank.get((b, a), 99)), np.log1p(min(rank.get((a, b), 99), rank.get((b, a), 99)))]); Y.append(frozenset((a, b)) in G); V.append(vid); P.append((a, b))
np.savez(out, X=np.array(X, float), Y=np.array(Y), V=np.array(V), P=np.array(P)); print(out, len(X), "pairs, positives", int(np.sum(Y)))
