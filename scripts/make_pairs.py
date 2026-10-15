import sys, json, itertools
import numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
src, mp, N, dst = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
outs = {o["video_id"]: o for o in map(json.loads, open(src))}
res = {}; t = pq.read_table(mp)
for i in range(t.num_rows):
    row = t.slice(i, 1).to_pylist()[0]; vid = row["video_id"]; o = outs.get(vid, {})
    H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
    kept = o.get("kept_obj_ids") or list(range(min(no, 40))); have = set()
    try:
        for r in json.loads(o["text"])["relationships"]: have.add(frozenset((kept[int(r[0])-1], kept[int(r[2])-1])))
    except Exception: pass
    frames = list(range(0, nf, 6)); D = {}; DL = {}; A = {}
    for ob in range(no):
        D[ob] = []; A[ob] = 0
        for f in frames:
            rle = {"size": [H, W], "counts": row["counts"][f*no + ob]}
            m = M.decode(rle)[::4, ::4].astype(bool) if M.area(rle) > 0 else None; D[ob].append(m); A[ob] += 0 if m is None else int(m.sum())
        DL[ob] = [dil(m) if m is not None else None for m in D[ob]]
    sc = {}
    for a, b in itertools.combinations(range(no), 2):
        if frozenset((a, b)) in have: continue
        c = [bool((DL[a][j] & D[b][j]).any()) for j in range(len(frames)) if D[a][j] is not None and D[b][j] is not None]
        if c and np.mean(c) > 0: sc[(a, b)] = np.mean(c) + 1e-9 * (A[a] + A[b])
    res[vid] = [list(p) for p, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:N]]
    del D, DL
json.dump(res, open(dst, "w")); print("videos", len(res), "pairs", sum(map(len, res.values())))
