"""P(pair stated | touching) vs P(pair stated | non-touching). touching = dilated masks (4x down, 2 px) touch in any sampled frame.
Usage: masks.parquet ann.parquet frame_step max_frames [max_videos]"""
import sys, json, itertools, numpy as np, pyarrow.parquet as pq
from pycocotools import mask as Mk
M, A, STEP, MAXF = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); NV = int(sys.argv[5]) if len(sys.argv) > 5 else 10**9
ann = {r["video_id"]: r for r in pq.read_table(A).to_pylist()}
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
c = dict(t_st=0, t_all=0, n_st=0, n_all=0); nv = 0
for r in pq.ParquetFile(M).iter_batches(batch_size=16):
    for r in r.to_pylist():
        v = str(r["video_id"])
        if v not in ann or nv >= NV: continue
        H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
        fr = list(range(0, nf, STEP)); fr = fr[:: max(1, len(fr) // MAXF)][:MAXF]
        D = np.zeros((no, len(fr), (H + 3) // 4, (W + 3) // 4), bool)
        for j, f in enumerate(fr):
            for o in range(no):
                cnt = r["counts"][f * no + o]
                if cnt: D[o, j] = Mk.decode({"size": [H, W], "counts": cnt})[::4, ::4][: D.shape[2], : D.shape[3]] > 0
        R = ann[v]["relationships"]; R = json.loads(R) if isinstance(R, str) else R
        st = {frozenset((int(x[0]), int(x[2]))) for x in R if int(x[0]) != int(x[2])}
        DL = np.stack([np.stack([dil(D[o, j]) for j in range(len(fr))]) for o in range(no)]) if no else D
        for a, b in itertools.combinations(range(no), 2):
            both = D[a].any((1, 2)) & D[b].any((1, 2))
            if not both.any(): continue
            touch = bool((DL[a] & D[b]).any())
            s = frozenset((a, b)) in st
            if touch: c["t_all"] += 1; c["t_st"] += s
            else: c["n_all"] += 1; c["n_st"] += s
        nv += 1
pt, pn = c["t_st"] / max(1, c["t_all"]), c["n_st"] / max(1, c["n_all"])
print(f"videos {nv}  P(stated|touching) {100*pt:.1f}% ({c['t_st']}/{c['t_all']})  P(stated|non-touching) {100*pn:.1f}% ({c['n_st']}/{c['n_all']})  ratio {pt/max(pn,1e-9):.1f}x  share of stated on touching {100*c['t_st']/max(1,c['t_st']+c['n_st']):.0f}%")
