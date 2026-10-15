"""Re-rank pair decodes by containment (max over directions of fraction of smaller-object pixels inside the other's bbox), from masks at 1 Hz."""
import sys, json, numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
src, mp, fps, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
mt = pq.read_table(mp); rows = {mt.slice(i,1).to_pylist()[0]["video_id"]: i for i in range(mt.num_rows)}
by = {}
for r in map(json.loads, open(src)): by.setdefault(r["video_id"], []).append(r)
with open(dst, "w") as fo:
    for v, L in by.items():
        row = mt.slice(rows[v], 1).to_pylist()[0]; H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
        fr = list(range(0, nf, max(1, int(round(fps))))); cache = {}
        def ms(x):
            if x not in cache: cache[x] = [M.decode({"size":[H,W],"counts":row["counts"][f*no+x]})[::4,::4].astype(bool) if M.area({"size":[H,W],"counts":row["counts"][f*no+x]})>0 else None for f in fr]
            return cache[x]
        def inb(a, b):
            vals = []
            for x, y in zip(ms(a), ms(b)):
                if x is None or y is None: continue
                ys, xs = np.nonzero(y); yy, xx = np.nonzero(x)
                if len(ys) == 0 or len(yy) == 0: continue
                vals.append(((xx >= xs.min()) & (xx <= xs.max()) & (yy >= ys.min()) & (yy <= ys.max())).mean())
            return np.mean(vals) if vals else 0.0
        sc = []
        for i, r in enumerate(L):
            a, b = r["pair"]; sc.append((max(inb(a, b), inb(b, a)), -r.get("rank", i), i))
        for _, _, i in sorted(sc, reverse=True): fo.write(json.dumps(L[i]) + "\n")
