import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Drop TraSeR relations whose subject/object masks never touch (a-priori rule, no tuning)."""
import sys, json, re, itertools, os
import numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
src, mp, dst = sys.argv[1], sys.argv[2], sys.argv[3]
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
mt = pq.read_table(mp); rows = {mt.slice(i,1).to_pylist()[0]["video_id"]: i for i in range(mt.num_rows)}
kept_n = dropped = 0
with open(dst, "w") as fo:
    for o in map(json.loads, open(src)):
        if "error" in o: fo.write(json.dumps(o) + "\n"); continue
        row = mt.slice(rows[o["video_id"]], 1).to_pylist()[0]; H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
        kept = o["kept_obj_ids"]
        rels = parse_rels(o["text"])
        frames = list(range(0, nf, int(os.environ.get("MASK_STEP", "6")))); cache = {}
        def masks(ob):
            if ob not in cache:
                cache[ob] = [M.decode({"size": [H, W], "counts": row["counts"][f*no+ob]})[::4, ::4].astype(bool) if M.area({"size": [H, W], "counts": row["counts"][f*no+ob]}) > 0 else None for f in frames]
            return cache[ob]
        out = []
        for r in rels:
            try: a, b = kept[int(r[0])-1], kept[int(r[2])-1]
            except Exception: out.append(r); continue
            A, B = masks(a), masks(b)
            touch = any(x is not None and y is not None and (dil(x) & y).any() for x, y in zip(A, B))
            if touch: out.append(r); kept_n += 1
            else: dropped += 1
        o = dict(o); o["text"] = json.dumps({"relationships": out}); fo.write(json.dumps(o) + "\n")
print("kept", kept_n, "dropped", dropped, file=sys.stderr)
