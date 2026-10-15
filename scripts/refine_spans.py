"""Training-free temporal refinement: intersect each predicted span with the 1 Hz frames where subject/object masks touch
(only if they touch somewhere inside the span; otherwise leave the span unchanged). Usage: in.jsonl masks.parquet mask_fps out.jsonl"""
import sys, json, os, numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from relparse import parse_rels
src, mp, fps, dst = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
MODE = os.environ.get("REFINE", "fragments")  # fragments | envelope
import re as _re
ONLY = _re.compile(os.environ["REFINE_ONLY"]) if os.environ.get("REFINE_ONLY") else None
mt = pq.read_table(mp); rows = {mt.slice(i, 1).to_pylist()[0]["video_id"]: i for i in range(mt.num_rows)}
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
changed = total = 0
with open(dst, "w") as fo:
    for o in map(json.loads, open(src)):
        if "error" in o or o["video_id"] not in rows: fo.write(json.dumps(o) + "\n"); continue
        row = mt.slice(rows[o["video_id"]], 1).to_pylist()[0]; H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
        step = max(1, int(round(fps))); fr = list(range(0, nf, step)); ts = [f / fps for f in fr]; cache = {}
        def ms(x):
            if x not in cache:
                cache[x] = [M.decode({"size": [H, W], "counts": row["counts"][f*no + x]})[::4, ::4].astype(bool) if 0 <= x < no and M.area({"size": [H, W], "counts": row["counts"][f*no + x]}) > 0 else None for f in fr]
            return cache[x]
        k = o["kept_obj_ids"]; out = []
        for r in parse_rels(o["text"]):
            try: s, ob = k[int(r[0]) - 1], k[int(r[2]) - 1]
            except Exception: out.append(r); continue
            total += 1
            if ONLY and not ONLY.search(r[1].lower()): out.append(r); continue
            A, B = ms(s), ms(ob)
            touch = [a is not None and b is not None and bool((dil(a) & b).any()) for a, b in zip(A, B)]
            new = []
            for sp in r[3]:
                if len(sp) != 2: continue
                a0, b0 = float(sp[0]), float(sp[1]); inside = [t for t, c in zip(ts, touch) if c and a0 - 0.5 <= t <= b0 + 0.5]
                if not inside: new.append([a0, b0]); continue
                if MODE == "envelope": new.append([min(inside), max(inside)]); continue
                # contiguous touching segments inside the span (gaps > 1.5 s split)
                seg = [inside[0], inside[0]]
                for t in inside[1:]:
                    if t - seg[1] <= 1.5: seg[1] = t
                    else: new.append(seg); seg = [t, t]
                new.append(seg)
            if new != [[float(x), float(y)] for x, y in r[3] if True]: changed += 1
            out.append([r[0], r[1], r[2], new or r[3]])
        o = dict(o); o["text"] = json.dumps({"relationships": out}); fo.write(json.dumps(o) + "\n")
print(f"relations {total}, spans changed {changed} ({100*changed/max(total,1):.0f}%)", file=sys.stderr)
