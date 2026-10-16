"""Apply the co-visibility span constraint (covis_clip.py) to a merged/dedup jsonl in GT-id space.
Usage: covis_apply.py {svg2|vidor|pvsg} IN.jsonl OUT.jsonl. Mask timeline = the frames TraSeR sees: every `stride`-th mask frame at `fps`."""
import sys, os, json, numpy as np, pyarrow.parquet as pq
from pycocotools import mask as M
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from relparse import parse_rels; from covis_clip import apply
W = "/home/karthik/svg2-killtest"; bn, src, dst = sys.argv[1:4]
CFG = {"svg2": (f"{W}/data/masks_vipseg.parquet", 3, lambda v: 2.0),
       "vidor": (f"{W}/data/vidor/prep/masks_vidor.parquet", 1, (lambda m: lambda v: m[v]["fps_eff"])(json.load(open(f"{W}/data/vidor/prep/vidor_meta.json")))),
       "pvsg": (f"{W}/data/pvsg/masks_pvsg.parquet", 1, None)}   # PVSG local masks: full 5 fps annotation -> fps = mask frames / duration
mp, stride, fps = CFG[bn]; NF = {}; DUR = {k: m["duration"] for k, m in json.load(open(f"{W}/kaggle/ds-pvsg/meta.json")).items()} if bn == "pvsg" else {}; rows = [json.loads(l) for l in open(src)]; need = {o["video_id"] for o in rows}
t = pq.read_table(mp); VIS = {}
for i in range(t.num_rows):
    r = t.slice(i, 1).to_pylist()[0]; v = str(r["video_id"])
    if v not in need: continue
    H, Wd, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
    NF[v] = nf
    VIS[v] = {o: np.array([M.area({"size": [H, Wd], "counts": r["counts"][f * no + o]}) > 0 for f in range(0, nf, stride)]) for o in range(no)}
n0 = n1 = 0
with open(dst, "w") as fo:
    for o in rows:
        if "error" in o or o["video_id"] not in VIS: fo.write(json.dumps(o) + "\n"); continue
        k = o["kept_obj_ids"]; rels = []
        for r in parse_rels(o["text"]):
            try: rels.append([k[int(r[0]) - 1], r[1], k[int(r[2]) - 1], r[3] if isinstance(r[3], list) else []])
            except Exception: pass
        v_ = o["video_id"]; out = apply(rels, VIS[v_], fps(v_) if fps else NF[v_] / DUR[v_]); n0 += len(rels); n1 += len(out)
        ix = {x: i + 1 for i, x in enumerate(k)}; o = dict(o)
        o["text"] = json.dumps({"relationships": [[ix[a], p, ix[b], sp] for a, p, b, sp in out]}); fo.write(json.dumps(o) + "\n")
print(bn, "relations", n0, "->", n1, file=sys.stderr)
