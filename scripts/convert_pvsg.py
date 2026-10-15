"""PVSG val -> SVG2-style gt_pvsg.parquet + masks_pvsg.parquet + JPEG frames (5 fps)."""
import json, zipfile, io, os, subprocess, sys, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as M
R = sys.argv[1]; d = json.load(open(f"{R}/pvsg.json")); FPS = 5.0
src = {v: s for s, sp in d["split"].items() for v in sp["val"]}
zips = {"vidor": "vidor_masks.zip", "epic_kitchen": "epic_kitchen_masks.zip", "ego4d": "ego4d_masks.zip"}
Z = {k: zipfile.ZipFile(f"{R}/{v}") for k, v in zips.items()}
names = {k: {} for k in Z}
for k, z in Z.items():
    for n in z.namelist():
        if n.endswith(".png"): names[k].setdefault(n.split("/")[-2], []).append(n)
data = {x["video_id"]: x for x in d["data"]}
gt_rows, mask_rows = [], []
for vid, s in sorted(src.items()):
    x = data[vid]; objs = x["objects"]; ids = sorted(o["object_id"] for o in objs); idx = {oid: i for i, oid in enumerate(ids)}
    pngs = sorted(names[s].get(vid, []))
    if not pngs: print("NO MASKS", vid); continue
    fdir = f"{R}/frames/{vid}"; os.makedirs(fdir, exist_ok=True)
    if not os.listdir(fdir):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", f"{R}/videos/{vid}.mp4", "-vf", "fps=5", "-q:v", "3", f"{fdir}/%05d.jpg"], check=True)
    nj = len(os.listdir(fdir)); nf = min(nj, len(pngs))
    first = np.array(Image.open(io.BytesIO(Z[s].read(pngs[0])))); H, W = first.shape[:2]
    counts = []
    for f in range(nf):
        a = np.array(Image.open(io.BytesIO(Z[s].read(pngs[f]))))
        for oid in ids:
            counts.append(M.encode(np.asfortranarray((a == oid).astype(np.uint8)))["counts"].decode())
    mask_rows.append(dict(video_id=vid, mask_height=H, mask_width=W, num_frames=nf, num_objects=len(ids), counts=counts))
    gobjs = [{f"object_{idx[o['object_id']]}": o["category"], "attributes": []} for o in objs]
    rels = [[idx[r[0]], r[2], idx[r[1]], [[a / FPS, b / FPS] for a, b in r[3]], ""] for r in x["relations"] if r[0] in idx and r[1] in idx]
    gt_rows.append(dict(video_id=vid, source=f"PVSG-{s}", objects=json.dumps(gobjs), relationships=json.dumps(rels)))
    print(vid, s, "jpg", nj, "png", len(pngs), "objs", len(ids), "rels", len(rels), flush=True)
pq.write_table(pa.Table.from_pylist(gt_rows), f"{R}/gt_pvsg.parquet"); pq.write_table(pa.Table.from_pylist(mask_rows), f"{R}/masks_pvsg.parquet")
print("CONVERTED", len(gt_rows))
