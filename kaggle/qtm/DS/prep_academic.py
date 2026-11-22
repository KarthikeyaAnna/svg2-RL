"""SVG2 academic splits (vidor | vidvrd) -> TraSeR-sampled JPEG frames + per-sampled-frame masks + SVG2 labels, in the
prep_*_train.py layout (frames/<v>/<j>.jpg, masks_train.parquet, labels_train.parquet, meta_train.json). Same frame rule as
prep_vidor_train.py / TraSeR training (data_qwen.video_decord): target = min(max(round(L), 4), 128); idx = unique(linspace(0, fc-1, target)).
Runs on Deathstar CPUs. Usage: prep_academic.py SRC=vidor [WORKERS=32] [LIMIT=0]"""
import os, sys, json, glob, zipfile, tempfile, time, concurrent.futures as cf
for a in sys.argv[1:]:
    if "=" in a: os.environ[a.split("=", 1)[0]] = a.split("=", 1)[1]
import numpy as np, cv2, pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
SRC = os.environ["SRC"]; NW = int(os.environ.get("WORKERS", "32")); LIMIT = int(os.environ.get("LIMIT", "0"))
B = "/media/nas_mount/karthikeya/svg2"; RAW = f"{B}/raw/{SRC}"; OUT = f"{B}/train/{SRC}"; os.makedirs(RAW, exist_ok=True); os.makedirs(f"{OUT}/frames", exist_ok=True)
RP = "UWGZQ/Synthetic_Visual_Genome2"; T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
VREPO, VPARTS = {"vidor": ("shangxd/vidor", [f"training-video-part{k}.zip" for k in range(1, 9)]),
                 "vidvrd": ("shangxd/imagenet-vidvrd", ["vidvrd-videos-part1.zip", "vidvrd-videos-part2.zip"])}[SRC]
NSH = {"vidor": 23, "vidvrd": 1}[SRC]
lab_p = hf_hub_download(RP, f"data/academic_datasets/{SRC}/train-00000-of-00001.parquet", repo_type="dataset", local_dir=RAW)
LAB = {str(r["video_id"]): r for r in pq.read_table(lab_p).to_pylist()}
EXCL = set(json.load(open(f"{B}/exclude_ids.json")))
zips = [hf_hub_download(VREPO, p, repo_type="dataset", local_dir=RAW) for p in VPARTS]
MEM = {}
for zp in zips:
    for n in zipfile.ZipFile(zp).namelist():
        if n.lower().endswith((".mp4", ".avi", ".mkv")): MEM[os.path.splitext(os.path.basename(n))[0]] = (zp, n)
log(SRC, "labels", len(LAB), "videos in zips", len(MEM), "labelled+video", len(set(LAB) & set(MEM)), "excluded", len(set(LAB) & EXCL))

def one(args):
    v, m = args
    try:
        zp, n = MEM[v]; fc = int(m["num_frames"])
        with zipfile.ZipFile(zp) as z, tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
            p = z.extract(n, td); cap = cv2.VideoCapture(p); fps = cap.get(cv2.CAP_PROP_FPS)
            nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not fps or fps <= 0: return v, None, f"no fps"
            L = fc / fps; idx = np.unique(np.linspace(0, fc - 1, min(max(round(L), 4), 128), dtype=int)).tolist(); want = set(idx)
            frames = {}; f = 0; last = None
            while f <= idx[-1]:
                ok, im = cap.read()
                if not ok: break
                if f in want: frames[f] = im
                last = im; f += 1
            cap.release()
        if f < fc and f <= idx[-1]: return v, None, f"decoded {f} < mask frames {fc} (container says {nvid})"
        fd = f"{OUT}/frames/{v}"; os.makedirs(fd, exist_ok=True)
        for j, fi in enumerate(idx): cv2.imwrite(f"{fd}/{j:05d}.jpg", frames[fi], [cv2.IMWRITE_JPEG_QUALITY, 90])
        no = int(m["num_objects"]); counts = [m["counts"][fi * no + o] for fi in idx for o in range(no)]
        mr = dict(video_id=v, mask_height=m["mask_height"], mask_width=m["mask_width"], num_frames=len(idx), num_objects=no, counts=counts)
        return v, dict(mask=mr, meta=dict(duration=L, sampled=len(idx), fps_eff=len(idx) / L, fc=fc, video_frames=nvid, fps=fps)), None
    except Exception as e: return v, None, repr(e)[:200]

M, Lb, META, ERR = [], [], {}, []; done = 0
with cf.ProcessPoolExecutor(NW) as ex:
    for i in range(NSH):
        sp = hf_hub_download(RP, f"masks/academic_datasets/{SRC}/train-{i:05d}-of-{NSH:05d}.parquet", repo_type="dataset", local_dir=RAW)
        rows = [r for r in pq.read_table(sp).to_pylist() if str(r["video_id"]) in LAB and str(r["video_id"]) in MEM and str(r["video_id"]) not in EXCL]
        if LIMIT: rows = rows[:max(0, LIMIT - done)]
        for v, res, err in ex.map(one, [(str(r["video_id"]), r) for r in rows], chunksize=2):
            if err: ERR.append(f"{v}: {err}"); continue
            M.append(res["mask"]); Lb.append(LAB[v]); META[v] = res["meta"]
        done += len(rows); del rows; log("shard", i, "ok", len(M), "errors", len(ERR))
        if LIMIT and done >= LIMIT: break
pq.write_table(pa.Table.from_pylist(M), f"{OUT}/masks_train.parquet"); pq.write_table(pa.Table.from_pylist(Lb), f"{OUT}/labels_train.parquet")
json.dump(META, open(f"{OUT}/meta_train.json", "w")); json.dump(dict(src=SRC, done=len(M), errors=len(ERR), err_examples=ERR[:30], seconds=round(time.time() - T0)), open(f"{OUT}/prep_summary.json", "w"))
log("DONE", SRC, "videos", len(M), "errors", len(ERR), ERR[:5])
