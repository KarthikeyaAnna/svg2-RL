"""PVD (svg2_pvd, GPT-5 labels; TraSeR's own training split) -> TraSeR-sampled JPEG frames + per-sampled-frame masks + labels
+ touching object pairs (dilated masks touch on >= 1 sampled frame; used to build the 'salient' target view).
PE-Video extended tar i <-> SVG2 masks/cleaned/pvd shard i. Usage: out_dir tar_indices(comma) [max_per_tar]"""
import sys, os, json, io, tarfile, tempfile, itertools, concurrent.futures as cf
import numpy as np, cv2, pyarrow as pa, pyarrow.parquet as pq
from pycocotools import mask as Mk
from huggingface_hub import hf_hub_download, HfFileSystem
OUT, TARS = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]; MAXN = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
R = os.path.expanduser("~/svg2-killtest"); RP = "UWGZQ/Synthetic_Visual_Genome2"; fs = HfFileSystem()
os.makedirs(f"{OUT}/frames", exist_ok=True)
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
want = set()
masks = {}
for t in TARS:
    p = hf_hub_download(RP, f"masks/cleaned/pvd/train-{t:05d}-of-00550.parquet", repo_type="dataset", local_dir=f"{R}/data/pvd_probe")
    for b in pq.ParquetFile(p).iter_batches(batch_size=64):
        for r in b.to_pylist(): masks[str(r["video_id"])] = r
labels = {}
def lab_shard(i):
    with fs.open(f"datasets/{RP}/data/cleaned/pvd/train-{i:05d}-of-00060.parquet") as f:
        t = pq.read_table(f); ids = [str(x) for x in t.column("video_id").to_pylist()]
        keep = [j for j, v in enumerate(ids) if v in masks]
        return t.take(keep).to_pylist() if keep else []
with cf.ThreadPoolExecutor(16) as ex:
    for rows in ex.map(lab_shard, range(60)):
        for r in rows: labels[str(r["video_id"])] = r
print("masks", len(masks), "labelled", len(labels), flush=True)
M, L, meta = [], [], {}; errs = 0
def one(v, data):
    m = masks[v]; fc = m["num_frames"]
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tf:
        tf.write(data); tf.flush(); cap = cv2.VideoCapture(tf.name); fps = cap.get(cv2.CAP_PROP_FPS)
        Ls = fc / fps; idx = np.unique(np.linspace(0, fc - 1, min(max(round(Ls), 4), 128), dtype=int)).tolist()
        fd = f"{OUT}/frames/{v}"; done = os.path.isdir(fd) and len(os.listdir(fd)) == len(idx)
        want = set(idx); frames = {}; n = 0
        while True:
            ok, im = cap.read()
            if not ok: break
            if n in want and not done: frames[n] = im
            n += 1
        cap.release()
    if n != fc: return None, f"{v} mask frames {fc} != decoded {n}"
    if not done:
        os.makedirs(fd, exist_ok=True)
        for j, f in enumerate(idx): cv2.imwrite(f"{fd}/{j:05d}.jpg", frames[f], [cv2.IMWRITE_JPEG_QUALITY, 90])
    del frames
    no, H, W = m["num_objects"], m["mask_height"], m["mask_width"]
    counts = [m["counts"][f * no + o] for f in idx for o in range(no)]
    D = np.zeros((no, len(idx), (H + 3) // 4, (W + 3) // 4), bool)
    for j in range(len(idx)):
        for o in range(no):
            c = counts[j * no + o]
            if c: D[o, j] = Mk.decode({"size": [H, W], "counts": c})[::4, ::4][: D.shape[2], : D.shape[3]] > 0
    touch = [[a, b] for a, b in itertools.combinations(range(no), 2)
             if any((dil(D[a, j]) & D[b, j]).any() for j in range(len(idx)) if D[a, j].any() and D[b, j].any())]
    lab = dict(labels[v]); lab["touching"] = json.dumps(touch)
    return dict(mask=dict(video_id=v, mask_height=H, mask_width=W, num_frames=len(idx), num_objects=no, counts=counts), label=lab,
                meta=dict(duration=Ls, sampled=len(idx), fps_eff=len(idx) / Ls, fc=fc)), None
for t in TARS:
    p = hf_hub_download("facebook/PE-Video", f"extended/{t:06d}.tar", repo_type="dataset", local_dir=f"{R}/data/pvd_probe")
    tfz = tarfile.open(p); mem = [x for x in tfz.getmembers() if x.name.endswith(".mp4") and x.name[:-4] in labels][:MAXN]
    for i0 in range(0, len(mem), 4):                       # 4 videos in flight (16 GB RAM machine)
        blobs = [(x.name[:-4], tfz.extractfile(x).read()) for x in mem[i0:i0 + 4]]
        with cf.ThreadPoolExecutor(4) as ex:
            for res, err in ex.map(lambda a: one(*a), blobs):
                if err: errs += 1; print("ERR", err, flush=True) if errs < 10 else None; continue
                M.append(res["mask"]); L.append(res["label"]); meta[res["mask"]["video_id"]] = res["meta"]
        del blobs
    print("tar", t, "done, total", len(M), "errors", errs, flush=True)
pq.write_table(pa.Table.from_pylist(M), f"{OUT}/masks_train.parquet"); pq.write_table(pa.Table.from_pylist(L), f"{OUT}/labels_train.parquet")
json.dump(meta, open(f"{OUT}/meta_train.json", "w")); print("DONE", len(M), "errors", errs, flush=True)
