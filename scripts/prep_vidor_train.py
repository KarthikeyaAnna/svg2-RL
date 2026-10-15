"""VidOR-train (TraSeR's own training split) -> TraSeR-sampled JPEG frames + per-sampled-frame masks (SVG2's released masks)
+ SVG2 labels, for SFT. Videos are read member-by-member from the remote HF zips (shangxd/vidor), no full download.
Frame rule = TraSeR training (data_qwen.video_decord): target = min(max(round(L), 4), 128); idx = unique(linspace(0, fc-1, target)).
SVG2 VidOR masks hold one entry per original frame, so we keep counts[idx].
Usage: out_dir n_dev n_train mask_shard_glob"""
import sys, os, json, glob, zipfile, random, tempfile, concurrent.futures as cf
import numpy as np, cv2, pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
OUT, NDEV, NTRAIN, SH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
R = os.path.expanduser("~/svg2-killtest")
fs = HfFileSystem(); BASE = "datasets/shangxd/vidor"
lab = {str(r["video_id"]): r for r in pq.read_table(f"{R}/data/svg2ann/vidor.parquet").to_pylist()}
excl = set(map(str, json.load(open(f"{R}/data/svg2ann/exclude_ids.json"))))
ann = zipfile.ZipFile(fs.open(f"{BASE}/training-annotation.zip"))
annp = {os.path.basename(n)[:-5]: n for n in ann.namelist() if n.endswith(".json")}
parts = {}
for k in range(1, 9):
    z = zipfile.ZipFile(fs.open(f"{BASE}/training-video-part{k}.zip"))
    for n in z.namelist():
        if n.endswith(".mp4"): parts[os.path.basename(n)[:-4]] = (k, n)
SHARDS = sorted(glob.glob(SH))
ids_all = [str(x) for p in SHARDS for x in pq.read_table(p, columns=["video_id"]).column(0).to_pylist()]
cand = sorted(v for v in ids_all if v in lab and v in parts and v in annp and v not in excl)
print("candidates", len(cand), "excluded", sum(v in excl for v in ids_all), flush=True)
masks = {}
random.Random(0).shuffle(cand); dev, train = cand[:NDEV], cand[NDEV:NDEV + NTRAIN]
os.makedirs(f"{OUT}/frames", exist_ok=True)
def one(v):
    try:
        d = json.loads(ann.read(annp[v])); fc, fps = int(d["frame_count"]), float(d["fps"]); L = fc / fps
        m = masks.pop(v)
        if m["num_frames"] != fc: return None, f"{v} mask frames {m['num_frames']} != {fc}"
        idx = np.unique(np.linspace(0, fc - 1, min(max(round(L), 4), 128), dtype=int)).tolist()
        k, n = parts[v]
        with zipfile.ZipFile(LOCALZIP[k]) as z, tempfile.TemporaryDirectory() as td:
            p = z.extract(n, td); cap = cv2.VideoCapture(p); frames = []; want = set(idx); f = 0; last = None
            while f <= idx[-1]:
                ok, im = cap.read()
                if not ok: break
                if f in want: frames.append(im)
                last = im; f += 1
            cap.release()
        while len(frames) < len(idx): frames.append(frames[-1] if frames else last)
        fd = f"{OUT}/frames/{v}"; os.makedirs(fd, exist_ok=True)
        for j, im in enumerate(frames): cv2.imwrite(f"{fd}/{j:05d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
        del frames
        no = m["num_objects"]; counts = [m["counts"][f * no + o] for f in idx for o in range(no)]
        mr = dict(video_id=v, mask_height=m["mask_height"], mask_width=m["mask_width"], num_frames=len(idx), num_objects=no, counts=counts)
        return dict(mask=mr, label=lab[v], meta=dict(duration=L, sampled=len(idx), fps_eff=len(idx) / L, fc=fc)), None
    except Exception as e:
        return None, f"{v} {e!r}"[:300]
# whole-part downloads (remote member reads were ~30 s/video); process each part's videos, then delete the part
from huggingface_hub import hf_hub_download
LOCALZIP = {}; RES = {}
split_of = {v: "dev" for v in dev}; split_of.update({v: "train" for v in train})
for k in sorted({parts[v][0] for v in split_of}):
    LOCALZIP[k] = hf_hub_download("shangxd/vidor", f"training-video-part{k}.zip", repo_type="dataset", local_dir=f"{R}/data/vidor_zip")
    vk = set(v for v in split_of if parts[v][0] == k)
    for sp in SHARDS:                                   # read only this part's mask rows (row filter), 4 videos in flight (16 GB RAM)
        rows = pq.read_table(sp, filters=[("video_id", "in", sorted(vk))]).to_pylist()
        for r in rows: masks[str(r["video_id"])] = r
        del rows
        batch = [v for v in list(masks) if v in vk and v not in RES]
        for i0 in range(0, len(batch), 4):
            b4 = batch[i0:i0 + 4]
            with cf.ThreadPoolExecutor(4) as ex:
                for v, (res, err) in zip(b4, ex.map(one, b4)):
                    if err: print("ERR", err, flush=True); continue
                    RES[v] = res
        print("part", k, "shard", os.path.basename(sp), "done total", len(RES), flush=True)
    os.remove(LOCALZIP[k]); print("part", k, "videos", len(vk), "done total", len(RES), flush=True)
for split, vids in (("dev", dev), ("train", train)):
    M, Lb, meta = [], [], {}
    for v in vids:
        if v not in RES: continue
        res = RES[v]; M.append(res["mask"]); Lb.append(res["label"]); meta[v] = res["meta"]
    pq.write_table(pa.Table.from_pylist(M), f"{OUT}/masks_{split}.parquet"); pq.write_table(pa.Table.from_pylist(Lb), f"{OUT}/labels_{split}.parquet")
    json.dump(meta, open(f"{OUT}/meta_{split}.json", "w")); print("DONE", split, len(M), flush=True)
