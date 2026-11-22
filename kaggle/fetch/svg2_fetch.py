# ==== SVG2 training-data fetcher for Kaggle CPU notebooks (internet ON). Produces EXACTLY the layout of prep_sav_train.py /
# prep_pvd_train.py (TraSeR frame sampling, per-sampled-frame masks, labels + touching pairs, meta), but packed for Kaggle's
# output limits: frames_NNN.tar (frames/<video>/<j>.jpg, <= PACK videos each) + masks_train.parquet + labels_train.parquet +
# meta_train.json + fetch_summary.json.
# Parallel at two levels: every source item (SA-V tar / PVD shard) is streamed by its own reader thread, and every video is
# decoded (cv2 frames + RLE masks + touching pairs) in a process pool with one worker per CPU core.
# Config globals prepended by build_fetch.py: MODE ("sav"|"pvd"), ITEMS, LINKS (sav), EXCLUDE, SAV_INDEX (sav), LIMIT, OUT.
import os, io, sys, json, time, tarfile, tempfile, itertools, threading, queue, urllib.request, collections
import concurrent.futures as cf
import numpy as np, cv2, pyarrow as pa, pyarrow.parquet as pq
try: from pycocotools import mask as Mk
except ImportError:
    import subprocess; subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pycocotools"], check=True); from pycocotools import mask as Mk
from huggingface_hub import hf_hub_download, HfFileSystem
RP = "UWGZQ/Synthetic_Visual_Genome2"; NCPU = os.cpu_count() or 4; PACK = 400; T0 = time.time()
TMP = os.environ.get("FETCH_TMP", "/tmp/svg2fetch"); os.makedirs(TMP, exist_ok=True); os.makedirs(OUT, exist_ok=True)
log = lambda *a: print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o

def process(v, data, m, label):
    """One video (runs in a worker process): TraSeR sampling -> JPEG bytes, sampled-frame masks, touching pairs, meta."""
    try:
        fc = int(m["num_frames"])
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=TMP) as tf:
            tf.write(data); tf.flush(); cap = cv2.VideoCapture(tf.name); fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0: return None, f"{v}: no fps"
            Ls = fc / fps; idx = np.unique(np.linspace(0, fc - 1, min(max(round(Ls), 4), 128), dtype=int)).tolist(); want = set(idx)
            frames = {}; n = 0
            while True:
                ok, im = cap.read()
                if not ok: break
                if n in want: frames[n] = im
                n += 1
            cap.release()
        if n != fc: return None, f"{v}: mask frames {fc} != decoded {n}"
        jpgs = [cv2.imencode(".jpg", frames[f], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() for f in idx]; del frames
        no, H, W = int(m["num_objects"]), int(m["mask_height"]), int(m["mask_width"])
        counts = [m["counts"][f * no + o] for f in idx for o in range(no)]
        D = np.zeros((no, len(idx), (H + 3) // 4, (W + 3) // 4), bool)
        for j in range(len(idx)):
            for o in range(no):
                c = counts[j * no + o]
                if c: D[o, j] = Mk.decode({"size": [H, W], "counts": c})[::4, ::4][: D.shape[2], : D.shape[3]] > 0
        touch = [[a, b] for a, b in itertools.combinations(range(no), 2)
                 if any((dil(D[a, j]) & D[b, j]).any() for j in range(len(idx)) if D[a, j].any() and D[b, j].any())]
        lab = dict(label); lab["touching"] = json.dumps(touch); lab["video_id"] = v
        return dict(video_id=v, jpgs=jpgs, mask=dict(video_id=v, mask_height=H, mask_width=W, num_frames=len(idx), num_objects=no, counts=counts),
                    label=lab, meta=dict(duration=Ls, sampled=len(idx), fps_eff=len(idx) / Ls, fc=fc)), None
    except Exception as e:
        return None, f"{v}: {type(e).__name__} {str(e)[:120]}"

def load_labels(ann_paths, wanted):
    fs = HfFileSystem(); out = {}
    def one(p):
        with fs.open(p) as f:
            t = pq.read_table(f); ids = [str(x) for x in t.column("video_id").to_pylist()]
            keep = [j for j, v in enumerate(ids) if v in wanted]
            return t.take(keep).to_pylist() if keep else []
    with cf.ThreadPoolExecutor(16) as ex:
        for rows in ex.map(one, ann_paths):
            for r in rows: out[str(r["video_id"])] = r
    return out

def load_masks(shard_files, wanted):
    """Download mask shards in parallel, keep only rows for wanted videos."""
    out = {}
    if not shard_files: return out
    def one(fn):
        p = hf_hub_download(RP, fn, repo_type="dataset", local_dir=f"{TMP}/masks"); rows = {}
        for b in pq.ParquetFile(p).iter_batches(batch_size=32):
            for r in b.to_pylist():
                if str(r["video_id"]) in wanted: rows[str(r["video_id"])] = r
        os.remove(p); return rows
    with cf.ThreadPoolExecutor(min(8, len(shard_files))) as ex:
        for rows in ex.map(one, shard_files): out.update(rows)
    return out

def _shard_ids(path):                                                  # (process pool) video_id column of one mask shard
    for attempt in range(5):
        try:
            with HfFileSystem().open(path, block_size=1 << 20) as f:
                return path.split("/")[-1], [str(x) for x in pq.ParquetFile(f).read(columns=["video_id"]).column(0).to_pylist()]
        except Exception as e: err = e; time.sleep(2 + 3 * attempt)
    raise err
def build_index(mode, n):
    """video -> mask shard file name, from the video_id column of every SVG2 mask shard (no local index files needed)."""
    idx = {}; files = [f"datasets/{RP}/masks/cleaned/{mode}/train-{i:05d}-of-{n:05d}.parquet" for i in range(n)]
    with cf.ProcessPoolExecutor(min(16, 2 * NCPU)) as ex:
        for fn, vs in ex.map(_shard_ids, files):
            for v in vs: idx[v] = fn
    log(f"{mode} mask index: {len(idx)} videos over {n} shards"); return idx

class ShardMasks:
    """Mask rows loaded on demand, one SVG2 shard at a time (a PE-Video tar touches only ~2 shards); small LRU cache."""
    def __init__(self, idx, mode, keep=8):
        self.idx, self.mode, self.keep = idx, mode, keep; self.cache = collections.OrderedDict(); self.lk = threading.Lock(); self.sl = collections.defaultdict(threading.Lock)
    def __getitem__(self, v):
        fn = self.idx[v]
        with self.sl[fn]:
            with self.lk: hit = self.cache.get(fn)
            if hit is None:
                p = hf_hub_download(RP, f"masks/cleaned/{self.mode}/{fn}", repo_type="dataset", local_dir=f"{TMP}/masks")
                t = pq.read_table(p); os.remove(p); hit = (t, {str(x): i for i, x in enumerate(t.column("video_id").to_pylist())})
                with self.lk:
                    self.cache[fn] = hit
                    while len(self.cache) > self.keep: self.cache.popitem(last=False)
        t, rows = hit; return t.slice(rows[v], 1).to_pylist()[0]

# ---------------------------------------------------------------- per-mode sources: (reader streams, wanted ids, masks, labels)
fsh = HfFileSystem()
if MODE == "sav" and "SAV_INDEX" not in globals(): SAV_INDEX = build_index("sav", 69)
if MODE == "sav":
    wanted = {v for v, sh in SAV_INDEX.items() if f"sav_{int(v.split('_')[1]) // 1000:03d}" in ITEMS and v not in EXCLUDE}
    shard_files = sorted({f"masks/cleaned/sav/{SAV_INDEX[v]}" for v in wanted})
    log("SA-V items", ITEMS, "wanted", len(wanted), "mask shards", len(shard_files))
    labels = load_labels([f"datasets/{RP}/data/cleaned/sav/train-00000-of-00001.parquet"], wanted)
    masks = load_masks(shard_files, wanted)
    def open_stream(item):
        return tarfile.open(fileobj=urllib.request.urlopen(LINKS[f"{item}.tar"], timeout=300), mode="r|")
    def vid_of(name):
        b = os.path.basename(name)
        return b[:-4] if b.endswith(".mp4") and b.startswith("sav_") else None
elif "PVD_WANTED" not in globals():                                     # PVD, no precomputed tar lists: index + labels here, masks lazily
    pidx = build_index("pvd", 550)
    class ArrowRows:                                                     # all ~545k PVD labels kept as Arrow (dicts were OOM-killed)
        def __init__(self, paths):
            fs = HfFileSystem(); ts = []
            for p_ in paths:
                with fs.open(p_) as f: ts.append(pq.read_table(f))
            self.t = pa.concat_tables(ts, promote_options="default"); self.ix = {str(x): i for i, x in enumerate(self.t.column("video_id").to_pylist())}
        def __contains__(self, v): return v in self.ix
        def __getitem__(self, v): return self.t.slice(self.ix[v], 1).to_pylist()[0]
    labels = ArrowRows([f"datasets/{RP}/data/cleaned/pvd/train-{i:05d}-of-00060.parquet" for i in range(60)])
    wanted = {v for v in pidx if v in labels and v not in EXCLUDE}; masks = ShardMasks(pidx, "pvd")
    log("PVD tars", ITEMS, "| SVG2-labelled + masked, not excluded:", len(wanted))
    def open_stream(item):
        return tarfile.open(fileobj=fsh.open(f"datasets/facebook/PE-Video/extended/{item:06d}.tar", block_size=8 << 20), mode="r|")
    def vid_of(name):
        b = os.path.basename(name); return b[:-4] if b.endswith(".mp4") else None
else:                                                                   # ITEMS = PE-Video tar indices; masks via PVD_INDEX
    wanted = {v for v in PVD_WANTED if v not in EXCLUDE}
    shard_files = sorted({f"masks/cleaned/pvd/train-{PVD_INDEX[v]:05d}-of-00550.parquet" for v in wanted})
    masks = load_masks(shard_files, wanted)
    labels = load_labels([f"datasets/{RP}/data/cleaned/pvd/train-{i:05d}-of-00060.parquet" for i in range(60)], wanted)
    log("PVD tars", ITEMS, "wanted", len(wanted), "mask shards", len(shard_files), "with masks", len(masks), "labelled", len(labels))
    def open_stream(item):
        return tarfile.open(fileobj=fsh.open(f"datasets/facebook/PE-Video/extended/{item:06d}.tar", block_size=8 << 20), mode="r|")
    def vid_of(name):
        b = os.path.basename(name); return b[:-4] if b.endswith(".mp4") else None
todo = wanted if isinstance(masks, ShardMasks) else {v for v in wanted if v in masks and v in labels}
log("videos with masks + labels:", len(todo), "| CPU workers:", NCPU)

# ---------------------------------------------------------------- readers (one thread per item) -> process pool -> packer
Q = queue.Queue(maxsize=4 * NCPU); readers_left = [len(ITEMS)]; lock = threading.Lock(); seen = collections.Counter()
def reader(item):
    for attempt in range(3):
        try:
            with open_stream(item) as tf:
                for mem in tf:
                    v = vid_of(mem.name)
                    if not mem.isfile() or v is None or v not in todo: continue
                    with lock:
                        if seen[v]: continue
                        seen[v] += 1
                        if LIMIT and sum(seen.values()) > LIMIT: break
                    data = tf.extractfile(mem).read(); Q.put((v, data, masks[v], labels[v]))
            break
        except Exception as e: log("reader retry", item, attempt, repr(e)[:150])
    with lock: readers_left[0] -= 1
    if readers_left[0] == 0: Q.put(None)
for it in ITEMS: threading.Thread(target=reader, args=(it,), daemon=True).start()

M, L, META, errs = [], [], {}, []; pack = None; npack = 0; inpack = 0
def add_result(res):
    global pack, npack, inpack
    if pack is None or inpack >= PACK:
        if pack is not None: pack.close()
        pack = tarfile.open(f"{OUT}/frames_{npack:03d}.tar", "w"); npack += 1; inpack = 0
    v = res["video_id"]
    for j, b in enumerate(res["jpgs"]):
        ti = tarfile.TarInfo(f"frames/{v}/{j:05d}.jpg"); ti.size = len(b); pack.addfile(ti, io.BytesIO(b))
    inpack += 1; M.append(res["mask"]); L.append(res["label"]); META[v] = res["meta"]
STALL = int(globals().get("STALL_SEC", 600)); last = [time.time()]; S_LOCK = threading.Lock(); finished = [False]
def finalize(stalled):
    """Write everything done so far and exit. Called by the main loop, or by the watchdog thread when nothing finished for
    STALL s -- the main thread itself can block (pvd-a / pvd-a2 froze inside the pool with no timeout firing)."""
    with S_LOCK:
        if finished[0]: return
        finished[0] = True
        if pack is not None: pack.close()
        pq.write_table(pa.Table.from_pylist(M), f"{OUT}/masks_train.parquet"); pq.write_table(pa.Table.from_pylist(L), f"{OUT}/labels_train.parquet")
        json.dump(META, open(f"{OUT}/meta_train.json", "w"))
        summ = dict(mode=MODE, items=ITEMS, wanted=len(todo), streamed=sum(seen.values()), done=len(M), errors=len(errs), err_examples=errs[:20],
                    stalled=stalled, frame_packs=npack, seconds=round(time.time() - T0))
        json.dump(summ, open(f"{OUT}/fetch_summary.json", "w")); log("FETCH DONE", json.dumps(summ))
    if stalled:
        for pr in list(getattr(ex, "_processes", {}).values()):
            try: pr.kill()
            except Exception: pass
    os._exit(0)                                                        # do not wait on hung reader threads / workers
def watchdog():
    while True:
        time.sleep(15)
        if time.time() - last[0] > STALL:
            log(f"STALL: no video finished for {STALL}s; saving {len(M)} done"); finalize(True)
threading.Thread(target=watchdog, daemon=True).start()
ex = cf.ProcessPoolExecutor(NCPU)
futs = set(); done_readers = False
while not done_readers or futs:
    while not done_readers and len(futs) < 2 * NCPU:
        try: item = Q.get(timeout=30)
        except queue.Empty: break                                  # a slow/stalled reader must not block finished work
        if item is None: done_readers = True; break
        v, data, mrow, lrow = item; futs.add(ex.submit(process, v, data, mrow, lrow))
    if futs:
        fin, futs = cf.wait(futs, timeout=60, return_when=cf.FIRST_COMPLETED)
        for f in fin:
            res, err = f.result(); last[0] = time.time()
            with S_LOCK:
                if finished[0]: break
                if err: errs.append(err)
                else: add_result(res)
        if fin and len(M) % 200 == 0: log("videos done", len(M), "errors", len(errs))
        if globals().get("TEST_MAIN_HANG") and len(M) >= TEST_MAIN_HANG: log("TEST: main thread hangs"); time.sleep(10 ** 6)
finalize(False)
