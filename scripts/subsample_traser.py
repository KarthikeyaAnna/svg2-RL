"""Keep only TraSeR's sampled frames (~1 fps, <=128) for frames + masks; write meta fps_eff. Usage: root fps masks.parquet frames_dir out_dir"""
import sys, os, json, shutil, numpy as np, pyarrow as pa, pyarrow.parquet as pq
R, FPS, mp, fd, out = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
os.makedirs(f"{out}/frames", exist_ok=True); rows = []; meta = {}
t = pq.read_table(mp)
for i in range(t.num_rows):
    r = t.slice(i, 1).to_pylist()[0]; vid = r["video_id"]; nf, no = r["num_frames"], r["num_objects"]
    L = nf / FPS; target = min(max(round(L), 4), 128); idx = np.unique(np.linspace(0, nf - 1, target, dtype=int)).tolist()
    jp = sorted(os.listdir(f"{fd}/{vid}"))[:nf]; od = f"{out}/frames/{vid}"; os.makedirs(od, exist_ok=True)
    for j, f in enumerate(idx): shutil.copy(f"{fd}/{vid}/{jp[f]}", f"{od}/{j:05d}.jpg")
    r["counts"] = [r["counts"][f * no + o] for f in idx for o in range(no)]; r["num_frames"] = len(idx); rows.append(r)
    meta[vid] = dict(duration=L, sampled=len(idx), fps_eff=len(idx) / L)
pq.write_table(pa.Table.from_pylist(rows), f"{out}/masks.parquet"); json.dump(meta, open(f"{out}/meta.json", "w")); print("done", len(rows))
