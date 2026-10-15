# VidOR val -> TraSeR-sampled JPEG frames + SAM2 box-prompted masks (RLE) + SVG2-style GT. T4 + internet.
import os, sys, json, glob, zipfile, subprocess, time, math, ast
subprocess.run(f"{sys.executable} -m pip install -q -U 'transformers>=4.57' huggingface_hub pycocotools", shell=True)
import numpy as np, torch, cv2, pyarrow as pa, pyarrow.parquet as pq
from pycocotools import mask as M
from huggingface_hub import hf_hub_download
from transformers import Sam2Model, Sam2Processor
W = "/kaggle/working"; T = "/kaggle/temp"; os.makedirs(T, exist_ok=True)
ann = zipfile.ZipFile(hf_hub_download("shangxd/vidor", "validation-annotation.zip", repo_type="dataset", local_dir=T))
vz = zipfile.ZipFile(hf_hub_download("shangxd/vidor", "validation-video.zip", repo_type="dataset", local_dir=T))
vnames = {os.path.basename(n)[:-4]: n for n in vz.namelist() if n.endswith(".mp4")}
dev = "cuda"; proc = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-small"); sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-small").to(dev).eval()
print("sam2 loaded", torch.cuda.get_device_name(0), flush=True)
def lit(x): return x if not isinstance(x, str) else ast.literal_eval(x)
gt_rows, mask_rows, meta = [], [], {}
anns = sorted(n for n in ann.namelist() if n.endswith(".json"))
n_fallback = 0
for i, an in enumerate(anns):
    t0 = time.time(); d = json.loads(ann.read(an)); vid = d["video_id"]
    try:
        fc, fps = int(d["frame_count"]), float(d["fps"]); L = fc / fps
        target = min(max(round(L), 4), 128); idx = np.unique(np.linspace(0, fc - 1, target, dtype=int)).tolist()
        objs = lit(d["subject/objects"]); tids = sorted(o["tid"] for o in objs); ti = {t: k for k, t in enumerate(tids)}
        traj = lit(d["trajectories"]); rels = lit(d["relation_instances"])
        p = vz.extract(vnames[vid], T); cap = cv2.VideoCapture(p); frames = []
        for f in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
            if not ok: im = frames[-1] if frames else np.zeros((int(d["height"]), int(d["width"]), 3), np.uint8)
            frames.append(im)
        cap.release(); os.remove(p)
        H, Wd = frames[0].shape[:2]; fdir = f"{W}/frames/{vid}"; os.makedirs(fdir, exist_ok=True)
        empty = M.encode(np.asfortranarray(np.zeros((H, Wd), np.uint8)))["counts"].decode()
        counts = []
        for j, (f, im) in enumerate(zip(idx, frames)):
            cv2.imwrite(f"{fdir}/{j:05d}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            boxes = {b["tid"]: b["bbox"] for b in (traj[f] if f < len(traj) else [])}
            row = [empty] * len(tids)
            if boxes:
                bl = [[boxes[t]["xmin"], boxes[t]["ymin"], boxes[t]["xmax"], boxes[t]["ymax"]] for t in boxes]
                try:
                    inp = proc(images=cv2.cvtColor(im, cv2.COLOR_BGR2RGB), input_boxes=[bl], return_tensors="pt").to(dev)
                    with torch.no_grad(): out = sam(**inp, multimask_output=False)
                    ms = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][:, 0].numpy() > 0
                except Exception as e:
                    n_fallback += 1; ms = []
                    for (x0, y0, x1, y1) in bl:
                        m = np.zeros((H, Wd), bool); m[int(y0):int(y1) + 1, int(x0):int(x1) + 1] = True; ms.append(m)
                    if n_fallback < 3: print("SAM2 fallback", vid, repr(e)[:300], flush=True)
                for t, m in zip(boxes, ms): row[ti[t]] = M.encode(np.asfortranarray(np.asarray(m, np.uint8)))["counts"].decode()
            counts += row
        mask_rows.append(dict(video_id=vid, mask_height=H, mask_width=Wd, num_frames=len(idx), num_objects=len(tids), counts=counts))
        gobjs = [{f"object_{ti[o['tid']]}": o["category"]} for o in objs]
        grel = [[ti[r["subject_tid"]], r["predicate"], ti[r["object_tid"]], [[r["begin_fid"] / fps, r["end_fid"] / fps]], ""] for r in rels]
        gt_rows.append(dict(video_id=vid, source="VidOR-val", objects=json.dumps(gobjs), relationships=json.dumps(grel)))
        meta[vid] = dict(duration=L, sampled=len(idx), fps_eff=len(idx) / L)
    except Exception as e:
        import traceback; print("VIDEO ERROR", vid, traceback.format_exc()[-400:], flush=True)
    if i % 25 == 0: print(i, vid, len(idx), "frames", round(time.time() - t0, 1), "s", "fallbacks", n_fallback, flush=True)
pq.write_table(pa.Table.from_pylist(gt_rows), f"{W}/gt_vidor.parquet"); pq.write_table(pa.Table.from_pylist(mask_rows), f"{W}/masks_vidor.parquet")
json.dump(meta, open(f"{W}/vidor_meta.json", "w")); print("PREP DONE", len(gt_rows), "fallbacks", n_fallback, flush=True)
