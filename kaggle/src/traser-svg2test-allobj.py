# Kill test: run released TraSeR on SVG2test (VIPSeg part) with GT masks; save raw outputs.
import glob, json, os, subprocess, sys, time, math
def find(pattern):
    hits = glob.glob(pattern, recursive=True); print("find", pattern, "->", hits[:3], flush=True)
    return hits[0]
ASSETS = os.path.dirname(find("/kaggle/input/**/TRASER/config.json"))
ROOT = os.path.dirname(ASSETS)
DATA = os.path.dirname(find("/kaggle/input/**/masks_vipseg.parquet"))
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
FPS = 2.0      # VIPSeg imgs are every 3rd frame of the 6 fps source; masks are 6 fps -> take mask frame 3j
model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
processor = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); processor.tokenizer = tokenizer
masks = {r["video_id"]: r for r in pq.read_table(f"{DATA}/masks_vipseg.parquet").to_pylist()}
out_path = "/kaggle/working/traser_svg2test_vipseg_allobj.jsonl"; done = set()
for vid in sorted(masks):
    if masks[vid]["num_objects"] <= 40 and vid != "856_coe8HkbRIk4": continue
    t0 = time.time(); rec = {"video_id": vid}
    try:
        frames = sorted(glob.glob(f"/kaggle/input/**/{vid}/*.jpg", recursive=True))
        m = masks[vid]; H, W, nf, no = m["mask_height"], m["mask_width"], m["num_frames"], m["num_objects"]
        rec.update(n_jpg=len(frames), mask_frames=nf, num_objects=no)
        total = len(frames); idx = I._frame_indices(total, total / FPS)
        video = np.stack([np.array(Image.open(frames[i]).convert("RGB").resize((W, H))) for i in idx])
        fps_eff = len(idx) / (total / FPS)
        import copy
        vp = copy.deepcopy(processor.video_processor); vp.max_pixels = I.VIDEO_MAX_FRAME_PIXELS; vp.min_pixels = I.VIDEO_MIN_FRAME_PIXELS
        vp.size["longest_edge"] = vp.max_pixels; vp.size["shortest_edge"] = vp.min_pixels
        pv = vp.preprocess(videos=video, return_tensors="pt")
        pixel_values_videos, video_grid_thw = pv["pixel_values_videos"], pv["video_grid_thw"][0]
        second_per_grid_ts = [processor.video_processor.temporal_patch_size / fps_eff]
        t_grid, h_patch, w_patch = (int(x) for x in video_grid_thw)
        n_video_tokens = t_grid * h_patch * w_patch // I.SPATIAL_MERGE_SIZE ** 2
        mask_data = [[{"size": [H, W], "counts": m["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
        rec["mask_frames_used"] = len(mask_data)
        obj_ids = list(range(no))[:100]
        obj_masks, kept = I.build_obj_masks(mask_data, obj_ids, idx.tolist(), h_patch * I.PATCH_SIZE, w_patch * I.PATCH_SIZE)
        rec.update(kept_obj_ids=kept, sampled=len(idx), n_video_tokens=n_video_tokens)
        _, per_obj_idx, _ = select_tokens(obj_masks=obj_masks, grid_thw=(t_grid, h_patch, w_patch), patch_size=I.PATCH_SIZE,
            spatial_merge_size=I.SPATIAL_MERGE_SIZE, temporal_patch_size=I.TEMPORAL_PATCH_SIZE, coverage_thresh=0.5, time_reduce="max", device="cpu")
        input_ids = I.build_prompt_ids(tokenizer, I.PROMPTS["scene_graph"], n_video_tokens).cuda()
        label_ids = tokenizer([I.OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per_obj_idx))], add_special_tokens=False)["input_ids"]
        gpw = int(4 / (I.TEMPORAL_PATCH_SIZE / I.BASE_INTERVAL))
        windows = [f"<{w*4} - {(w+1)*4} sec>" for w in range(math.ceil(t_grid / gpw))]
        window_ids = tokenizer(windows, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            embeds, pos, am, rope, _, _, _ = rearrange_token(model=model, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                pixel_values_videos=pixel_values_videos.cuda().to(torch.bfloat16), video_grid_thw=video_grid_thw[None].cuda(),
                image_grid_thw=None, pixel_values=None, second_per_grid_ts=torch.tensor(second_per_grid_ts),
                obj_token_indices_per_sample=[[i.cuda() for i in per_obj_idx]], obj_traj_start_id=int(model.config.obj_traj_start_id),
                obj_traj_end_id=int(model.config.obj_traj_end_id), text_token_ids_per_sample=[[torch.tensor(x) for x in label_ids]],
                timestamp_token_ids_per_batch=[[torch.tensor(x) for x in window_ids]], grids_per_temporal_window_per_batch=[gpw])
            rec["prefill_len"] = int(embeds.shape[1])
            gen = model.generate(inputs_embeds=embeds, position_ids=pos, attention_mask=am.long(), rope_deltas=rope, max_new_tokens=8192, do_sample=False)
        rec["n_gen_tokens"] = int(gen.shape[1]); rec["text"] = tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        import traceback; rec["error"] = traceback.format_exc()[-2000:]
    rec["sec"] = round(time.time() - t0, 1)
    print(vid, {k: rec.get(k) for k in ["n_jpg","mask_frames","num_objects","sampled","prefill_len","n_gen_tokens","sec"]}, rec.get("error","")[-300:], flush=True)
    with open(out_path, "a") as f: f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)

# ---------------- judge stage (same session, consolidate GPU use) ----------------
del model; import gc; gc.collect(); torch.cuda.empty_cache()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle/src")
import pyarrow.parquet as pq
gt = {r["video_id"]: r for r in pq.read_table(find("/kaggle/input/**/gt_vipseg.parquet")).to_pylist()}
outs = [json.loads(l) for l in open(find("/kaggle/input/**/traser_svg2test_vipseg.jsonl"))] + [json.loads(l) for l in open(out_path)]
exec(compile('# Lenient label judge (stand-in for SVG2\'s GPT-4o-mini aligner): classify (GT label, predicted label) pairs.\nimport glob, json, os, re, sys, torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\nCATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]\ndef build_pairs(outs, gt):\n    pairs = set()\n    for o in outs:\n        if "error" in o or o["video_id"] not in gt: continue\n        kept = o["kept_obj_ids"]; g = gt[o["video_id"]]\n        gobj = {int(k.split("_")[1]): str(v) for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}\n        try: pred = json.loads(o["text"])\n        except Exception: continue\n        for ob in pred.get("objects", []):\n            for k, v in ob.items():\n                m = re.match(r"object[ _](\\d+)", k)\n                if m and int(m.group(1)) - 1 < len(kept):\n                    gl = gobj.get(kept[int(m.group(1)) - 1])\n                    if gl: pairs.add(("object", gl.lower().strip(), str(v).lower().strip()))\n        P = {}\n        for r in pred.get("relationships", []):\n            try: P.setdefault((kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1), []).append(str(r[1]))\n            except Exception: pass\n        for s, p, ob, *_ in json.loads(g["relationships"]):\n            for q in P.get((s, ob), []):\n                if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))\n    return sorted(pairs)\ndef run_judge(pairs, model_dir, out_path, bs=64):\n    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"\n    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()\n    res = {}\n    for i in range(0, len(pairs), bs):\n        batch = pairs[i:i + bs]; prompts = []\n        for kind, g, p in batch:\n            msg = (f"You are a strict lexical matcher for video scene graph evaluation. Compare a ground-truth {kind} label with a predicted {kind} label "\n                   f"and classify their relationship into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\\n"\n                   f"Ground truth: \\"{g}\\"\\nPrediction: \\"{p}\\"\\nAnswer with the category name only.")\n            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False))\n        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")\n        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=8, do_sample=False)\n        for (kind, g, p), t in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):\n            t = t.strip().lower(); res[f"{kind}|||{g}|||{p}"] = next((c for c in CATS if t.startswith(c.split("/")[0])), "mismatch:" + t[:20])\n        print("judged", min(i + bs, len(pairs)), "/", len(pairs), flush=True)\n    json.dump(res, open(out_path, "w"))\n', 'judge_inline', 'exec'))
pairs = build_pairs(outs, gt); print("pairs to judge", len(pairs), flush=True)
run_judge(pairs, os.path.dirname(find("/kaggle/input/**/Qwen3*8*/**/config.json") if glob.glob("/kaggle/input/**/Qwen3*8*/**/config.json", recursive=True) else find("/kaggle/input/**/qwen-3/**/config.json")), "/kaggle/working/judge_cache.json")
print("JUDGE DONE", flush=True)
