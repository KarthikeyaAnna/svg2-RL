
import glob, json, os, subprocess, sys, time, math, copy, random, itertools
def find(p):
    h = glob.glob(p, recursive=True); print("find", p, "->", h[:2], flush=True); return h[0]
DSNAME = "pvsg-val-traser"
ASSETS = os.path.dirname(find("/kaggle/input/**/TRASER/config.json")); ROOT = os.path.dirname(ASSETS)
DATA = os.path.dirname(find(f"/kaggle/input/**/{DSNAME}/**/meta.json") if glob.glob(f"/kaggle/input/**/{DSNAME}/**/meta.json", recursive=True) else find("/kaggle/input/**/meta.json"))
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as M
from transformers import AutoProcessor, AutoTokenizer
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
processor = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); processor.tokenizer = tokenizer
meta = json.load(open(f"{DATA}/meta.json")); gt = {r["video_id"]: r for r in pq.read_table(f"{DATA}/gt.parquet").to_pylist()}
mt = pq.read_table(f"{DATA}/masks.parquet"); rowidx = {mt.slice(i, 1).to_pylist()[0]["video_id"]: i for i in range(mt.num_rows)}
OUT = "/kaggle/working"; KSAL, KRND = 40, 20
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o
def run(objs, prompt, ctx, max_new):
    pvv, vgt, idx, spg, t_grid, h_patch, w_patch, n_vt, mask_data = ctx
    obj_masks, kept = I.build_obj_masks(mask_data, objs, idx, h_patch * I.PATCH_SIZE, w_patch * I.PATCH_SIZE)
    _, per, _ = select_tokens(obj_masks=obj_masks, grid_thw=(t_grid, h_patch, w_patch), patch_size=I.PATCH_SIZE, spatial_merge_size=I.SPATIAL_MERGE_SIZE,
                              temporal_patch_size=I.TEMPORAL_PATCH_SIZE, coverage_thresh=0.5, time_reduce="max", device="cpu")
    ids = I.build_prompt_ids(tokenizer, I.PROMPTS[prompt], n_vt).cuda()
    lab = tokenizer([I.OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per))], add_special_tokens=False)["input_ids"]
    gpw = int(4 / (I.TEMPORAL_PATCH_SIZE / I.BASE_INTERVAL))
    win = tokenizer([f"<{w*4} - {(w+1)*4} sec>" for w in range(math.ceil(t_grid / gpw))], add_special_tokens=False)["input_ids"]
    with torch.no_grad():
        e, pos, am, rope, _, _, _ = rearrange_token(model=model, input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values_videos=pvv, video_grid_thw=vgt[None].cuda(),
            image_grid_thw=None, pixel_values=None, second_per_grid_ts=torch.tensor(spg), obj_token_indices_per_sample=[[i.cuda() for i in per]],
            obj_traj_start_id=int(model.config.obj_traj_start_id), obj_traj_end_id=int(model.config.obj_traj_end_id),
            text_token_ids_per_sample=[[torch.tensor(x) for x in lab]], timestamp_token_ids_per_batch=[[torch.tensor(x) for x in win]], grids_per_temporal_window_per_batch=[gpw])
        g = model.generate(inputs_embeds=e, position_ids=pos, attention_mask=am.long(), rope_deltas=rope, max_new_tokens=max_new, do_sample=False)
    return kept, tokenizer.decode(g[0], skip_special_tokens=True), int(e.shape[1])
def covered(kept, text):
    s = set()
    try:
        for r in json.loads(text)["relationships"]:
            try: s.add(frozenset((kept[int(r[0]) - 1], kept[int(r[2]) - 1])))
            except Exception: pass
    except Exception: pass
    return s
rng = random.Random(0)
for vid in sorted(meta):
    t0 = time.time()
    try:
        row = mt.slice(rowidx[vid], 1).to_pylist()[0]; H, W, nf, no = row["mask_height"], row["mask_width"], row["num_frames"], row["num_objects"]
        frames = sorted(glob.glob(f"{DATA}/**/frames/{vid}/*.jpg", recursive=True))[:nf]; total = len(frames); FPS = meta[vid]["fps_eff"]
        idx = I._frame_indices(total, total / FPS).tolist()
        video = np.stack([np.array(Image.open(frames[i]).convert("RGB").resize((W, H))) for i in idx])
        vp = copy.deepcopy(processor.video_processor); vp.max_pixels = I.VIDEO_MAX_FRAME_PIXELS; vp.min_pixels = I.VIDEO_MIN_FRAME_PIXELS
        vp.size["longest_edge"] = vp.max_pixels; vp.size["shortest_edge"] = vp.min_pixels
        pv = vp.preprocess(videos=video, return_tensors="pt"); vgt = pv["video_grid_thw"][0]; t_grid, h_patch, w_patch = (int(x) for x in vgt)
        spg = [processor.video_processor.temporal_patch_size / (len(idx) / (total / FPS))]
        n_vt = t_grid * h_patch * w_patch // I.SPATIAL_MERGE_SIZE ** 2
        mask_data = [[{"size": [H, W], "counts": row["counts"][f * no + o]} for o in range(no)] for f in range(nf)][:total]
        ctx = (pv["pixel_values_videos"].cuda().to(torch.bfloat16), vgt, idx, spg, t_grid, h_patch, w_patch, n_vt, mask_data)
    except Exception as e:
        import traceback; print(vid, "VIDEO ERROR", traceback.format_exc()[-400:], flush=True); continue
    caps = [40] + ([100] if no > 40 else [])
    full = {}
    for cap in caps:
        rec = {"video_id": vid, "cap": cap}
        try:
            rec["kept_obj_ids"], rec["text"], rec["prefill_len"] = run(list(range(no))[:cap], "scene_graph", ctx, 8192); full[cap] = rec
        except Exception as e:
            import traceback; rec["error"] = traceback.format_exc()[-800:]
        with open(f"{OUT}/full_cap{cap}.jsonl", "a") as f: f.write(json.dumps(rec) + chr(10))
    best = full.get(100) or full.get(40)
    have = covered(best["kept_obj_ids"], best["text"]) if best and "text" in best else set()
    # geometric salience (contact) at the sampled frames
    D = {}; DL = {}; A = {}
    for ob in range(no):
        D[ob] = []; A[ob] = 0
        for f in idx:
            rle = mask_data[f][ob]
            m = M.decode(rle)[::4, ::4].astype(bool) if M.area(rle) > 0 else None; D[ob].append(m); A[ob] += 0 if m is None else int(m.sum())
        DL[ob] = [dil(m) if m is not None else None for m in D[ob]]
    sc = {}; uncovered = []
    for a, b in itertools.combinations(range(no), 2):
        if frozenset((a, b)) in have or A[a] == 0 or A[b] == 0: continue
        uncovered.append([a, b])
        c = [bool((DL[a][j] & D[b][j]).any()) for j in range(len(idx)) if D[a][j] is not None and D[b][j] is not None]
        if c and np.mean(c) > 0: sc[(a, b)] = np.mean(c) + 1e-9 * (A[a] + A[b])
    sal = [list(p) for p, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:KSAL]]
    rng.shuffle(uncovered); rnd = uncovered[:KRND]
    for name, plist in [("pairs_sal", sal), ("pairs_rnd", rnd)]:
        for rank, (a, b) in enumerate(plist):
            rec = {"video_id": vid, "pair": [a, b], "rank": rank}
            try: rec["kept_obj_ids"], rec["text"], _ = run(sorted([a, b]), "scene_graph", ctx, 1024)
            except Exception as e:
                import traceback; rec["error"] = traceback.format_exc()[-500:]
            with open(f"{OUT}/{name}.jsonl", "a") as f: f.write(json.dumps(rec) + chr(10))
    print(vid, "objs", no, "sal", len(sal), "rnd", len(rnd), "sec", round(time.time() - t0, 1), flush=True)
print("INFER DONE", flush=True)
del model; import gc; gc.collect(); torch.cuda.empty_cache()
outs = []
for fn in ["full_cap40", "full_cap100", "pairs_sal", "pairs_rnd"]:
    if os.path.exists(f"{OUT}/{fn}.jsonl"): outs += [json.loads(l) for l in open(f"{OUT}/{fn}.jsonl")]
exec(compile('# Lenient label judge (stand-in for SVG2\'s GPT-4o-mini aligner): classify (GT label, predicted label) pairs.\nimport glob, json, os, re, sys, torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\nCATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]\ndef build_pairs(outs, gt):\n    pairs = set()\n    for o in outs:\n        if "error" in o or o["video_id"] not in gt: continue\n        kept = o["kept_obj_ids"]; g = gt[o["video_id"]]\n        gobj = {int(k.split("_")[1]): str(v) for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}\n        try: pred = json.loads(o["text"])\n        except Exception: continue\n        for ob in pred.get("objects", []):\n            for k, v in ob.items():\n                m = re.match(r"object[ _](\\d+)", k)\n                if m and int(m.group(1)) - 1 < len(kept):\n                    gl = gobj.get(kept[int(m.group(1)) - 1])\n                    if gl: pairs.add(("object", gl.lower().strip(), str(v).lower().strip()))\n        P = {}\n        for r in pred.get("relationships", []):\n            try: P.setdefault((kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1), []).append(str(r[1]))\n            except Exception: pass\n        for s, p, ob, *_ in json.loads(g["relationships"]):\n            for q in P.get((s, ob), []):\n                if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))\n    return sorted(pairs)\ndef run_judge(pairs, model_dir, out_path, bs=64):\n    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"\n    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()\n    res = {}\n    for i in range(0, len(pairs), bs):\n        batch = pairs[i:i + bs]; prompts = []\n        for kind, g, p in batch:\n            msg = (f"You are a strict lexical matcher for video scene graph evaluation. Compare a ground-truth {kind} label with a predicted {kind} label "\n                   f"and classify their relationship into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\\n"\n                   f"Ground truth: \\"{g}\\"\\nPrediction: \\"{p}\\"\\nAnswer with the category name only.")\n            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False))\n        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")\n        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=8, do_sample=False)\n        for (kind, g, p), t in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):\n            t = t.strip().lower(); res[f"{kind}|||{g}|||{p}"] = next((c for c in CATS if t.startswith(c.split("/")[0])), "mismatch:" + t[:20])\n        print("judged", min(i + bs, len(pairs)), "/", len(pairs), flush=True)\n    json.dump(res, open(out_path, "w"))\n', 'judge_inline', 'exec'))
pairs = build_pairs(outs, gt); print("pairs to judge", len(pairs), flush=True)
run_judge(pairs, os.path.dirname(find("/kaggle/input/**/qwen-3/**/config.json")), f"{OUT}/judge_cache.json")
print("JUDGE DONE", flush=True)
