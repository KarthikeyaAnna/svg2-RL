import glob, os, sys, subprocess, time
print("QE SCRIPT START", time.ctime(), flush=True)

def find(p, root="/kaggle/input"):
    t0 = time.time()
    clean_p = p.lstrip("*").lstrip("/")
    direct = [
        f"/kaggle/input/notebooks/karthikeyaanna/traser-qtm-v4c2/{clean_p}",
        f"/kaggle/input/traser-qtm-v4c2/{clean_p}",
        f"/kaggle/input/notebooks/srikarthikeyaa/traser-assets/{clean_p}",
        f"/kaggle/input/notebooks/karthikeyaanna/svg2-ds-mirror/{clean_p}",
        f"/kaggle/input/traser-assets/{clean_p}",
        f"/kaggle/input/svg2-ds-mirror/{clean_p}",
        f"/kaggle/input/datasets/srikarthikeyaa/{clean_p}",
        f"/kaggle/input/notebooks/srikarthikeyaa/{clean_p}",
        f"/kaggle/input/{clean_p}"
    ]
    for d in direct:
        if os.path.exists(d):
            print(f"find direct {p} -> {d} ({time.time()-t0:.3f}s)", flush=True)
            return d
    for pat in [f"/kaggle/input/*/*/{clean_p}", f"/kaggle/input/*/{clean_p}"]:
        hits = sorted(glob.glob(pat))
        if hits:
            print(f"find shallow {p} -> {hits[0]} ({time.time()-t0:.3f}s)", flush=True)
            return hits[0]
    fn = os.path.basename(p)
    hits = sorted(glob.glob(f"/kaggle/input/*/*/{fn}") + glob.glob(f"/kaggle/input/*/*/*/{fn}") + glob.glob(f"/kaggle/input/*/*/*/*/{fn}"))
    if hits:
        print(f"find depth-4 {p} -> {hits[0]} ({time.time()-t0:.3f}s)", flush=True)
        return hits[0]
    raise FileNotFoundError(f"Could not find {p} anywhere in /kaggle/input (checked in {time.time()-t0:.3f}s)")

ASSETS = os.path.dirname(find("TRASER/config.json")); ROOT = os.path.dirname(ASSETS)
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
LOCAL = False; OUT = "/kaggle/working"
import json as _j
_qc = []
for _p in glob.glob("/kaggle/input/**/config.json", recursive=True):
    try: _mt = _j.load(open(_p)).get("model_type")
    except Exception: _mt = None
    _d = os.path.dirname(_p); _w = glob.glob(_d + "/*.safetensors")
    if "qwen" in _p.lower() or _mt == "qwen2_5_vl": print("config candidate", _p, _mt, len(_w), flush=True)
    if _mt == "qwen2_5_vl" and _w and "TRASER" not in _p: _qc.append(_d)
if _qc:
    QWEN_DIR = sorted(_qc)[0]
else:
    raise FileNotFoundError("Could not find base Qwen2.5-VL weights")
PROC_DIR = f"{ROOT}/Qwen2.5-VL-3B-Instruct" if os.path.exists(f"{ROOT}/Qwen2.5-VL-3B-Instruct/preprocessor_config.json") else QWEN_DIR
print("PROC_DIR", PROC_DIR, flush=True)
print("QWEN_DIR", QWEN_DIR, flush=True)
def _opt(p):
    r = sorted(glob.glob(f"/kaggle/input/{p}", recursive=True)); return os.path.dirname(r[0]) if r else None
TRAIN_DIRS = {k: d for k, d in (("vidor", _opt("**/train_vidor/**/meta_train.json")), ("pvd", _opt("**/train_pvd/**/meta_train.json")),
              ("sav", _opt("**/traser-train-sav/**/meta_train.json"))) if d}            # optional: test decode does not need them
CANDS = find("**/qtm_cands.json"); _tc = glob.glob("/kaggle/input/**/qtm_test_cands.json", recursive=True); TEST_CANDS = _tc[0] if _tc else None
SV = os.path.dirname(find("**/traser-svg2sav/**/masks_sav.parquet")); VIP = find("**/masks_vipseg.parquet")
VIDOR_DIR = os.path.dirname(find("**/vidor_meta.json")); PVSG_DIR = os.path.dirname(find("**/pvsg-val-traser/**/meta.json"))
def _root(p, slug):
    parts = p.split("/"); return "/".join(parts[:parts.index(slug) + 1]) if slug in parts else os.path.dirname(p)
BENCHES = {"svg2": dict(masks=VIP, gt=find("**/gt_vipseg.parquet"), root=_root(VIP, "svg2test-vipseg"), stride=3, fps=2.0),
           "sav": dict(masks=f"{SV}/masks_sav.parquet", gt=f"{SV}/gt_sav.parquet", root=SV, stride=12, fps=2.0),
           "vidor": dict(masks=f"{VIDOR_DIR}/masks_vidor.parquet", gt=f"{VIDOR_DIR}/gt_vidor.parquet", root=VIDOR_DIR, stride=1, meta=f"{VIDOR_DIR}/vidor_meta.json"),
           "pvsg": dict(masks=f"{PVSG_DIR}/masks.parquet", gt=f"{PVSG_DIR}/gt.parquet", root=PVSG_DIR, stride=1, meta=f"{PVSG_DIR}/meta.json")}
N_DEV, LIMIT_VIDS, EPOCHS, ACCUM, LOG_EVERY, TRAIN_SECONDS = 40, 10 ** 9, 2, 8, 20, 6.0 * 3600
MAX_FRAMES, MAX_PIXELS, MAX_LEN, TURNS_PER_SEQ, NEG_RATIO = 32, 336 * 588, 9000, 12, 0.5
MAX_NEW_GRAPH, MAX_NEW_PAIR, PAIR_BATCH, GRAPH_BATCH, EVAL_SECONDS = 1024, 256, 16, 8, 11 * 3600
ARM, TAG = 'QE', 'v4c2'
LORA_R, LR, LR_MARK = 64, 1e-4, 1e-3
DEV_DEADLINE = 7.5 * 3600 if not LOCAL else 10 ** 9
GR_LR, GR_STEPS, GR_VIDEOS, GR_PAIRS, G_SAMPLES, W_NONE = 2e-5, 120, 4, 4, 8, 0.5
ZERO_SHOT = TAG == "q0"
_W = "QG" if (ARM == "QE" and TAG == "qg") else "QS"                     # which trained weights QE / QG start from
QS_LORA = None if ZERO_SHOT or ARM not in ("QE", "QG") else (find(f"**/lora_{_W}.pt") if not LOCAL else OUT + f"/lora_{_W}.pt")
QS_MARKS = None if QS_LORA is None else (find(f"**/marks_{_W}.pt") if not LOCAL else OUT + f"/marks_{_W}.pt")

SKIP_DEV = True
# ==== QTM: base Qwen2.5-VL-3B + Token Marks (Omni-RGPT, CVPR 2025, re-implemented) for video scene graphs.
# Plan: plans/qwen_base_pairs.md. Objects are pointed at by K learnable mark vectors: mark k is ADDED to every merged visual
# token (2 frames x 28x28 px) covered by object k's mask, and the same vector replaces the text token <|mark_k|>.
# Output format = TraSeR's ({"objects": [{"object k": name}], "relationships": [[s, p, o, [[t0, t1]]]]}), so the repo scorer works.
import os, re, json, math, time, random, glob, gc, collections
import numpy as np, torch, torch.nn as nn, pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as Mk

K_MARKS, COVER = 40, 0.3
SYS = "You are a helpful assistant."
MARK_TOKENS = [f"<|mark_{k}|>" for k in range(1, K_MARKS + 1)]
TASK_GRAPH = ('Output the video scene graph as JSON: {"objects": [{"object 1": name}, ...], "relationships": '
              '[[subject, predicate, object, [[start_sec, end_sec]]], ...]}. Use the object numbers above.')
TASK_PAIR = ('Output the relationships between object {i} and object {j} as JSON: {{"objects": [{{"object {i}": name}}, '
             '{{"object {j}": name}}], "relationships": [[subject, predicate, object, [[start_sec, end_sec]]], ...]}}. '
             'Use an empty relationships list if there is none.')
TASK_VIS = "When is object {k} visible? Answer as JSON [[start_sec, end_sec], ...]."
TASK_TOUCH = "When do object {i} and object {j} touch? Answer as JSON [[start_sec, end_sec], ...], or [] if never."

# ---------------------------------------------------------------- data access (same layout as the e1 kernels)
def load_rows(path): return {str(r["video_id"]): r for r in pq.read_table(path).to_pylist()}

def list_frames(root, v):
    for d in (f"{root}/frames/{v}", f"{root}/{v}"):
        f = sorted(glob.glob(f"{d}/*.jpg"))
        if f: return f
    hits = [p for p in glob.glob(f"{root}/**/{v}", recursive=True) if os.path.isdir(p)]
    return sorted(glob.glob(f"{hits[0]}/*.jpg")) if hits else []

class VideoSrc:
    """One video: jpg frames (already TraSeR-sampled), RLE masks at those frames (every `stride`-th mask row), fps of the jpgs."""
    def __init__(self, vid, mask_row, frames, fps, stride=1, max_obj=K_MARKS):
        r = mask_row; self.vid, self.H, self.W = vid, int(r["mask_height"]), int(r["mask_width"])
        self.no = int(r["num_objects"]); nf = int(r["num_frames"]); self.counts = r["counts"]
        self.rows = list(range(0, nf, stride))[:len(frames)]; self.frames = frames[:len(self.rows)]
        self.fps = float(fps); self.kept = list(range(min(self.no, max_obj)))
    def mask(self, fi, ob):
        rle = {"size": [self.H, self.W], "counts": self.counts[self.rows[fi] * self.no + ob]}
        if isinstance(rle["counts"], str): rle["counts"] = rle["counts"].encode()
        return Mk.decode(rle).astype(bool)

def pick_frames(n, max_frames):
    idx = list(range(n)) if n <= max_frames else sorted(set(np.linspace(0, n - 1, max_frames).round().astype(int).tolist()))
    if len(idx) % 2: idx = idx + [idx[-1]]                          # Qwen2.5-VL merges frame pairs
    return idx

def target_hw(H, W, max_pixels):
    s = min(1.0, math.sqrt(max_pixels / (H * W)))
    return max(56, int(round(H * s / 28)) * 28), max(56, int(round(W * s / 28)) * 28)

def prepare_video(src, max_frames, max_pixels):
    """-> dict(frames uint8 [T,h,w,3], fps_used, times (sec of each used frame), cover [t2, h/28, w/28, n_obj] float, kept)."""
    idx = pick_frames(len(src.frames), max_frames); h, w = target_hw(src.H, src.W, max_pixels)
    fr = np.stack([np.array(Image.open(src.frames[i]).convert("RGB").resize((w, h), Image.BILINEAR)) for i in idx])
    n = len(src.kept); gh, gw = h // 28, w // 28; cov = np.zeros((len(idx) // 2, gh, gw, n), np.float32)
    for fi_pos, fi in enumerate(idx):
        for j, ob in enumerate(src.kept):
            m = src.mask(fi, ob)
            if not m.any(): continue
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
            c = m.reshape(gh, 28, gw, 28).mean((1, 3))
            cov[fi_pos // 2, :, :, j] = np.maximum(cov[fi_pos // 2, :, :, j], c)
    span = (len(src.frames) - 1) / src.fps if len(src.frames) > 1 else 1.0
    fps_used = (len(idx) - 1) / span if span > 0 and len(idx) > 1 else src.fps
    return dict(frames=fr, fps_used=max(fps_used, 0.05), times=[i / src.fps for i in idx], cover=cov, kept=src.kept)

def mark_map(cover):
    """cover [t2, gh, gw, n] -> binary [t2*gh*gw, K] (raster order of Qwen2.5-VL merged video tokens). Cells >= COVER, plus the
    best cell of each frame pair where the object is visible at all (small objects still get a mark)."""
    t2, gh, gw, n = cover.shape; M = (cover >= COVER).astype(np.float32)
    flat = cover.reshape(t2, gh * gw, n)
    for t in range(t2):
        for j in range(n):
            if flat[t, :, j].max() > 0 and not M[t].reshape(gh * gw, n)[:, j].any():
                c = int(flat[t, :, j].argmax()); M[t, c // gw, c % gw, j] = 1.0
    out = np.zeros((t2 * gh * gw, K_MARKS), np.float32); out[:, :n] = M.reshape(-1, n)
    return torch.from_numpy(out)

# ---------------------------------------------------------------- Token Marks on the model
class TokenMarks(nn.Module):
    def __init__(self, model, tok, K=K_MARKS):
        super().__init__()
        D = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
        q, _ = torch.linalg.qr(torch.randn(D, K)); self.M = nn.Parameter(q.t().contiguous().float())     # orthogonal rows
        self.scaled = False; self.maps = None; self.cache = None
        self.ids = torch.tensor(tok.convert_tokens_to_ids(MARK_TOKENS)); assert int(self.ids.min()) >= 151643
        self.base_id = int(self.ids[0]); assert (self.ids == torch.arange(self.base_id, self.base_id + K)).all()
        inner = model.model; orig = inner.get_video_features; self.orig = orig
        def gvf(pixel_values_videos, video_grid_thw=None):
            feats = self.cache if self.cache is not None else orig(pixel_values_videos, video_grid_thw)
            if not self.scaled:                                     # match the marks' norm to half a visual token's norm
                with torch.no_grad(): self.M.mul_(0.5 * float(feats[0].float().norm(dim=-1).median()))
                self.scaled = True
            maps = self.maps; assert maps is not None and len(maps) == len(feats), (len(maps or []), len(feats))
            return tuple(f + (mp.to(f.device, self.M.dtype) @ self.M).to(f.dtype) for f, mp in zip(feats, maps))
        inner.get_video_features = gvf
        def hook(mod, inp, out):
            ids = inp[0]; rel = ids - self.base_id; sel = (rel >= 0) & (rel < K)
            if sel.any():
                out = out.clone(); out[sel] = self.M[rel[sel]].to(out.dtype)
            return out
        model.get_input_embeddings().register_forward_hook(hook)

def add_mark_tokens(tok):
    n = tok.add_special_tokens({"additional_special_tokens": MARK_TOKENS}, replace_additional_special_tokens=False)
    return n

# ---------------------------------------------------------------- prompts / targets
def objects_line(n): return "Objects: " + ", ".join(f"object {k} <|mark_{k}|>" for k in range(1, n + 1)) + "."

def build_chat(proc, n_obj, turns, first_extra=""):
    """turns: list of (user_text, assistant_text or None). The video + objects line go into the first user turn."""
    msgs = [{"role": "system", "content": SYS}]
    for i, (u, a) in enumerate(turns):
        content = ([{"type": "video"}, {"type": "text", "text": first_extra + objects_line(n_obj) + "\n" + u}] if i == 0 else [{"type": "text", "text": u}])
        msgs.append({"role": "user", "content": content})
        if a is not None: msgs.append({"role": "assistant", "content": [{"type": "text", "text": a}]})
    return proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=turns[-1][1] is None)

def names_of(label_objects):
    out = {}
    for ob in label_objects:
        k = next((x for x in ob if x.startswith("object_")), None)
        if k is not None: out[int(k.split("_")[1])] = str(ob[k])
    return out

def graph_target(label, kept):
    pos = {o: i + 1 for i, o in enumerate(kept)}; nm = names_of(json.loads(label["objects"]))
    objs = [{f"object {pos[o]}": nm.get(o, "object")} for o in kept]
    rels = [[pos[int(r[0])], str(r[1]), pos[int(r[2])], r[3]] for r in json.loads(label["relationships"]) if int(r[0]) in pos and int(r[2]) in pos]
    return json.dumps({"objects": objs, "relationships": rels})

def pair_target(label, kept, a, b):
    pos = {o: i + 1 for i, o in enumerate(kept)}; nm = names_of(json.loads(label["objects"]))
    rels = [[pos[int(r[0])], str(r[1]), pos[int(r[2])], r[3]] for r in json.loads(label["relationships"]) if {int(r[0]), int(r[2])} == {a, b}]
    return json.dumps({"objects": [{f"object {pos[a]}": nm.get(a, "object")}, {f"object {pos[b]}": nm.get(b, "object")}], "relationships": rels})

def spans_from_bool(times, on, step):
    out, st = [], None
    for t, x in zip(times, on):
        if x and st is None: st = t
        if not x and st is not None: out.append([round(st, 1), round(prev + step, 1)]); st = None
        prev = t
    if st is not None: out.append([round(st, 1), round(prev + step, 1)])
    return out

def free_tasks(src, prep, rng, n_vis=2, n_touch=2):
    """Label-free binding tasks from the masks (same videos, no new data): visibility and contact spans."""
    cov = prep["cover"]; t2 = cov.shape[0]; step = 2.0 / prep["fps_used"]; times = [prep["times"][2 * t] for t in range(t2)]
    vis = cov.reshape(t2, -1, cov.shape[-1]).max(1) > 0                # [t2, n]
    out = []; n = vis.shape[1]
    for j in rng.sample(range(n), min(n_vis, n)):
        out.append((TASK_VIS.format(k=j + 1), json.dumps(spans_from_bool(times, vis[:, j], step))))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (vis[:, i] & vis[:, j]).any()]
    for i, j in rng.sample(pairs, min(n_touch, len(pairs))):
        both = cov[..., i] > 0; other = cov[..., j] > 0                  # cell-level touch: adjacent or shared occupied cells
        dil = both.copy(); dil[:, 1:] |= both[:, :-1]; dil[:, :-1] |= both[:, 1:]; dil[:, :, 1:] |= both[:, :, :-1]; dil[:, :, :-1] |= both[:, :, 1:]
        on = (dil & other).reshape(t2, -1).any(1)
        out.append((TASK_TOUCH.format(i=i + 1, j=j + 1), json.dumps(spans_from_bool(times, on, step))))
    return out

# ---------------------------------------------------------------- tensors for one sequence
def encode(proc, text, prep, mm):
    """-> model inputs for one sequence (batch 1) + the mark map; labels mask only assistant answers."""
    enc = proc(text=[text], videos=[prep["frames"]], return_tensors="pt",
               videos_kwargs=dict(fps=prep["fps_used"], do_sample_frames=False, min_pixels=28 * 28, max_pixels=10 ** 8))
    nvt = int((enc["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|video_pad|>")).sum())
    assert nvt == mm.shape[0], (nvt, mm.shape)
    return enc

def assistant_labels(ids):
    """Loss on assistant answer tokens (+ <|im_end|>) only."""
    lab = torch.full_like(ids, -100); s = ids[0].tolist(); i = 0
    while i < len(s) - 2:
        if s[i] == 151644 and s[i + 1] == 77091 and s[i + 2] == 198:     # <|im_start|>assistant\n
            j = i + 3
            while j < len(s) and s[j] != 151645: j += 1
            lab[0, i + 3:min(j + 1, len(s))] = ids[0, i + 3:min(j + 1, len(s))]; i = j
        i += 1
    return lab

# ---------------------------------------------------------------- generation helpers
from transformers import LogitsProcessor, LogitsProcessorList
class GenOnlyRepetitionPenalty(LogitsProcessor):
    """Repetition penalty over GENERATED tokens only (TraSeR's convention). HF's built-in also penalises every prompt token,
    which suppresses JSON punctuation/digits when the prompt itself is JSON-like."""
    def __init__(self, prompt_len, penalty=1.05): self.n, self.p = prompt_len, penalty
    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.n:]
        if gen.shape[1] == 0: return scores
        sc = scores.gather(1, gen); sc = torch.where(sc < 0, sc * self.p, sc / self.p)
        return scores.scatter(1, gen, sc)
def _lp(enc): return LogitsProcessorList([GenOnlyRepetitionPenalty(enc["input_ids"].shape[1])])

def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception:
        try: return json.loads(m.group(0)[:m.group(0).rfind("]") + 1] + "]}")
        except Exception: return None

@torch.no_grad()
def generate(model, proc, tm, prep, mm, texts, max_new, feats_cache=None):
    """Greedy decode of several prompts that share ONE video (pair questions): the video is encoded once and reused."""
    tok = proc.tokenizer; tok.padding_side = "left"
    enc = proc(text=texts, videos=[prep["frames"]] * len(texts), return_tensors="pt", padding=True,
               videos_kwargs=dict(fps=prep["fps_used"], do_sample_frames=False, min_pixels=28 * 28, max_pixels=10 ** 8)).to(model.device)
    if feats_cache is None:                                              # encode the video once, without marks
        per = enc["pixel_values_videos"].shape[0] // len(texts)
        feats_cache = [f.detach() for f in tm.orig(enc["pixel_values_videos"][:per], enc["video_grid_thw"][:1])]
    tm.cache = list(feats_cache) * len(texts); tm.maps = [mm] * len(texts)
    model.model.rope_deltas = None
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, logits_processor=_lp(enc))
    tm.cache = None; tm.maps = None
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out], feats_cache

@torch.no_grad()
def generate_multi(model, proc, tm, preps, mms, texts, max_new):
    """Greedy decode of one prompt per video for several DIFFERENT videos in one batch (graph mode)."""
    tok = proc.tokenizer; tok.padding_side = "left"
    fps = [p["fps_used"] for p in preps]
    enc = proc(text=texts, videos=[p["frames"] for p in preps], return_tensors="pt", padding=True,
               videos_kwargs=dict(fps=fps, do_sample_frames=False, min_pixels=28 * 28, max_pixels=10 ** 8)).to(model.device)
    tm.cache = None; tm.maps = list(mms); model.model.rope_deltas = None
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, logits_processor=_lp(enc))
    tm.maps = None
    return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

"""Minimal LoRA SFT for TraSeR that reuses FastTraSeR's exact prompt construction (cpu_stage + gpu_stage), so training inputs
are identical to inference inputs. Targets follow TraSeR's training format (data_qwen._renumber_objects_and_relationships):
{"objects": [{"object k": label, "attributes": [...]?}], "relationships": [[s, p, o, [[a, b], ...]], ...]} with objects renumbered
1..K in kept order, then "<|im_end|>".

Style tags: the prompt key selects a tagged prompt (register_style_prompts). LoRA is merged into the base weights after training
(merge_lora), so the merged model runs in the fast engine unchanged."""
import json, math, torch, torch.nn as nn

CAMERA_ID = -1
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=64, alpha=128):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


def apply_lora(language_model, r=64, alpha=128, targets=LORA_TARGETS, freeze=None):
    """freeze: the full model (vision tower, resamplers, embeddings, lm_head are frozen too)."""
    for p in (freeze or language_model).parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(language_model.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                setattr(mod, t, LoRALinear(child, r, alpha)); n += 1
    return n


def lora_params(module):
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


def lora_state_dict(module):
    return {f"{name}.{k}": getattr(m, k).detach().cpu() for name, m in module.named_modules() if isinstance(m, LoRALinear) for k in ("A", "B")}


def load_lora_state_dict(module, sd):
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            m.A.data.copy_(sd[f"{name}.A"]); m.B.data.copy_(sd[f"{name}.B"])


def merge_lora(module):
    """Fold W += scale * B @ A and put the plain nn.Linear back."""
    for name, mod in list(module.named_modules()):
        for t in LORA_TARGETS:
            child = getattr(mod, t, None)
            if isinstance(child, LoRALinear):
                W = child.base.weight
                W.data += (child.scale * (child.B @ child.A)).to(W.dtype)
                setattr(mod, t, child.base)



import json, re
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\]?)')
NUM2 = re.compile(r'\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]')
def parse_rels(text):
    try:
        r = json.loads(text)["relationships"]
        if isinstance(r, list): return r
    except Exception: pass
    out = []
    for a, p, b, s in REL_RX.findall(text):
        spans = [[float(x), float(y)] for x, y in NUM2.findall(s)]
        out.append([int(a), p, int(b), spans])
    return out

"""Graph-matching reward against TraSeR's own training labels (fixed a priori, benchmark-agnostic).

R = F1_rel + 0.25 * object_accuracy - 0.2 * [output not a complete JSON object]
- Relations: each predicted relation [s, p, o, spans] is matched one-to-one (greedy by temporal IoU) to a label relation with the same
  ordered object pair and the same normalized predicate. A match earns its temporal IoU (soft credit, 0..1). Duplicates and repeats
  cannot match twice, so they count as false positives. F1 = 2PR/(P+R) with P = credit/#pred, R = credit/#label.
- Objects: predicted "object k" label equals the label of kept[k-1] after normalization.
Prediction ids are 1-based over `kept` (the prompt's object order); label ids are the original object ids."""
import json, re

def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("_", " ").strip())

def secs(spans):
    out = set()
    for ab in spans or []:
        try: a, b = float(ab[0]), float(ab[1])
        except Exception: continue
        if b < a: a, b = b, a
        x = a
        while x <= b + 1e-6:
            out.add(round(x * 2) / 2); x += 0.5
    return out

def tiou(a, b):
    A, B = secs(a), secs(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0

def pred_objects(text):
    out = {}
    for m in re.finditer(r'\{\s*"object (\d+)"\s*:\s*"([^"]*)"', text):
        out[int(m.group(1))] = m.group(2)
    return out

def reward(text, kept, label_objects, label_rels, w_obj=0.25, w_fmt=0.2):
    """Returns (R, details). label_objects: [{"object_<id>": name, ...}]; label_rels: [[s, p, o, spans, ...]]."""
    idx = {oid: i + 1 for i, oid in enumerate(kept)}
    G = []
    for r in label_rels:
        s, o = int(r[0]), int(r[2])
        if s in idx and o in idx: G.append((idx[s], norm(r[1]), idx[o], r[3]))
    P = []
    for r in parse_rels(text):
        try: P.append((int(r[0]), norm(r[1]), int(r[2]), r[3]))
        except Exception: continue
    cand = []
    for i, (s, p, o, sp) in enumerate(P):
        for j, (gs, gp, go, gsp) in enumerate(G):
            if s == gs and o == go and p == gp:
                cand.append((tiou(sp, gsp), i, j))
    used_p, used_g, credit = set(), set(), 0.0
    for c, i, j in sorted(cand, reverse=True):
        if i in used_p or j in used_g or c <= 0: continue
        used_p.add(i); used_g.add(j); credit += c
    prec = credit / len(P) if P else 0.0
    rec = credit / len(G) if G else (1.0 if not P else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    lab = {}
    for ob in label_objects:
        k = next((k for k in ob if k.startswith("object_")), None)
        if k is not None: lab[int(k.split("_")[1])] = norm(ob[k])
    po = pred_objects(text)
    obj_acc = sum(1 for k, oid in enumerate(kept) if norm(po.get(k + 1, "")) == lab.get(oid, "\0")) / max(1, len(kept))
    complete = text.rstrip().endswith("}")
    try: json.loads(text); valid = True
    except Exception: valid = False
    R = f1 + w_obj * obj_acc - (0.0 if (complete and valid) else w_fmt)
    return R, dict(f1=f1, p=prec, r=rec, obj=obj_acc, n_pred=len(P), n_label=len(G), valid=valid)


def reward_g4(text, kept, label_objects, label_rels, touching_pairs=None,
              w_obj=0.25, w_fmt=0.2, lam_fp=0.15):
    """G4 reward: base F1 + geometry-gated false-positive penalty.
    
    touching_pairs: set of frozenset({obj_id_a, obj_id_b}) for pairs
                    whose masks make dilated contact on any frame.
    """
    R_base, d = reward(text, kept, label_objects, label_rels, w_obj, w_fmt)
    if touching_pairs is None:
        touching_pairs = set()
    idx = {oid: i + 1 for i, oid in enumerate(kept)}
    labelled_pairs = set()
    for r in label_rels:
        try:
            s, o = int(r[0]), int(r[2])
            if s in idx and o in idx:
                labelled_pairs.add(frozenset((s, o)))
        except Exception:
            pass
    
    n_geo_fp = 0
    for r in parse_rels(text):
        try:
            s, o = int(r[0]), int(r[2])
            s_oid = kept[s - 1] if 1 <= s <= len(kept) else -1
            o_oid = kept[o - 1] if 1 <= o <= len(kept) else -1
            pair = frozenset((s_oid, o_oid))
            if pair not in labelled_pairs and pair not in touching_pairs:
                n_geo_fp += 1
        except Exception:
            pass
    
    n_pred = d["n_pred"]
    penalty = lam_fp * n_geo_fp / max(1, n_pred)
    R = R_base - penalty
    d["n_geo_fp"] = n_geo_fp
    d["penalty"] = round(penalty, 4)
    return R, d


# ==== QTM run: ARM = "QS" (SFT: Token Marks + LoRA on all pairs/graphs/free tasks, then dev graph decode)
#                ARM = "QE" (test decode: graph mode + pair mode for QS weights; Q0 zero-shot box-text graph mode)
# Config globals (ARM, paths, sizes) are prepended by build_qtm.py.
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
T0 = time.time(); INFO = {"arm": ARM}
dev = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- model
proc = AutoProcessor.from_pretrained(PROC_DIR); tok = proc.tokenizer; add_mark_tokens(tok)
if LOCAL:
    from transformers import Qwen2_5_VLConfig
    c = json.load(open(f"{PROC_DIR}/config.json"))
    c.update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, max_window_layers=2)
    c["vision_config"].update(depth=2, hidden_size=64, out_hidden_size=512, num_heads=2, intermediate_size=128, fullatt_block_indexes=[1])
    if "text_config" in c: c["text_config"].update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, layer_types=["full_attention"] * 2)
    if "layer_types" in c: c["layer_types"] = ["full_attention"] * 2
    cfg = Qwen2_5_VLConfig(**c); cfg._attn_implementation = "sdpa"; torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(cfg).float().eval()
else:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(QWEN_DIR, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
model.generation_config.temperature = None; model.generation_config.top_p = None; model.generation_config.top_k = None
tm = TokenMarks(model, tok).to(dev); lm = model.model.language_model
print("model ready", f"{time.time()-T0:.0f}s", flush=True)

def load_train_sources():
    srcs = {}
    for name, d in TRAIN_DIRS.items():
        masks, labels, meta = load_rows(f"{d}/masks_train.parquet"), load_rows(f"{d}/labels_train.parquet"), json.load(open(f"{d}/meta_train.json"))
        vids = [v for v in sorted(masks) if v in labels and list_frames(d, v)]
        srcs[name] = dict(dir=d, masks=masks, labels=labels, meta=meta, vids=vids)
    rng = random.Random(0); tr, dv = {}, {}                            # identical to split_train_dev (e1 kernels)
    for name, S in srcs.items():
        vv = list(S["vids"]); rng.shuffle(vv); dv[name] = vv[:N_DEV]; tr[name] = vv[N_DEV:]
    return srcs, tr, dv

def train_src(S, v):
    return VideoSrc(v, S["masks"][v], list_frames(S["dir"], v), S["meta"][v]["fps_eff"], stride=1)

def video_turns(name, S, v, prep, rng, cands):
    L = S["labels"][v]; kept = prep["kept"]; ks = set(kept); pos = {o: i + 1 for i, o in enumerate(kept)}
    rels = json.loads(L["relationships"])
    lab = sorted({tuple(sorted((int(r[0]), int(r[2])))) for r in rels if int(r[0]) in ks and int(r[2]) in ks and int(r[0]) != int(r[2])})
    turns = []
    for a, b in lab:
        i, j = sorted((pos[a], pos[b])); turns.append((TASK_PAIR.format(i=i, j=j), pair_target(L, kept, a, b)))
    labset = set(lab); nneg = math.ceil(NEG_RATIO * len(lab)); negs = []
    for e in cands.get(name, {}).get(v, []):
        a, b, p = int(e[0]), int(e[1]), float(e[2])
        if p >= 0.3 and a in ks and b in ks and tuple(sorted((a, b))) not in labset and len(negs) < nneg:
            i, j = sorted((pos[a], pos[b])); negs.append((TASK_PAIR.format(i=i, j=j), pair_target(L, kept, a, b)))
    return turns, negs

# ---------------------------------------------------------------- SFT
def run_sft():
    srcs, tr, dv = load_train_sources(); cands = json.load(open(CANDS))
    nl = apply_lora(lm, r=LORA_R, alpha=2 * LORA_R, freeze=model); tm.to(dev)
    loras = lora_params(lm); opt = torch.optim.AdamW([{"params": loras, "lr": LR}, {"params": [tm.M], "lr": LR_MARK}], weight_decay=0.0)
    items = [(n, v) for n in sorted(tr) for v in tr[n]][:LIMIT_VIDS * len(tr)]; random.Random(1).shuffle(items)
    print(f"SFT videos {len(items)} | lora layers {nl} | dev {[(n, len(x)) for n, x in dv.items()]}", flush=True)
    print("DEV IDS", json.dumps({n: dv[n] for n in sorted(dv)}), flush=True)
    total_steps = max(1, EPOCHS * len(items) * 3 // ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
    if not LOCAL: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train(); rng = random.Random(2); stats = collections.defaultdict(list); nseq = 0; step = 0
    import concurrent.futures as cf
    def prep_item(it):
        n, v = it; S = srcs[n]
        try:
            src = train_src(S, v); prep = prepare_video(src, MAX_FRAMES, MAX_PIXELS); return n, v, prep
        except Exception as e: return n, v, e
    stop = False
    for ep in range(EPOCHS):
        def bounded(ex, its, depth=24):                                   # at most `depth` prepared videos in RAM
            q = collections.deque()
            for it in its:
                q.append(ex.submit(prep_item, it))
                if len(q) >= depth: yield q.popleft().result()
            while q: yield q.popleft().result()
        with cf.ThreadPoolExecutor(6) as ex:
            for n, v, prep in bounded(ex, items):
                if isinstance(prep, Exception): stats["prep_err"].append(1); continue
                S = srcs[n]; mm = mark_map(prep["cover"]); nobj = len(prep["kept"])
                pos_t, neg_t = video_turns(n, S, v, prep, rng, cands); free = free_tasks(None, prep, rng)
                rest = pos_t + neg_t + free; rng.shuffle(rest)
                seqs = [[(TASK_GRAPH, graph_target(S["labels"][v], prep["kept"]))]] + [rest[i:i + TURNS_PER_SEQ] for i in range(0, len(rest), TURNS_PER_SEQ)]
                for sq in seqs:
                    text = build_chat(proc, nobj, sq)
                    try:
                        enc = encode(proc, text, prep, mm); labels = assistant_labels(enc["input_ids"])
                        if enc["input_ids"].shape[1] > MAX_LEN: stats["too_long"].append(1); continue
                        tm.maps = [mm]; tm.cache = None
                        out = model(**{k: v_.to(dev) for k, v_ in enc.items()}, labels=labels.to(dev))
                        (out.loss / ACCUM).backward(); stats["loss_graph" if len(sq) == 1 and sq[0][0] == TASK_GRAPH else "loss_turns"].append(float(out.loss))
                        stats["len"].append(enc["input_ids"].shape[1]); nseq += 1
                    except torch.OutOfMemoryError:
                        opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); stats["oom"].append(1); continue
                    finally:
                        tm.maps = None
                    if nseq % 2000 == 0: print("PROGRESS", json.dumps({"epoch": ep, "seq": nseq, "sec": round(time.time() - T0)}), flush=True)
                    if nseq % ACCUM == 0:
                        torch.nn.utils.clip_grad_norm_(loras + [tm.M], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                        if step % LOG_EVERY == 0:
                            print("SFT", json.dumps({"step": step, "seq": nseq, "sec": round(time.time() - T0), **{k: round(float(np.mean(x[-200:])), 4) for k, x in stats.items()}}), flush=True)
                    if time.time() - T0 > TRAIN_SECONDS: stop = True; break
                if stop: break
        if stop: print("time budget reached", flush=True); break
    model.eval()
    if not LOCAL: model.gradient_checkpointing_disable()
    torch.save(lora_state_dict(lm), f"{OUT}/lora_{ARM}.pt"); torch.save(tm.M.detach().cpu(), f"{OUT}/marks_{ARM}.pt")
    INFO.update(train_seconds=round(time.time() - T0), steps=step, sequences=nseq, **{k: round(float(np.mean(x)), 4) for k, x in stats.items()})
    # dev graph decode (single pass) -> scored locally with engine/reward.py (exact)
    with open(f"{OUT}/dev_graph.jsonl", "w") as fo:
        for n in sorted(dv):
            for v in dv[n][:LIMIT_VIDS]:
                if time.time() - T0 > DEV_DEADLINE: INFO["dev_skipped"] = INFO.get("dev_skipped", 0) + 1; continue   # never risk the 12 h kill
                try:
                    prep = prepare_video(train_src(srcs[n], v), MAX_FRAMES, MAX_PIXELS); mm = mark_map(prep["cover"])
                    txt, _ = generate(model, proc, tm, prep, mm, [build_chat(proc, len(prep["kept"]), [(TASK_GRAPH, None)])], MAX_NEW_GRAPH)
                    fo.write(json.dumps({"source": n, "video_id": v, "kept_obj_ids": prep["kept"], "text": txt[0]}) + "\n")
                except Exception as e:
                    fo.write(json.dumps({"source": n, "video_id": v, "error": str(e)[:200]}) + "\n")
    print("DEV DONE", flush=True)

# ---------------------------------------------------------------- test decode
def bench_sources():
    out = {}
    for bn, b in BENCHES.items():
        masks, gt = load_rows(b["masks"]), load_rows(b["gt"]); meta = json.load(open(b["meta"])) if b.get("meta") else None
        vids = [v for v in sorted(masks) if v in gt and list_frames(b["root"], v)][:LIMIT_VIDS]
        out[bn] = dict(b=b, masks=masks, meta=meta, vids=vids)
    return out

def bench_src(B, v):
    b = B["b"]; fps = B["meta"][v]["fps_eff"] if B["meta"] else b["fps"]
    return VideoSrc(v, B["masks"][v], list_frames(b["root"], v), fps, stride=b["stride"])

Q0_COMPACT = " Output compact JSON on a single line (no code fence, no indentation); give each object a short name."
def boxes_line(prep, n_times=8):
    """Q0 (zero-shot) object reference: per-object boxes (pixels of the model's input frames) at up to n_times moments."""
    cov = prep["cover"]; t2 = cov.shape[0]; ts = sorted(set(np.linspace(0, t2 - 1, min(n_times, t2)).round().astype(int).tolist()))
    lines = []
    for j in range(cov.shape[-1]):
        parts = []
        for t in ts:
            ys, xs = np.nonzero(cov[t, :, :, j] > 0)
            if len(ys): parts.append(f"t={prep['times'][2 * t]:.0f}s [{xs.min()*28},{ys.min()*28},{(xs.max()+1)*28},{(ys.max()+1)*28}]")
        lines.append(f"object {j + 1}: " + ("; ".join(parts) if parts else "not visible"))
    return "Objects (bounding boxes [x1,y1,x2,y2] in pixels of the video frames):\n" + "\n".join(lines) + "\n"

def run_eval():
    BS = bench_sources(); tcands = json.load(open(TEST_CANDS)) if TEST_CANDS else {}
    weights = {}
    if QS_LORA:
        apply_lora(lm, r=LORA_R, alpha=2 * LORA_R, freeze=model); load_lora_state_dict(lm, torch.load(QS_LORA, map_location=dev))
        tm.M.data.copy_(torch.load(QS_MARKS, map_location=dev)); tm.scaled = True; tm.to(dev)
    if not ZERO_SHOT and not globals().get("SKIP_DEV") and TRAIN_DIRS:  # held-out dev graphs with the corrected decoder (pre-registered dev check)
        srcs, tr, dv = load_train_sources()
        with open(f"{OUT}/dev_graph_{TAG}.jsonl", "w") as fo:
            for n in sorted(dv):
                for v in dv[n][:LIMIT_VIDS]:
                    try:
                        prep = prepare_video(train_src(srcs[n], v), MAX_FRAMES, MAX_PIXELS); mm = mark_map(prep["cover"])
                        txt, _ = generate(model, proc, tm, prep, mm, [build_chat(proc, len(prep["kept"]), [(TASK_GRAPH, None)])], MAX_NEW_GRAPH)
                        fo.write(json.dumps({"source": n, "video_id": v, "kept_obj_ids": prep["kept"], "text": txt[0]}) + "\n")
                    except Exception as e: fo.write(json.dumps({"source": n, "video_id": v, "error": str(e)[:200]}) + "\n")
        print("DEV DONE", flush=True)
    for bn, B in BS.items():
        t1 = time.time(); fg = open(f"{OUT}/{bn}_graph_{TAG}.jsonl", "w"); fp = open(f"{OUT}/{bn}_pairs_{TAG}.jsonl", "w"); npairs = 0
        if ZERO_SHOT:                                                    # graph mode only: batch several videos per generate call
            for c0 in range(0, len(B["vids"]), GRAPH_BATCH):
                if time.time() - T0 > EVAL_SECONDS: INFO.setdefault("skipped", {})[bn] = len(B["vids"]) - c0; break
                ch = B["vids"][c0:c0 + GRAPH_BATCH]; preps, mms, prompts, ok = [], [], [], []
                for v in ch:
                    try:
                        pr = prepare_video(bench_src(B, v), MAX_FRAMES, MAX_PIXELS); preps.append(pr); mms.append(torch.zeros(mark_map(pr["cover"]).shape)); ok.append(v)
                        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": boxes_line(pr) + TASK_GRAPH + Q0_COMPACT}]}]
                        prompts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                    except Exception as e: fg.write(json.dumps({"video_id": v, "error": str(e)[:300]}) + "\n")
                try: outs = generate_multi(model, proc, tm, preps, mms, prompts, MAX_NEW_GRAPH) if ok else []
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); outs = [generate_multi(model, proc, tm, [p], [m], [t], MAX_NEW_GRAPH)[0] for p, m, t in zip(preps, mms, prompts)]
                for v, pr, o in zip(ok, preps, outs): fg.write(json.dumps({"video_id": v, "kept_obj_ids": pr["kept"], "text": o}) + "\n")
                fg.flush(); gc.collect(); torch.cuda.empty_cache()
            fg.close(); fp.close(); INFO[bn] = dict(videos=len(B["vids"]), seconds=round(time.time() - t1)); print("BENCH", bn, INFO[bn], flush=True)
            continue
        for v in B["vids"]:
            if time.time() - T0 > EVAL_SECONDS: INFO.setdefault("skipped", {})[bn] = INFO.get("skipped", {}).get(bn, 0) + 1; continue
            try:
                prep = prepare_video(bench_src(B, v), MAX_FRAMES, MAX_PIXELS); kept = prep["kept"]; n = len(kept)
                if ZERO_SHOT:
                    mm = torch.zeros(mark_map(prep["cover"]).shape)                 # no marks: objects given as box text
                    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "video"}, {"type": "text", "text": boxes_line(prep) + TASK_GRAPH + Q0_COMPACT}]}]
                    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                else:
                    mm = mark_map(prep["cover"]); prompt = build_chat(proc, n, [(TASK_GRAPH, None)])
                g, feats = generate(model, proc, tm, prep, mm, [prompt], MAX_NEW_GRAPH)
                fg.write(json.dumps({"video_id": v, "kept_obj_ids": kept, "text": g[0]}) + "\n")
                if ZERO_SHOT or not tcands: continue
                J = parse_json(g[0]) or {}; men = set()
                for r in J.get("relationships", []) if isinstance(J, dict) else []:
                    try: men.add(frozenset((kept[int(r[0]) - 1], kept[int(r[2]) - 1])))
                    except Exception: pass
                ks = set(kept); pos = {o: i + 1 for i, o in enumerate(kept)}
                C = [e for e in tcands.get(bn, {}).get(v, []) if float(e[2]) >= 0.5 and int(e[0]) in ks and int(e[1]) in ks and frozenset((int(e[0]), int(e[1]))) not in men][:40]
                # batch only prompts of IDENTICAL token length (no padding: padded multi-row generation was corrupt on Kaggle, §94)
                allp = [(e, build_chat(proc, n, [(TASK_PAIR.format(i=min(pos[int(e[0])], pos[int(e[1])]), j=max(pos[int(e[0])], pos[int(e[1])])), None)])) for e in C]
                bylen = collections.defaultdict(list)
                for e, q in allp: bylen[len(tok(q)["input_ids"])].append((e, q))
                groups = [g[c0:c0 + PAIR_BATCH] for g in bylen.values() for c0 in range(0, len(g), PAIR_BATCH)]
                for grp in groups:
                    ch = [e for e, _ in grp]; prompts = [q for _, q in grp]
                    outs, _ = generate(model, proc, tm, prep, mm, prompts, MAX_NEW_PAIR, feats_cache=feats)
                    for e, o in zip(ch, outs):
                        fp.write(json.dumps({"video_id": v, "pair": [int(e[0]), int(e[1])], "p": float(e[2]), "kept_obj_ids": kept, "text": o}) + "\n"); npairs += 1
            except Exception as e:
                fg.write(json.dumps({"video_id": v, "error": str(e)[:300]}) + "\n")
            gc.collect(); torch.cuda.empty_cache()
        fg.close(); fp.close(); INFO[bn] = dict(videos=len(B["vids"]), pairs=npairs, seconds=round(time.time() - t1))
        print("BENCH", bn, INFO[bn], flush=True)


# ---------------------------------------------------------------- GRPO on pairs (QG): same reward as PG, starting from QS
def pair_reward_q(text, kept, a, b, L):
    """PG's pair reward with global object numbering: labelled pair -> rel F1 (engine/reward.py rule) + 0.25 x name accuracy over
    the TWO objects - 0.2 if the answer is not complete valid JSON; unlabelled pair -> W_NONE if no relation else 0."""
    rels = json.loads(L["relationships"]); pr = [r for r in rels if {int(r[0]), int(r[2])} == {a, b}]
    npred = len(parse_rels(text))
    if not pr: return (W_NONE if npred == 0 else 0.0), dict(labelled=False, none=npred == 0)
    _, d = reward(text, kept, json.loads(L["objects"]), pr, w_obj=0.0, w_fmt=0.0)
    pos = {o: i + 1 for i, o in enumerate(kept)}; nm = names_of(json.loads(L["objects"])); po = pred_objects(text)
    obj = np.mean([norm(po.get(pos[o], "")) == norm(nm.get(o, "\0")) for o in (a, b)])
    try: json.loads(text.strip()); ok = text.strip().endswith("}")
    except Exception: ok = False
    return d["f1"] + 0.25 * obj - (0.0 if ok else 0.2), dict(labelled=True, none=npred == 0, f1=d["f1"])

def seq_logprob(enc, n_prompt, mm, feats):
    """Mean log-prob of the completion tokens of each row (rows share one video; prompt length n_prompt incl. left padding)."""
    B = enc["input_ids"].shape[0]; tm.cache = list(feats) * B; tm.maps = [mm] * B; model.model.rope_deltas = None
    keep = enc["input_ids"].shape[1] - n_prompt + 1                      # logits only for the completion positions
    out = model(**enc, logits_to_keep=keep); tm.cache = None; tm.maps = None
    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    tgt = enc["input_ids"][:, n_prompt:]; m = enc["attention_mask"][:, n_prompt:].float()
    tl = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (tl * m).sum(1) / m.sum(1).clamp(min=1)

def run_grpo():
    srcs, tr, dv = load_train_sources(); cands = json.load(open(CANDS))
    apply_lora(lm, r=LORA_R, alpha=2 * LORA_R, freeze=model); load_lora_state_dict(lm, torch.load(QS_LORA, map_location=dev))
    tm.M.data.copy_(torch.load(QS_MARKS, map_location=dev)); tm.scaled = True; tm.to(dev)
    loras = lora_params(lm); opt = torch.optim.AdamW([{"params": loras, "lr": GR_LR}, {"params": [tm.M], "lr": GR_LR}], weight_decay=0.0)
    rng = random.Random(3); vids = [(n, v) for n in sorted(tr) for v in tr[n]]; rng.shuffle(vids)
    if not LOCAL: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tok.padding_side = "left"; hist = []; vi = 0
    for step in range(1, GR_STEPS + 1):
        if time.time() - T0 > TRAIN_SECONDS: print("time budget reached", step, flush=True); break
        st = collections.defaultdict(list); opt.zero_grad(set_to_none=True); nupd = 0
        for _ in range(GR_VIDEOS):                                      # GR_VIDEOS videos x GR_PAIRS pairs x G samples per step
            n, v = vids[vi % len(vids)]; vi += 1; S = srcs[n]
            try: prep = prepare_video(train_src(S, v), MAX_FRAMES, MAX_PIXELS)
            except Exception: continue
            mm = mark_map(prep["cover"]); kept = prep["kept"]; L = S["labels"][v]
            pos_t, neg_t = video_turns(n, S, v, prep, rng, cands)
            pool = [(t, True) for t in pos_t] + [(t, False) for t in neg_t]; rng.shuffle(pool)
            feats = None
            for (q, _tgt), _lab in pool[:GR_PAIRS]:
                prompt = build_chat(proc, len(kept), [(q, None)])
                i, j = [int(x) for x in re.findall(r"object (\d+)", q)[:2]]; a, b = kept[i - 1], kept[j - 1]
                model.eval()
                with torch.no_grad():
                    enc = proc(text=[prompt], videos=[prep["frames"]], return_tensors="pt", videos_kwargs=dict(fps=prep["fps_used"], do_sample_frames=False, min_pixels=784, max_pixels=10 ** 8)).to(dev)
                    if feats is None: feats = [f.detach() for f in tm.orig(enc["pixel_values_videos"], enc["video_grid_thw"])]
                    encG = dict(input_ids=enc["input_ids"].repeat(G_SAMPLES, 1), attention_mask=enc["attention_mask"].repeat(G_SAMPLES, 1),
                                pixel_values_videos=enc["pixel_values_videos"].repeat(G_SAMPLES, 1), video_grid_thw=enc["video_grid_thw"].repeat(G_SAMPLES, 1),
                                second_per_grid_ts=enc["second_per_grid_ts"].repeat(G_SAMPLES))      # G copies = G samples (video encoded once, cached)
                    tm.cache = list(feats) * G_SAMPLES; tm.maps = [mm] * G_SAMPLES; model.model.rope_deltas = None
                    gen = model.generate(**encG, max_new_tokens=MAX_NEW_PAIR, do_sample=True, temperature=1.0, top_p=0.95)
                    tm.cache = None; tm.maps = None
                np_ = enc["input_ids"].shape[1]; texts = [tok.decode(g[np_:], skip_special_tokens=True) for g in gen]
                rd = [pair_reward_q(t, kept, a, b, L) for t in texts]; rs = np.array([r for r, _ in rd])
                if globals().get("NOISE_REWARD"): rs = rs + np.random.rand(len(rs))           # local smoke only: exercise the update path
                lab = rd[0][1]["labelled"]; st["R_lab" if lab else "R_neg"].append(rs.mean()); st["none_lab" if lab else "none_neg"].append(np.mean([d["none"] for _, d in rd]))
                if rs.std() < 1e-6: st["zero_var"].append(1); continue
                st["zero_var"].append(0); adv = torch.tensor(rs - rs.mean(), device=dev, dtype=torch.float32)
                model.train()
                att = (gen != tok.pad_token_id).long(); att[:, :np_] = enc["attention_mask"].expand(G_SAMPLES, -1)
                encg = dict(input_ids=gen, attention_mask=att, pixel_values_videos=enc["pixel_values_videos"].repeat(G_SAMPLES, 1),
                            video_grid_thw=enc["video_grid_thw"].repeat(G_SAMPLES, 1), second_per_grid_ts=enc["second_per_grid_ts"].repeat(G_SAMPLES))
                try:
                    lp = seq_logprob(encg, np_, mm, feats); loss = -(adv * lp).mean() / (GR_VIDEOS * GR_PAIRS)
                    loss.backward(); nupd += 1
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache(); st["oom"].append(1)
        if nupd: torch.nn.utils.clip_grad_norm_(loras + [tm.M], 1.0); opt.step()
        rec = dict(step=step, upd=nupd, sec=round(time.time() - T0), **{k: round(float(np.mean(x)), 4) for k, x in st.items()}); hist.append(rec)
        print("QG", json.dumps(rec), flush=True)
    model.eval()
    if not LOCAL: model.gradient_checkpointing_disable()
    torch.save(lora_state_dict(lm), f"{OUT}/lora_QG.pt"); torch.save(tm.M.detach().cpu(), f"{OUT}/marks_QG.pt"); json.dump(hist, open(f"{OUT}/qg_hist.json", "w"))
    INFO.update(steps=len(hist), train_seconds=round(time.time() - T0))

if ARM == "QS": run_sft()
elif ARM == "QG": run_grpo()
elif ARM == "QE": run_eval()
json.dump(INFO, open(f"{OUT}/info_{ARM}_{TAG}.json", "w")); print("QTM DONE", json.dumps(INFO), flush=True)
