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

PREFIX = b"SVG2_PACKED_0001"
TAR_INDEX = {}                                                            # root -> {video: [(tar path, byte offset, size), ...]}
def _tar_sig(root):
    fs = sorted(set(glob.glob(f"{root}/frames_*.bin") + glob.glob(f"{root}/frames_*.tar*")))
    return [(p, os.path.getsize(p), int(os.path.getmtime(p))) for p in fs]

def index_fetch_dir(root):
    """Cached wrapper: reuse {root}/.tar_index_v1.pkl when the archive list/sizes/mtimes are unchanged (indexing over NAS takes ~5 min/source)."""
    import pickle
    cp, sig = f"{root}/.tar_index_v1.pkl", _tar_sig(root)
    try:
        with open(cp, "rb") as fh: c = pickle.load(fh)
        if c["sig"] == sig and c["index"]:
            TAR_INDEX[root] = c["index"]; print("tar index cache hit", root, flush=True); return len(c["index"])
    except Exception:
        pass
    n = _index_fetch_dir_scan(root)
    if TAR_INDEX.get(root):
        try:
            with open(cp + ".tmp", "wb") as fh: pickle.dump({"sig": sig, "index": TAR_INDEX[root]}, fh, protocol=5)
            os.replace(cp + ".tmp", cp)
        except Exception as e:
            print("tar index cache not written:", e, flush=True)
    return n

def _index_fetch_dir_scan(root):
    """Packed fetch output (frames_NNN.tar / frames_NNN.bin): index every frame's byte range once; frames are then read with seek+read (thread-safe)."""
    import tarfile
    idx = collections.defaultdict(list)
    bins = sorted(glob.glob(f"{root}/frames_*.bin"))
    tars = sorted(glob.glob(f"{root}/frames_*.tar*"))
    for tp in sorted(set(bins + tars)):
        try:
            with open(tp, "rb") as fh:
                head = fh.read(len(PREFIX))
                if head == PREFIX:
                    fh.seek(len(PREFIX))
                else:
                    fh.seek(0)
                with tarfile.open(fileobj=fh, mode="r:*") as tf:
                    for m in tf:
                        if m.isfile() and m.name.startswith("frames/"):
                            parts = m.name.split("/")
                            if len(parts) == 3:
                                _, v, fn = parts; idx[v].append((fn, tp, m.offset_data, m.size))
        except Exception as e:
            print(f"tar error {tp}: {e}", flush=True)
    if idx:
        TAR_INDEX[root] = {v: [(tp, off, sz) for _, tp, off, sz in sorted(L)] for v, L in idx.items()}
        return len(TAR_INDEX[root])
    vids = {os.path.basename(p) for p in glob.glob(f"{root}/frames/*") if os.path.isdir(p)}
    return len(vids)

def open_frame(f):
    if isinstance(f, tuple):
        import io
        with open(f[0], "rb") as fh: fh.seek(f[1]); return Image.open(io.BytesIO(fh.read(f[2])))
    return Image.open(f)

def list_frames(root, v):
    if root in TAR_INDEX and TAR_INDEX[root]: return TAR_INDEX[root].get(v, [])
    for d in (f"{root}/frames/{v}", f"{root}/{v}"):
        f = sorted(glob.glob(f"{d}/*.jpg"))
        if f: return f
    hits = [p for p in glob.glob(f"{root}/*/{v}") if os.path.isdir(p)]
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
    fr = np.stack([np.array(open_frame(src.frames[i]).convert("RGB").resize((w, h), Image.BILINEAR)) for i in idx])
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

def encode_fast(proc, text, prep, mm, vid=None):
    """encode() with the video processed ONCE per video: the first call runs the processor and returns the video tensors
    (`vid`); later sequences of the same video only tokenize the text, with <|video_pad|> expanded by hand as the processor
    does (checked against encode() at run time, VERIFY_FAST)."""
    if vid is None:
        enc = encode(proc, text, prep, mm); return enc, {k: x for k, x in enc.items() if k not in ("input_ids", "attention_mask")}
    t = proc.tokenizer([text.replace("<|video_pad|>", "<|video_pad|>" * mm.shape[0], 1)], return_tensors="pt")
    return {"input_ids": t["input_ids"], "attention_mask": t["attention_mask"], **vid}, vid

def pack_turns(tok, turns, budget):
    """Greedy chunks of turns whose estimated text length fits `budget` tokens (all turns of a video in one sequence when they fit)."""
    out, cur, n = [], [], 0
    for q, a in turns:
        c = len(tok(q)["input_ids"]) + len(tok(a)["input_ids"]) + 12
        if cur and n + c > budget: out.append(cur); cur, n = [], 0
        cur.append((q, a)); n += c
    return out + ([cur] if cur else [])

def answer_loss(model, enc, labels):
    """Same value as model(**enc, labels=labels).loss, but lm_head logits only for answer positions (not the ~2.5k video tokens)."""
    lab = labels[0]; f = int((lab != -100).nonzero()[0]); keep = lab.shape[0] - f + 1
    out = model(**enc, logits_to_keep=keep)
    return torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), lab[f:].to(out.logits.device), ignore_index=-100)

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
