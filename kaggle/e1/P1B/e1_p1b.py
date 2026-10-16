import glob, json, os, subprocess, sys, time, math, random, itertools, collections, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def find(p, root="/kaggle/input"):
    h = sorted(glob.glob(f"{root}/{p}", recursive=True)); print("find", p, "->", h[:2], flush=True); return h[0]
ASSETS = os.path.dirname(find("**/TRASER/config.json")); ROOT = os.path.dirname(ASSETS); OUT = "/kaggle/working"
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
"""Fast batched inference for TraSeR (released checkpoint, unchanged weights).

Speed-ups (each smoke-tested in engine/smoke_test.py):
  1. Left-padded multi-sequence batching of inputs_embeds with per-row rope_delta correction.
  2. Length-aware planning: sort by expected output length (tokens ~ objects) and pack batches under a KV-memory budget.
  3. Safe per-batch token cap: 300 + 200 * max_objects (no clean output on our benchmarks exceeds it).
  4. Loop stop (LoopStop3): attribute list > 30 items, or a 300-char chunk repeated >= 2 times inside the relationships section,
     checked every `every` tokens on decoded text (fast tokenizer).
  5. Prefill computes logits only for the last position (avoids a [B, L, 151k] tensor).
  6. CPU preparation (frame loading, video preprocessing, mask decoding) runs in a background thread pool.
  7. Pair decodes (two-object queries) are batched the same way.
  8. GQA-aware masked decode attention (install_gqa_decode_attention): with a padding mask, transformers' SDPA path calls
     repeat_kv, copying the whole KV cache 8x per layer per step (measured 4-5x slower decode). For q_len == 1 we instead fold
     the 8 query heads sharing a KV head into the query-length axis, so SDPA reads the cache once with the padding mask.
  9. LoopStop4 = LoopStop3 + novelty stall (among the last 40 relation tuples, fewer than 8 unseen (s,p,o) triples).
 10. SlotDecoder: continuous batching. A finished sequence frees its slot and a queued one is prefilled and merged into the
     running KV cache (left-padded, trimmed to the longest active row) every `every` steps, so a batch no longer waits for
     its longest output. Greedy + the checkpoint's repetition penalty (1.05), identical to HF generate token for token.
 11. Vision features are computed once per video (C["vfeat"]) and reused by every pair query of that video
     (rearrange_token otherwise re-runs the vision encoder on the whole video per call).
"""
import math, re, time, copy, collections, concurrent.futures as cf
import numpy as np, torch
from transformers import StoppingCriteria, StoppingCriteriaList

KV_BYTES_PER_TOKEN = 36 * 2 * 2 * 128 * 2  # layers * (k,v) * kv_heads * head_dim * bf16


def gqa_decode_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, _orig=None, **kw):
    """query [B, H, 1, d], key/value [B, Hkv, S, d], attention_mask bool/additive [B, 1, 1, S'] (S' >= S)."""
    B, H, Lq, d = query.shape; Hkv = key.shape[1]
    if Lq != 1 or attention_mask is None or H == Hkv or attention_mask.ndim != 4 or attention_mask.shape[2] != 1:
        return _orig(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, is_causal=is_causal, **kw)
    q = query.reshape(B, Hkv, H // Hkv, d)                      # heads h = g * n_rep + r share KV head g (repeat_kv layout)
    m = attention_mask[:, :, :, : key.shape[-2]]
    o = torch.nn.functional.scaled_dot_product_attention(q.contiguous(), key.contiguous(), value.contiguous(), attn_mask=m,
                                                         dropout_p=0.0, scale=scaling, is_causal=False)
    return o.reshape(B, H, 1, d).transpose(1, 2).contiguous(), None


def install_gqa_decode_attention():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS["sdpa"] = lambda *a, **k: gqa_decode_sdpa(*a, _orig=sdpa_attention_forward, **k)


class LastOnlyHead(torch.nn.Module):
    """During prefill only the last position's logits are needed by generate()."""
    def __init__(self, lm):
        super().__init__(); self.lm = lm
    def forward(self, h):
        return self.lm(h[:, -1:, :]) if h.shape[1] > 1 else self.lm(h)


class LoopStop3(StoppingCriteria):
    def __init__(self, fast_tok, every=64, chunk=300, reps=2, max_attr=30, window_tokens=100000):
        self.tok, self.every, self.chunk, self.reps, self.max_attr, self.window = fast_tok, every, chunk, reps, max_attr, window_tokens
        self.steps, self.stopped = 0, None

    @staticmethod
    def text_fires(t, chunk=300, reps=2, max_attr=30):
        lo = t.rfind('"attributes": [')
        if lo >= 0 and "]" not in t[lo:] and t[lo:].count('", "') > max_attr:
            return True
        r = t.find('"relationships"')
        if r >= 0:
            rel = t[r:]
            if len(rel) > chunk and rel[:-chunk].count(rel[-chunk:]) >= reps:
                return True
        return False

    def __call__(self, input_ids, scores, **kw):
        self.steps += 1
        if self.stopped is None:
            self.stopped = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if self.steps % self.every == 0:
            texts = self.tok.batch_decode(input_ids[:, -self.window:], skip_special_tokens=True)  # stall rule sees this window only
            for i, t in enumerate(texts):
                if not self.stopped[i] and self.text_fires(t, self.chunk, self.reps, self.max_attr):
                    self.stopped[i] = True
        return self.stopped.clone()


class LoopStop4(LoopStop3):
    RX = re.compile(r'\[\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,')

    @staticmethod
    def text_fires(t, chunk=300, reps=2, max_attr=30, win=40, min_new=8):
        if LoopStop3.text_fires(t, chunk, reps, max_attr):
            return True
        r = t.find('"relationships"')
        if r < 0:
            return False
        tup = LoopStop4.RX.findall(t[r:])
        if len(tup) >= win:
            seen = set(tup[:-win]); new = 0
            for x in tup[-win:]:
                if x not in seen: new += 1; seen.add(x)
            return new < min_new
        return False


def token_cap(n_objects, lo=300, per_obj=200, hard=8192):
    return int(min(hard, lo + per_obj * n_objects))


def plan_batches(items, max_batch=32, kv_budget_gb=40.0):
    """items: list of dicts with 'key', 'prefill', 'n_obj'. Sort by expected output length (desc) and pack greedily so that
    B * (max_prefill + max_cap) * KV_BYTES_PER_TOKEN <= budget."""
    budget = kv_budget_gb * 1e9
    order = sorted(items, key=lambda x: (x["n_obj"], x["prefill"]), reverse=True)
    batches, cur = [], []
    for it in order:
        cand = cur + [it]
        L = max(c["prefill"] for c in cand); C = max(token_cap(c["n_obj"]) for c in cand)
        if cur and (len(cand) > max_batch or len(cand) * (L + C) * KV_BYTES_PER_TOKEN > budget):
            batches.append(cur); cur = [it]
        else:
            cur = cand
    if cur: batches.append(cur)
    return batches


def pad_batch(Ps, device):
    """Left-pad prepared single-sequence inputs. Returns inputs_embeds, 3D position_ids, attention_mask, rope_deltas.
    Decode positions are cache_position + rope_delta; left padding shifts cache_position by `pad`, so rope_delta -= pad."""
    L = max(P["e"].shape[1] for P in Ps); B = len(Ps); D = Ps[0]["e"].shape[2]
    e = torch.zeros((B, L, D), dtype=Ps[0]["e"].dtype, device=device)
    am = torch.zeros((B, L), dtype=torch.long, device=device)
    pos = torch.zeros((3, B, L), dtype=Ps[0]["pos"].dtype, device=device)
    rope = torch.zeros((B, 1), dtype=Ps[0]["rope"].dtype, device=device)
    for i, P in enumerate(Ps):
        n = P["e"].shape[1]; pad = L - n
        e[i, pad:] = P["e"][0]; am[i, pad:] = 1; pos[:, i, pad:] = P["pos"][:, 0]; rope[i] = P["rope"][0] - pad
    return e, pos, am, rope


class SlotDecoder:
    """Greedy continuous-batching decoder over prepared items (dicts from FastTraSeR.gpu_stage)."""
    def __init__(self, model, fast_tok, max_slots=32, kv_budget_gb=40.0, every=64, loop_stop=True, eos=(151645, 151643), log=print,
                 temperature=0.0, top_p=1.0, seed=0):
        self.m, self.ftok, self.B, self.budget, self.every, self.loop_stop, self.log = model, fast_tok, max_slots, kv_budget_gb * 1e9, every, loop_stop, log
        self.temperature, self.top_p = temperature, top_p
        self.gen = None if temperature == 0 else torch.Generator(device=model.device).manual_seed(seed)
        self.eos = torch.tensor(eos)
        rp = getattr(model.generation_config, "repetition_penalty", None)
        self.rp = float(rp) if rp and rp != 1.0 else None
        self.lm = model.model.language_model; self.emb = model.model.get_input_embeddings()
        self.head = model.lm_head.lm if isinstance(model.lm_head, LastOnlyHead) else model.lm_head

    def _logits(self, h, seen):
        x = self.head(h[:, -1, :]).float()
        if self.rp is not None:                                   # HF RepetitionPenaltyLogitsProcessor over generated tokens
            x = torch.where(seen, torch.where(x < 0, x * self.rp, x / self.rp), x)
        return x

    def _pick(self, x):
        """Greedy (temperature 0) or sampling with HF's order: repetition penalty (in _logits) -> temperature -> top-p."""
        if self.temperature == 0:
            return x.argmax(-1)
        x = x / self.temperature
        if self.top_p < 1.0:
            sv, si = torch.sort(x, descending=True, dim=-1)
            cp = torch.softmax(sv, -1).cumsum(-1)
            drop = cp - torch.softmax(sv, -1) > self.top_p          # keep the smallest prefix with mass >= top_p
            sv = sv.masked_fill(drop, float("-inf"))
            x = torch.full_like(x, float("-inf")).scatter(-1, si, sv)
        return torch.multinomial(torch.softmax(x, -1), 1, generator=self.gen).squeeze(-1)

    @staticmethod
    def _fit(t, S):                                               # left-pad / left-trim the sequence axis (dim -2 or -1) to S
        d = -2 if t.dim() == 4 else -1; n = t.shape[d]
        if n == S: return t
        if n > S: return t.narrow(d, n - S, S)
        shp = list(t.shape); shp[d] = S - n
        return torch.cat([torch.zeros(shp, dtype=t.dtype, device=t.device), t], dim=d)

    def run(self, items):
        """items: list of (key, P, cap). Returns {key: dict(text, n_tokens, stop)}."""
        from transformers import DynamicCache
        dev = items[0][1]["e"].device; V = self.head.out_features
        q = collections.deque(sorted(items, key=lambda x: (x[2], x[1]["prefill"]), reverse=True))   # longest-first
        rows, cache, am, out, step, t0 = [], None, None, {}, 0, time.time()
        while q or rows:
            # ---- maintenance: retire finished rows, admit queued ones
            done = [i for i, r in enumerate(rows) if r["stop"]]
            for i in done:
                r = rows[i]; out[r["key"]] = dict(text=self.ftok.decode(r["gen"], skip_special_tokens=True), n_tokens=len(r["gen"]), stop=r["stop"], ids=list(r["gen"]))
            keep = [i for i, r in enumerate(rows) if not r["stop"]]
            if done and keep:
                idx = torch.tensor(keep, device=dev)
                for L in cache.layers: L.keys, L.values = L.keys[idx], L.values[idx]
                am = am[idx]; rows = [rows[i] for i in keep]
            elif done:
                rows, cache, am = [], None, None
            new = []
            def need(rs): return max((r["len"] + r["cap"] - len(r["gen"])) for r in rs) if rs else 0
            while q and len(rows) + len(new) < self.B:
                cand = new + [q[0]]
                worst = max(need(rows), max(P["prefill"] + c for _, P, c in cand))
                if (rows or new) and (len(rows) + len(cand)) * worst * KV_BYTES_PER_TOKEN > self.budget: break
                new.append(q.popleft())
            if new:
                # Prefill each new sequence on its own: unpadded -> no attention mask -> fused causal SDPA. A padded multi-row
                # prefill makes transformers materialise [B, H, L, L] attention (26 GB OOM on long PVSG prompts).
                Lmax = max(P["e"].shape[1] for _, P, _ in new); caches, firsts = [], []
                with torch.no_grad():
                    for _, P, _ in new:
                        n_ = P["e"].shape[1]; c1 = DynamicCache()
                        h = self.lm(inputs_embeds=P["e"], position_ids=P["pos"], attention_mask=torch.ones((1, n_), dtype=torch.long, device=dev),
                                    past_key_values=c1, use_cache=True, cache_position=torch.arange(n_, device=dev)).last_hidden_state
                        firsts.append(self._pick(self._logits(h, torch.zeros((1, V), dtype=torch.bool, device=dev)))[0]); caches.append(c1)
                    first = torch.stack(firsts)
                    c_n = caches[0]
                    for Ls in zip(*[c.layers for c in caches]):
                        Ls[0].keys = torch.cat([self._fit(L.keys, Lmax) for L in Ls], 0)
                        Ls[0].values = torch.cat([self._fit(L.values, Lmax) for L in Ls], 0)
                    am_n = torch.cat([self._fit(torch.ones((1, P["e"].shape[1]), dtype=torch.long, device=dev), Lmax) for _, P, _ in new], 0)
                    del caches
                nr = []
                for j, (k, P, c) in enumerate(new):
                    n = P["e"].shape[1]
                    nr.append(dict(key=k, cap=c, gen=[int(first[j])], tok=int(first[j]), pos=n + int(P["rope"].reshape(-1)[0]),
                                   len=n + 1, stop=None, seen=None))
                if rows:
                    S = max(max(r["len"] - 1 for r in rows), Lmax)     # current cache length needed (tokens already in cache)
                    for Lo, Ln in zip(cache.layers, c_n.layers):
                        Lo.keys = torch.cat([self._fit(Lo.keys, S), self._fit(Ln.keys, S)], 0)
                        Lo.values = torch.cat([self._fit(Lo.values, S), self._fit(Ln.values, S)], 0)
                    am = torch.cat([self._fit(am, S), self._fit(am_n, S)], 0)
                else:
                    cache, am = c_n, am_n
                rows += nr
                for r in rows: r["seen"] = None                          # rebuilt below
                self.log(f"admit {len(new)} -> active {len(rows)} cache_len {am.shape[1]} queue {len(q)} step {step} t {time.time()-t0:.1f}s")
            if not rows: break
            seen = torch.zeros((len(rows), V), dtype=torch.bool, device=dev)
            for i, r in enumerate(rows): seen[i, torch.tensor(r["gen"], device=dev)] = True
            for i, r in enumerate(rows):                                   # first token may already be EOS / cap 1
                if r["gen"][-1] in self.eos.tolist(): r["stop"] = "eos"
                elif len(r["gen"]) >= r["cap"]: r["stop"] = "cap"
            # ---- decode `every` steps
            for _ in range(self.every):
                if all(r["stop"] for r in rows): break
                ids = torch.tensor([[r["tok"]] for r in rows], device=dev)
                pos = torch.tensor([r["pos"] for r in rows], device=dev).view(1, -1, 1).expand(3, -1, 1)
                am = torch.cat([am, torch.ones((len(rows), 1), dtype=am.dtype, device=dev)], 1)
                with torch.no_grad():
                    h = self.lm(inputs_embeds=self.emb(ids), position_ids=pos, attention_mask=am, past_key_values=cache, use_cache=True,
                                cache_position=torch.tensor([am.shape[1] - 1], device=dev)).last_hidden_state
                    nxt = self._pick(self._logits(h, seen))
                seen[torch.arange(len(rows), device=dev), nxt] = True
                nl = nxt.tolist(); step += 1
                for i, r in enumerate(rows):
                    r["pos"] += 1; r["len"] += 1
                    if r["stop"]: continue
                    r["tok"] = nl[i]; r["gen"].append(nl[i])
                    if nl[i] in (151645, 151643): r["stop"] = "eos"
                    elif len(r["gen"]) >= r["cap"]: r["stop"] = "cap"
            if self.loop_stop:
                for r in rows:
                    if not r["stop"] and LoopStop4.text_fires(self.ftok.decode(r["gen"], skip_special_tokens=True)): r["stop"] = "loop"
        self.log(f"slot decoder: {len(out)} sequences, {step} steps, {sum(o['n_tokens'] for o in out.values())} tokens, {time.time()-t0:.1f}s")
        return out


class FastTraSeR:
    def __init__(self, model, processor, tok, fast_tok, I, rearrange_token, select_tokens, device="cuda", dtype=torch.bfloat16):
        self.m, self.proc, self.tok, self.ftok, self.I = model, processor, tok, fast_tok, I
        self.rearrange, self.select, self.dev, self.dtype = rearrange_token, select_tokens, device, dtype
        if not isinstance(self.m.lm_head, LastOnlyHead):
            self.m.lm_head = LastOnlyHead(self.m.lm_head)
        self.vp = copy.deepcopy(processor.video_processor)
        self.vp.max_pixels = I.VIDEO_MAX_FRAME_PIXELS; self.vp.min_pixels = I.VIDEO_MIN_FRAME_PIXELS
        self.vp.size["longest_edge"] = self.vp.max_pixels; self.vp.size["shortest_edge"] = self.vp.min_pixels

    # ---------- preparation ----------
    def cpu_stage(self, load_frames, total, fps, mask_data, obj_ids, H, W):
        """load_frames(idx_list) -> np.uint8 [N,H,W,3]; mask_data: per video frame list of per-object RLE dicts."""
        I = self.I
        idx = I._frame_indices(total, total / fps)
        video = load_frames(idx.tolist())
        pv = self.vp.preprocess(videos=video, return_tensors="pt")
        vgt = pv["video_grid_thw"][0]; tg, hp, wp = (int(x) for x in vgt)
        spg = [self.proc.video_processor.temporal_patch_size / (len(idx) / (total / fps))]
        om, kept = I.build_obj_masks(mask_data, obj_ids, idx.tolist(), hp * I.PATCH_SIZE, wp * I.PATCH_SIZE)
        return dict(pv=pv["pixel_values_videos"], vgt=vgt, tg=tg, hp=hp, wp=wp, spg=spg, om=om, kept=kept, mask_data=mask_data, idx=idx.tolist())

    def gpu_stage(self, C, obj_subset=None, prompt="scene_graph"):
        """Token selection + trajectory arrangement. obj_subset: list of positions in C['kept'] (for pair queries)."""
        I = self.I
        om, kept = C["om"], C["kept"]
        if obj_subset is not None:
            om = om[obj_subset]; kept = [kept[j] for j in obj_subset]
        _, per, _ = self.select(obj_masks=om.to(self.dev), grid_thw=(C["tg"], C["hp"], C["wp"]), patch_size=I.PATCH_SIZE,
                                spatial_merge_size=I.SPATIAL_MERGE_SIZE, temporal_patch_size=I.TEMPORAL_PATCH_SIZE,
                                coverage_thresh=0.5, time_reduce="max", device=self.dev)
        nvt = C["tg"] * C["hp"] * C["wp"] // I.SPATIAL_MERGE_SIZE ** 2
        ids = I.build_prompt_ids(self.tok, I.PROMPTS[prompt], nvt).to(self.dev)
        lab = self.tok([I.OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per))], add_special_tokens=False)["input_ids"]
        gpw = int(4 / (I.TEMPORAL_PATCH_SIZE / I.BASE_INTERVAL))
        win = self.tok([f"<{w*4} - {(w+1)*4} sec>" for w in range(math.ceil(C["tg"] / gpw))], add_special_tokens=False)["input_ids"]
        mm = self.m.model
        with torch.no_grad():
            if C.get("vfeat") is None:                                  # needs C["pv"]; prepare() drops pv after this
                C["vfeat"] = mm.get_video_features(C["pv"].to(self.dev, dtype=mm.visual.dtype), C["vgt"][None].to(self.dev))
            mm.get_video_features = lambda *a, **k: C["vfeat"]       # instance override: reuse cached features
            try:
                e, pos, am, rope, _, _, _ = self.rearrange(model=self.m, input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values_videos=torch.empty(0), video_grid_thw=C["vgt"][None].to(self.dev), image_grid_thw=None, pixel_values=None,
                second_per_grid_ts=torch.tensor(C["spg"]), obj_token_indices_per_sample=[[i.to(self.dev) for i in per]],
                obj_traj_start_id=int(self.m.config.obj_traj_start_id), obj_traj_end_id=int(self.m.config.obj_traj_end_id),
                text_token_ids_per_sample=[[torch.tensor(x) for x in lab]], timestamp_token_ids_per_batch=[[torch.tensor(x) for x in win]],
                grids_per_temporal_window_per_batch=[gpw])
            finally:
                del mm.get_video_features
        return dict(e=e, pos=pos, rope=rope, kept=kept, prefill=int(e.shape[1]), n_obj=len(kept))

    # ---------- generation ----------
    def generate(self, Ps, loop_stop=True, cap=None):
        e, pos, am, rope = pad_batch(Ps, self.dev)
        cap = cap or max(token_cap(P["n_obj"]) for P in Ps)
        kw = dict(stopping_criteria=StoppingCriteriaList([LoopStop4(self.ftok)])) if loop_stop else {}
        with torch.no_grad():
            g = self.m.generate(inputs_embeds=e, position_ids=pos, attention_mask=am, rope_deltas=rope, do_sample=False,
                                max_new_tokens=cap, **kw)
        return self.tok.batch_decode(g, skip_special_tokens=True), g

    def prepare(self, jobs, cpu_fn, workers=6, prompt="scene_graph"):
        """CPU stage in a thread pool, GPU stage in order. Returns (prepared, cpu_keep)."""
        prepared, cpu_keep = {}, {}
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {k: ex.submit(lambda k=k: self.cpu_stage(**cpu_fn(k))) for k in jobs}
            for k in jobs:
                C = futs[k].result(); prepared[k] = self.gpu_stage(C, prompt=prompt)
                C.pop("pv", None); C.pop("mask_data", None)              # pixels no longer needed once vision features are cached (host RAM)
                cpu_keep[k] = {kk: C[kk] for kk in ("om", "kept", "tg", "hp", "wp", "spg", "vgt", "idx", "vfeat")}
        return prepared, cpu_keep

    def run_slots(self, jobs, cpu_fn, max_slots=32, kv_budget_gb=40.0, loop_stop=True, every=64, workers=6, log=print, prompt="scene_graph"):
        """Like run(), but decodes with SlotDecoder (continuous batching). Returns {key: dict(text, kept, C, n_tokens, stop)}."""
        t0 = time.time(); prepared, cpu_keep = self.prepare(jobs, cpu_fn, workers, prompt); t1 = time.time()
        dec = SlotDecoder(self.m, self.ftok, max_slots, kv_budget_gb, every, loop_stop, log=log)
        res = dec.run([(k, P, token_cap(P["n_obj"])) for k, P in prepared.items()])
        log(f"prep {t1-t0:.1f}s, decode {time.time()-t1:.1f}s, total {time.time()-t0:.1f}s for {len(jobs)} videos")
        return {k: dict(res[k], kept=prepared[k]["kept"], C=cpu_keep[k]) for k in jobs}

    def run(self, jobs, cpu_fn, max_batch=32, kv_budget_gb=40.0, loop_stop=True, workers=6, log=print):
        """jobs: list of keys. cpu_fn(key) -> dict(args for cpu_stage). Returns {key: dict(text, kept)}.
        CPU stage for all jobs runs in a thread pool (overlaps with GPU work); GPU stage + generation run in planned batches."""
        t0 = time.time(); out = {}
        with cf.ThreadPoolExecutor(workers) as ex:
            futs = {k: ex.submit(lambda k=k: self.cpu_stage(**cpu_fn(k))) for k in jobs}
            prepared = {}
            cpu_keep = {}
            for k in jobs:
                C = futs[k].result(); prepared[k] = self.gpu_stage(C)
                cpu_keep[k] = {kk: C[kk] for kk in ("om", "kept", "tg", "hp", "wp", "spg", "vgt", "pv", "idx", "vfeat")}
        t1 = time.time()
        items = [dict(key=k, prefill=P["prefill"], n_obj=P["n_obj"]) for k, P in prepared.items()]
        for b in plan_batches(items, max_batch, kv_budget_gb):
            tb = time.time(); keys = [x["key"] for x in b]
            texts, g = self.generate([prepared[k] for k in keys], loop_stop=loop_stop)
            for k, t in zip(keys, texts): out[k] = dict(text=t, kept=prepared[k]["kept"], C=cpu_keep[k])
            log(f"batch n={len(keys)} prefill_max={max(x['prefill'] for x in b)} objs_max={max(x['n_obj'] for x in b)} gen_len={g.shape[1]} sec={time.time()-tb:.1f}")
        log(f"prep {t1-t0:.1f}s, total {time.time()-t0:.1f}s for {len(jobs)} videos")
        return out

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


STYLE_TEXT = {"vidor": " Annotation style: VidOR.", "salient": " Annotation style: salient."}


def register_style_prompts(I):
    base = I.PROMPTS["scene_graph"]
    head, tail = base.split(":\n<video>", 1)
    for k, s in STYLE_TEXT.items():
        I.PROMPTS[f"scene_graph_{k}"] = head + "." + s.rstrip(".") + ":\n<video>" + tail


def build_target(objects, relationships, kept, keep_rel=None):
    """objects: list of {"object_<id>": label, "attributes"?: [...]}; relationships: [[s, p, o, spans, ...]];
    kept: original object ids in prompt order (Object 1 = kept[0]); keep_rel(s, o) -> bool filters relations (salient view).
    Returns JSON text in TraSeR's output format, or None if nothing to supervise."""
    lab = {}
    for ob in objects:
        k = next(k for k in ob if k.startswith("object_"))
        lab[int(k.split("_")[1])] = ob
    idmap = {oid: i + 1 for i, oid in enumerate(kept)}
    objs = []
    for oid in kept:
        ob = lab.get(oid)
        if ob is None:
            return None
        k = next(k for k in ob if k.startswith("object_"))
        d = {f"object {idmap[oid]}": ob[k]}
        if "attributes" in ob:
            d["attributes"] = ob["attributes"]
        objs.append(d)
    rels = []
    for r in relationships:
        s, p, o, spans = int(r[0]), r[1], int(r[2]), r[3]
        ns = CAMERA_ID if s == CAMERA_ID else idmap.get(s)
        no = CAMERA_ID if o == CAMERA_ID else idmap.get(o)
        if ns is None or no is None:
            continue
        if keep_rel is not None and not keep_rel(s, o):
            continue
        rels.append([ns, p, no, spans])
    return json.dumps({"objects": objs, "relationships": rels}, ensure_ascii=False)


def sample_loss(eng, P, target_text):
    """P: dict from eng.gpu_stage (prompt embeddings, 3D positions, rope delta). Teacher-forced loss on target + <|im_end|>.
    Target token j sits at position n + rope + j, exactly where greedy decoding would place it."""
    m, tok = eng.m, eng.tok
    ids = tok(target_text + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(P["e"].device)
    T = ids.shape[1]; n = P["e"].shape[1]
    emb = m.model.get_input_embeddings()(ids).to(P["e"].dtype)
    x = torch.cat([P["e"], emb[:, :-1]], 1)
    tpos = (n + P["rope"].reshape(-1)[0] + torch.arange(T - 1, device=x.device)).view(1, 1, -1).expand(3, 1, -1)
    pos = torch.cat([P["pos"], tpos.to(P["pos"].dtype)], 2)
    h = m.model.language_model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=x.device),
                                use_cache=False).last_hidden_state
    head = m.lm_head.lm if hasattr(m.lm_head, "lm") else m.lm_head
    logits = head(h[:, n - 1:]).float()
    return nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), ids.view(-1)), T

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

# Lenient label judge (stand-in for SVG2's GPT-4o-mini aligner): classify (GT label, predicted label) pairs.
import glob, json, os, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
CATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]
def build_pairs(outs, gt):
    pairs = set()
    for o in outs:
        if "error" in o or o["video_id"] not in gt: continue
        kept = o["kept_obj_ids"]; g = gt[o["video_id"]]
        gobj = {int(k.split("_")[1]): str(v) for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}
        try: pred = json.loads(o["text"])
        except Exception: continue
        for ob in pred.get("objects", []):
            for k, v in ob.items():
                m = re.match(r"object[ _](\d+)", k)
                if m and int(m.group(1)) - 1 < len(kept):
                    gl = gobj.get(kept[int(m.group(1)) - 1])
                    if gl: pairs.add(("object", gl.lower().strip(), str(v).lower().strip()))
        P = {}
        for r in pred.get("relationships", []):
            try: P.setdefault((kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1), []).append(str(r[1]))
            except Exception: pass
        for s, p, ob, *_ in json.loads(g["relationships"]):
            for q in P.get((s, ob), []):
                if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))
    return sorted(pairs)
def run_judge(pairs, model_dir, out_path, bs=64):
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    res = {}
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]; prompts = []
        for kind, g, p in batch:
            msg = (f"You are a strict lexical matcher for video scene graph evaluation. Compare a ground-truth {kind} label with a predicted {kind} label "
                   f"and classify their relationship into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\n"
                   f"Ground truth: \"{g}\"\nPrediction: \"{p}\"\nAnswer with the category name only.")
            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        for (kind, g, p), t in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            t = t.strip().lower(); res[f"{kind}|||{g}|||{p}"] = next((c for c in CATS if t.startswith(c.split("/")[0])), "mismatch:" + t[:20])
        print("judged", min(i + bs, len(pairs)), "/", len(pairs), flush=True)
    json.dump(res, open(out_path, "w"))

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

"""On-policy GRPO for TraSeR (LoRA), rewards from the graph-matching reward against TraSeR's own training labels.
Choices fixed a priori: G rollouts per video (temperature 1.0, top-p 0.95, the checkpoint's repetition penalty), group-mean-centred
advantages without std scaling and a constant length normaliser (Dr. GRPO, arXiv 2503.20783), one gradient update per batch (purely
on-policy, so no importance ratio / clipping), KL to the frozen reference via the k3 estimator (reference = LoRA switched off)."""
import torch, torch.nn as nn

def set_lora_enabled(module, on):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.scale = m._s if on else 0.0

def _remember_scale(module):
    for m in module.modules():
        if isinstance(m, LoRALinear) and not hasattr(m, "_s"):
            m._s = m.scale

def token_logprobs(eng, P, ids):
    """Log-probabilities of the sampled token ids (list[int]) under the current model, teacher-forced at the decode positions."""
    m = eng.m; dev = P["e"].device
    t = torch.tensor([ids], device=dev); n = P["e"].shape[1]; T = t.shape[1]
    emb = m.model.get_input_embeddings()(t).to(P["e"].dtype)
    x = torch.cat([P["e"], emb[:, :-1]], 1)
    tpos = (n + P["rope"].reshape(-1)[0] + torch.arange(T - 1, device=dev)).view(1, 1, -1).expand(3, 1, -1)
    pos = torch.cat([P["pos"], tpos.to(P["pos"].dtype)], 2)
    h = m.model.language_model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev),
                                use_cache=False).last_hidden_state
    head = m.lm_head.lm if hasattr(m.lm_head, "lm") else m.lm_head
    logits = head(h[:, n - 1:]).float()
    return torch.log_softmax(logits, -1).gather(-1, t[..., None]).squeeze(-1)[0]          # [T]

def grpo_loss(eng, P, ids, adv, beta=0.02, norm_len=4096):
    """Loss for one rollout (call backward per rollout to bound memory). Returns (loss, kl_mean)."""
    lm = eng.m.model.language_model; _remember_scale(lm)
    if beta > 0:
        with torch.no_grad():
            set_lora_enabled(lm, False); ref = token_logprobs(eng, P, ids); set_lora_enabled(lm, True)
    lp = token_logprobs(eng, P, ids)
    pg = -(adv * lp).sum() / norm_len
    if beta > 0:
        d = ref - lp; kl = torch.exp(d) - d - 1                                            # k3 estimator, per token
        return pg + beta * kl.sum() / norm_len, float(kl.mean())
    return pg, 0.0

model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tok = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); ftok = AutoTokenizer.from_pretrained(ASSETS, use_fast=True)
proc = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cuda", dtype=torch.bfloat16)
install_gqa_decode_attention(); register_style_prompts(I)
def gpu_sync(): torch.cuda.synchronize()
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
# benchmark data roots
SVG2_MASKS = find("**/masks_vipseg.parquet"); SVG2_GT = find("**/gt_vipseg.parquet"); SVG2_FR = os.path.dirname(os.path.dirname(SVG2_MASKS))
VIDOR_DIR = os.path.dirname(find("**/vidor_meta.json")); PVSG_DIR = os.path.dirname(find("**/pvsg-val-traser/**/meta.json") if glob.glob("/kaggle/input/**/pvsg-val-traser/**/meta.json", recursive=True) else find("**/pvsg*/**/meta.json"))
TRAIN_DIR = os.path.dirname(find("**/meta_train.json")) if glob.glob("/kaggle/input/**/meta_train.json", recursive=True) else None
JUDGE_DIR = os.path.dirname(find("**/qwen-3/**/config.json")) if glob.glob("/kaggle/input/**/qwen-3/**/config.json", recursive=True) else None
print("dirs", SVG2_FR, VIDOR_DIR, PVSG_DIR, TRAIN_DIR, JUDGE_DIR, flush=True)

ARM = "P1B"; BENCHES = ["svg2", "vidor", "pvsg"]; LIMIT_VIDS = 10**9
PAIRS_FILE = find("**/pairlists_p1.json")

# ==== P1: released TraSeR decodes learned-scorer pair lists (no training). Compared offline against A0's contact-ranked and random pairs.
T = {"arm": ARM}
def slug_root(p, slug):
    parts = p.split("/"); return "/".join(parts[:parts.index(slug) + 1]) if slug in parts else os.path.dirname(p)
FRAMEDIRS = collections.defaultdict(list)
for dp, dn, fn in os.walk("/kaggle/input" if not globals().get("FAKE_FRAMES") else R):
    if any(f.endswith(".jpg") for f in fn): FRAMEDIRS[os.path.basename(dp)].append(dp)
def frames_of(v, root):
    ds = [d for d in FRAMEDIRS.get(v, []) if d.startswith(root)]
    if ds: return sorted(glob.glob(f"{ds[0]}/*.jpg"))
    return ["FAKE"] * 400 if globals().get("FAKE_FRAMES") else []
def load_rows(path): return {str(r["video_id"]): r for r in pq.read_table(path).to_pylist()}
SVG2_ROOT = slug_root(SVG2_MASKS, "svg2test-vipseg")
B = {
  "svg2": dict(masks=load_rows(SVG2_MASKS), gt=load_rows(SVG2_GT), root=SVG2_ROOT, stride=3, fps=lambda v: 2.0),
  "vidor": dict(masks=load_rows(f"{VIDOR_DIR}/masks_vidor.parquet"), gt=load_rows(f"{VIDOR_DIR}/gt_vidor.parquet"), root=VIDOR_DIR, stride=1,
                fps=(lambda m: (lambda v: m[v]["fps_eff"]))(json.load(open(f"{VIDOR_DIR}/vidor_meta.json")))),
  "pvsg": dict(masks=load_rows(f"{PVSG_DIR}/masks.parquet"), gt=load_rows(f"{PVSG_DIR}/gt.parquet"), root=PVSG_DIR, stride=1,
               fps=(lambda m: (lambda v: m[v]["fps_eff"]))(json.load(open(f"{PVSG_DIR}/meta.json")))),
}
def make_cpu_fn(b, max_obj=40):
    def cpu_fn(v):
        r = b["masks"][v]; H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
        files = frames_of(v, b["root"]); s = b["stride"]; total = min(len(files), len(range(0, nf, s)))
        md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, s)][:total]
        if files[0] == "FAKE":
            fr = np.random.default_rng(0).integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8); lf = lambda idx: fr[idx]
        else: lf = lambda idx: np.stack([np.array(Image.open(files[i]).convert("RGB").resize((W, H))) for i in idx])
        return dict(load_frames=lf, total=total, fps=b["fps"](v), mask_data=md, obj_ids=list(range(no))[:max_obj], H=H, W=W)
    return cpu_fn
PL = json.load(open(PAIRS_FILE))
CHUNKS = {"svg2": 96, "vidor": 96, "pvsg": 16}
for bn in BENCHES:
    b = B[bn]; vids = sorted(v for v in PL[bn] if v in b["masks"] and frames_of(v, b["root"]) and PL[bn][v])[:LIMIT_VIDS]
    cpu_fn = make_cpu_fn(b); t0 = time.time(); n = 0
    with open(f"{OUT}/{bn}_pairs_learned.jsonl", "w") as fo:
        for c0 in range(0, len(vids), CHUNKS[bn]):
            ch = vids[c0:c0 + CHUNKS[bn]]
            prepared, keep = eng.prepare(ch, cpu_fn, 6, "scene_graph")
            jobs = []
            for v in ch:
                kept = keep[v]["kept"]; pos = {oid: i for i, oid in enumerate(kept)}
                for rk, e in enumerate(PL[bn][v]):                       # entries [a, b] or [a, b, prob]
                    a, b_ = int(e[0]), int(e[1])
                    if a in pos and b_ in pos: jobs.append((v, rk, [pos[a], pos[b_]]))
            Pp = [eng.gpu_stage(keep[v], obj_subset=sub, prompt="scene_graph") for v, _, sub in jobs]
            if Pp:
                res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
                for i, ((v, rk, sub), p) in enumerate(zip(jobs, Pp)):
                    fo.write(json.dumps({"video_id": v, "rank": rk, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"]}) + "\n"); n += 1
            del prepared, keep, Pp; gc.collect(); torch.cuda.empty_cache()
            print(f"{bn} {min(c0 + CHUNKS[bn], len(vids))}/{len(vids)} pairs {n} {time.time()-t0:.0f}s", flush=True)
    T[bn] = dict(videos=len(vids), pairs=n, seconds=round(time.time() - t0, 1)); print("BENCH DONE", bn, T[bn], flush=True)
    json.dump(T, open(f"{OUT}/timing_{ARM}.json", "w"))
if JUDGE_DIR:
    del model; gc.collect(); torch.cuda.empty_cache(); pairs = set()
    for bn in BENCHES:
        gt = B[bn]["gt"]
        for l in open(f"{OUT}/{bn}_pairs_learned.jsonl"):
            o = json.loads(l); kept = o["kept_obj_ids"]; g = gt[o["video_id"]]; P_ = collections.defaultdict(list)
            for r in parse_rels(o["text"]):
                try: P_[(kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1)].append(str(r[1]))
                except Exception: pass
            for s, p, ob, *_ in json.loads(g["relationships"]):
                for q in P_.get((s, ob), []):
                    if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))
    pairs = sorted(pairs); print("pairs to judge", len(pairs), flush=True); run_judge(pairs, JUDGE_DIR, f"{OUT}/judge_{ARM}.json")
print("P1 DONE", json.dumps(T), flush=True)
