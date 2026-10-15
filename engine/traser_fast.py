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
                e, pos, am_n, rope = pad_batch([P for _, P, _ in new], dev)
                c_n = DynamicCache()
                with torch.no_grad():
                    h = self.lm(inputs_embeds=e, position_ids=pos, attention_mask=am_n, past_key_values=c_n, use_cache=True,
                                cache_position=torch.arange(e.shape[1], device=dev)).last_hidden_state
                    seen = torch.zeros((len(new), V), dtype=torch.bool, device=dev)
                    first = self._pick(self._logits(h, seen))
                nr = []
                for j, (k, P, c) in enumerate(new):
                    n = P["e"].shape[1]
                    nr.append(dict(key=k, cap=c, gen=[int(first[j])], tok=int(first[j]), pos=n + int(P["rope"].reshape(-1)[0]),
                                   len=n + 1, stop=None, seen=None))
                if rows:
                    S = max(max(r["len"] - 1 for r in rows), e.shape[1])     # current cache length needed (tokens already in cache)
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
