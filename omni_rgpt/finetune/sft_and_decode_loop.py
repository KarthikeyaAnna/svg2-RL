# ==== QTM run: ARM = "QS" (SFT: Token Marks + LoRA on all pairs/graphs/free tasks, then dev graph decode)
#                ARM = "QE" (test decode: graph mode + pair mode for QS weights; Q0 zero-shot box-text graph mode)
# Config globals (ARM, paths, sizes) are prepended by build_qtm.py.
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
T0 = time.time(); INFO = {"arm": ARM}
import threading, subprocess as _sp
def _monitor():                                                           # resource line every 60 s (sessions were dying with empty logs)
    while True:
        try:
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30
            mem = _sp.run("free -g | awk '/Mem/{print $3\"/\"$2}'", shell=True, capture_output=True, text=True).stdout.strip()
            dsk = _sp.run("df -BG --output=used /kaggle/working /tmp 2>/dev/null | tail -2 | tr '\\n' ' '", shell=True, capture_output=True, text=True).stdout.strip()
            gpu = (torch.cuda.memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30) if torch.cuda.is_available() else (0, 0)
            print(f"MON t={time.time() - T0:.0f}s rss={rss:.1f}G sys_mem={mem}G disk(work,tmp)={dsk} gpu={gpu[0]:.1f}/{gpu[1]:.1f}G threads={threading.active_count()}", flush=True)
        except Exception as e: print("MON err", e, flush=True)
        time.sleep(60)
threading.Thread(target=_monitor, daemon=True).start()
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

def add_fetch_sources():
    """Extra TraSeR-split training data from the CPU fetch notebooks (packed frames). Appended AFTER the original sources, so
    the original dev split (Random(0), sequential per source) is unchanged."""
    for d in FETCH_DIRS:
        sm = json.load(open(f"{d}/fetch_summary.json")); name = f"{sm['mode']}_{os.path.basename(d.rstrip('/'))}"
        n = index_fetch_dir(d); TRAIN_DIRS[name] = d; print("fetch source", name, "videos", n, flush=True)
if globals().get("FETCH_DIRS"): add_fetch_sources()

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
    if globals().get("RESUME_LORA"):                                      # continue from earlier SFT weights (QS v2)
        load_lora_state_dict(lm, torch.load(RESUME_LORA, map_location=dev)); tm.M.data.copy_(torch.load(RESUME_MARKS, map_location=dev)); tm.scaled = True
        print("resumed from", RESUME_LORA, flush=True)
    loras = lora_params(lm); opt = torch.optim.AdamW([{"params": loras, "lr": LR}, {"params": [tm.M], "lr": LR_MARK}], weight_decay=0.0)
    items = [(n, v) for n in sorted(tr) if globals().get("TRAIN_FILTER", "") in n for v in tr[n]][:LIMIT_VIDS * len(tr)]; random.Random(1).shuffle(items)   # v4: new videos only
    print(f"SFT videos {len(items)} | lora layers {nl} | dev {[(n, len(x)) for n, x in dv.items()]}", flush=True)
    print("DEV IDS", json.dumps({n: dv[n] for n in sorted(dv)}), flush=True)
    FAST, PACK, CKPT = globals().get("FAST_SFT", False), globals().get("PACK_TURNS", False), globals().get("GRAD_CKPT", True)
    total_steps = max(1, EPOCHS * len(items) * (2 if PACK else 3) // ACCUM)
    CHUNKED = globals().get("CHUNKED", False); start = 0; step0 = 0          # chained runs: each kernel trains the next slice of `items`
    if globals().get("RESUME_STATE"):
        st = torch.load(RESUME_STATE, map_location=dev); opt.load_state_dict(st["opt"]); start, step0 = st["items_done"], st["step"]
        assert st["n_items"] == len(items), (st["n_items"], len(items)); print("resume state: items_done", start, "step", step0, flush=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + step0 + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s + step0) / total_steps)))))
    if not LOCAL and CKPT: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    print("SFT speed config", json.dumps(dict(fast=FAST, pack=PACK, ckpt=CKPT, accum=ACCUM, max_len=MAX_LEN)), flush=True)
    model.train(); rng = random.Random(2); stats = collections.defaultdict(list); nseq = 0; step = 0; nvid = 0; nverify = 0
    TB = time.time() if (globals().get("SPEED") or CHUNKED) else T0         # budget counts from the first training step
    if start: rng = random.Random(2 + start)
    ncons = 0
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
                if len(q) >= depth:
                    tw = time.time(); r = q.popleft().result(); stats["wait_s"].append(time.time() - tw); yield r   # data-loading stall
            while q: yield q.popleft().result()
        with cf.ThreadPoolExecutor(6) as ex:
            for n, v, prep in bounded(ex, items[start:]):
                ncons += 1
                if isinstance(prep, Exception): stats["prep_err"].append(1); continue
                S = srcs[n]; mm = mark_map(prep["cover"]); nobj = len(prep["kept"])
                pos_t, neg_t = video_turns(n, S, v, prep, rng, cands); free = free_tasks(None, prep, rng)
                rest = pos_t + neg_t + free; rng.shuffle(rest)
                if PACK: chunks = pack_turns(tok, rest, MAX_LEN - mm.shape[0] - 8 * nobj - 300)   # video + objects line + template margin
                else: chunks = [rest[i:i + TURNS_PER_SEQ] for i in range(0, len(rest), TURNS_PER_SEQ)]
                seqs = [[(TASK_GRAPH, graph_target(S["labels"][v], prep["kept"]))]] + chunks   # graph alone: never sees pair answers
                vid = None; feats = None; nvid += 1; stats["seq_per_vid"].append(len(seqs))
                for sq in seqs:
                    text = build_chat(proc, nobj, sq)
                    try:
                        if FAST:
                            first = vid is None; enc, vid = encode_fast(proc, text, prep, mm, vid)
                            if first: vid = {k: x.to(dev) for k, x in vid.items()}
                            elif nverify < 20:                           # the hand-expanded tokens must equal the processor's
                                ref = encode(proc, text, prep, mm)["input_ids"]; nverify += 1
                                assert torch.equal(ref, enc["input_ids"]), ("encode_fast mismatch", ref.shape, enc["input_ids"].shape)
                        else: enc = encode(proc, text, prep, mm)
                        labels = assistant_labels(enc["input_ids"])
                        if enc["input_ids"].shape[1] > MAX_LEN: stats["too_long"].append(1); continue
                        stats["too_long"].append(0); tm.maps = [mm]; enc = {k: v_.to(dev) for k, v_ in enc.items()}
                        if FAST:
                            if feats is None:                            # frozen vision tower: once per video, no graph
                                with torch.no_grad(): feats = [f.detach() for f in tm.orig(enc["pixel_values_videos"], enc["video_grid_thw"])]
                            tm.cache = list(feats); loss = answer_loss(model, enc, labels)
                            if len(stats["verify_loss"]) < 3:                # same loss as the plain HF path
                                with torch.no_grad():
                                    tm.cache = None; ref = model(**enc, labels=labels.to(dev)).loss; tm.cache = list(feats)
                                stats["verify_loss"].append(abs(float(ref) - float(loss)))
                                print("VERIFY loss fast %.5f ref %.5f" % (float(loss), float(ref)), flush=True)
                        else:
                            tm.cache = None; loss = model(**enc, labels=labels.to(dev)).loss
                        (loss / ACCUM).backward(); stats["loss_graph" if len(sq) == 1 and sq[0][0] == TASK_GRAPH else "loss_turns"].append(float(loss))
                        stats["len"].append(enc["input_ids"].shape[1]); nseq += 1
                    except torch.OutOfMemoryError:
                        opt.zero_grad(set_to_none=True); tm.cache = None; feats = None; torch.cuda.empty_cache(); stats["oom"].append(1)
                        if not CKPT and not LOCAL:                       # fall back to checkpointing for the rest of the run
                            CKPT = True; model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                            print("OOM -> gradient checkpointing ON", flush=True)
                        continue
                    finally:
                        tm.maps = None; tm.cache = None
                    if nseq % 2000 == 0: print("PROGRESS", json.dumps({"epoch": ep, "seq": nseq, "videos": nvid, "sec": round(time.time() - T0)}), flush=True)
                    if nseq % ACCUM == 0:
                        torch.nn.utils.clip_grad_norm_(loras + [tm.M], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                        if step % LOG_EVERY == 0:
                            print("SFT", json.dumps({"step": step, "seq": nseq, "videos": nvid, "sec": round(time.time() - T0), **{k: round(float(np.mean(x[-200:])), 4) for k, x in stats.items()}}), flush=True)
                    if time.time() - TB > TRAIN_SECONDS or time.time() - T0 > globals().get("HARD_CAP", 10 ** 9): stop = True; break   # HARD_CAP: wall clock from session start (12 h kill)
                if stop: break
        if stop: print("time budget reached", flush=True); break
    model.eval()
    if not LOCAL: model.gradient_checkpointing_disable()
    torch.save(lora_state_dict(lm), f"{OUT}/lora_{ARM}.pt"); torch.save(tm.M.detach().cpu(), f"{OUT}/marks_{ARM}.pt")
    if CHUNKED:
        torch.save({"opt": opt.state_dict(), "step": step0 + step, "items_done": start + ncons, "n_items": len(items)}, f"{OUT}/train_state.pt")
        INFO.update(items_done=start + ncons, n_items=len(items))
        print("CHUNK DONE items", start, "->", start + ncons, "of", len(items), flush=True)
    INFO.update(train_seconds=round(time.time() - T0), steps=step, sequences=nseq, videos=nvid, peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1) if dev == "cuda" else 0, **{k: round(float(np.mean(x)), 4) for k, x in stats.items()})
    # dev graph decode (single pass) -> scored locally with engine/reward.py (exact)
    with open(f"{OUT}/dev_graph.jsonl", "w") as fo:
        last = not CHUNKED or start + ncons >= len(items)                 # chained runs: dev decode only after the final slice
        for n in sorted(dv):
            if not last or n not in globals().get("DEV_SOURCES", dv): continue
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
