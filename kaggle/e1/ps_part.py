# ---------------- PS arm: pair-decoder adapter with "none" abstention (pair-conditioned SFT on TraSeR's own training data)
def pairsft_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV)
    PLS = json.load(open(PAIRSFT_FILE)); rng = random.Random(0); items = []
    for name in sorted(tr):
        trset = set(tr[name])
        for v, e in PLS.get(name, {}).items():
            if v not in trset: continue
            items += [(name, v, tuple(p), 1) for p in e["pos"]] + [(name, v, tuple(p), 0) for p in e["neg"]]
    rng.shuffle(items); items = items[:MAX_PAIR_SAMPLES]
    nl = apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    params = lora_params(model); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    steps = math.ceil(len(items) / ACCUM); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    print(f"PAIR-SFT samples={len(items)} pos={sum(x[3] for x in items)} lora={nl} steps={steps}", flush=True)
    t0 = time.time(); losses = []; skipped = 0; cache = {}
    for i, (name, v, (a, b_), lab) in enumerate(items):
        if v not in cache:
            if len(cache) > 64: cache.clear()
            cache[v] = eng.cpu_stage(**srcs[name]["cpu_fn"](v))
        C = cache[v]; kept = C["kept"]; pos = {oid: j for j, oid in enumerate(kept)}
        if a not in pos or b_ not in pos: skipped += 1; continue
        with torch.no_grad(): P = eng.gpu_stage(C, obj_subset=[pos[a], pos[b_]], prompt="scene_graph")
        L = srcs[name]["lab"][v]; objs = json.loads(L["objects"]); rels = json.loads(L["relationships"])
        tgt = build_target(objs, rels, [a, b_], keep_rel=lambda s_, o_: {s_, o_} <= {a, b_})
        if tgt is None: skipped += 1; continue
        loss, _ = sample_loss(eng, P, tgt); (loss / ACCUM).backward(); losses.append(float(loss))
        if (i + 1) % ACCUM == 0 or i + 1 == len(items):
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if i % 200 == 0: print(f"pair-sft {i}/{len(items)} loss {np.mean(losses[-200:]) if losses else 0:.4f} skipped {skipped} {time.time()-t0:.0f}s", flush=True)
    model.eval(); model.model.language_model.gradient_checkpointing_disable(); cache.clear()
    # dev: abstention behaviour on held-out training videos (labelled vs hard-negative pairs)
    dstat = collections.Counter()
    for name in sorted(dv):
        for v in dv[name][:20]:
            e = PLS.get(name, {}).get(v)
            if not e: continue
            C = eng.cpu_stage(**srcs[name]["cpu_fn"](v)); pos = {oid: j for j, oid in enumerate(C["kept"])}
            for lab, lst in ((1, e["pos"]), (0, e["neg"])):
                for a, b_ in lst:
                    if a not in pos or b_ not in pos: continue
                    P = eng.gpu_stage(C, obj_subset=[pos[a], pos[b_]], prompt="scene_graph")
                    t = SlotDecoder(model, ftok, 1, 40.0, 64, False, log=lambda s: None).run([(0, P, 512)])[0]["text"]
                    empty = len(parse_rels(t)) == 0; dstat[f"{name}_{'pos' if lab else 'neg'}_n"] += 1; dstat[f"{name}_{'pos' if lab else 'neg'}_none"] += empty
    print("PAIR-SFT dev abstention", dict(dstat), flush=True)
    torch.save(lora_state_dict(model.model.language_model), f"{OUT}/lora_{ARM}.pt"); merge_lora(model.model.language_model)
    T.update(train_seconds=round(time.time() - t0, 1), samples=len(items), skipped=skipped, loss_first=round(float(np.mean(losses[:100])), 4),
             loss_last=round(float(np.mean(losses[-100:])), 4), dev_abstention=dict(dstat)); torch.cuda.empty_cache()
    decode_eval_pairs()
def decode_eval_pairs():
    """Decode the evaluation pair lists (same lists as P1B) with the current model."""
    PL = json.load(open(PAIRS_FILE)); CH = {"svg2": 96, "vidor": 96, "pvsg": 16}
    for bn in PAIR_BENCHES:
        b = B[bn]; vids = sorted(v for v in PL[bn] if v in b["masks"] and frames_of(v, b["root"]) and PL[bn][v])[:LIMIT_VIDS]
        cpu_fn = make_cpu_fn(b); t1 = time.time(); n = 0
        with open(f"{OUT}/{bn}_pairs_learned.jsonl", "w") as fo:
            for c0 in range(0, len(vids), CH[bn]):
                ch = vids[c0:c0 + CH[bn]]; prepared, keep = eng.prepare(ch, cpu_fn, 6, "scene_graph"); jobs = []
                for v in ch:
                    pos = {oid: j for j, oid in enumerate(keep[v]["kept"])}
                    for rk, e in enumerate(PL[bn][v]):
                        a, b_ = int(e[0]), int(e[1])
                        if a in pos and b_ in pos: jobs.append((v, rk, [pos[a], pos[b_]]))
                Pp = [eng.gpu_stage(keep[v], obj_subset=sub, prompt="scene_graph") for v, _, sub in jobs]
                if Pp:
                    res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
                    for i, ((v, rk, sub), p) in enumerate(zip(jobs, Pp)):
                        fo.write(json.dumps({"video_id": v, "rank": rk, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"]}) + "\n"); n += 1
                del prepared, keep, Pp; gc.collect(); torch.cuda.empty_cache()
        T[bn] = dict(videos=len(vids), pairs=n, seconds=round(time.time() - t1, 1)); print("PAIRS DONE", bn, T[bn], flush=True)
