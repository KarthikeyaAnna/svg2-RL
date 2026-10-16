# ---------------- PG arm: pair-level GRPO (level 1 of pair-factored policy optimisation), warm-started from the PS adapter
def pair_reward(text, a, b_, objs, rels):
    pr = [r for r in rels if {int(r[0]), int(r[2])} == {a, b_}]
    npred = len(parse_rels(text))
    if not pr:                                           # unlabelled hard negative: reward a correct abstention
        return (W_NONE if npred == 0 else 0.0), dict(none=npred == 0, labelled=False)
    R_, d_ = reward(text, [a, b_], objs, pr)
    return R_, dict(none=npred == 0, labelled=True, f1=d_["f1"])
def pg_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV)
    PLS = json.load(open(PAIRSFT_FILE)); SEED_ = globals().get("TRAIN_SEED", 0); rng = random.Random(1000 * SEED_); pool = []
    for name in sorted(tr):
        trset = set(tr[name])
        for v, e in PLS.get(name, {}).items():
            if v in trset: pool += [(name, v, tuple(p)) for p in e["pos"] + e["neg"]]
    rng.shuffle(pool)
    nl = apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    if not globals().get("SKIP_WARMSTART"):
        load_lora_state_dict(model.model.language_model, torch.load(find("**/lora_PS.pt"), map_location="cuda"))
    params = lora_params(model); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0); lm = model.model.language_model
    print(f"PAIR-GRPO pool={len(pool)} lora={nl} warm-start=PS", flush=True)
    t0 = time.time(); hist = []; ptr = 0; cache = {}
    for step in range(1, NSTEPS + 1):
        if time.time() - t0 > TRAIN_SECONDS: print("time budget reached", step, flush=True); break
        batch = pool[ptr:ptr + BATCH_PAIRS]; ptr += BATCH_PAIRS
        if len(batch) < BATCH_PAIRS: ptr = 0; rng.shuffle(pool); continue
        model.eval(); jobs = []
        with torch.no_grad():
            for name, v, (a, b_) in batch:
                if v not in cache:
                    if len(cache) > 48: cache.clear()
                    cache[v] = eng.cpu_stage(**srcs[name]["cpu_fn"](v))
                C = cache[v]; pos = {oid: j for j, oid in enumerate(C["kept"])}
                if a in pos and b_ in pos: jobs.append((name, v, a, b_, eng.gpu_stage(C, obj_subset=[pos[a], pos[b_]], prompt="scene_graph")))
            ro = SlotDecoder(model, ftok, 128, 40.0, 64, False, log=lambda s: None, temperature=TEMP, top_p=TOP_P, seed=step + 100000 * SEED_).run(
                [(f"{k}#{g}", J[4], 512) for k, J in enumerate(jobs) for g in range(G)])
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
        st = collections.defaultdict(list); nupd = 0
        for k, (name, v, a, b_, P) in enumerate(jobs):
            L = srcs[name]["lab"][v]; objs = json.loads(L["objects"]); rels = json.loads(L["relationships"])
            rd = [pair_reward(ro[f"{k}#{g}"]["text"], a, b_, objs, rels) for g in range(G)]
            rs = np.array([r for r, _ in rd]); lab = rd[0][1]["labelled"]
            st["R_lab" if lab else "R_neg"].append(rs.mean()); st["none_lab" if lab else "none_neg"].append(np.mean([d["none"] for _, d in rd]))
            if rs.std() < 1e-6: st["zero_var"].append(1); continue
            st["zero_var"].append(0); adv = rs - rs.mean()
            for g in range(G):
                ids = ro[f"{k}#{g}"]["ids"]
                if not ids: continue
                try:
                    loss, _ = grpo_loss(eng, P, ids, float(adv[g]), beta=0.0, norm_len=NORM_LEN)
                    (loss / (len(jobs) * G)).backward(); nupd += 1
                except torch.OutOfMemoryError: torch.cuda.empty_cache()
        if nupd: torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        lm.gradient_checkpointing_disable(); model.eval()
        rec = dict(step=step, upd=nupd, sec=round(time.time() - t0), **{k: round(float(np.mean(x)), 4) for k, x in st.items()})
        hist.append(rec); print("PG", json.dumps(rec), flush=True)
        del ro, jobs; gc.collect(); torch.cuda.empty_cache()
    json.dump(hist, open(f"{OUT}/pg_hist.json", "w")); cache.clear()
    # same dev abstention probe as PS
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
                    dstat[f"{name}_{'pos' if lab else 'neg'}_n"] += 1; dstat[f"{name}_{'pos' if lab else 'neg'}_none"] += len(parse_rels(t)) == 0
    print("PG dev abstention", dict(dstat), flush=True)
    torch.save(lora_state_dict(lm), f"{OUT}/lora_{ARM}.pt"); merge_lora(lm)
    T.update(train_seconds=round(time.time() - t0, 1), steps=len(hist), dev_abstention=dict(dstat)); torch.cuda.empty_cache()
    decode_eval_pairs()
