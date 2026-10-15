# ---------------- GRPO arm (G1): on-policy RL on TraSeR's own training data, graph-matching reward (engine/reward.py)
def load_sources():
    srcs = {}
    for name, d in TRAIN_DIRS.items():
        tm = load_rows(f"{d}/masks_train.parquet"); tl = load_rows(f"{d}/labels_train.parquet"); me = json.load(open(f"{d}/meta_train.json"))
        vids = [v for v in sorted(tm) if v in tl and frames_of(v, d)]
        srcs[name] = dict(lab=tl, cpu_fn=make_cpu_fn(dict(masks=tm, root=d, stride=1, fps=(lambda me: lambda v: me[v]["fps_eff"])(me)), max_obj=40), vids=vids)
    return srcs
def split_train_dev(srcs, n_dev):
    rng = random.Random(0); tr, dv = {}, {}
    for name, S_ in srcs.items():
        vv = list(S_["vids"]); rng.shuffle(vv); dv[name] = vv[:n_dev]; tr[name] = vv[n_dev:]
    return tr, dv
def dev_reward(srcs, dv):
    out = {}
    for name, vids in dv.items():
        prep, _ = eng.prepare(vids, srcs[name]["cpu_fn"], 6, "scene_graph")
        res = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in vids])
        rs = [reward(res[v]["text"], prep[v]["kept"], json.loads(srcs[name]["lab"][v]["objects"]), json.loads(srcs[name]["lab"][v]["relationships"])) for v in vids]
        out[name] = dict(R=float(np.mean([r for r, _ in rs])), f1=float(np.mean([d["f1"] for _, d in rs])), n_pred=float(np.mean([d["n_pred"] for _, d in rs])),
                         valid=float(np.mean([d["valid"] for _, d in rs])), loop=float(np.mean([res[v]["stop"] == "loop" for v in vids])))
        del prep; gc.collect(); torch.cuda.empty_cache()
    return out
def grpo_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV)
    nl = apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    params = lora_params(model); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    lm = model.model.language_model; hist = []; t0 = time.time()
    model.eval(); d0 = dev_reward(srcs, dv); hist.append(dict(step=0, dev=d0)); print("GRPO dev step 0", json.dumps(d0), flush=True)
    rng = random.Random(1); names = sorted(tr)
    for step in range(1, NSTEPS + 1):
        if time.time() - t0 > TRAIN_SECONDS: print("time budget reached at step", step, flush=True); break
        batch = [(n_, rng.choice(tr[n_])) for n_ in (names * BATCH_PROMPTS)[:BATCH_PROMPTS]]      # equal weight per source
        model.eval(); tg0 = time.time()
        with torch.no_grad():
            Ps = {v: eng.gpu_stage(eng.cpu_stage(**srcs[n_]["cpu_fn"](v)), prompt="scene_graph") for n_, v in batch}
            dec = SlotDecoder(model, ftok, 64, 40.0, 64, True, log=lambda s: None, temperature=TEMP, top_p=TOP_P, seed=step)
            ro = dec.run([(f"{v}#{g}", Ps[v], token_cap(Ps[v]["n_obj"])) for _, v in batch for g in range(G)])
        tg = time.time(); stats = collections.defaultdict(list); nupd = 0
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
        for n_, v in batch:
            lab = srcs[n_]["lab"][v]; ob = json.loads(lab["objects"]); rl = json.loads(lab["relationships"])
            rs = np.array([reward(ro[f"{v}#{g}"]["text"], Ps[v]["kept"], ob, rl)[0] for g in range(G)])
            stats[f"R_{n_}"].append(rs.mean()); stats["std"].append(rs.std())
            if rs.std() < 1e-6: continue
            adv = rs - rs.mean()
            for g in range(G):
                ids = ro[f"{v}#{g}"]["ids"]
                if not ids or abs(adv[g]) < 1e-9: continue
                loss, kl = grpo_loss(eng, Ps[v], ids, float(adv[g]), beta=BETA, norm_len=NORM_LEN)
                (loss / (BATCH_PROMPTS * G)).backward(); stats["kl"].append(kl); nupd += 1
        if nupd:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        lm.gradient_checkpointing_disable(); model.eval()
        rec = dict(step=step, upd=nupd, t_gen=round(tg - tg0, 1), t_upd=round(time.time() - tg, 1), sec=round(time.time() - t0), **{k: float(np.mean(v_)) for k, v_ in stats.items()})
        hist.append(rec); print("GRPO", json.dumps(rec), flush=True)
        del Ps, ro; gc.collect(); torch.cuda.empty_cache()
        if step % EVAL_EVERY == 0:
            d = dev_reward(srcs, dv); hist.append(dict(step=step, dev=d)); print("GRPO dev step", step, json.dumps(d), flush=True)
            json.dump(hist, open(f"{OUT}/grpo_hist.json", "w"))
    d = dev_reward(srcs, dv); hist.append(dict(step="final", dev=d)); print("GRPO dev final", json.dumps(d), flush=True)
    json.dump(hist, open(f"{OUT}/grpo_hist.json", "w"))
    torch.save(lora_state_dict(lm), f"{OUT}/lora_{ARM}.pt"); merge_lora(lm)
    T.update(train_seconds=round(time.time() - t0, 1), steps=len([h for h in hist if "upd" in h]), dev_start=d0, dev_final=d); torch.cuda.empty_cache()
