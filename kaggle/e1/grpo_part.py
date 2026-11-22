# ---------------- GRPO arm (G1-G4): on-policy RL on TraSeR's own training data, graph-matching reward (engine/reward.py)
def dil(m, k=2):
    o = m.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1): o |= np.roll(np.roll(m, dy, 0), dx, 1)
    return o

def compute_touching_pairs(C, k_dilate=2):
    """Returns set of frozenset({kept[a], kept[b]}) for pairs with dilated mask contact."""
    kept = C["kept"]
    om = C["om"][:, :, ::4, ::4]
    if hasattr(om, "numpy"):
        om = om.numpy()
    om = om > 0
    touching = set()
    for a in range(len(kept)):
        for b in range(a + 1, len(kept)):
            for j in range(om.shape[1]):
                ma, mb = om[a, j], om[b, j]
                if not ma.any() or not mb.any():
                    continue
                da = dil(ma, k_dilate)
                if (da & mb).any():
                    touching.add(frozenset((kept[a], kept[b])))
                    break
    return touching

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

DEVN = [0]
def build_anchor(ro, v, G, kr, kept, rl):
    """G6 anchor graph. ANCHOR='union': every relation of the G rollouts that matches a label (exact, soft-tIoU credit > 0), deduplicated
    by (subject, predicate, object) keeping the best-credited instance, in the model's own text; ANCHOR='label': all label relations on
    kept objects. Objects list: the best-reward rollout's. Returns JSON text or None."""
    best = int(np.argmax([r for r, _ in kr]))
    try: objs = json.loads(ro[f"{v}#{best}"]["text"]).get("objects", [])
    except Exception: objs = []
    R = {}
    if ANCHOR == "label":
        idx = {o: i + 1 for i, o in enumerate(kept)}
        rels = [[idx[int(x[0])], str(x[1]), idx[int(x[2])], x[3]] for x in rl if int(x[0]) in idx and int(x[2]) in idx]
    else:
        for g in range(G):
            t = ro[f"{v}#{g}"]["text"]
            for c0, c1, k_, _ in kr[g][1]["rels"]:
                if k_ <= 0: continue
                x = None
                for cand in (t[c0:c1] + "]", t[c0:c1]):
                    try: x = json.loads(cand); break
                    except Exception: pass
                if not isinstance(x, list) or len(x) < 4: continue
                key = (int(x[0]), norm(str(x[1])), int(x[2]))
                if key not in R or k_ > R[key][0]: R[key] = (k_, x)
        rels = [x for _, x in sorted(R.values(), key=lambda z: (z[1][0], z[1][2]))]
    return json.dumps({"objects": objs, "relationships": rels}) if rels else None
def dev_reward(srcs, dv):
    out = {}
    for name, vids in dv.items():
        prep, cpu_keep = eng.prepare(vids, srcs[name]["cpu_fn"], 6, "scene_graph")
        res = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in vids])
        if globals().get("ARM") == "G4":
            touching = {v: compute_touching_pairs(cpu_keep[v]) for v in vids}
            rs = [reward_g4(res[v]["text"], prep[v]["kept"], json.loads(srcs[name]["lab"][v]["objects"]), json.loads(srcs[name]["lab"][v]["relationships"]), touching[v], lam_fp=globals().get("LAM_FP", 0.15)) for v in vids]
            out[name] = dict(R=float(np.mean([r for r, _ in rs])), f1=float(np.mean([d["f1"] for _, d in rs])), n_pred=float(np.mean([d["n_pred"] for _, d in rs])),
                             n_geo_fp=float(np.mean([d.get("n_geo_fp", 0) for _, d in rs])),
                             penalty=float(np.mean([d.get("penalty", 0.0) for _, d in rs])),
                             valid=float(np.mean([d["valid"] for _, d in rs])), loop=float(np.mean([res[v]["stop"] == "loop" for v in vids])))
        elif globals().get("ARM") == "G5":
            rs = [reward(res[v]["text"], prep[v]["kept"], json.loads(srcs[name]["lab"][v]["objects"]), json.loads(srcs[name]["lab"][v]["relationships"])) for v in vids]
            kk = [kappa_reward(res[v]["text"], prep[v]["kept"], json.loads(srcs[name]["lab"][v]["objects"]), json.loads(srcs[name]["lab"][v]["relationships"]), KT)[1]["rel"] for v in vids]
            out[name] = dict(R=float(np.mean([r for r, _ in rs])), f1=float(np.mean([d["f1"] for _, d in rs])), kappa_rel=float(np.mean(kk)), n_pred=float(np.mean([d["n_pred"] for _, d in rs])),
                             valid=float(np.mean([d["valid"] for _, d in rs])), loop=float(np.mean([res[v]["stop"] == "loop" for v in vids])))
            DEVN[0] += 0 if name != sorted(dv)[0] else 1
            with open(f"{OUT}/dev_greedy.jsonl", "a") as fo:
                for v in vids: fo.write(json.dumps({"eval": DEVN[0], "source": name, "video_id": v, "kept_obj_ids": prep[v]["kept"], "text": res[v]["text"]}) + "\n")
        else:
            rs = [reward(res[v]["text"], prep[v]["kept"], json.loads(srcs[name]["lab"][v]["objects"]), json.loads(srcs[name]["lab"][v]["relationships"])) for v in vids]
            out[name] = dict(R=float(np.mean([r for r, _ in rs])), f1=float(np.mean([d["f1"] for _, d in rs])), n_pred=float(np.mean([d["n_pred"] for _, d in rs])),
                             valid=float(np.mean([d["valid"] for _, d in rs])), loop=float(np.mean([res[v]["stop"] == "loop" for v in vids])))
        del prep, cpu_keep; gc.collect(); torch.cuda.empty_cache()
    return out

def grpo_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV)
    nl = apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    if globals().get("WARMSTART_LORA") and not globals().get("SKIP_WARMSTART"):
        ws = WARMSTART_LORA if isinstance(WARMSTART_LORA, str) else find("**/lora_G3.pt")
        load_lora_state_dict(model.model.language_model, torch.load(ws, map_location="cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Warm-started LoRA from {ws}", flush=True)
    params = lora_params(model); opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    lm = model.model.language_model; hist = []; t0 = time.time()
    model.eval(); d0 = dev_reward(srcs, dv); hist.append(dict(step=0, dev=d0)); print("GRPO dev step 0", json.dumps(d0), flush=True)
    SEED_ = globals().get("TRAIN_SEED", 0)
    if SEED_: torch.manual_seed(SEED_)
    rng = random.Random(1 + 1000 * SEED_); names = sorted(tr); skipped_long = 0; oom = 0
    for step in range(1, NSTEPS + 1):
        if time.time() - t0 > TRAIN_SECONDS: print("time budget reached at step", step, flush=True); break
        if globals().get("SOURCE_WEIGHTS"):                                                    # e.g. TraSeR's natural mix
            srcs_ = rng.choices(names, weights=[SOURCE_WEIGHTS[n_] for n_ in names], k=BATCH_PROMPTS)
        else:
            srcs_ = (names * BATCH_PROMPTS)[:BATCH_PROMPTS]                                    # equal weight per source
        batch = [(n_, rng.choice(tr[n_])) for n_ in srcs_]
        model.eval(); tg0 = time.time()
        with torch.no_grad():
            Cs = {v: eng.cpu_stage(**srcs[n_]["cpu_fn"](v)) for n_, v in batch}
            touching = {v: compute_touching_pairs(Cs[v]) for v in Cs} if globals().get("ARM") == "G4" else {}
            Ps = {v: eng.gpu_stage(Cs[v], prompt="scene_graph") for v in Cs}
            del Cs
            long_ = [v for v in Ps if Ps[v]["prefill"] > MAX_TRAIN_PREFILL]                  # a-priori memory rule
            skipped_long += len(long_); batch = [(n_, v) for n_, v in batch if v not in long_]
            for v in long_: del Ps[v]; touching.pop(v, None)
            if not batch: continue
            dec = SlotDecoder(model, ftok, 64, 40.0, 64, True, log=lambda s: None, temperature=TEMP, top_p=TOP_P, seed=step + 100000 * SEED_)
            ro = dec.run([(f"{v}#{g}", Ps[v], token_cap(Ps[v]["n_obj"])) for _, v in batch for g in range(G)])
        tg = time.time(); stats = collections.defaultdict(list); nupd = 0
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
        for n_, v in batch:
            lab = srcs[n_]["lab"][v]; ob = json.loads(lab["objects"]); rl = json.loads(lab["relationships"])
            if globals().get("ARM") == "G4":
                lam_fp = globals().get("LAM_FP", 0.15)
                res_pairs = [reward_g4(ro[f"{v}#{g}"]["text"], Ps[v]["kept"], ob, rl, touching[v], lam_fp=lam_fp) for g in range(G)]
                rs = np.array([r for r, _ in res_pairs])
                stats[f"geo_fp_{n_}"].append(float(np.mean([d["n_geo_fp"] for _, d in res_pairs])))
                stats[f"pen_{n_}"].append(float(np.mean([d["penalty"] for _, d in res_pairs])))
            elif globals().get("ARM") == "G5":
                kr = [kappa_reward(ro[f"{v}#{g}"]["text"], Ps[v]["kept"], ob, rl, KT, loo=True) for g in range(G)]
                anc = build_anchor(ro, v, G, kr, Ps[v]["kept"], rl) if globals().get("ANCHOR") else None
                if anc is not None:
                    ro[f"{v}#anchor"] = dict(text=anc, ids=ftok.encode(anc, add_special_tokens=False) + [151645])
                    kr.append(kappa_reward(anc, Ps[v]["kept"], ob, rl, KT, loo=True)); stats["anchor_rels"].append(float(kr[-1][1]["n_pred"]))
                rs = np.array([r for r, _ in kr]); allm = [x[3] for _, d in kr for x in d["rels"]]
                rel_base = float(np.mean(allm)) if allm else 0.0
                stats[f"rel_{n_}"].append(float(np.mean([d["rel"] for _, d in kr]))); stats[f"f1x_{n_}"].append(float(np.mean([d["f1_exact"] for _, d in kr])))
                stats[f"npred_{n_}"].append(float(np.mean([d["n_pred"] for _, d in kr])))
            else:
                rs = np.array([reward(ro[f"{v}#{g}"]["text"], Ps[v]["kept"], ob, rl)[0] for g in range(G)])
            stats[f"R_{n_}"].append(rs.mean()); stats["std"].append(rs.std())
            if rs.std() < 1e-6 and not (globals().get("ARM") == "G5" and CREDIT_W > 0): continue
            adv = rs - rs.mean(); keys = [f"{v}#{g}" for g in range(G)] + ([f"{v}#anchor"] if f"{v}#anchor" in ro else [])
            for g, key in enumerate(keys):
                ids = ro[key]["ids"]
                if not ids or (abs(adv[g]) < 1e-9 and globals().get("ARM") != "G5"): continue
                a_g = float(adv[g])
                if globals().get("ARM") == "G5" and CREDIT_W > 0:
                    a_tok, okk = token_advantages(ids, ftok, a_g, kr[g][1]["rels"], rel_base, w=CREDIT_W); stats["credit_ok"].append(float(okk))
                    a_g = torch.tensor(a_tok, dtype=torch.float32, device=Ps[v]["e"].device)
                try:
                    loss, kl = grpo_loss(eng, Ps[v], ids, a_g, beta=BETA, norm_len=NORM_LEN)
                    (loss / (BATCH_PROMPTS * len(keys))).backward(); stats["kl"].append(kl); nupd += 1
                except torch.OutOfMemoryError:
                    oom += 1; loss = None; torch.cuda.empty_cache()
        if nupd:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        lm.gradient_checkpointing_disable(); model.eval()
        rec = dict(step=step, upd=nupd, skipped_long=skipped_long, oom=oom, t_gen=round(tg - tg0, 1), t_upd=round(time.time() - tg, 1), sec=round(time.time() - t0), **{k: float(np.mean(v_)) for k, v_ in stats.items()})
        hist.append(rec); print("GRPO", json.dumps(rec), flush=True)
        del Ps, ro; gc.collect(); torch.cuda.empty_cache()
        if step % EVAL_EVERY == 0:
            d = dev_reward(srcs, dv); hist.append(dict(step=step, dev=d)); print("GRPO dev step", step, json.dumps(d), flush=True)
            json.dump(hist, open(f"{OUT}/grpo_hist.json", "w"))
    d = dev_reward(srcs, dv); hist.append(dict(step="final", dev=d)); print("GRPO dev final", json.dumps(d), flush=True)
    json.dump(hist, open(f"{OUT}/grpo_hist.json", "w"))
    torch.save(lora_state_dict(lm), f"{OUT}/lora_{ARM}.pt"); merge_lora(lm)
    T.update(train_seconds=round(time.time() - t0, 1), steps=len([h for h in hist if "upd" in h]), dev_start=d0, dev_final=d); torch.cuda.empty_cache()
