# ---------------- PH arm: training data for the reward-trained pair head (head v2). On TraSeR's own training videos, mirror the test
# pipeline exactly: released-model greedy graph (adapter off) -> the ranker's top PH_TOPN pairs the graph does not mention (out-of-fold
# ranker probabilities, PH_CANDS) -> PG pair decode (adapter on). Targets are computed offline from the training labels.
def ph_arm():
    srcs = load_sources(); CND = json.load(open(PH_CANDS)); lm = model.model.language_model
    apply_lora(lm, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    sd = torch.load(PG_LORA, map_location=model.device); load_lora_state_dict(lm, sd)
    k0 = sorted(k for k in sd if k.endswith(".B"))[0]; m0 = lm.get_submodule(k0[:-2])
    assert torch.equal(m0.B.data.float().cpu(), sd[k0].float().cpu()) and float(sd[k0].abs().max()) > 0, "PG adapter not loaded"
    print("PH loaded PG adapter", len(sd), "tensors", flush=True); del sd
    for m in lm.modules():
        if isinstance(m, LoRALinear): m._s = m.scale
    model.eval(); n_pairs = 0
    with open(f"{OUT}/train_full.jsonl", "w") as ff, open(f"{OUT}/train_pairs.jsonl", "w") as fp:
        for name in sorted(srcs):
            vids = [v for v in srcs[name]["vids"] if v in CND.get(name, {})][:LIMIT_VIDS]; t1 = time.time()
            for c0 in range(0, len(vids), 32):
                ch = vids[c0:c0 + 32]
                with torch.no_grad():
                    prep, keep = eng.prepare(ch, srcs[name]["cpu_fn"], 6, "scene_graph")
                    set_lora_enabled(lm, False)
                    g = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in ch])
                    jobs = []
                    for v in ch:
                        kept = prep[v]["kept"]; pos = {oid: j for j, oid in enumerate(kept)}; men = set()
                        for r in parse_rels(globals().get("SMOKE_DEV_TEXT") or g[v]["text"]):
                            try: a, b_ = int(r[0]), int(r[2])
                            except Exception: continue
                            if 1 <= a <= len(kept) and 1 <= b_ <= len(kept): men.add(frozenset((kept[a - 1], kept[b_ - 1])))
                        ff.write(json.dumps({"source": name, "video_id": v, "kept_obj_ids": kept, "text": g[v]["text"], "stop": g[v]["stop"],
                                             "ids": g[v]["ids"], "lp": [round(x, 4) for x in g[v]["lp"]]}) + "\n")
                        cand = [e for e in CND[name][v] if int(e[0]) in pos and int(e[1]) in pos and frozenset((int(e[0]), int(e[1]))) not in men][:PH_TOPN]
                        jobs += [(v, rk, float(e[2]), [pos[int(e[0])], pos[int(e[1])]]) for rk, e in enumerate(cand)]
                    Pp = [eng.gpu_stage(keep[v], obj_subset=sub, prompt="scene_graph") for v, _, _, sub in jobs]
                    set_lora_enabled(lm, True)
                    if Pp:
                        res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
                        for i, ((v, rk, pr, sub), p) in enumerate(zip(jobs, Pp)):
                            fp.write(json.dumps({"source": name, "video_id": v, "rank": rk, "p": pr, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"],
                                                 "ids": res[i]["ids"], "lp": [round(x, 4) for x in res[i]["lp"]]}) + "\n")
                        n_pairs += len(Pp)
                ff.flush(); fp.flush(); del prep, keep, g, Pp; gc.collect(); torch.cuda.empty_cache()
                print(f"PH {name} {min(c0 + 32, len(vids))}/{len(vids)} pairs {n_pairs} {time.time()-t1:.0f}s", flush=True)
            T[f"ph_{name}"] = dict(videos=len(vids), seconds=round(time.time() - t1, 1)); json.dump(T, open(f"{OUT}/timing_{ARM}.json", "w"))
    T["ph_pairs"] = n_pairs; print("PH DONE", T, flush=True)
    # test: re-decode the P1B evaluation pair lists with PG (greedy, deterministic -> same texts as out/e1_pg) + per-token log-probs
    set_lora_enabled(lm, True); PL = json.load(open(PAIRS_FILE)); CH = {"svg2": 96, "vidor": 96, "pvsg": 16}
    for bn in PAIR_BENCHES:
        b = B[bn]; vids = sorted(v for v in PL[bn] if v in b["masks"] and frames_of(v, b["root"]) and PL[bn][v])[:LIMIT_VIDS]
        cpu_fn = make_cpu_fn(b); t1 = time.time(); n = 0
        with open(f"{OUT}/{bn}_pairs_learned.jsonl", "w") as fo:
            for c0 in range(0, len(vids), CH[bn]):
                ch = vids[c0:c0 + CH[bn]]; jobs = []
                with torch.no_grad():
                    prepared, keep = eng.prepare(ch, cpu_fn, 6, "scene_graph")
                    for v in ch:
                        pos = {oid: j for j, oid in enumerate(keep[v]["kept"])}
                        for rk, e in enumerate(PL[bn][v]):
                            a, b_ = int(e[0]), int(e[1])
                            if a in pos and b_ in pos: jobs.append((v, rk, [pos[a], pos[b_]]))
                    Pp = [eng.gpu_stage(keep[v], obj_subset=sub, prompt="scene_graph") for v, _, sub in jobs]
                    if Pp:
                        res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
                        for i, ((v, rk, sub), p) in enumerate(zip(jobs, Pp)):
                            fo.write(json.dumps({"video_id": v, "rank": rk, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"],
                                                 "ids": res[i]["ids"], "lp": [round(x, 4) for x in res[i]["lp"]]}) + "\n"); n += 1
                del prepared, keep, Pp; gc.collect(); torch.cuda.empty_cache()
        T[f"ph_test_{bn}"] = dict(videos=len(vids), pairs=n, seconds=round(time.time() - t1, 1)); print("PH TEST DONE", bn, T[f"ph_test_{bn}"], flush=True)
