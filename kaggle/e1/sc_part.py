# ---------------- SC arm: K sampled graphs per video (the GRPO rollout distribution) + the greedy graph, for offline aggregation
# (vote consensus / MBR, RESEARCH_LOG §63). SC_LORA = whole-graph adapter (G3) or None (released model = the control).
# Sets: dev (held-out training videos, split_train_dev) + the three test sets. One prefill per video; samples decoded in one batched
# SlotDecoder run with keys v#k, exactly as GRPO rollouts (temperature TEMP, top-p TOP_P, the checkpoint's repetition penalty).
def sc_arm():
    if SC_LORA:
        apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
        sd = torch.load(SC_LORA, map_location=model.device); load_lora_state_dict(model.model.language_model, sd)
        k0 = sorted(k for k in sd if k.endswith(".B"))[0]; m0 = model.model.language_model.get_submodule(k0[:-2])
        assert torch.equal(m0.B.data.float().cpu(), sd[k0].float().cpu()) and float(sd[k0].abs().max()) > 0, "adapter not loaded"
        print("SC loaded adapter", SC_LORA, len(sd), "tensors", flush=True); del sd
        merge_lora(model.model.language_model)
    model.eval(); srcs = load_sources(); _, dv = split_train_dev(srcs, N_DEV); jobs = []
    if globals().get("SC_OAC"): oac_arm(srcs, dv); T["oac_done"] = True
    for name in sorted(dv): jobs.append(("dev", name, dv[name], srcs[name]["cpu_fn"], 16))
    for bn in SC_BENCHES:
        b = B[bn]; jobs.append((bn, bn, sorted(v for v in b["masks"] if v in b["gt"] and frames_of(v, b["root"]))[:LIMIT_VIDS], make_cpu_fn(b), {"pvsg": 8}.get(bn, 32)))
    ci = 0
    for set_, name, vids, cpu_fn, ch_n in jobs:
        t1 = time.time(); stops = collections.Counter()
        with open(f"{OUT}/{set_}_samples.jsonl", "a") as fo:
            for c0 in range(0, len(vids), ch_n):
                ch = vids[c0:c0 + ch_n]; ci += 1
                with torch.no_grad():
                    prep, _ = eng.prepare(ch, cpu_fn, 6, "scene_graph")
                    g0 = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in ch])
                    gs = SlotDecoder(model, ftok, 64, 40.0, 64, True, log=lambda s: None, temperature=TEMP, top_p=TOP_P, seed=ci).run(
                        [(f"{v}#{k}", prep[v], token_cap(prep[v]["n_obj"])) for v in ch for k in range(1, SC_K + 1)])
                for v in ch:
                    for k in range(SC_K + 1):
                        r = g0[v] if k == 0 else gs[f"{v}#{k}"]; stops[(k > 0, r["stop"])] += 1
                        fo.write(json.dumps({"source": name, "video_id": v, "kept_obj_ids": prep[v]["kept"], "k": k, "text": r["text"], "stop": r["stop"]}) + "\n")
                fo.flush(); del prep, g0, gs; gc.collect(); torch.cuda.empty_cache()
                print(f"SC {set_}/{name} {min(c0 + ch_n, len(vids))}/{len(vids)} {time.time()-t1:.0f}s", flush=True)
        T[f"sc_{set_}_{name}"] = dict(videos=len(vids), seconds=round(time.time() - t1, 1), stops={f"{'s' if a else 'g'}_{b_}": n for (a, b_), n in stops.items()})
        json.dump(T, open(f"{OUT}/timing_{ARM}.json", "w")); print("SC DONE", set_, name, T[f"sc_{set_}_{name}"], flush=True)
