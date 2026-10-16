# ---------------- PV arm: the PG pair decoder applied to pairs TraSeR already mentions (verify / replace probe).
# Dev (held-out training videos): base full graph (adapter off) + PG decode of every mentioned pair -> scored locally.
# Test: PG decode of every pair mentioned in the released-model A0 graphs (PAIRS_FILE built from A0 outputs).
def pv_arm():
    srcs = load_sources(); tr, dv = split_train_dev(srcs, N_DEV); devC = {}
    with open(f"{OUT}/dev_full.jsonl", "w") as fo:
        for name in sorted(dv):
            prep, keep = eng.prepare(dv[name], srcs[name]["cpu_fn"], 6, "scene_graph")
            res = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in dv[name]])
            for v in dv[name]:
                fo.write(json.dumps({"source": name, "video_id": v, "kept_obj_ids": prep[v]["kept"], "text": res[v]["text"], "stop": res[v]["stop"]}) + "\n")
                devC[(name, v)] = (keep[v], prep[v]["kept"], globals().get("SMOKE_DEV_TEXT") or res[v]["text"])
            del prep, res; gc.collect(); torch.cuda.empty_cache()
    print("PV dev full graphs", len(devC), flush=True)
    apply_lora(model.model.language_model, r=LORA_R, alpha=2 * LORA_R, freeze=model)
    sd = torch.load(PG_LORA, map_location=model.device); load_lora_state_dict(model.model.language_model, sd)
    k0 = sorted(k for k in sd if k.endswith(".B"))[0]; m0 = model.model.language_model.get_submodule(k0[:-2])
    assert torch.equal(m0.B.data.float().cpu(), sd[k0].float().cpu()) and float(sd[k0].abs().max()) > 0, "PG adapter not loaded"
    print("PV loaded PG adapter", len(sd), "tensors", flush=True); del sd
    merge_lora(model.model.language_model); model.eval(); t0 = time.time(); n = 0
    with open(f"{OUT}/dev_pairs_mentioned.jsonl", "w") as fo:
        jobs = []
        for (name, v), (C, kept, text) in devC.items():
            S = set()
            for r in parse_rels(text):
                try: a, b_ = int(r[0]), int(r[2])
                except Exception: continue
                if 1 <= a <= len(kept) and 1 <= b_ <= len(kept) and a != b_: S.add(tuple(sorted((a - 1, b_ - 1))))
            S = sorted(S)
            jobs += [(name, v, C, [a, b_]) for a, b_ in S]
        for c0 in range(0, len(jobs), 1024):
            ch = jobs[c0:c0 + 1024]; Pp = [eng.gpu_stage(C, obj_subset=sub, prompt="scene_graph") for _, _, C, sub in ch]
            res = SlotDecoder(model, ftok, 64, 40.0, 64, False, log=lambda s: None).run([(i, p, 1024) for i, p in enumerate(Pp)])
            for i, ((name, v, _, _), p) in enumerate(zip(ch, Pp)):
                fo.write(json.dumps({"source": name, "video_id": v, "pair": p["kept"], "kept_obj_ids": p["kept"], "text": res[i]["text"]}) + "\n"); n += 1
            del Pp, res; gc.collect(); torch.cuda.empty_cache()
    T["dev"] = dict(videos=len(devC), pairs=n, seconds=round(time.time() - t0, 1)); print("PV dev pairs", T["dev"], flush=True)
    devC.clear(); gc.collect(); torch.cuda.empty_cache()
    decode_eval_pairs()
