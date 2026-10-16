# ---------------- HX arm: per-object latents of the released model for a learned pair-proposal head (no decoding).
# One prefill per video (all kept objects, scene_graph prompt). For three decoder layers (1/3, 2/3, last): mean hidden state over
# each object's block (<obj_traj_start> ... <obj_traj_end>) and the final prompt token's state (attends to every object).
def hx_arm():
    lm = model.model.language_model; nL = len(lm.layers); LAY = [nL // 3 - 1, 2 * nL // 3 - 1, nL - 1]; cap = {}
    def hook(i):
        def f(m, a, o): cap[i] = (o[0] if isinstance(o, tuple) else o).detach()
        return f
    hs = [lm.layers[i].register_forward_hook(hook(i)) for i in LAY]
    st, en = int(model.config.obj_traj_start_id), int(model.config.obj_traj_end_id)
    def feats(P):
        n = P["e"].shape[1]; dev = P["e"].device
        with torch.no_grad():
            lm(inputs_embeds=P["e"], position_ids=P["pos"], attention_mask=torch.ones((1, n), dtype=torch.long, device=dev), use_cache=False,
               cache_position=torch.arange(n, device=dev))
        ids = P["ids"][0]; s_ = (ids == st).nonzero().flatten().tolist(); e_ = (ids == en).nonzero().flatten().tolist()
        assert len(s_) == len(e_) == P["n_obj"], (len(s_), len(e_), P["n_obj"])
        obj = torch.stack([torch.stack([cap[i][0, a:b + 1].float().mean(0) for i in LAY]) for a, b in zip(s_, e_)]).half().cpu()
        ctx = torch.stack([cap[i][0, -1].float() for i in LAY]).half().cpu(); cap.clear(); return obj, ctx
    jobs = []
    if "train" in HX_SETS:
        srcs = load_sources(); jobs += [(f"train_{n}", srcs[n]["vids"][:LIMIT_VIDS], srcs[n]["cpu_fn"], 16) for n in sorted(srcs)]
    for bn in HX_BENCHES:
        b = B[bn]; jobs.append((bn, sorted(v for v in b["masks"] if v in b["gt"] and frames_of(v, b["root"]))[:LIMIT_VIDS], make_cpu_fn(b), {"pvsg": 8}.get(bn, 16)))
    for name, vids, cpu_fn, ch_n in jobs:
        out = {}; t1 = time.time()
        for c0 in range(0, len(vids), ch_n):
            prepared, keep = eng.prepare(vids[c0:c0 + ch_n], cpu_fn, 6, "scene_graph")
            for v, P in prepared.items():
                obj, ctx = feats(P); out[v] = dict(kept=list(P["kept"]), obj=obj, ctx=ctx)
            del prepared, keep; gc.collect(); torch.cuda.empty_cache()
            if c0 % (ch_n * 20) == 0: print(f"HX {name} {min(c0 + ch_n, len(vids))}/{len(vids)} {time.time()-t1:.0f}s", flush=True)
        torch.save(dict(layers=LAY, data=out), f"{OUT}/lat_{name}.pt"); T[f"hx_{name}"] = dict(videos=len(out), seconds=round(time.time() - t1, 1))
        print("HX DONE", name, T[f"hx_{name}"], flush=True)
    for h in hs: h.remove()
