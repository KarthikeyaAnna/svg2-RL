# ---------------- OAC kill test (RESEARCH_LOG §65): object-ablation credit of each predicted relation, released model, dev videos.
# For the greedy graph of each dev video: teacher-forced log-prob of every generated token (no repetition penalty) with the full input,
# and again with one object's visual latents (the video-token positions inside its <obj_traj_start>..<obj_traj_end> block) replaced by
# the mean latent over all objects; block headers, object numbering and timestamps are kept. One ablated pass per object that appears
# in a relation. Per relation: token positions of its predicate + span text; offline: delta = min over {subject, object} of the drop.
def oac_arm(srcs, dv):
    st, en, vt = int(model.config.obj_traj_start_id), int(model.config.obj_traj_end_id), int(model.config.video_token_id)
    emb = model.model.get_input_embeddings(); head = model.lm_head.lm if isinstance(model.lm_head, LastOnlyHead) else model.lm_head
    lm = model.model.language_model
    def tf_lp(e, P, gen):
        n = e.shape[1]; p0 = n + int(P["rope"].reshape(-1)[0]); g = torch.tensor(gen[:-1], device=e.device)
        x = torch.cat([e, emb(g[None]).to(e.dtype)], 1)
        pos = torch.cat([P["pos"], torch.arange(p0, p0 + len(gen) - 1, device=e.device).view(1, 1, -1).expand(3, 1, -1)], 2)
        h = lm(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones((1, x.shape[1]), dtype=torch.long, device=e.device), use_cache=False,
               cache_position=torch.arange(x.shape[1], device=e.device)).last_hidden_state[0, n - 1:]
        out = []
        for c0 in range(0, h.shape[0], 512):
            lp = torch.log_softmax(head(h[c0:c0 + 512]).float(), -1)
            out.append(lp.gather(1, torch.tensor(gen[c0:c0 + 512], device=e.device).view(-1, 1)).squeeze(1))
        return torch.cat(out).cpu()
    REL = re.compile(r'\[\s*(\d+)\s*,\s*("[^"]*"\s*,\s*\d+\s*,\s*\[(?:\s*\[[^\]]*\]\s*,?)*\s*\])\s*\]')
    with open(f"{OUT}/oac_dev.jsonl", "w") as fo:
        for name in sorted(dv):
            t1 = time.time(); prep, _ = eng.prepare(dv[name], srcs[name]["cpu_fn"], 6, "scene_graph")
            g = SlotDecoder(model, ftok, 32, 40.0, 64, True, log=lambda s: None).run([(v, prep[v], token_cap(prep[v]["n_obj"])) for v in dv[name]])
            for v in dv[name]:
                P = prep[v]; gen = g[v]["ids"] if not globals().get("SMOKE_OAC_TEXT") else ftok.encode(SMOKE_OAC_TEXT, add_special_tokens=False)
                if len(gen) < 2: continue
                text = ftok.decode(gen, skip_special_tokens=True); ends = []
                for j in range(len(gen)): ends.append(len(ftok.decode(gen[:j + 1], skip_special_tokens=True)))
                starts = [0] + ends[:-1]; rels = []
                for m in REL.finditer(text):
                    a, b_ = m.start(2), m.end(2); toks = [j for j in range(len(gen)) if starts[j] < b_ and ends[j] > a]
                    s_ = int(m.group(1)); o_ = int(re.match(r'"[^"]*"\s*,\s*(\d+)', m.group(2)).group(1))
                    if toks and 1 <= s_ <= P["n_obj"] and 1 <= o_ <= P["n_obj"]: rels.append((s_, o_, m.group(0), toks))
                if not rels: continue
                ids = P["ids"][0]; S_ = (ids == st).nonzero().flatten().tolist(); E_ = (ids == en).nonzero().flatten().tolist()
                assert len(S_) == len(E_) == P["n_obj"], (len(S_), len(E_), P["n_obj"])
                lat = [[j for j in range(a, b + 1) if int(ids[j]) == vt] for a, b in zip(S_, E_)]
                assert all(lat), "object block without latent positions"
                allp = torch.tensor([j for L in lat for j in L], device=P["e"].device); mu = P["e"][0, allp].mean(0)
                with torch.no_grad():
                    base = tf_lp(P["e"], P, gen); abl = {}
                    for k in sorted({x for s_, o_, _, _ in rels for x in (s_, o_)}):
                        e2 = P["e"].clone(); e2[0, torch.tensor(lat[k - 1], device=e2.device)] = mu; abl[k] = tf_lp(e2, P, gen); del e2
                for s_, o_, txt, toks in rels:
                    t = torch.tensor(toks)
                    fo.write(json.dumps({"source": name, "video_id": v, "kept_obj_ids": P["kept"], "rel": txt, "s": s_, "o": o_, "n_tok": len(toks),
                                         "lp": round(float(base[t].mean()), 4), "lp_s": round(float(abl[s_][t].mean()), 4),
                                         "lp_o": round(float(abl[o_][t].mean()), 4)}) + "\n")
                fo.write(json.dumps({"source": name, "video_id": v, "kept_obj_ids": P["kept"], "graph": text}) + "\n")
                del base, abl; torch.cuda.empty_cache()
            del prep, g; gc.collect(); torch.cuda.empty_cache(); print(f"OAC {name} {len(dv[name])} videos {time.time()-t1:.0f}s", flush=True)
