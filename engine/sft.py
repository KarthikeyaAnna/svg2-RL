"""Minimal LoRA SFT for TraSeR that reuses FastTraSeR's exact prompt construction (cpu_stage + gpu_stage), so training inputs
are identical to inference inputs. Targets follow TraSeR's training format (data_qwen._renumber_objects_and_relationships):
{"objects": [{"object k": label, "attributes": [...]?}], "relationships": [[s, p, o, [[a, b], ...]], ...]} with objects renumbered
1..K in kept order, then "<|im_end|>".

Style tags: the prompt key selects a tagged prompt (register_style_prompts). LoRA is merged into the base weights after training
(merge_lora), so the merged model runs in the fast engine unchanged."""
import json, math, torch, torch.nn as nn

CAMERA_ID = -1
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=64, alpha=128):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scale).to(x.dtype)


def apply_lora(language_model, r=64, alpha=128, targets=LORA_TARGETS, freeze=None):
    """freeze: the full model (vision tower, resamplers, embeddings, lm_head are frozen too)."""
    for p in (freeze or language_model).parameters():
        p.requires_grad_(False)
    n = 0
    for name, mod in list(language_model.named_modules()):
        for t in targets:
            child = getattr(mod, t, None)
            if isinstance(child, nn.Linear):
                setattr(mod, t, LoRALinear(child, r, alpha)); n += 1
    return n


def lora_params(module):
    return [p for m in module.modules() if isinstance(m, LoRALinear) for p in (m.A, m.B)]


def lora_state_dict(module):
    return {f"{name}.{k}": getattr(m, k).detach().cpu() for name, m in module.named_modules() if isinstance(m, LoRALinear) for k in ("A", "B")}


def load_lora_state_dict(module, sd):
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            m.A.data.copy_(sd[f"{name}.A"]); m.B.data.copy_(sd[f"{name}.B"])


def merge_lora(module):
    """Fold W += scale * B @ A and put the plain nn.Linear back."""
    for name, mod in list(module.named_modules()):
        for t in LORA_TARGETS:
            child = getattr(mod, t, None)
            if isinstance(child, LoRALinear):
                W = child.base.weight
                W.data += (child.scale * (child.B @ child.A)).to(W.dtype)
                setattr(mod, t, child.base)


STYLE_TEXT = {"vidor": " Annotation style: VidOR.", "salient": " Annotation style: salient."}


def register_style_prompts(I):
    base = I.PROMPTS["scene_graph"]
    head, tail = base.split(":\n<video>", 1)
    for k, s in STYLE_TEXT.items():
        I.PROMPTS[f"scene_graph_{k}"] = head + "." + s.rstrip(".") + ":\n<video>" + tail


def build_target(objects, relationships, kept, keep_rel=None):
    """objects: list of {"object_<id>": label, "attributes"?: [...]}; relationships: [[s, p, o, spans, ...]];
    kept: original object ids in prompt order (Object 1 = kept[0]); keep_rel(s, o) -> bool filters relations (salient view).
    Returns JSON text in TraSeR's output format, or None if nothing to supervise."""
    lab = {}
    for ob in objects:
        k = next(k for k in ob if k.startswith("object_"))
        lab[int(k.split("_")[1])] = ob
    idmap = {oid: i + 1 for i, oid in enumerate(kept)}
    objs = []
    for oid in kept:
        ob = lab.get(oid)
        if ob is None:
            return None
        k = next(k for k in ob if k.startswith("object_"))
        d = {f"object {idmap[oid]}": ob[k]}
        if "attributes" in ob:
            d["attributes"] = ob["attributes"]
        objs.append(d)
    rels = []
    for r in relationships:
        s, p, o, spans = int(r[0]), r[1], int(r[2]), r[3]
        ns = CAMERA_ID if s == CAMERA_ID else idmap.get(s)
        no = CAMERA_ID if o == CAMERA_ID else idmap.get(o)
        if ns is None or no is None:
            continue
        if keep_rel is not None and not keep_rel(s, o):
            continue
        rels.append([ns, p, no, spans])
    return json.dumps({"objects": objs, "relationships": rels}, ensure_ascii=False)


def sample_loss(eng, P, target_text):
    """P: dict from eng.gpu_stage (prompt embeddings, 3D positions, rope delta). Teacher-forced loss on target + <|im_end|>.
    Target token j sits at position n + rope + j, exactly where greedy decoding would place it."""
    m, tok = eng.m, eng.tok
    ids = tok(target_text + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"].to(P["e"].device)
    T = ids.shape[1]; n = P["e"].shape[1]
    emb = m.model.get_input_embeddings()(ids).to(P["e"].dtype)
    x = torch.cat([P["e"], emb[:, :-1]], 1)
    tpos = (n + P["rope"].reshape(-1)[0] + torch.arange(T - 1, device=x.device)).view(1, 1, -1).expand(3, 1, -1)
    pos = torch.cat([P["pos"], tpos.to(P["pos"].dtype)], 2)
    h = m.model.language_model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=x.device),
                                use_cache=False).last_hidden_state
    head = m.lm_head.lm if hasattr(m.lm_head, "lm") else m.lm_head
    logits = head(h[:, n - 1:]).float()
    return nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), ids.view(-1)), T
