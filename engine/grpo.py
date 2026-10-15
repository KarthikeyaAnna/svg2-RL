"""On-policy GRPO for TraSeR (LoRA), rewards from the graph-matching reward against TraSeR's own training labels.
Choices fixed a priori: G rollouts per video (temperature 1.0, top-p 0.95, the checkpoint's repetition penalty), group-mean-centred
advantages without std scaling and a constant length normaliser (Dr. GRPO, arXiv 2503.20783), one gradient update per batch (purely
on-policy, so no importance ratio / clipping), KL to the frozen reference via the k3 estimator (reference = LoRA switched off)."""
import torch, torch.nn as nn
from sft import LoRALinear

def set_lora_enabled(module, on):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.scale = m._s if on else 0.0

def _remember_scale(module):
    for m in module.modules():
        if isinstance(m, LoRALinear) and not hasattr(m, "_s"):
            m._s = m.scale

def token_logprobs(eng, P, ids):
    """Log-probabilities of the sampled token ids (list[int]) under the current model, teacher-forced at the decode positions."""
    m = eng.m; dev = P["e"].device
    t = torch.tensor([ids], device=dev); n = P["e"].shape[1]; T = t.shape[1]
    emb = m.model.get_input_embeddings()(t).to(P["e"].dtype)
    x = torch.cat([P["e"], emb[:, :-1]], 1)
    tpos = (n + P["rope"].reshape(-1)[0] + torch.arange(T - 1, device=dev)).view(1, 1, -1).expand(3, 1, -1)
    pos = torch.cat([P["pos"], tpos.to(P["pos"].dtype)], 2)
    h = m.model.language_model(inputs_embeds=x, position_ids=pos, attention_mask=torch.ones(x.shape[:2], dtype=torch.long, device=dev),
                                use_cache=False).last_hidden_state
    head = m.lm_head.lm if hasattr(m.lm_head, "lm") else m.lm_head
    logits = head(h[:, n - 1:]).float()
    return torch.log_softmax(logits, -1).gather(-1, t[..., None]).squeeze(-1)[0]          # [T]

def grpo_loss(eng, P, ids, adv, beta=0.02, norm_len=4096):
    """Loss for one rollout (call backward per rollout to bound memory). Returns (loss, kl_mean)."""
    lm = eng.m.model.language_model; _remember_scale(lm)
    if beta > 0:
        with torch.no_grad():
            set_lora_enabled(lm, False); ref = token_logprobs(eng, P, ids); set_lora_enabled(lm, True)
    lp = token_logprobs(eng, P, ids)
    pg = -(adv * lp).sum() / norm_len
    if beta > 0:
        d = ref - lp; kl = torch.exp(d) - d - 1                                            # k3 estimator, per token
        return pg + beta * kl.sum() / norm_len, float(kl.mean())
    return pg, 0.0
