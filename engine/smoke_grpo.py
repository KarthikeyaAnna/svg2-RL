"""CPU smoke test of engine/grpo.py with the tiny TraSeR: logprobs match decoding, LoRA-off == base, KL=0 at init, and GRPO increases
a synthetic reward (fraction of rollout tokens equal to a chosen id)."""
import sys, os, json
R = os.path.expanduser("~/svg2-killtest"); sys.path.insert(0, f"{R}/SVG2/traser"); sys.path.insert(0, f"{R}/engine")
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
from traser_fast import FastTraSeR, SlotDecoder, install_gqa_decode_attention, pad_batch
from sft import apply_lora, lora_params
from grpo import token_logprobs, grpo_loss, set_lora_enabled, _remember_scale
torch.manual_seed(0); torch.set_num_threads(8); FAIL = []
def check(n, ok, d=""):
    print(("PASS " if ok else "FAIL ") + n, d, flush=True)
    if not ok: FAIL.append(n)
cfg = json.load(open(f"{R}/traser_cfg/config.json"))
cfg.update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, max_window_layers=2,
           temporal_resampler_n_latents=4, object_resampler_n_latents=4, resampler_depth=1)
cfg["vision_config"].update(depth=2, hidden_size=64, out_hidden_size=512, num_heads=2, intermediate_size=128, fullatt_block_indexes=[1])
if "text_config" in cfg: cfg["text_config"].update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512)
config = Qwen2_5_VLConfig(**cfg); config._attn_implementation = "sdpa"
model = TRASER(config).float().eval()
tok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=False); ftok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=True)
proc = AutoProcessor.from_pretrained(f"{R}/qwen_proc"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cpu", dtype=torch.float32); install_gqa_decode_attention()
mt = {r["video_id"]: r for r in pq.read_table(f"{R}/data/masks_vipseg.parquet").to_pylist()}
v = sorted(mt, key=lambda k: mt[k]["num_objects"])[5]; r = mt[v]
H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]; total = (nf + 2) // 3
md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
fr = np.random.default_rng(0).integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8)
C = eng.cpu_stage(lambda idx: fr[idx], total, 2.0, md, list(range(no)), H, W); P = eng.gpu_stage(C)
model.generation_config.repetition_penalty = 1.0
# G1: teacher-forced logprobs == log_softmax of generate() logits at the sampled tokens
e_, pos_, am_, rope_ = pad_batch([P], "cpu")
with torch.no_grad():
    g = model.generate(inputs_embeds=e_, position_ids=pos_, attention_mask=am_, rope_deltas=rope_, do_sample=False, max_new_tokens=8, output_logits=True, return_dict_in_generate=True)
    ids = g.sequences[0].tolist(); ref = torch.stack([torch.log_softmax(x[0].float(), -1)[t] for x, t in zip(g.logits, ids)])
    lp = token_logprobs(eng, P, ids)
check("G1 token_logprobs == decode log-probs", torch.allclose(lp, ref, atol=1e-4), f"max diff {(lp-ref).abs().max():.2e}")
# G2: LoRA (after a perturbation) switched off == base; KL 0 at zero-init
apply_lora(model.model.language_model, r=8, alpha=16, freeze=model); _remember_scale(model.model.language_model)
with torch.no_grad(): l0, kl0 = grpo_loss(eng, P, ids, adv=1.0, beta=0.02)
check("G2 KL is 0 at LoRA zero-init", abs(kl0) < 1e-6, f"{kl0:.2e}")
for p_ in lora_params(model): p_.data.add_(0.05 * torch.randn_like(p_))
with torch.no_grad():
    set_lora_enabled(model.model.language_model, False); lp_off = token_logprobs(eng, P, ids); set_lora_enabled(model.model.language_model, True)
    lp_on = token_logprobs(eng, P, ids)
check("G2 LoRA-off reproduces base log-probs", torch.allclose(lp_off, lp, atol=1e-5) and not torch.allclose(lp_on, lp, atol=1e-3))
for p_ in lora_params(model):                                     # back to zero-init B for the learning test
    if p_.shape[1] == 8 and p_.shape[0] != 8: p_.data.zero_()
# G3: GRPO raises a synthetic reward. Make the toy policy peaked (scale lm_head) so a handful of tokens carry the mass, then reward
# the 2nd most frequent sampled token: present but not dominant, the regime GRPO operates in.
with torch.no_grad(): model.lm_head.lm.weight.mul_(6.0)
import collections
d0 = SlotDecoder(model, ftok, max_slots=16, every=8, loop_stop=False, log=lambda s: None, temperature=1.0, seed=99).run([(f"z{i}", P, 12) for i in range(16)])
freq = collections.Counter(x for o in d0.values() for x in o["ids"]); TARGET = freq.most_common(2)[1][0]
print("toy policy top tokens:", freq.most_common(4))
def rew(t): return sum(1 for x in t if x == TARGET) / max(1, len(t))
opt = torch.optim.AdamW(lora_params(model), lr=5e-3); hist = []; gn = []
for step in range(25):
    dec = SlotDecoder(model, ftok, max_slots=8, every=8, loop_stop=False, log=lambda s: None, temperature=1.0, top_p=1.0, seed=step)
    out = dec.run([(f"g{i}", P, 12) for i in range(8)])
    rs = np.array([rew(out[f"g{i}"]["ids"]) for i in range(8)]); hist.append(rs.mean())
    model.train()
    for i in range(8):
        loss, _ = grpo_loss(eng, P, out[f"g{i}"]["ids"], adv=float(rs[i] - rs.mean()), beta=0.0, norm_len=12); loss.backward()
    gn.append(float(sum(p_.grad.norm() ** 2 for p_ in lora_params(model) if p_.grad is not None) ** 0.5))
    opt.step(); opt.zero_grad(); model.eval()
print("mean synthetic reward per step:", [round(float(x), 2) for x in hist]); print("grad norms:", [f"{x:.1e}" for x in gn[:5]])
check("G3 GRPO increases the reward", np.mean(hist[-5:]) > np.mean(hist[:5]) + 0.15, f"{np.mean(hist[:5]):.2f} -> {np.mean(hist[-5:]):.2f}")
print("\nGRPO SMOKE:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
