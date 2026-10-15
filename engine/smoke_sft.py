"""CPU smoke test of engine/sft.py with the tiny TraSeR: target format, LoRA zero-init equivalence, loss decreases, positions match
greedy decoding, merge equivalence, merged model decodes in FastTraSeR/SlotDecoder."""
import sys, os, json, copy
R = os.path.expanduser("~/svg2-killtest"); sys.path.insert(0, f"{R}/SVG2/traser"); sys.path.insert(0, f"{R}/engine")
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
from traser_fast import FastTraSeR, SlotDecoder, install_gqa_decode_attention
from sft import apply_lora, lora_params, merge_lora, build_target, sample_loss, register_style_prompts, lora_state_dict, load_lora_state_dict
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
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cpu", dtype=torch.float32)
install_gqa_decode_attention(); register_style_prompts(I)
check("S0 style prompts registered", "Annotation style: VidOR" in I.PROMPTS["scene_graph_vidor"] and I.PROMPTS["scene_graph_vidor"].endswith(":\n<video>\n"), repr(I.PROMPTS["scene_graph_vidor"]))

# ---- one real SVG2test video (masks + human GT) as a stand-in training sample, random frames
mt = {r["video_id"]: r for r in pq.read_table(f"{R}/data/masks_vipseg.parquet").to_pylist()}
gt = {r["video_id"]: r for r in pq.read_table(f"{R}/data/gt_vipseg.parquet").to_pylist()}
v = sorted(mt, key=lambda k: mt[k]["num_objects"])[5]; r = mt[v]
H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]; total = (nf + 2) // 3
md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
fr = np.random.default_rng(0).integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8)
C = eng.cpu_stage(lambda idx: fr[idx], total, 2.0, md, list(range(no)), H, W)
objs = json.loads(gt[v]["objects"]); rels = json.loads(gt[v]["relationships"])
tgt = build_target(objs, rels, C["kept"])
d = json.loads(tgt)
check("S1 target objects renumbered in kept order", [list(o)[0] for o in d["objects"]] == [f"object {i+1}" for i in range(len(C["kept"]))])
check("S1 relations remapped", all(1 <= x[0] <= len(C["kept"]) and 1 <= x[2] <= len(C["kept"]) for x in d["relationships"]) and len(d["relationships"]) > 0, f"{len(d['relationships'])} rels of {len(rels)}")
first = int(rels[0][0])
sal = build_target(objs, rels, C["kept"], keep_rel=lambda s, o: s != first)
check("S1 keep_rel filter applied", len(json.loads(sal)["relationships"]) == sum(int(x[0]) != first for x in rels) < len(d["relationships"]))
tgt = tgt[:600]  # keep the CPU test short: truncate the target text (format irrelevant for the loss mechanics)

# ---- S2: teacher-forced logits at target positions == greedy decode logits (positions/rope correct)
P = eng.gpu_stage(C, prompt="scene_graph_vidor")
eng_gen = eng.generate([P], loop_stop=False, cap=6)[1][0].tolist()
greedy_txt = tok.decode(eng_gen)
with torch.no_grad():
    l_g, T = sample_loss(eng, P, greedy_txt.replace("<|im_end|>", ""))
ids = tok(greedy_txt.replace("<|im_end|>", "") + "<|im_end|>", add_special_tokens=False)["input_ids"]
# (random-model text like "\n\n\n" re-tokenizes differently; real targets are tokenized once, so only S2b matters)
# verify argmax of teacher-forced logits reproduces greedy tokens (penalty off in tiny config)
m = model; n = P["e"].shape[1]
with torch.no_grad():
    t_ids = torch.tensor([eng_gen])
    emb = m.model.get_input_embeddings()(t_ids)
    x = torch.cat([P["e"], emb[:, :-1]], 1)
    tpos = (n + P["rope"].reshape(-1)[0] + torch.arange(len(eng_gen) - 1)).view(1, 1, -1).expand(3, 1, -1)
    h = m.model.language_model(inputs_embeds=x, position_ids=torch.cat([P["pos"], tpos], 2), use_cache=False).last_hidden_state
    am = m.lm_head.lm(h[:, n - 1:]).argmax(-1)[0].tolist()
check("S2b teacher forcing argmax == greedy decode", am == eng_gen, f"{am} vs {eng_gen}")
# stronger: full logits at every target position equal generate()'s per-step scores (positions + rope exactly right)
from traser_fast import pad_batch
e_, pos_, am_, rope_ = pad_batch([P], "cpu")
with torch.no_grad():
    g = model.generate(inputs_embeds=e_, position_ids=pos_, attention_mask=am_, rope_deltas=rope_, do_sample=False, max_new_tokens=6,
                       output_logits=True, return_dict_in_generate=True, repetition_penalty=1.0)
    tf = m.lm_head.lm(h[:, n - 1:])[0]
gl = torch.stack([x[0] for x in g.logits])
check("S2c teacher-forced logits == generate logits at all steps", torch.allclose(tf, gl, atol=1e-4), f"max diff {(tf-gl).abs().max():.2e}")
# and a position-shift sanity: shifting target positions by +1 must change the logits (test is sensitive)
with torch.no_grad():
    h2 = m.model.language_model(inputs_embeds=x, position_ids=torch.cat([P["pos"], tpos + 1], 2), use_cache=False).last_hidden_state
    tf2 = m.lm_head.lm(h2[:, n - 1:])[0]
check("S2c sensitive to a 1-position shift", not torch.allclose(tf2[1:], gl[1:], atol=1e-4), f"diff {(tf2-gl).abs().max():.2e}")

# ---- S3: LoRA zero-init leaves outputs unchanged; S4 loss decreases; S5 merge == unmerged; S6 merged decodes
with torch.no_grad(): base_loss, _ = sample_loss(eng, P, tgt)
nl = apply_lora(model.model.language_model, r=8, alpha=16, freeze=model)
with torch.no_grad(): l0, _ = sample_loss(eng, P, tgt)
check("S3 LoRA wrapped and zero-init identical", nl == 2 * 7 and torch.allclose(l0, base_loss, atol=1e-6), f"{nl} modules, {base_loss:.4f} vs {l0:.4f}")
opt = torch.optim.AdamW(lora_params(model), lr=5e-3); model.train()
L = []
for step in range(8):
    loss, T = sample_loss(eng, P, tgt); loss.backward(); opt.step(); opt.zero_grad(); L.append(float(loss))
model.eval()
check("S4 loss decreases", L[-1] < L[0] - 1.0 and all(b <= a + 1e-3 for a, b in zip(L, L[1:])), f"{[round(x,3) for x in L]}")
check("S4 only LoRA params trainable", all(not p.requires_grad for n_, p in model.named_parameters() if not (n_.endswith(".A") or n_.endswith(".B"))))
sd = lora_state_dict(model.model.language_model)
with torch.no_grad(): lt, _ = sample_loss(eng, P, tgt)
merge_lora(model.model.language_model)
with torch.no_grad(): lm_, _ = sample_loss(eng, P, tgt)
check("S5 merged loss == LoRA loss", abs(float(lt) - float(lm_)) < 1e-4, f"{float(lt):.5f} vs {float(lm_):.5f}")
check("S5 no LoRA modules left", not any(type(x).__name__ == "LoRALinear" for x in model.modules()))
P2 = eng.gpu_stage(C, prompt="scene_graph_vidor")
out = SlotDecoder(model, ftok, max_slots=2, every=8, loop_stop=False, log=lambda s: None).run([("a", P2, 12)])
check("S6 merged model decodes via SlotDecoder", out["a"]["n_tokens"] == 12 or out["a"]["stop"] == "eos", str(out["a"])[:120])
check("S7 LoRA state dict size", len(sd) == 2 * nl)
print("\nSFT SMOKE:", "ALL PASS" if not FAIL else f"FAILED {FAIL}")
