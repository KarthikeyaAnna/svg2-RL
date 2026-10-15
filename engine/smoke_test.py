"""CPU smoke test of engine/traser_fast.py with a tiny randomly initialised TraSeR (same code paths, fp32, deterministic).
Checks: (T1) padded multi-video batch == one-at-a-time generation, token for token; (T2) LastOnlyHead does not change outputs;
(T3) LoopStop3 stops only the looping row in a batch; (T4) batch planner respects sorting/budget; (T5) pair (object-subset) path."""
import sys, os, json, time
R = os.path.expanduser("~/svg2-killtest")
sys.path.insert(0, f"{R}/SVG2/traser"); sys.path.insert(0, f"{R}/engine")
import numpy as np, torch, pyarrow.parquet as pq
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
from traser_fast import SlotDecoder, FastTraSeR, LoopStop3, LoopStop4, install_gqa_decode_attention, gqa_decode_sdpa, LastOnlyHead, plan_batches, token_cap, KV_BYTES_PER_TOKEN
torch.manual_seed(0); torch.set_num_threads(8)
FAIL = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail, flush=True)
    if not ok: FAIL.append(name)

# ---- tiny TraSeR with the real config structure
cfg = json.load(open(f"{R}/traser_cfg/config.json"))
# 4 query heads share 1 KV head (n_rep 4) so the GQA decode path is exercised; head_dim stays 128 for mrope
cfg.update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, max_window_layers=2,
           temporal_resampler_n_latents=4, object_resampler_n_latents=4, resampler_depth=1)
cfg["vision_config"].update(depth=2, hidden_size=64, out_hidden_size=512, num_heads=2, intermediate_size=128, fullatt_block_indexes=[1])
if "text_config" in cfg: cfg["text_config"].update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512)
config = Qwen2_5_VLConfig(**cfg); config._attn_implementation = "sdpa"
model = TRASER(config).float().eval()
tok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=False); ftok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=True)
proc = AutoProcessor.from_pretrained(f"{R}/qwen_proc"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cpu", dtype=torch.float32)

# ---- T0: GQA decode attention == transformers SDPA (repeat_kv) on random tensors with a left-padding mask
from transformers.integrations.sdpa_attention import sdpa_attention_forward
class _M: num_key_value_groups = 8; is_causal = True
q = torch.randn(3, 16, 1, 128); k = torch.randn(3, 2, 50, 128); v = torch.randn(3, 2, 50, 128)
am = torch.ones(3, 1, 1, 50, dtype=torch.bool); am[0, ..., :17] = False; am[2, ..., :3] = False
ref, _ = sdpa_attention_forward(_M(), q, k, v, am, scaling=128 ** -0.5)
new, _ = gqa_decode_sdpa(_M(), q, k, v, am, scaling=128 ** -0.5, _orig=sdpa_attention_forward)
check("T0 GQA decode attention matches repeat_kv SDPA", ref.shape == new.shape and torch.allclose(ref, new, atol=1e-5), f"max diff {(ref-new).abs().max():.2e}")
install_gqa_decode_attention()

# ---- real SVG2test masks (3 videos with different object counts), random frames
mt = pq.read_table(f"{R}/data/masks_vipseg.parquet").to_pylist()
mt = sorted(mt, key=lambda r: r["num_objects"])
pick = [mt[2], mt[len(mt) // 2], mt[-12]]
def cpu_args(r, n_obj=None):
    H, W, nf, no = r["mask_height"], r["mask_width"], r["num_frames"], r["num_objects"]
    total = (nf + 2) // 3
    md = [[{"size": [H, W], "counts": r["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
    rng = np.random.default_rng(len(r["video_id"]))
    frames = rng.integers(0, 255, (total, H // 4, W // 4, 3), dtype=np.uint8)
    return dict(load_frames=lambda idx: frames[idx], total=total, fps=2.0, mask_data=md, obj_ids=list(range(no))[: (n_obj or 40)], H=H, W=W)
t = time.time()
C = [eng.cpu_stage(**cpu_args(r, 12)) for r in pick]
P = [eng.gpu_stage(c) for c in C]
print("prepared", [(r["video_id"], p["prefill"], p["n_obj"]) for r, p in zip(pick, P)], f"{time.time()-t:.1f}s")
check("different prefill lengths (padding exercised)", len({p["prefill"] for p in P}) == 3, [p["prefill"] for p in P])

CAP = 24
def strip(row):
    row = row.tolist(); return row[: next((i for i, x in enumerate(row) if x in (151643, 151645)), len(row))]
# ---- T1: padded batch == single
single = [strip(eng.generate([p], loop_stop=False, cap=CAP)[1][0]) for p in P]
_, gb = eng.generate(P, loop_stop=False, cap=CAP)
batched = [strip(r) for r in gb]
for i in range(3):
    check(f"T1 batch row {i} == single", single[i] == batched[i], f"len {len(single[i])}/{len(batched[i])} first diff {next((j for j,(a,b) in enumerate(zip(single[i],batched[i])) if a!=b), None)}")
# reversed order (padding on different rows)
_, gr = eng.generate(P[::-1], loop_stop=False, cap=CAP)
check("T1 reversed batch order", [strip(r) for r in gr][::-1] == single)

# ---- T2: LastOnlyHead vs full head
full = model.lm_head.lm; model.lm_head = full
ref = [strip(eng.generate([p], loop_stop=False, cap=CAP)[1][0]) for p in P[:1]]
model.lm_head = LastOnlyHead(full)
check("T2 LastOnlyHead identical", ref[0] == single[0])

# ---- T3: LoopStop3 row-wise on real texts (looping vs clean), fed as token ids
outs = {json.loads(l)["video_id"]: json.loads(l) for l in open(f"{R}/out/traser_svg2test_vipseg.jsonl")}
loop_txt = outs["276_3HgBHBOnpbg"]["text"][:6000]          # attribute-list loop ("white","black",...)
clean_txt = outs["359_4ZPKJtcNGZE"]["text"]
ids = [ftok(t, add_special_tokens=False)["input_ids"] for t in (loop_txt, clean_txt)]
n = max(map(len, ids)); pad = ftok.pad_token_id or 151643
mat = torch.tensor([x + [pad] * (n - len(x)) for x in ids])
ls = LoopStop3(ftok, every=64)
fired_at = [None, None]
for s in range(64, n + 64, 64):
    st = None
    for _ in range(64): st = ls(mat[:, :min(s, n)], None)
    for i in range(2):
        if st[i] and fired_at[i] is None: fired_at[i] = min(s, n)
check("T3 loop row stopped", fired_at[0] is not None, f"at token {fired_at[0]} of {len(ids[0])}")
check("T3 clean row not stopped", fired_at[1] is None, f"clean len {len(ids[1])}")
# LoopStop4 novelty stall: fan-out loop (new object ids, same few triples cycling) vs. a clean relation list
fan = '"relationships": [' + ", ".join(f'[{1 + i % 3}, "next to", {2 + i % 5}, [0.0, 4.0]]' for i in range(60))
clean = '"relationships": [' + ", ".join(f'[{i}, "next to", {i + 1}, [0.0, 4.0]]' for i in range(60))
check("T3b LoopStop4 stall fires on cycling triples", LoopStop4.text_fires(fan))
check("T3b LoopStop4 stall silent on distinct triples", not LoopStop4.text_fires(clean))
check("T3b LoopStop4 silent on real clean output", not LoopStop4.text_fires(clean_txt))

# ---- T4: planner on realistic sizes from the saved SVG2test run
items = [dict(key=o["video_id"], prefill=o.get("prefill_len", 3000), n_obj=len(o["kept_obj_ids"])) for o in outs.values() if "kept_obj_ids" in o]
bs = plan_batches(items, max_batch=32, kv_budget_gb=40.0)
ok = all(len(b) * (max(x["prefill"] for x in b) + max(token_cap(x["n_obj"]) for x in b)) * KV_BYTES_PER_TOKEN <= 40e9 or len(b) == 1 for b in bs)
check("T4 planner budget respected", ok, f"{len(bs)} batches, sizes {[len(b) for b in bs]}, obj ranges {[(min(x['n_obj'] for x in b), max(x['n_obj'] for x in b)) for b in bs]}")
check("T4 all videos planned once", sorted(x["key"] for b in bs for x in b) == sorted(x["key"] for x in items))

# ---- T5: pair path (object subset of an already-prepared video) is valid and batched == single
Pp = [eng.gpu_stage(C[1], obj_subset=[a, b]) for a, b in [(0, 1), (2, 5), (3, 4)]]
check("T5 pair prepare two objects", all(p["n_obj"] == 2 for p in Pp), [p["kept"] for p in Pp])
s1 = [strip(eng.generate([p], loop_stop=False, cap=CAP)[1][0]) for p in Pp]
_, gp = eng.generate(Pp, loop_stop=False, cap=CAP)
check("T5 pair batch == single", [strip(r) for r in gp] == s1)
# ---- T7: cached vision features give the exact same pair inputs as recomputing them
fresh = {k: v for k, v in C[1].items() if k != "vfeat"}
check("T7 vision features cached after full prepare", C[1].get("vfeat") is not None)
for a, b in [(0, 1), (3, 4)]:
    x = eng.gpu_stage(C[1], obj_subset=[a, b]); y = eng.gpu_stage(dict(fresh), obj_subset=[a, b])
    check(f"T7 pair ({a},{b}) cached == recomputed", torch.equal(x["e"], y["e"]) and torch.equal(x["pos"], y["pos"]) and torch.equal(x["rope"], y["rope"]))
check("T7 get_video_features override removed", "get_video_features" not in vars(model.model))

# ---- T6: SlotDecoder (continuous batching, mid-run refills, repetition penalty 1.05) == HF generate one at a time
model.generation_config.repetition_penalty = 3.0   # strong, so the test can detect a wrong penalty (checkpoint uses 1.05)
items = [("v0", P[0], 24), ("v1", P[1], 9), ("v2", P[2], 17), ("p0", Pp[0], 5), ("p1", Pp[1], 30), ("p2", Pp[2], 12)]
ref = {k: strip(eng.generate([p], loop_stop=False, cap=c)[1][0]) for k, p, c in items}
nop = SlotDecoder(model, ftok, max_slots=6, every=64, loop_stop=False, log=lambda s: None); nop.rp = None
nopo = nop.run(items)
check("T6 test is sensitive: dropping the repetition penalty changes some output", any(ftok.decode(ref[k], skip_special_tokens=True) != nopo[k]["text"] for k, _, _ in items))
for slots, every in [(2, 4), (3, 7), (6, 64)]:
    dec = SlotDecoder(model, ftok, max_slots=slots, every=every, loop_stop=False, log=lambda s: None)
    got = dec.run(items)
    ok = all(ftok.decode(ref[k], skip_special_tokens=True) == got[k]["text"] and got[k]["n_tokens"] == len(ref[k]) for k, _, _ in items)
    bad = [k for k, _, _ in items if ftok.decode(ref[k], skip_special_tokens=True) != got[k]["text"]]
    check(f"T6 SlotDecoder slots={slots} every={every} == single generate", ok, f"mismatch {bad} lens {[(got[k]['n_tokens'], len(ref[k])) for k,_,_ in items]}")
# ---- T8: sampling decoder: rollouts differ across group members, reproducible per seed, greedy unaffected
model.generation_config.repetition_penalty = 1.05
sd = lambda seed: SlotDecoder(model, ftok, max_slots=8, every=8, loop_stop=False, log=lambda s: None, temperature=1.0, top_p=0.95, seed=seed)
grp = [(f"g{i}", P[0], 16) for i in range(6)]
a = sd(0).run(grp); b = sd(0).run(grp)
check("T8 sampled rollouts differ within a group", len({a[k]["text"] for k, _, _ in grp}) > 1)
check("T8 same seed reproduces rollouts", all(a[k]["text"] == b[k]["text"] for k, _, _ in grp))
print("\nSMOKE TEST:", "ALL PASS" if not FAIL else f"FAILED: {FAIL}")
