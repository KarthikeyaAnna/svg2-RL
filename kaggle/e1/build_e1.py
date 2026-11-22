"""Usage: python build_e1.py ARM out.py [local]   (ARM H0 uses h0_body.py)"""
import sys
R = "/home/karthik/svg2-killtest"; arm, out = sys.argv[1], sys.argv[2]; local = len(sys.argv) > 3
head = open(f"{R}/kaggle/e1/e1_head_{'local' if local else 'kaggle'}.py").read()
for k, f in [("ENGINE_SRC", "engine/traser_fast.py"), ("SFT_SRC", "engine/sft.py"), ("RELPARSE_SRC", "scripts/relparse.py"), ("JUDGE_SRC", "kaggle/e1/judge_src.py"), ("REWARD_SRC", "engine/reward.py"), ("GRPO_SRC", "engine/grpo.py"), ("CREDIT_SRC", "engine/credit.py")]:
    src = open(f"{R}/{f}").read()
    if f.endswith("relparse.py") or f.endswith("sft.py"): src = src.replace("from __future__ import annotations\n", "")
    if f.endswith("reward.py"): src = src.replace("from relparse import parse_rels\n", "")
    if f.endswith("grpo.py"): src = src.replace("from sft import LoRALinear\n", "")
    if f.endswith("credit.py"): src = src.replace("from relparse import parse_rels, REL_RX\n", "").replace("from reward import norm, tiou, reward\n", "")
    head = head.replace(k + "\n", src + "\n", 1)
cfg = open(f"{R}/kaggle/e1/cfg_{arm}.py").read()
if local:
    cfg = "\n".join(l for l in cfg.splitlines() if not l.startswith("TRAIN_DIRS"))
    if arm == "PHA":
        td = f'{{"pvd": "{R}/data/smoke_train_pvd"}}'
    elif arm == "PHB":
        td = f'{{"vidor": "{R}/data/smoke_train_vidor"}}'
    else:
        td = f'{{"vidor": "{R}/data/smoke_train_vidor", "pvd": "{R}/data/smoke_train_pvd"}}' if arm != "A0" else "{}"
    cfg += f'\nLIMIT_VIDS, MAX_PER_SOURCE, ACCUM, N_DEV = 2, 2, 2, 1\nTRAIN_DIRS = {td}\n'
if __import__("os").environ.get("TRAIN_SEED"): cfg += f"\nTRAIN_SEED = {int(__import__('os').environ['TRAIN_SEED'])}\n"
body = {"H0": "h0_body.py", "P1": "p1_body.py", "P1B": "p1_body.py"}.get(arm, "e1_body.py")
if local and arm == "H0": cfg += "\nN_PER_SOURCE, G = 2, 3\n"
if local and arm == "PG":
    cfg = cfg.replace('find("**/pairsft_train.json")', repr("/home/karthik/svg2-killtest/data/ds_pairsft/pairsft_train.json")).replace('find("**/pairlists_eval.json")', repr("/home/karthik/svg2-killtest/data/ds_pairsft/pairlists_eval.json")) + "\nNSTEPS, BATCH_PAIRS, G, LORA_R, LIMIT_VIDS, N_DEV = 2, 3, 3, 8, 1, 1\n_reward0 = reward\ndef reward(*a, **k):\n    R_, d_ = _reward0(*a, **k); return R_ + random.random(), d_\nSKIP_WARMSTART = True\n"
if local and arm == "PV":
    cfg = cfg.replace('find("**/pairlists_mentioned.json")', repr("/home/karthik/svg2-killtest/data/ds_pv/pairlists_mentioned.json")).replace('find("**/lora_PG.pt")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_pv_smoke.pt")) + "\nLORA_R, LIMIT_VIDS, N_DEV = 8, 1, 1\nimport copy\n_lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)\nfor _n, _p in _lm.named_parameters():\n    if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)\ntorch.save(lora_state_dict(_lm), PG_LORA); del _lm\nSMOKE_DEV_TEXT = json.dumps({'objects': [], 'relationships': [[1, 'on', 2, [[0, 3]]], [2, 'under', 1, [[0, 3]]], [1, 'next to', 3, [[0, 3]]], [3, 'x', 9, [[0, 1]]]]})\n"
if local and arm == "HX": cfg += "\nLIMIT_VIDS = 2\n"
if local and arm in ("PH", "PHA", "PHB"):
    cfg = cfg.replace('find("**/ph_cands.json")', repr(f"{R}/data/ds_ph/ph_cands.json")).replace('find("**/lora_PG.pt")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_pv_smoke.pt")).replace('find("**/pairlists_eval.json")', repr(f"{R}/data/ds_pairsft/pairlists_eval.json")) + "\nLORA_R, LIMIT_VIDS = 8, 2\nSMOKE_DEV_TEXT = json.dumps({'objects': [], 'relationships': [[1, 'on', 2, [[0, 3]]]]})\n"
if local and arm == "SV":
    sm = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_pv_smoke.pt"
    cfg = cfg.replace('find("**/traser-svg2sav/**/masks_sav.parquet")', repr(f"{R}/data/ds_svg2sav/masks_sav.parquet")).replace('find("**/traser-svg2sav/**/gt_sav.parquet")', repr(f"{R}/data/ds_svg2sav/gt_sav.parquet")).replace('find("**/traser-svg2sav/**/sav_pair_cands.json")', repr(f"{R}/data/ds_svg2sav/sav_pair_cands.json")).replace('find("**/lora_PG.pt")', repr(sm)).replace('find("**/lora_G3.pt")', repr(sm)) + "\nLORA_R, LIMIT_VIDS = 8, 2\nimport copy\n_lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)\nfor _n, _p in _lm.named_parameters():\n    if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)\ntorch.save(lora_state_dict(_lm), SV_PG); del _lm\n"
if local and arm == "KT":
    cfg = cfg.replace('find("**/ph_cands.json")', repr(f"{R}/data/ds_ph/ph_cands.json")) + "\nKT_NTRAIN, N_DEV = 1, 1\nSMOKE_OAC_TEXT = json.dumps({'objects': [{'object 1': 'man', 'attributes': ['tall']}, {'object 2': 'cup', 'attributes': []}, {'object 3': 'table', 'attributes': ['wood']}], 'relationships': [[1, 'holding', 2, [[0.0, 3.5]]], [2, 'on', 3, [[1.0, 2.0]]]]})\n"
if local and arm == "SVP":
    sm = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_svp_smoke.pt"
    cfg = cfg.replace('find("**/traser-svg2sav/**/masks_sav.parquet")', repr(f"{R}/data/ds_svg2sav/masks_sav.parquet")).replace('find("**/traser-svg2sav/**/gt_sav.parquet")', repr(f"{R}/data/ds_svg2sav/gt_sav.parquet")).replace('find("**/svg2sav_pairlists.json")', repr(f"{R}/out/e1_sv/svg2sav_pairlists.json")).replace('find("**/lora_PS.pt")', repr(sm)) + "\nLORA_R, LIMIT_VIDS = 8, 2\nimport copy\n_lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)\nfor _n, _p in _lm.named_parameters():\n    if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)\ntorch.save(lora_state_dict(_lm), SVP_PS); del _lm\n"
if local and arm == "VS": cfg += "\nVS_EXTRA, N_DEV, LIMIT_VIDS = 2, 2, 3\nSMOKE_VS_ANY_DONOR = True\nSMOKE_VS_TEXT = json.dumps({'objects': [{'object 1': 'man', 'attributes': ['tall']}, {'object 2': 'cup', 'attributes': []}], 'relationships': [[1, 'holding', 2, [[0.0, 3.5]]]]})\n"
if local and arm == "SCB": cfg += "\nSMOKE_OAC_TEXT = json.dumps({'objects': [{'object 1': 'man'}], 'relationships': [[1, 'holding', 2, [[0.0, 3.5]]], [2, 'on', 3, [[1.0, 2.0], [3.0, 4.0]]]]})\n"
if local and arm == "DT":
    cfg = cfg.replace('find("**/lora_G3.pt")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_dt_smoke.pt")) + "\nLORA_R, LIMIT_VIDS, N_DEV, SC_K = 8, 2, 1, 3\nimport copy\n_lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)\nfor _n, _p in _lm.named_parameters():\n    if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)\ntorch.save(lora_state_dict(_lm), SC_LORA); del _lm\n"
if local and arm == "DS":
    cfg = cfg.replace('find("**/distill_targets.json")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/distill_smoke/distill_targets.json")).replace('find("**/traser-svg2sav/**/masks_sav.parquet")', repr(f"{R}/data/ds_svg2sav/masks_sav.parquet")).replace('find("**/traser-svg2sav/**/gt_sav.parquet")', repr(f"{R}/data/ds_svg2sav/gt_sav.parquet")) + "\nMAX_PER_SOURCE, EPOCHS, LORA_R, LIMIT_VIDS, N_DEV = 2, 1, 8, 1, 1\n"
if local and arm in ("SC", "SCB"):
    cfg = cfg.replace('find("**/lora_G3.pt")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/lora_pv_smoke.pt")) + "\nLORA_R, LIMIT_VIDS, N_DEV, SC_K = 8, 1, 1, 3\nif SC_LORA:\n    import copy\n    _lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)\n    for _n, _p in _lm.named_parameters():\n        if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)\n    torch.save(lora_state_dict(_lm), SC_LORA); del _lm\n"
if local and arm == "PS":
    cfg = cfg.replace('find("**/pairsft_train.json")', repr("/home/karthik/svg2-killtest/data/ds_pairsft/pairsft_train.json")).replace('find("**/pairlists_eval.json")', repr("/home/karthik/svg2-killtest/data/ds_pairsft/pairlists_eval.json")) + "\nMAX_PAIR_SAMPLES, ACCUM, LORA_R, LIMIT_VIDS, N_DEV = 6, 2, 8, 1, 1\n"
if local and arm in ("P1", "P1B"):
    pl = "/home/karthik/svg2-killtest/data/ds_pairlists_" + ("p1b" if arm == "P1B" else "p1") + "/pairlists_p1.json"
    cfg = cfg.replace('find("**/pairlists_p1.json")', repr(pl)) + "\nLIMIT_VIDS = 2\n"
if local and arm in ("G1", "G2", "G3"): cfg += "\nNSTEPS, EVAL_EVERY, N_DEV, G, BATCH_PROMPTS, LORA_R, LIMIT_VIDS = 2, 1, 1, 3, 2, 8, 1\n_reward0 = reward\ndef reward(*a, **k):\n    R_, d_ = _reward0(*a, **k); return R_ + random.random(), d_   # local smoke only: varied rewards exercise the update path\n"
if local and arm in ("G5", "G5X", "G6", "G6L"):
    cfg = "\n".join(l for l in cfg.splitlines() if not l.startswith("TRAIN_DIRS") and not l.startswith("              \"sav\"")) 
    cfg = cfg.replace('find("**/jt_pairs.json")', repr(f"{R}/data/ds_jt/jt_pairs.json")).replace('find("**/judge_JT.json")', repr("/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/judge_JT_smoke.json"))
    cfg += f"""
TRAIN_DIRS = {{"vidor": "{R}/data/smoke_train_vidor", "pvd": "{R}/data/smoke_train_pvd", "sav": "{R}/data/smoke_train_sav"}}
NSTEPS, EVAL_EVERY, N_DEV, G, BATCH_PROMPTS, LORA_R, LIMIT_VIDS = 2, 1, 1, 3, 3, 8, 1
"""
if local and arm == "G4":
    smoke_g3_lora = "/tmp/smoke_lora_g3.pt"
    cfg = cfg.replace('find("**/lora_G3.pt")', repr(smoke_g3_lora))
    cfg += f"""
NSTEPS, EVAL_EVERY, N_DEV, G, BATCH_PROMPTS, LORA_R, LIMIT_VIDS = 2, 1, 1, 3, 2, 8, 1
if WARMSTART_LORA:
    import copy
    _lm = copy.deepcopy(model.model.language_model); apply_lora(_lm, r=8, alpha=16, freeze=_lm)
    for _n, _p in _lm.named_parameters():
        if _n.endswith('.A') or _n.endswith('.B'): torch.nn.init.normal_(_p, std=0.02)
    torch.save(lora_state_dict(_lm), WARMSTART_LORA); del _lm
_reward_g4_0 = reward_g4
def reward_g4(*a, **k):
    R_, d_ = _reward_g4_0(*a, **k); return R_ + random.random(), d_
"""

bsrc = open(f"{R}/kaggle/e1/{body}").read().replace("# GRPO_PART\n", open(f"{R}/kaggle/e1/grpo_part.py").read() + "\n" + open(f"{R}/kaggle/e1/ps_part.py").read() + "\n" + open(f"{R}/kaggle/e1/pg_part.py").read() + "\n" + open(f"{R}/kaggle/e1/pv_part.py").read() + "\n" + open(f"{R}/kaggle/e1/hx_part.py").read() + "\n" + open(f"{R}/kaggle/e1/sc_part.py").read() + "\n" + open(f"{R}/kaggle/e1/ph_part.py").read() + "\n" + open(f"{R}/kaggle/e1/oac_part.py").read() + "\n" + open(f"{R}/kaggle/e1/kt_part.py").read() + "\n" + open(f"{R}/kaggle/e1/sv_part.py").read() + "\n" + open(f"{R}/kaggle/e1/vs_part.py").read() + "\n" + open(f"{R}/kaggle/e1/svp_part.py").read() + "\n")
open(out, "w").write(head + "\n" + cfg + "\n" + bsrc)
