"""Usage: python build_e1.py ARM out.py [local]   (ARM H0 uses h0_body.py)"""
import sys
R = "/home/karthik/svg2-killtest"; arm, out = sys.argv[1], sys.argv[2]; local = len(sys.argv) > 3
head = open(f"{R}/kaggle/e1/e1_head_{'local' if local else 'kaggle'}.py").read()
for k, f in [("ENGINE_SRC", "engine/traser_fast.py"), ("SFT_SRC", "engine/sft.py"), ("RELPARSE_SRC", "scripts/relparse.py"), ("JUDGE_SRC", "kaggle/e1/judge_src.py"), ("REWARD_SRC", "engine/reward.py"), ("GRPO_SRC", "engine/grpo.py")]:
    src = open(f"{R}/{f}").read()
    if f.endswith("relparse.py") or f.endswith("sft.py"): src = src.replace("from __future__ import annotations\n", "")
    if f.endswith("reward.py"): src = src.replace("from relparse import parse_rels\n", "")
    if f.endswith("grpo.py"): src = src.replace("from sft import LoRALinear\n", "")
    head = head.replace(k + "\n", src + "\n", 1)
cfg = open(f"{R}/kaggle/e1/cfg_{arm}.py").read()
if local:
    cfg = "\n".join(l for l in cfg.splitlines() if not l.startswith("TRAIN_DIRS")) + f'''
LIMIT_VIDS, MAX_PER_SOURCE, ACCUM, N_DEV = 2, 2, 2, 1
TRAIN_DIRS = {{"vidor": "{R}/data/smoke_train_vidor", "pvd": "{R}/data/smoke_train_pvd"}} if ARM != "A0" else {{}}
'''
body = "h0_body.py" if arm == "H0" else "e1_body.py"
if local and arm == "H0": cfg += "\nN_PER_SOURCE, G = 2, 3\n"
if local and arm == "G1": cfg += "\nNSTEPS, EVAL_EVERY, N_DEV, G, BATCH_PROMPTS, LORA_R, LIMIT_VIDS = 2, 1, 1, 3, 2, 8, 1\n_reward0 = reward\ndef reward(*a, **k):\n    R_, d_ = _reward0(*a, **k); return R_ + random.random(), d_   # local smoke only: varied rewards exercise the update path\n"
bsrc = open(f"{R}/kaggle/e1/{body}").read().replace("# GRPO_PART\n", open(f"{R}/kaggle/e1/grpo_part.py").read() + "\n")
open(out, "w").write(head + "\n" + cfg + "\n" + bsrc)
