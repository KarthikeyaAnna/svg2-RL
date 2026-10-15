import glob, json, os, subprocess, sys, time, math, random, itertools, collections, gc
R = os.path.expanduser("~/svg2-killtest"); OUT = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/e1_smoke"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, f"{R}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
ENGINE_SRC
SFT_SRC
RELPARSE_SRC
JUDGE_SRC
REWARD_SRC
GRPO_SRC
cfg = json.load(open(f"{R}/traser_cfg/config.json"))
cfg.update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, max_window_layers=2, temporal_resampler_n_latents=4, object_resampler_n_latents=4, resampler_depth=1)
cfg["vision_config"].update(depth=2, hidden_size=64, out_hidden_size=512, num_heads=2, intermediate_size=128, fullatt_block_indexes=[1])
if "text_config" in cfg: cfg["text_config"].update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512)
config = Qwen2_5_VLConfig(**cfg); config._attn_implementation = "sdpa"; torch.manual_seed(0)
model = TRASER(config).float().eval()
tok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=False); ftok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=True)
proc = AutoProcessor.from_pretrained(f"{R}/qwen_proc"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cpu", dtype=torch.float32)
install_gqa_decode_attention(); register_style_prompts(I)
def gpu_sync(): pass
_orig_run = SlotDecoder.run
def _run(self, items): return _orig_run(self, [(k, P, min(c, 12)) for k, P, c in items])
SlotDecoder.run = _run
_orig_tgt = build_target
def build_target(*a, **k):
    t = _orig_tgt(*a, **k); return None if t is None else t[:300]     # short targets on CPU
FAKE_FRAMES = True
SVG2_MASKS = f"{R}/data/masks_vipseg.parquet"; SVG2_GT = f"{R}/data/gt_vipseg.parquet"
VIDOR_DIR = f"{R}/data/vidor/prep"; PVSG_DIR = f"{R}/kaggle/ds-pvsg"; TRAIN_DIR = f"{R}/data/train_vidor"; JUDGE_DIR = None
# local file layout differs from the Kaggle datasets: alias names
import shutil
