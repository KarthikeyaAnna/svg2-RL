import glob, json, os, sys, time
R = os.path.expanduser("~/svg2-killtest")
sys.path.insert(0, f"{R}/SVG2/traser")
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
ENGINE_SRC
cfg = json.load(open(f"{R}/traser_cfg/config.json"))
cfg.update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512, max_window_layers=2, temporal_resampler_n_latents=4, object_resampler_n_latents=4, resampler_depth=1)
cfg["vision_config"].update(depth=2, hidden_size=64, out_hidden_size=512, num_heads=2, intermediate_size=128, fullatt_block_indexes=[1])
if "text_config" in cfg: cfg["text_config"].update(hidden_size=512, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=1, intermediate_size=512)
config = Qwen2_5_VLConfig(**cfg); config._attn_implementation = "sdpa"; torch.manual_seed(0)
model = TRASER(config).float().eval()
tok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=False); ftok = AutoTokenizer.from_pretrained(f"{R}/traser_cfg", use_fast=True)
proc = AutoProcessor.from_pretrained(f"{R}/qwen_proc"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cpu", dtype=torch.float32)
install_gqa_decode_attention()
MASKS = f"{R}/data/masks_vipseg.parquet"; FRAMES = "/nonexistent"; OUT = "/tmp/fast6_smoke"; os.makedirs(OUT, exist_ok=True)
NVID, MAXB, KVGB, KPAIRS, MAXB_PAIRS = 4, 2, 40.0, 3, 4
import re as _re
def parse_rel_ids(text): return [(int(a), int(b)) for a, b in _re.findall(r'\[\s*(\d+)\s*,\s*"[^"]*"\s*,\s*(\d+)\s*,', text)]
# tiny random model never stops: cap generation for the smoke run
_orig_gen = eng.generate
def _gen(Ps, loop_stop=True, cap=None): return _orig_gen(Ps, loop_stop=loop_stop, cap=16)
eng.generate = _gen
_orig_run = SlotDecoder.run
def _run(self, items): return _orig_run(self, [(k, P, min(c, 16)) for k, P, c in items])
SlotDecoder.run = _run
