import glob, json, os, subprocess, sys, time
def find(p):
    h = glob.glob(p, recursive=True); print("find", p, "->", h[:2], flush=True); return h[0]
ASSETS = os.path.dirname(find("/kaggle/input/**/TRASER/config.json")); ROOT = os.path.dirname(ASSETS)
MASKS = find("/kaggle/input/**/masks_vipseg.parquet"); FRAMES = "/kaggle/input"; OUT = "/kaggle/working"
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
ENGINE_SRC
model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tok = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); ftok = AutoTokenizer.from_pretrained(ASSETS, use_fast=True)
proc = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cuda", dtype=torch.bfloat16)
install_gqa_decode_attention()
NVID, MAXB, KVGB, KPAIRS, MAXB_PAIRS = 10**9, 32, 40.0, 20, 64
import re as _re
def parse_rel_ids(text): return [(int(a), int(b)) for a, b in _re.findall(r'\[\s*(\d+)\s*,\s*"[^"]*"\s*,\s*(\d+)\s*,', text)]
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
