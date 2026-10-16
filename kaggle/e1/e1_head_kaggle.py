import glob, json, os, subprocess, sys, time, math, random, itertools, collections, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def find(p, root="/kaggle/input"):
    h = sorted(glob.glob(f"{root}/{p}", recursive=True)); print("find", p, "->", h[:2], flush=True); return h[0]
ASSETS = os.path.dirname(find("**/TRASER/config.json")); ROOT = os.path.dirname(ASSETS); OUT = "/kaggle/working"
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
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
model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
tok = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); ftok = AutoTokenizer.from_pretrained(ASSETS, use_fast=True)
proc = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct"); proc.tokenizer = tok
eng = FastTraSeR(model, proc, tok, ftok, I, rearrange_token, select_tokens, device="cuda", dtype=torch.bfloat16)
install_gqa_decode_attention(); register_style_prompts(I)
def gpu_sync(): torch.cuda.synchronize()
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
# benchmark data roots
SVG2_MASKS = find("**/masks_vipseg.parquet"); SVG2_GT = find("**/gt_vipseg.parquet"); SVG2_FR = os.path.dirname(os.path.dirname(SVG2_MASKS))
VIDOR_DIR = os.path.dirname(find("**/vidor_meta.json")); PVSG_DIR = os.path.dirname(find("**/pvsg-val-traser/**/meta.json") if glob.glob("/kaggle/input/**/pvsg-val-traser/**/meta.json", recursive=True) else find("**/pvsg*/**/meta.json"))
TRAIN_DIR = os.path.dirname(find("**/meta_train.json")) if glob.glob("/kaggle/input/**/meta_train.json", recursive=True) else None
JUDGE_DIR = os.path.dirname(find("**/qwen-3/**/config.json")) if glob.glob("/kaggle/input/**/qwen-3/**/config.json", recursive=True) else None
print("dirs", SVG2_FR, VIDOR_DIR, PVSG_DIR, TRAIN_DIR, JUDGE_DIR, flush=True)
