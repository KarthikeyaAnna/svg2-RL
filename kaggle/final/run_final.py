# Kill test: run released TraSeR on SVG2test (VIPSeg part) with GT masks; save raw outputs.
import glob, json, os, subprocess, sys, time, math
def find(pattern):
    hits = glob.glob(pattern, recursive=True); print("find", pattern, "->", hits[:3], flush=True)
    return hits[0]
ASSETS = os.path.dirname(find("/kaggle/input/**/TRASER/config.json"))
ROOT = os.path.dirname(ASSETS)
DATA = os.path.dirname(find("/kaggle/input/**/masks_vipseg.parquet"))
subprocess.run(f"{sys.executable} -m pip install -q --no-index --find-links {ROOT}/wheels transformers==4.54.1 'tokenizers<0.22' 'huggingface_hub<1.0' safetensors pycocotools", shell=True, check=True)
sys.path.insert(0, f"{ROOT}/SVG2/traser")
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
import inference as I
from traser_train.train.modeling_traser import TRASER
from traser_train.train.token_arrangement import rearrange_token
from traser_train.train.token_selection import select_tokens
print("torch", torch.__version__, torch.cuda.get_device_name(0), flush=True)
FPS = 2.0      # VIPSeg imgs are every 3rd frame of the 6 fps source; masks are 6 fps -> take mask frame 3j
model = TRASER.from_pretrained(ASSETS, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
processor = AutoProcessor.from_pretrained(f"{ROOT}/Qwen2.5-VL-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(ASSETS, use_fast=False); processor.tokenizer = tokenizer
masks = {r["video_id"]: r for r in pq.read_table(f"{DATA}/masks_vipseg.parquet").to_pylist()}

PAIRS = json.loads('{"1016_8J41CsGYhNI": [[1, 3], [2, 3], [1, 13], [3, 13], [3, 11], [2, 10], [2, 14], [11, 13], [3, 10], [3, 14], [1, 12], [1, 9], [2, 6], [3, 4], [3, 12], [3, 7], [3, 8], [10, 13], [13, 14], [10, 11]], "1047_-_XOsbwGZgg": [[7, 10], [7, 8], [6, 7], [7, 24], [7, 26], [5, 7], [7, 42], [7, 36], [7, 33], [7, 39], [7, 9], [7, 17], [7, 16], [7, 51], [4, 7], [7, 37], [7, 20], [7, 38], [7, 30], [7, 32]], "1086_iywqpda7d8k": [[0, 3], [0, 6], [0, 7], [0, 5], [0, 8], [2, 6], [1, 6], [1, 7], [1, 5], [3, 6], [3, 8], [6, 7], [5, 6], [4, 5], [3, 7], [2, 5]], "1125_Cbv34yDZ8gc": [[6, 25], [6, 7], [0, 15], [15, 17], [15, 18], [6, 10], [1, 15], [0, 16], [6, 24], [6, 34], [6, 26], [5, 15], [0, 18], [15, 32], [14, 15], [12, 15], [11, 15], [1, 16], [0, 1], [16, 45]], "1187_AqH9dWCvTkY": [[0, 24], [0, 25], [0, 23], [0, 3], [0, 4], [0, 21], [0, 31], [12, 13], [12, 22], [12, 20], [12, 25], [6, 12], [12, 19], [12, 26], [12, 35], [5, 12], [3, 12], [4, 12], [12, 27], [12, 18]], "1190_UmpN6eLjN8w": [[3, 5], [1, 5], [5, 6], [5, 7], [4, 5], [5, 17], [2, 5], [0, 5], [5, 8], [5, 11], [5, 9], [5, 16], [5, 12], [5, 13], [5, 10], [5, 14], [5, 15], [1, 7], [2, 3], [3, 11]], "1251__MCY-QkVaZw": [[0, 3], [1, 3], [2, 3], [3, 9], [3, 8], [3, 7], [3, 10], [3, 5], [3, 4], [3, 6], [0, 10], [2, 4], [2, 6], [9, 10], [3, 11], [2, 5], [0, 1], [1, 11], [7, 9]], "1253_TPxVj7do42I": [[2, 20], [2, 17], [2, 13], [1, 20], [1, 19], [0, 15], [1, 13], [5, 10], [0, 16], [4, 10], [4, 5], [8, 19], [15, 16], [12, 20], [15, 17], [13, 20], [14, 20], [8, 12], [10, 12], [10, 11]], "1256_Y_19xu4yTms": [[7, 14], [7, 24], [2, 7], [7, 12], [7, 22], [3, 7], [7, 10], [7, 13], [7, 28], [7, 16], [7, 18], [7, 11], [6, 14], [2, 6], [6, 8], [6, 12], [6, 21], [6, 26], [3, 6], [6, 10]], "1261_wsHfWwHXgLs": [[0, 6], [0, 4], [3, 5], [3, 7], [5, 7]], "1268_h58xqlUE5xA": [[4, 5], [1, 2], [6, 7], [0, 7], [7, 8], [7, 9], [0, 1], [0, 10], [0, 9], [8, 10], [8, 9], [0, 2], [2, 10]], "1275_ARcg-EyKWrA": [[0, 51], [0, 2], [7, 12], [12, 51], [0, 30], [0, 36], [0, 39], [0, 40], [0, 43], [0, 38], [0, 57], [0, 37], [0, 45], [0, 56], [0, 58], [0, 31], [0, 49], [0, 29], [0, 24], [0, 4]], "1308_-C_XboJfD0g": [[0, 10], [0, 1], [0, 5], [0, 13], [0, 3], [0, 9], [0, 6], [0, 4], [0, 8], [0, 7], [0, 12], [0, 11], [0, 2], [1, 10], [5, 10], [10, 13], [3, 10], [9, 10], [6, 10], [4, 10]], "1352_nt-UZxGk9Bg": [[0, 1], [1, 10], [1, 11], [1, 2], [1, 6], [1, 3], [1, 12], [1, 7], [1, 5], [1, 8], [1, 4], [3, 10], [4, 10], [2, 6], [2, 12], [2, 7], [2, 13], [6, 13], [3, 5], [7, 12]], "1480_WAGXu9_6Uv0": [[26, 27], [23, 27], [11, 12], [10, 12], [0, 12], [27, 36], [18, 27], [27, 28], [13, 27], [11, 26], [15, 26], [27, 43], [27, 41], [27, 45], [27, 40], [27, 42], [27, 44], [12, 36], [22, 26], [12, 29]], "1503_gwrvCEAZVl0": [[3, 4], [3, 18], [3, 13], [3, 17], [3, 15], [3, 19], [1, 3], [3, 5], [0, 3], [3, 16], [3, 12], [2, 3], [3, 7], [3, 14], [3, 9], [3, 10], [3, 8], [4, 18], [4, 13], [13, 18]], "1638_MhoaeR88gm4": [[39, 56], [1, 14], [14, 57], [39, 52], [0, 14], [21, 57], [4, 14], [5, 14], [52, 56], [39, 40], [6, 56], [39, 47], [1, 5], [39, 46], [39, 45], [39, 61], [39, 44], [39, 54], [39, 58], [39, 55]], "1691_md-DpQ4HX7Q": [[1, 12], [5, 12], [12, 18], [12, 17], [12, 20], [12, 14], [3, 12], [12, 19], [4, 12], [10, 12], [8, 12], [7, 12], [9, 12], [11, 12], [1, 5], [1, 3], [1, 6], [2, 4], [15, 18], [15, 20]], "1717_lh_2_1duNgw": [[7, 8], [0, 8], [8, 11], [2, 8], [4, 8], [3, 8], [8, 52], [1, 8], [8, 55], [8, 54], [8, 16], [8, 53], [8, 58], [6, 8], [8, 9], [8, 57], [8, 17], [8, 32], [8, 31], [8, 56]], "1757_0jsMPnghnck": [[1, 6], [1, 9], [1, 8], [1, 7], [5, 9], [5, 7], [4, 9], [5, 8], [12, 14], [14, 15], [8, 9], [7, 8], [4, 19], [5, 18], [15, 16], [12, 13], [5, 19], [7, 9], [16, 17], [12, 16]], "179_mha1KKixPts": [[8, 10], [7, 8], [8, 13], [8, 11], [8, 9], [8, 12], [0, 8], [2, 8], [4, 8], [5, 8], [1, 8], [3, 8], [8, 15], [8, 14], [8, 17], [10, 13], [10, 12], [7, 13], [7, 11], [2, 10]], "1903_cFq3flHndS0": [[2, 8], [2, 15], [2, 6], [2, 11], [0, 7], [0, 4], [3, 8], [3, 16], [3, 15], [3, 18], [3, 6], [3, 14], [3, 11], [3, 12], [3, 17], [3, 13], [8, 15], [15, 16], [6, 8], [16, 18]], "1936_gvKlIkjfP0Q": [[4, 12], [4, 14], [4, 35], [4, 5], [3, 4], [4, 33], [4, 24], [4, 44], [4, 45], [2, 4], [4, 6], [4, 54], [4, 7], [4, 10], [4, 13], [4, 34], [4, 9], [4, 32], [4, 23], [4, 43]], "2143_6OMR3X7IcZ0": [[1, 11], [1, 8], [1, 6], [1, 3], [2, 14], [2, 10], [2, 11], [2, 8], [2, 7], [2, 3], [8, 14], [2, 15], [2, 5], [10, 11], [0, 11], [0, 8], [4, 14], [13, 14], [0, 6], [6, 11]], "2225_6acPX_00M9Q": [[2, 5], [3, 10], [3, 8], [6, 17], [0, 15], [0, 14], [0, 13], [0, 12], [0, 16], [7, 8], [1, 8], [13, 14], [12, 13], [1, 7], [8, 10], [12, 14], [4, 5], [2, 17], [5, 7], [7, 10]], "226_n7YpGfnTqoY": [[1, 2], [1, 11], [1, 6], [1, 8], [1, 10], [1, 19], [2, 11], [2, 6], [6, 11], [3, 11], [11, 17], [2, 18], [2, 19], [11, 16], [6, 10], [6, 16], [3, 18], [1, 7], [1, 14], [5, 15]], "241_oEkly9vzEGQ": [[2, 3], [2, 7], [2, 13], [0, 2], [2, 11], [2, 4], [1, 2], [2, 12], [2, 23], [2, 24], [2, 18], [2, 25], [2, 5], [6, 13], [4, 21], [4, 8], [15, 17], [4, 5], [21, 25], [9, 23]], "246_QcRqBBAiC4o": [[22, 45], [22, 25], [22, 24], [9, 45], [9, 25], [22, 23], [9, 24], [9, 47], [22, 27], [22, 43], [22, 35], [9, 27], [8, 22], [5, 22], [15, 22], [1, 22], [9, 35], [22, 29], [7, 22], [22, 30]], "254_-7d3nOFx1V8": [[0, 3], [3, 11], [0, 10], [0, 18], [0, 6], [3, 10], [0, 20], [0, 22], [0, 14], [0, 17], [3, 9], [3, 24], [0, 19], [0, 21], [0, 15], [0, 27], [0, 12], [0, 23], [3, 5], [3, 25]], "276_3HgBHBOnpbg": [[9, 10], [1, 10], [0, 10], [4, 10], [10, 15], [6, 10], [10, 36], [10, 24], [10, 23], [10, 14], [10, 35], [10, 27], [10, 30], [10, 28], [7, 10], [10, 13], [10, 11], [10, 39], [10, 38], [10, 37]], "279_qw5ySRNNfNM": [[1, 46], [2, 46], [0, 46], [28, 46], [46, 57], [46, 64], [46, 63], [39, 46], [3, 46], [46, 53], [5, 46], [46, 50], [4, 46], [46, 69], [46, 77], [46, 68], [26, 46], [46, 62], [46, 60], [46, 88]], "285_EP_blwEf2K8": [[8, 13], [0, 8], [8, 22], [8, 16], [4, 8], [8, 27], [6, 8], [8, 24], [8, 14], [8, 15], [8, 26], [8, 40], [8, 47], [8, 19], [13, 22], [13, 27], [13, 24], [13, 14], [13, 15], [10, 13]], "308_7WhzIsqPQW8": [[3, 5], [2, 3], [1, 3], [0, 3], [3, 37], [3, 39], [3, 38], [3, 30], [3, 33], [3, 34], [3, 41], [5, 27], [6, 27], [7, 27], [9, 27], [8, 27], [13, 27], [11, 27], [12, 27], [19, 27]], "339_j2gELsuQ3Cg": [[0, 3], [1, 4], [1, 6], [0, 4], [3, 8], [2, 3], [2, 4], [2, 6], [4, 6], [8, 9], [4, 5], [4, 7], [4, 13], [6, 7], [6, 13], [7, 13], [0, 1], [4, 8], [3, 4], [0, 5]], "359_4ZPKJtcNGZE": [[1, 12], [0, 11], [0, 22], [0, 19], [0, 21], [0, 20], [0, 18], [3, 4], [4, 11], [4, 12], [4, 7], [4, 8], [4, 6], [4, 24], [4, 23], [4, 19], [4, 15], [4, 9], [4, 20], [4, 18]], "365_JFqiSr9A-Go": [[6, 13], [11, 13], [12, 13], [13, 35], [13, 36], [13, 16], [13, 39], [9, 13], [13, 33], [13, 34], [13, 37], [13, 18], [4, 13], [2, 13], [0, 13], [13, 38], [6, 11], [13, 42], [1, 13], [13, 32]], "419_CykhOKWEAjo": [[6, 8], [8, 14], [5, 8], [3, 8], [0, 8], [8, 22], [8, 9], [8, 16], [7, 8], [8, 11], [8, 13], [8, 10], [8, 24], [8, 26], [8, 39], [8, 17], [8, 20], [8, 27], [8, 25], [8, 12]], "434_eOTP9yrATX8": [[3, 8], [3, 9], [3, 11], [3, 4], [3, 10], [3, 12], [0, 4], [1, 4], [6, 11], [4, 12], [3, 5], [3, 6], [0, 5], [0, 6], [7, 10]], "465_nbJ_SLWUDxk": [[0, 29], [29, 31], [1, 29], [29, 38], [18, 29], [29, 35], [3, 29], [23, 29], [25, 29], [5, 29], [27, 29], [29, 36], [30, 31], [16, 30], [14, 30], [30, 33], [30, 34], [0, 31], [1, 31], [1, 16]], "470_BiIqH60-A1M": [[5, 6], [1, 5], [5, 23], [5, 10], [5, 34], [5, 11], [5, 12], [4, 6], [4, 7], [4, 19], [4, 9], [4, 18], [4, 33], [4, 30], [4, 24], [4, 23], [4, 14], [4, 34], [4, 28], [4, 21]], "474_b-Qlvj48YQw": [[2, 23], [2, 27], [0, 23], [0, 27], [1, 61], [1, 60], [1, 34], [1, 54], [1, 57], [1, 63], [1, 59], [1, 58], [1, 62], [1, 38], [1, 53], [1, 55], [1, 56], [1, 48], [1, 50], [1, 51]], "520_JbMXRRGOEkk": [[3, 4], [3, 5], [4, 20], [4, 9], [4, 8], [4, 10], [4, 15], [4, 19], [4, 7], [4, 13], [4, 22], [4, 12], [4, 14], [4, 23], [4, 16], [3, 21], [3, 8], [3, 17], [3, 18], [3, 16]], "547_7E-Xian95Qk": [[3, 4], [0, 4], [4, 11], [2, 4], [4, 7], [0, 11], [1, 4], [3, 10], [0, 9], [0, 8], [2, 11], [0, 10], [0, 1], [5, 16], [5, 12], [5, 21], [2, 8], [5, 15], [5, 13], [5, 18]], "551_VHxQVmG1pOA": [[23, 24], [0, 8], [5, 23], [5, 24], [23, 34], [24, 34], [0, 40], [6, 23], [6, 24], [7, 23], [7, 24], [23, 27], [24, 27], [23, 28], [24, 28], [23, 30], [24, 30], [23, 31], [24, 31], [23, 29]], "562_OA61thiz9wU": [[1, 2], [2, 5], [2, 3], [1, 18], [1, 6], [2, 4], [1, 34], [1, 35], [2, 18], [2, 19], [0, 4], [4, 5], [0, 6], [5, 6], [0, 7], [5, 22], [5, 19], [5, 21], [4, 25], [4, 30]], "628_nQRJD435Fh4": [[0, 14], [0, 6], [0, 20], [6, 14], [14, 20], [6, 20], [14, 28], [0, 15], [1, 14], [14, 15], [5, 14], [14, 19], [14, 30], [0, 13], [0, 27], [13, 14], [14, 27], [0, 31], [14, 31], [6, 15]], "642_ljk5b80TmkE": [[0, 7], [0, 8], [0, 15], [0, 14], [0, 16], [1, 7], [21, 22], [3, 7], [7, 9], [7, 12], [6, 7], [4, 7], [5, 7], [5, 21], [1, 18], [8, 13], [1, 8], [2, 17], [1, 2], [5, 17]], "66_927BvkIZglw": [[5, 6], [1, 5], [6, 7], [1, 7], [0, 7], [7, 19], [3, 7], [1, 6], [4, 8], [4, 15], [4, 13], [4, 14], [4, 9], [4, 10], [4, 11], [4, 12], [2, 4], [4, 18], [3, 6], [0, 17]], "670_JboU-y2LdkU": [[0, 19], [1, 2], [3, 19], [4, 19], [3, 4], [2, 19], [5, 19], [6, 19], [4, 5], [11, 19], [4, 12], [5, 13], [9, 10], [10, 22], [10, 20], [10, 21], [9, 22], [9, 20], [8, 17], [20, 22]], "700_zkhPzSZcRtQ": [[2, 6], [2, 4], [2, 20], [2, 21], [2, 15], [0, 2], [2, 7], [2, 22], [2, 3], [2, 5], [1, 2], [2, 8], [2, 10], [2, 9], [2, 18], [2, 17], [6, 20], [6, 23], [6, 7], [6, 22]], "714_nooF6zlfzMI": [[1, 4], [2, 4], [1, 7], [6, 7], [2, 3], [2, 12], [2, 8], [2, 11], [3, 12], [3, 8], [8, 12], [4, 12], [4, 9], [3, 9], [1, 3], [5, 11], [2, 5], [0, 6], [1, 2], [1, 11]], "722__ajUvCkhVcI": [[3, 5], [3, 4], [3, 14], [2, 3], [3, 8], [3, 6], [3, 7], [3, 12], [3, 10], [3, 17], [4, 5], [5, 11], [5, 7], [1, 11], [4, 6], [0, 9], [1, 17], [0, 8], [0, 10], [11, 14]], "742_ctcOIuSzy-s": [[0, 7], [3, 4], [2, 4], [4, 5], [4, 9], [4, 11], [3, 5], [5, 6], [6, 9], [6, 8], [2, 10], [2, 7], [5, 9], [7, 9], [8, 10], [10, 12], [13, 14], [8, 14], [8, 13], [8, 12]], "744_1X6KvqPjk6I": [[2, 6], [6, 7], [6, 17], [6, 31], [1, 5], [5, 9], [4, 5], [1, 2], [2, 4], [2, 16], [2, 21], [2, 20], [2, 26], [2, 22], [2, 14], [7, 17], [4, 7], [7, 27], [7, 29], [7, 30]], "752_RWBJGgqDpwk": [[3, 25], [25, 30], [25, 28], [2, 25], [25, 50], [5, 25], [6, 25], [4, 25], [25, 35], [0, 25], [25, 38], [25, 59], [25, 40], [25, 46], [23, 25], [25, 58], [25, 51], [22, 25], [25, 57], [25, 33]], "754_TYUV8DYWe8k": [[3, 7], [1, 7], [2, 7], [6, 11], [6, 19], [6, 16], [6, 10], [6, 15], [6, 18], [6, 17], [6, 24], [6, 13], [6, 23], [6, 12], [5, 9], [3, 9], [1, 9], [2, 9], [5, 12], [4, 27]], "761_liWqb_am68c": [[1, 12], [3, 11], [3, 12], [3, 6], [4, 5], [4, 8], [4, 7], [2, 12], [0, 10], [11, 12], [8, 11], [7, 8], [10, 11], [10, 12], [4, 11], [1, 11], [1, 9], [3, 9], [1, 6], [5, 11]], "766_m1Vdl-EMY1E": [[0, 2], [0, 1], [0, 4], [0, 15], [0, 17], [0, 5], [0, 12], [0, 7], [0, 10], [0, 8], [0, 6], [0, 3], [0, 11], [1, 2], [2, 19], [2, 3], [1, 7], [1, 10], [1, 19], [1, 8]], "778_3PmDn84laac": [[1, 2], [2, 15], [2, 14], [2, 3], [2, 4], [0, 1], [2, 13], [2, 20], [2, 21], [2, 18], [2, 19], [2, 16], [2, 17], [1, 14], [1, 4], [1, 13], [1, 10], [4, 15], [15, 19], [0, 5]], "839_SS_1452uWvg": [[0, 3], [0, 68], [0, 33], [0, 60], [0, 65], [0, 41], [0, 40], [0, 66], [0, 49], [0, 70], [0, 50], [0, 52], [0, 53], [0, 57], [0, 58], [0, 69], [0, 56], [0, 55], [0, 54], [0, 51]], "856_coe8HkbRIk4": [[6, 12], [4, 6], [6, 8], [6, 17], [6, 16], [6, 7], [3, 6], [6, 13], [0, 6], [6, 24], [6, 19], [6, 18], [6, 20], [6, 11], [6, 23], [6, 14], [1, 6], [6, 9], [4, 12], [8, 12]], "888_BS3hab7EtAg": [[6, 8], [3, 6], [4, 6], [5, 6], [6, 19], [6, 21], [6, 22], [6, 25], [6, 23], [3, 8], [0, 8], [4, 8], [5, 8], [7, 8], [8, 10], [8, 19], [8, 14], [8, 16], [8, 17], [8, 18]], "914_f4HgijyAEYs": [[0, 1], [1, 6], [5, 6], [1, 7], [1, 3], [4, 16], [6, 7], [3, 6], [4, 5], [9, 16], [2, 5], [5, 9], [2, 6], [5, 8], [5, 10], [5, 11], [1, 18], [6, 8], [2, 3], [7, 9]], "950_94nfEhq6S5w": [[6, 10], [1, 6], [0, 6], [1, 10], [0, 10], [6, 9], [6, 27], [6, 13], [6, 12], [6, 25], [5, 6], [6, 15], [6, 23], [4, 6], [6, 14], [6, 24], [6, 7], [9, 10], [6, 43], [6, 19]], "973_ceJ5D6wluX0": [[2, 3], [0, 3], [3, 4], [1, 3], [3, 5], [3, 6], [0, 1], [0, 5]], "976_U19VojbI0h4": [[7, 59], [4, 7], [7, 22], [6, 59], [7, 33], [7, 8], [7, 34], [7, 32], [6, 16], [0, 6], [7, 24], [5, 7], [7, 23], [7, 64], [7, 49], [59, 61], [7, 47], [7, 63], [7, 11], [7, 10]], "987_g0mln-jiQTw": [[1, 2], [2, 8], [0, 1], [1, 8], [0, 8], [2, 7], [1, 7], [2, 6], [2, 4], [1, 10], [1, 3], [1, 6], [1, 4], [1, 5], [8, 9], [6, 7], [4, 7], [4, 6], [8, 10]]}')
out_path = "/kaggle/working/traser_pairs_sg.jsonl"
import copy
for vid in sorted(PAIRS):
    if not PAIRS[vid]: continue
    t0 = time.time()
    try:
        frames = sorted(glob.glob(f"/kaggle/input/**/{vid}/*.jpg", recursive=True))
        m = masks[vid]; H, W, nf, no = m["mask_height"], m["mask_width"], m["num_frames"], m["num_objects"]
        total = len(frames); idx = I._frame_indices(total, total / FPS)
        video = np.stack([np.array(Image.open(frames[i]).convert("RGB").resize((W, H))) for i in idx])
        fps_eff = len(idx) / (total / FPS)
        vp = copy.deepcopy(processor.video_processor); vp.max_pixels = I.VIDEO_MAX_FRAME_PIXELS; vp.min_pixels = I.VIDEO_MIN_FRAME_PIXELS
        vp.size["longest_edge"] = vp.max_pixels; vp.size["shortest_edge"] = vp.min_pixels
        pv = vp.preprocess(videos=video, return_tensors="pt")
        pixel_values_videos, video_grid_thw = pv["pixel_values_videos"], pv["video_grid_thw"][0]
        second_per_grid_ts = [processor.video_processor.temporal_patch_size / fps_eff]
        t_grid, h_patch, w_patch = (int(x) for x in video_grid_thw)
        n_video_tokens = t_grid * h_patch * w_patch // I.SPATIAL_MERGE_SIZE ** 2
        mask_data = [[{"size": [H, W], "counts": m["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
        input_ids = I.build_prompt_ids(tokenizer, I.PROMPTS["scene_graph"], n_video_tokens).cuda()
        gpw = int(4 / (I.TEMPORAL_PATCH_SIZE / I.BASE_INTERVAL))
        window_ids = tokenizer([f"<{w*4} - {(w+1)*4} sec>" for w in range(math.ceil(t_grid / gpw))], add_special_tokens=False)["input_ids"]
        pvv = pixel_values_videos.cuda().to(torch.bfloat16)
    except Exception as e:
        import traceback; print(vid, "VIDEO ERROR", traceback.format_exc()[-300:], flush=True); continue
    for a, b in PAIRS[vid]:
        rec = {"video_id": vid, "pair": [a, b]}
        try:
            obj_masks, kept = I.build_obj_masks(mask_data, sorted([a, b]), idx.tolist(), h_patch * I.PATCH_SIZE, w_patch * I.PATCH_SIZE)
            rec["kept_obj_ids"] = kept
            _, per_obj_idx, _ = select_tokens(obj_masks=obj_masks, grid_thw=(t_grid, h_patch, w_patch), patch_size=I.PATCH_SIZE,
                spatial_merge_size=I.SPATIAL_MERGE_SIZE, temporal_patch_size=I.TEMPORAL_PATCH_SIZE, coverage_thresh=0.5, time_reduce="max", device="cpu")
            label_ids = tokenizer([I.OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per_obj_idx))], add_special_tokens=False)["input_ids"]
            with torch.no_grad():
                embeds, pos, am, rope, _, _, _ = rearrange_token(model=model, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    pixel_values_videos=pvv, video_grid_thw=video_grid_thw[None].cuda(), image_grid_thw=None, pixel_values=None,
                    second_per_grid_ts=torch.tensor(second_per_grid_ts), obj_token_indices_per_sample=[[i.cuda() for i in per_obj_idx]],
                    obj_traj_start_id=int(model.config.obj_traj_start_id), obj_traj_end_id=int(model.config.obj_traj_end_id),
                    text_token_ids_per_sample=[[torch.tensor(x) for x in label_ids]], timestamp_token_ids_per_batch=[[torch.tensor(x) for x in window_ids]],
                    grids_per_temporal_window_per_batch=[gpw])
                gen = model.generate(inputs_embeds=embeds, position_ids=pos, attention_mask=am.long(), rope_deltas=rope, max_new_tokens=1024, do_sample=False)
            rec["text"] = tokenizer.decode(gen[0], skip_special_tokens=True)
        except Exception as e:
            import traceback; rec["error"] = traceback.format_exc()[-800:]
        with open(out_path, "a") as f: f.write(json.dumps(rec) + chr(10))
    print(vid, len(PAIRS[vid]), "pairs", round(time.time() - t0, 1), "s", flush=True)
print("PAIRS SG DONE", flush=True)

out_path = "/kaggle/working/traser_svg2test_vipseg_allobj.jsonl"; done = set()
for vid in sorted(masks):
    if masks[vid]["num_objects"] <= 40 and vid != "856_coe8HkbRIk4": continue
    t0 = time.time(); rec = {"video_id": vid}
    try:
        frames = sorted(glob.glob(f"/kaggle/input/**/{vid}/*.jpg", recursive=True))
        m = masks[vid]; H, W, nf, no = m["mask_height"], m["mask_width"], m["num_frames"], m["num_objects"]
        rec.update(n_jpg=len(frames), mask_frames=nf, num_objects=no)
        total = len(frames); idx = I._frame_indices(total, total / FPS)
        video = np.stack([np.array(Image.open(frames[i]).convert("RGB").resize((W, H))) for i in idx])
        fps_eff = len(idx) / (total / FPS)
        import copy
        vp = copy.deepcopy(processor.video_processor); vp.max_pixels = I.VIDEO_MAX_FRAME_PIXELS; vp.min_pixels = I.VIDEO_MIN_FRAME_PIXELS
        vp.size["longest_edge"] = vp.max_pixels; vp.size["shortest_edge"] = vp.min_pixels
        pv = vp.preprocess(videos=video, return_tensors="pt")
        pixel_values_videos, video_grid_thw = pv["pixel_values_videos"], pv["video_grid_thw"][0]
        second_per_grid_ts = [processor.video_processor.temporal_patch_size / fps_eff]
        t_grid, h_patch, w_patch = (int(x) for x in video_grid_thw)
        n_video_tokens = t_grid * h_patch * w_patch // I.SPATIAL_MERGE_SIZE ** 2
        mask_data = [[{"size": [H, W], "counts": m["counts"][f * no + o]} for o in range(no)] for f in range(0, nf, 3)][:total]
        rec["mask_frames_used"] = len(mask_data)
        obj_ids = list(range(no))[:100]
        obj_masks, kept = I.build_obj_masks(mask_data, obj_ids, idx.tolist(), h_patch * I.PATCH_SIZE, w_patch * I.PATCH_SIZE)
        rec.update(kept_obj_ids=kept, sampled=len(idx), n_video_tokens=n_video_tokens)
        _, per_obj_idx, _ = select_tokens(obj_masks=obj_masks, grid_thw=(t_grid, h_patch, w_patch), patch_size=I.PATCH_SIZE,
            spatial_merge_size=I.SPATIAL_MERGE_SIZE, temporal_patch_size=I.TEMPORAL_PATCH_SIZE, coverage_thresh=0.5, time_reduce="max", device="cpu")
        input_ids = I.build_prompt_ids(tokenizer, I.PROMPTS["scene_graph"], n_video_tokens).cuda()
        label_ids = tokenizer([I.OBJECT_LABEL_TEMPLATE.format(i=k + 1) for k in range(len(per_obj_idx))], add_special_tokens=False)["input_ids"]
        gpw = int(4 / (I.TEMPORAL_PATCH_SIZE / I.BASE_INTERVAL))
        windows = [f"<{w*4} - {(w+1)*4} sec>" for w in range(math.ceil(t_grid / gpw))]
        window_ids = tokenizer(windows, add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            embeds, pos, am, rope, _, _, _ = rearrange_token(model=model, input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                pixel_values_videos=pixel_values_videos.cuda().to(torch.bfloat16), video_grid_thw=video_grid_thw[None].cuda(),
                image_grid_thw=None, pixel_values=None, second_per_grid_ts=torch.tensor(second_per_grid_ts),
                obj_token_indices_per_sample=[[i.cuda() for i in per_obj_idx]], obj_traj_start_id=int(model.config.obj_traj_start_id),
                obj_traj_end_id=int(model.config.obj_traj_end_id), text_token_ids_per_sample=[[torch.tensor(x) for x in label_ids]],
                timestamp_token_ids_per_batch=[[torch.tensor(x) for x in window_ids]], grids_per_temporal_window_per_batch=[gpw])
            rec["prefill_len"] = int(embeds.shape[1])
            gen = model.generate(inputs_embeds=embeds, position_ids=pos, attention_mask=am.long(), rope_deltas=rope, max_new_tokens=8192, do_sample=False)
        rec["n_gen_tokens"] = int(gen.shape[1]); rec["text"] = tokenizer.decode(gen[0], skip_special_tokens=True)
    except Exception as e:
        import traceback; rec["error"] = traceback.format_exc()[-2000:]
    rec["sec"] = round(time.time() - t0, 1)
    print(vid, {k: rec.get(k) for k in ["n_jpg","mask_frames","num_objects","sampled","prefill_len","n_gen_tokens","sec"]}, rec.get("error","")[-300:], flush=True)
    with open(out_path, "a") as f: f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)

# ---------------- judge stage (same session, consolidate GPU use) ----------------
del model; import gc; gc.collect(); torch.cuda.empty_cache()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "/kaggle/src")
import pyarrow.parquet as pq
gt = {r["video_id"]: r for r in pq.read_table(find("/kaggle/input/**/gt_vipseg.parquet")).to_pylist()}
outs = [json.loads(l) for l in open(find("/kaggle/input/**/traser_svg2test_vipseg.jsonl"))] + [json.loads(l) for l in open(out_path)]
outs += [json.loads(l) for l in open(find("/kaggle/input/**/traser_pairs.jsonl"))]
outs += [json.loads(l) for l in open("/kaggle/working/traser_pairs_sg.jsonl")]
exec(compile('# Lenient label judge (stand-in for SVG2\'s GPT-4o-mini aligner): classify (GT label, predicted label) pairs.\nimport glob, json, os, re, sys, torch\nfrom transformers import AutoTokenizer, AutoModelForCausalLM\nCATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]\ndef build_pairs(outs, gt):\n    pairs = set()\n    for o in outs:\n        if "error" in o or o["video_id"] not in gt: continue\n        kept = o["kept_obj_ids"]; g = gt[o["video_id"]]\n        gobj = {int(k.split("_")[1]): str(v) for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}\n        try: pred = json.loads(o["text"])\n        except Exception: continue\n        for ob in pred.get("objects", []):\n            for k, v in ob.items():\n                m = re.match(r"object[ _](\\d+)", k)\n                if m and int(m.group(1)) - 1 < len(kept):\n                    gl = gobj.get(kept[int(m.group(1)) - 1])\n                    if gl: pairs.add(("object", gl.lower().strip(), str(v).lower().strip()))\n        P = {}\n        for r in pred.get("relationships", []):\n            try: P.setdefault((kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1), []).append(str(r[1]))\n            except Exception: pass\n        for s, p, ob, *_ in json.loads(g["relationships"]):\n            for q in P.get((s, ob), []):\n                if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))\n    return sorted(pairs)\ndef run_judge(pairs, model_dir, out_path, bs=64):\n    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"\n    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()\n    res = {}\n    for i in range(0, len(pairs), bs):\n        batch = pairs[i:i + bs]; prompts = []\n        for kind, g, p in batch:\n            msg = (f"You are a strict lexical matcher for video scene graph evaluation. Compare a ground-truth {kind} label with a predicted {kind} label "\n                   f"and classify their relationship into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\\n"\n                   f"Ground truth: \\"{g}\\"\\nPrediction: \\"{p}\\"\\nAnswer with the category name only.")\n            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False))\n        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")\n        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=8, do_sample=False)\n        for (kind, g, p), t in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):\n            t = t.strip().lower(); res[f"{kind}|||{g}|||{p}"] = next((c for c in CATS if t.startswith(c.split("/")[0])), "mismatch:" + t[:20])\n        print("judged", min(i + bs, len(pairs)), "/", len(pairs), flush=True)\n    json.dump(res, open(out_path, "w"))\n', 'judge_inline', 'exec'))
pairs = build_pairs(outs, gt); print("pairs to judge", len(pairs), flush=True)
run_judge(pairs, os.path.dirname(find("/kaggle/input/**/Qwen3*8*/**/config.json") if glob.glob("/kaggle/input/**/Qwen3*8*/**/config.json", recursive=True) else find("/kaggle/input/**/qwen-3/**/config.json")), "/kaggle/working/judge_cache.json")
print("JUDGE DONE", flush=True)
