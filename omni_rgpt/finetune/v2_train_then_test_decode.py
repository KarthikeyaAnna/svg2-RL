"""QS v6 = QS v5 minus the 11 bad SVG2 PVD label files (0 spatial relations; RESEARCH_LOG §106-110), one A6000, ONE job:
  0) train: qs_v6_ds.py, init v4c2, same sources/recipe/order seed as v5, bad-file videos removed after the dev split,
     SKIP_DEV=1 (no dev decode, as v5). Resumes automatically from out/qs_v6/train_state.pt if restarted.
  1) smoke decode (4 SVG2test videos, 2 workers) -> out/qe_v6_par_smoke
  2) full test decode (svg2, sav, vidor, pvsg), settings identical to v5's (sdpa, 1024 / 256 tokens) -> out/qe_v6_par"""
import json, os, subprocess, sys, time
B = "/media/nas_mount/karthikeya/svg2"; P = sys.executable; N = 2
SOURCES = "vidor,vidvrd,fetch3-pvd,fetch4-sav-1,fetch4-sav-2,fetch4-sav-3,fetch4-sav-4"
# ---- 0) training
def train_done():
    import torch
    p = f"{B}/out/qs_v6/train_state.pt"
    if not os.path.exists(p): return False
    st = torch.load(p, map_location="cpu", weights_only=False); return st["items_done"] >= st["n_items"]
if not train_done():
    print("TRAIN start", time.ctime(), flush=True); t = time.time()
    r = subprocess.run([P, f"{B}/qs_v6_ds.py", f"SOURCES={SOURCES}", "SKIP_DEV=1"], env=dict(os.environ, PYTHONUNBUFFERED="1"))
    print("TRAIN rc", r.returncode, "h", round((time.time() - t) / 3600, 2), flush=True)
    if r.returncode != 0 or not train_done(): print("TRAIN NOT COMPLETE -> stop (GPU released)", flush=True); sys.exit(3)
else: print("TRAIN already complete -> decode", flush=True)
BASE = [P, f"{B}/eval_v5_ds_shard.py", f"SOURCES={SOURCES}", "TAG=v6", f"LORA={B}/out/qs_v6/lora_QS.pt",
        f"MARKS={B}/out/qs_v6/marks_QS.pt", "SKIP_DEV=1", "MAX_NEW_GRAPH=1024", "MAX_NEW_PAIR=256", "ATTN=sdpa"]
BENCHES = ["svg2", "sav", "vidor", "pvsg"]


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def valid_rate(R):
    ok = 0
    for r in R:
        if "error" in r: continue
        try: json.loads(r["text"]); ok += 1
        except Exception: pass
    return ok / len(R) if R else 0.0


def run_workers(out, extra):
    procs = []
    for i in range(N):
        env = dict(os.environ, SHARD=f"{i}/{N}", PYTHONUNBUFFERED="1")
        procs.append(subprocess.Popen(BASE + extra + [f"OUT={out}/s{i}"], env=env))
        time.sleep(20)                                                   # stagger model load
    return [p.wait() for p in procs]


def merge(out, benches):
    for bn in benches:
        for kind in ("graph", "pairs"):
            R = [r for i in range(N) for r in rows(f"{out}/s{i}/{bn}_{kind}_v6.jsonl")]
            R.sort(key=lambda r: str(r["video_id"]))                     # stable: keeps each video's pair order
            with open(f"{out}/{bn}_{kind}_v6.jsonl", "w") as f:
                for r in R: f.write(json.dumps(r) + "\n")
            print("CHECK", bn, kind, "rows", len(R), "videos", len({r["video_id"] for r in R}),
                  "errors", sum("error" in r for r in R), "valid", round(valid_rate(R), 3), flush=True)


# ---- 1) smoke
SM = f"{B}/out/qe_v6_par_smoke"; print("SMOKE start", time.ctime(), flush=True); t = time.time()
rc = run_workers(SM, ["BENCH=svg2", "LIMIT=4"]); merge(SM, ["svg2"])
G, PR = rows(f"{SM}/svg2_graph_v6.jsonl"), rows(f"{SM}/svg2_pairs_v6.jsonl")
old = {r["video_id"]: r["text"] for r in rows(f"{B}/out/qe_v5_par_smoke/svg2_graph_v5.jsonl")}
same = [old[r["video_id"]] == r["text"] for r in G if r["video_id"] in old]
print("SMOKE rc", rc, "min", round((time.time() - t) / 60, 1), "| graphs", len(G), "valid", round(valid_rate(G), 3),
      "| pairs", len(PR), "valid", round(valid_rate(PR), 3), "| graph text identical to v5 smoke (expect few):", sum(same), "of", len(same), flush=True)
if any(rc) or len(G) < 4 or len(PR) < 5 or valid_rate(G) < 0.5 or valid_rate(PR) < 0.9:
    print("SMOKE FAILED -> no full decode (GPU released)", flush=True); sys.exit(4)

# ---- 2) full decode
OUT = f"{B}/out/qe_v6_par"; print("RUN full test decode, 2 workers", time.ctime(), flush=True); t = time.time()
rc = run_workers(OUT, ["BENCH=" + ",".join(BENCHES)])
print("END full rc", rc, "min", round((time.time() - t) / 60, 1), flush=True)
merge(OUT, BENCHES)
sys.exit(1 if any(rc) else 0)
