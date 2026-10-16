"""Score E1-style runs against the fast-engine A0 baseline on all three benchmarks (dedup, per-span tIoU, Qwen3-8B judge caches merged,
paired per-video bootstrap) + output-style diagnostics. Usage:
  score_runs.py full RUN_DIR [RUN_DIR ...]     # full-graph outputs {bench}_full.jsonl of training arms
  score_runs.py pairs P1_DIR                   # A0 full + top-K learned pairs vs contact-ranked (A0 pairs_sal) vs random (A0 pairs_rnd)"""
import sys, os, json, glob, subprocess, re, collections
import numpy as np
W = "/home/karthik/svg2-killtest"; S = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/score"
os.makedirs(S, exist_ok=True); sys.path.insert(0, f"{W}/scripts"); from relparse import parse_rels
PY = [f"{W}/venv/bin/python", "-I"]; ENV = dict(os.environ, TIOU="perspan")
GT = {"svg2": f"{W}/data/gt_vipseg.parquet", "vidor": f"{W}/data/vidor/prep/gt_vidor.parquet", "pvsg": f"{W}/data/pvsg/gt_pvsg.parquet"}
A0 = {"svg2": f"{W}/out/e1_a0", "vidor": f"{W}/out/e1_a0", "pvsg": f"{W}/out/e1_a0p"}
BASEJ = {"svg2": f"{W}/out/judge_all.json", "vidor": f"{W}/out/vidor/judge_cache.json", "pvsg": f"{W}/out/pvsg/judge_cache.json"}
CLEAN = f"{W}/data/pvsg/clean_val_ids.json"

def run(*a): return subprocess.run(PY + list(a), capture_output=True, text=True, env=ENV).stdout
def judge(bn, extra_dirs):
    j = json.load(open(BASEJ[bn]))
    for d in [A0[bn]] + list(extra_dirs):
        for f in glob.glob(f"{d}/judge_*.json"): j.update(json.load(open(f)))
    p = f"{S}/judge_{bn}.json"; json.dump(j, open(p, "w")); return p
def dd(src, tag):
    dst = f"{S}/{tag}.dd.jsonl"; run(f"{W}/scripts/dedup.py", src, dst); return dst
def boot(a, b, bn, j, clean=False):
    out = run(f"{W}/scripts/bootstrap.py", a, b, GT[bn], j, *([CLEAN] if clean else [])).strip().splitlines()
    return out[-1] if out else "(no output)"
def style(path):
    n, bid, nt, dup, stops = [], [], [], [], collections.Counter()
    for l in open(path):
        o = json.loads(l); stops[o.get("stop", "?")] += 1
        R = [(str(r[0]), str(r[1]).lower().strip(), str(r[2])) for r in parse_rels(o["text"]) if isinstance(r, (list, tuple)) and len(r) >= 3]; n.append(len(R))
        pr = collections.defaultdict(set)
        for s, p, ob in R: pr[frozenset((s, ob))].add((s, ob))
        bid.append(np.mean([len(v) == 2 for k, v in pr.items() if len(k) == 2]) if pr else 0)
        nt.append(np.mean([p in ("next to", "next_to") for _, p, _ in R]) if R else 0)
        dup.append(1 - len(set(R)) / max(1, len(R)))
    return f"rels/video {np.mean(n):5.1f} | both-dir {np.mean(bid):.2f} | 'next to' {np.mean(nt):.2f} | dup {np.mean(dup):.3f} | stops {dict(stops)}"

def full(dirs):
    for bn in ("svg2", "vidor", "pvsg"):
        j = judge(bn, dirs); base = dd(f"{A0[bn]}/{bn}_full.jsonl", f"a0_{bn}")
        print(f"\n===== {bn}   A0: {style(f'{A0[bn]}/{bn}_full.jsonl')}")
        for d in dirs:
            f = f"{d}/{bn}_full.jsonl"
            if not os.path.exists(f): print(f"  {os.path.basename(d)}: missing"); continue
            b = dd(f, f"{os.path.basename(d)}_{bn}")
            print(f"  {os.path.basename(d)}: {style(f)}\n    all  : {boot(base, b, bn, j)}")
            if bn == "pvsg": print(f"    clean: {boot(base, b, bn, j, clean=True)}")

def pairs(p1):
    for bn in ("svg2", "vidor", "pvsg"):
        j = judge(bn, [p1]); base_full = f"{A0[bn]}/{bn}_full.jsonl"; base = dd(base_full, f"a0_{bn}")
        learned = [json.loads(l) for l in open(f"{p1}/{bn}_pairs_learned.jsonl")]
        learned.sort(key=lambda o: (o["video_id"], o["rank"]))
        lp = f"{S}/{bn}_learned_ordered.jsonl"; open(lp, "w").write("".join(json.dumps(o) + "\n" for o in learned))
        print(f"\n===== {bn}  (base = A0 full graph; each row adds top-K pairs from one ranking)")
        for K in (10, 20, 40):
            for name, pf in (("learned", lp), ("contact", f"{A0[bn]}/{bn}_pairs_sal.jsonl"), ("random", f"{A0[bn]}/{bn}_pairs_rnd.jsonl")):
                if name == "random" and K > 20: continue
                m = f"{S}/{bn}_{name}{K}.jsonl"; run(f"{W}/scripts/merge_pairs.py", base_full, pf, str(K), m)
                b = dd(m, f"{bn}_{name}{K}")
                print(f"  +{K:2d} {name:8s}: {boot(base, b, bn, j)}")
                if bn == "pvsg": print(f"          clean   : {boot(base, b, bn, j, clean=True)}")

if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    full(args) if mode == "full" else pairs(args[0])
