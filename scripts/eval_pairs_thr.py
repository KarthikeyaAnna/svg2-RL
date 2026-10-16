"""A0 full graph + decoded learned pairs selected by ranker probability >= tau (probabilities stored in the pair-list JSON).
Usage: eval_pairs_thr.py PAIRS_DIR PAIRLIST_JSON tau [tau ...]"""
import sys, os, json, subprocess, glob
W = "/home/karthik/svg2-killtest"; SC = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/score"; os.makedirs(SC, exist_ok=True)
PY = [f"{W}/venv/bin/python", "-I"]; ENV = dict(os.environ, TIOU="perspan")
GT = {"svg2": f"{W}/data/gt_vipseg.parquet", "vidor": f"{W}/data/vidor/prep/gt_vidor.parquet", "pvsg": f"{W}/data/pvsg/gt_pvsg.parquet"}
A0 = {"svg2": f"{W}/out/e1_a0/svg2_full.jsonl", "vidor": f"{W}/out/e1_a0/vidor_full.jsonl", "pvsg": f"{W}/out/e1_a0p/pvsg_full.jsonl"}
BASEJ = {"svg2": f"{W}/out/judge_all.json", "vidor": f"{W}/out/vidor/judge_cache.json", "pvsg": f"{W}/out/pvsg/judge_cache.json"}
pdir, plist, taus = sys.argv[1], json.load(open(sys.argv[2])), [float(x) for x in sys.argv[3:]]
FULL = {bn: (os.environ.get("FULL_DIR") + f"/{bn}_full.jsonl" if os.environ.get("FULL_DIR") else A0[bn]) for bn in A0}   # base graph to add pairs to
tag = os.path.basename(pdir.rstrip("/")) + ("_on_" + os.path.basename(os.environ["FULL_DIR"].rstrip("/")) if os.environ.get("FULL_DIR") else "")
def run(*a): return subprocess.run(PY + list(a), capture_output=True, text=True, env=ENV).stdout
for bn in ("svg2", "vidor", "pvsg"):
    j = json.load(open(BASEJ[bn]))
    for f in glob.glob(f"{W}/out/e1_a0*/judge_*.json") + glob.glob(f"{pdir}/judge_*.json") + (glob.glob(os.environ["FULL_DIR"] + "/judge_*.json") if os.environ.get("FULL_DIR") else []): j.update(json.load(open(f)))
    jp = f"{SC}/judge_{tag}_{bn}.json"; json.dump(j, open(jp, "w"))
    base = f"{SC}/a0_{bn}.dd.jsonl"; run(f"{W}/scripts/dedup.py", A0[bn], base)
    prob = {(v, int(e[0]), int(e[1])): e[2] for v, L in plist[bn].items() for e in L}
    dec = sorted((json.loads(l) for l in open(f"{pdir}/{bn}_pairs_learned.jsonl")), key=lambda o: (o["video_id"], o["rank"]))
    nv = len({o["video_id"] for o in dec})
    for tau in taus:
        keep = [o for o in dec if prob.get((o["video_id"], *sorted(o["pair"])), 0) >= tau]
        pf = f"{SC}/{tag}_{bn}_t{tau}.pairs.jsonl"; open(pf, "w").write("".join(json.dumps(o) + "\n" for o in keep))
        mf = f"{SC}/{tag}_{bn}_t{tau}.jsonl"; run(f"{W}/scripts/merge_pairs.py", FULL[bn], pf, "40", mf)
        dd = f"{SC}/{tag}_{bn}_t{tau}.dd.jsonl"; run(f"{W}/scripts/dedup.py", mf, dd)
        for clean in ([False, True] if bn == "pvsg" else [False]):
            out = run(f"{W}/scripts/bootstrap.py", base, dd, GT[bn], jp, *([f"{W}/data/pvsg/clean_val_ids.json"] if clean else [])).strip().splitlines()[-1]
            print(f"{bn:5s}{' clean' if clean else '      '} tau={tau}: pairs/video {len(keep)/max(1,nv):4.1f} | {out.split(' | videos')[0]}")
