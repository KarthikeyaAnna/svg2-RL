"""Uniform grid: every post-processing system on every benchmark, same scoring (dedup, per-span tIoU, judge) + paired bootstrap vs TraSeR.
Writes a JSON with P/R/F1 and significance per (benchmark, system)."""
import subprocess, json, os, re, sys
W = "/home/karthik/svg2-killtest"; S = "/tmp/claude-1000/-home-karthik/bb3aaa05-9eee-4a51-ad81-9ac5dda25ab2/scratchpad/grid"; os.makedirs(S, exist_ok=True)
PY = [f"{W}/venv/bin/python", "-I"]; ENV = dict(os.environ, TIOU="perspan")
B = {
 "svg2": dict(base=f"{W}/out/traser_svg2test_vipseg.jsonl", allobj=f"{W}/out/allobj_combined.jsonl", filt=f"{W}/out/allobj_filt.jsonl",
              sal=f"{W}/out/pairs_sg_40_ordered.jsonl", rnd=f"{W}/out/rand/traser_pairs_random.jsonl", gt=f"{W}/data/gt_vipseg.parquet", judge=f"{W}/out/judge_all.json"),
 "pvsg": dict(base=f"{W}/out/pvsg/full_cap40.jsonl", allobj=f"{W}/out/pvsg/allobj.jsonl", filt=f"{W}/out/pvsg/allobj_filt.jsonl", sal=f"{W}/out/pvsg/pairs_sal.jsonl",
              rnd=f"{W}/out/pvsg/pairs_rnd.jsonl", gt=f"{W}/data/pvsg/gt_pvsg.parquet", judge=f"{W}/out/pvsg/judge_cache.json"),
 "vidor": dict(base=f"{W}/out/vidor/full_cap40.jsonl", allobj=f"{W}/out/vidor/allobj.jsonl", filt=f"{W}/out/vidor/allobj_filt.jsonl", sal=f"{W}/out/vidor/pairs_sal.jsonl",
               rnd=f"{W}/out/vidor/pairs_rnd.jsonl", gt=f"{W}/data/vidor/prep/gt_vidor.parquet", judge=f"{W}/out/vidor/judge_cache.json"),
}
def run(*a): return subprocess.run(PY + list(a), capture_output=True, text=True, env=ENV).stdout
def merge(base, pairs, n, dst): run(f"{W}/scripts/merge_pairs.py", base, pairs, str(n), dst); return dst
def inv(src, dst): run(f"{W}/scripts/add_inverse.py", src, dst); return dst
def dd(src, dst): run(f"{W}/scripts/dedup.py", src, dst); return dst
res = {}
for bn, b in B.items():
    d = f"{S}/{bn}"; os.makedirs(d, exist_ok=True)
    systems = {
        "TraSeR": b["base"], "Contact filter": b["filt"],
        "+20 random": merge(b["allobj"], b["rnd"], 20, f"{d}/rnd20.jsonl"),
        "+20 salient": merge(b["allobj"], b["sal"], 20, f"{d}/sal20.jsonl"),
        "+40 salient": merge(b["allobj"], b["sal"], 40, f"{d}/sal40.jsonl"),
        "Filter + 20 salient": merge(b["filt"], b["sal"], 20, f"{d}/fsal20.jsonl"),
        "Both directions": inv(b["base"], f"{d}/inv.jsonl"),
    }
    systems["+20 salient + both directions"] = inv(systems["+20 salient"], f"{d}/sal20_inv.jsonl")
    base_dd = dd(b["base"], f"{d}/base_dd.jsonl")
    for clean in ([False, True] if bn == "pvsg" else [False]):
        key = bn + ("_clean" if clean else ""); res[key] = {}
        for name, f in systems.items():
            fd = dd(f, f"{d}/{re.sub('[^a-z0-9]+', '_', name.lower())}_dd.jsonl")
            args = [f"{W}/scripts/bootstrap.py", base_dd, fd, b["gt"], b["judge"]] + ([f"{W}/data/pvsg/clean_val_ids.json"] if clean else [])
            out = run(*args).strip().splitlines()[-1] if name != "TraSeR" else ""
            if name == "TraSeR":
                o = run(*[f"{W}/scripts/bootstrap.py", base_dd, base_dd, b["gt"], b["judge"]] + ([f"{W}/data/pvsg/clean_val_ids.json"] if clean else [])).strip().splitlines()[-1]
                out = o
            m = {k: re.search(rf"{k}:\s+([\d.]+)\s+->\s+([\d.]+)\s+delta\s+([+-][\d.]+)\s+95% CI \[\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\]\s*(\*?)", out) for k in ("P", "R", "F")}
            res[key][name] = {k: dict(v=float(x.group(2)), d=float(x.group(3)), lo=float(x.group(4)), hi=float(x.group(5)), sig=bool(x.group(6))) for k, x in m.items() if x}
            print(key, name, {k: (v["v"], v["d"], v["sig"]) for k, v in res[key][name].items()}, flush=True)
json.dump(res, open(f"{W}/out/grid_eval.json", "w"), indent=1)
