"""Video-id leakage check between training pools and the three test sets.
VidOR ids are compared by the numeric suffix (VidOR folder prefix '0001_' is dropped in SVG2's vidor parquet).
Usage: check_leakage.py <pvsg.json> <vidor validation-annotation.zip> <svg2 dir with vidor/vipseg/sav_train/sav_test/vip_test parquets> [pool_ids.txt ...] [dev:dev_ids.txt ...]
Prints every intersection; exits 1 if any EXTRA pool (one id per line, e.g. a self-training pool) intersects a test set."""
import sys, json, zipfile, pyarrow.parquet as pq
pv, vz, S = sys.argv[1], sys.argv[2], sys.argv[3].rstrip("/") + "/"
import re
def sfx(s):  # normalise only VidOR-style ids ("0001_4164158586" / "0001/4164158586" -> "4164158586"); others unchanged
    s = str(s).replace("/", "_"); return s.split("_")[1] if re.fullmatch(r"\d{4}_\d+", s) else s
d = json.load(open(pv))
pv_val = {sfx(v) for s in d["split"] for v in d["split"][s]["val"]}            # PVSG-val (all sources; non-VidOR ids pass through)
pv_tr = {sfx(v) for s in d["split"] for v in d["split"][s]["train"]}
vor_val = {n.split("/")[-1][:-5] for n in zipfile.ZipFile(vz).namelist() if n.endswith(".json")}
ids = lambda f: {sfx(x) for x in pq.read_table(S + f, columns=["video_id"]).column("video_id").to_pylist()}
TEST = {"PVSG-val": pv_val, "VidOR-val": vor_val, "SVG2test-VIPSeg": ids("vip_test.parquet"), "SVG2test-SAV": ids("sav_test.parquet")}
POOL = {"TraSeR-vidor-train": ids("vidor.parquet"), "TraSeR-vipseg-train": ids("vipseg.parquet"), "TraSeR-sav-train": ids("sav_train.parquet"), "PVSG-train": pv_tr}
for a in sys.argv[4:]:  # "dev:file" = held-out dev ids, treated like a test set; plain file = a training pool
    ids_ = {sfx(l.strip()) for l in open(a.split(":", 1)[-1]) if l.strip()}
    if a.startswith("dev:"): TEST["DEV:" + a[4:]] = ids_
    else: POOL["EXTRA:" + a] = ids_
bad = 0
for pn, P in POOL.items():
    for tn, T in TEST.items():
        n = len(P & T); print(f"{pn:28s} ({len(P):6d}) ∩ {tn:16s} ({len(T):4d}) = {n}")
        if n and pn.startswith("EXTRA"): bad = 1
# exclusion list for any pool drawn from VidOR train: PVSG-val ∪ VidOR-val ids
json.dump(sorted(pv_val | vor_val), open(S + "exclude_ids.json", "w")); print("wrote", S + "exclude_ids.json")
sys.exit(bad)
