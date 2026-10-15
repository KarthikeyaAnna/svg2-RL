"""Object accuracy with judge cache: TraSeR label vs GT label for the same mask id."""
import sys, json, re, pyarrow.parquet as pq
outs = [json.loads(l) for l in open(sys.argv[1])]; gt = {r["video_id"]: r for r in pq.read_table(sys.argv[2]).to_pylist()}
J = json.load(open(sys.argv[3])); n = strict = len_ = miss_judge = 0
for o in outs:
    if "error" in o: continue
    g = gt[o["video_id"]]; kept = o["kept_obj_ids"]
    gobj = {int(k.split("_")[1]): str(v).lower().strip() for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}
    try: pred = json.loads(o["text"]).get("objects", [])
    except Exception: continue
    plab = {}
    for ob in pred:
        for k, v in ob.items():
            m = re.match(r"object[ _](\d+)", k)
            if m and int(m.group(1)) - 1 < len(kept): plab[kept[int(m.group(1)) - 1]] = str(v).lower().strip()
    for oid in kept:
        if oid not in gobj: continue
        n += 1; p = plab.get(oid)
        if p is None: continue
        p = re.sub(r"\s*\(uncertain\)\s*$", "", p).strip()
        if p == gobj[oid]: strict += 1; len_ += 1; continue
        c = J.get(f"object|||{gobj[oid]}|||{p}")
        if c is None: miss_judge += 1
        elif not c.startswith("mismatch"): len_ += 1
print(f"objects {n} | strict acc {100*strict/n:.1f}% | lenient (judge) acc {100*len_/n:.1f}% | unjudged {miss_judge}  (paper TraSeR SVG2test object: 79.0 lenient, 28.8 strict)")
