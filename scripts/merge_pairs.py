import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Merge pair-conditioned TraSeR relations into baseline outputs (GT id space). Usage: base pairs_out topN dst"""
import sys, json, re, collections
base = [json.loads(l) for l in open(sys.argv[1])]
pairs = collections.defaultdict(list)
for r in map(json.loads, open(sys.argv[2])): pairs[r["video_id"]].append(r)
N = int(sys.argv[3]); stats = collections.Counter()
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
with open(sys.argv[4], "w") as fo:
    for o in base:
        if "error" in o: fo.write(json.dumps(o) + "\n"); continue
        pred = {"relationships": parse_rels(o["text"])}
        kept = list(o["kept_obj_ids"]); rels = []
        for r in pred.get("relationships", []):
            try: rels.append([kept[int(r[0])-1] if int(r[0]) > 0 else -1, r[1], kept[int(r[2])-1] if int(r[2]) > 0 else -1, r[3]])
            except Exception: pass
        for pr in pairs.get(o["video_id"], [])[:N]:
            stats["pairs"] += 1
            if "error" in pr: stats["pair_error"] += 1; continue
            pk = pr["kept_obj_ids"]
            prel = parse_rels(pr["text"])
            got = 0
            for r in prel:
                try:
                    s, ob = int(r[0]), int(r[2])
                    if s < 1 or ob < 1 or s == ob: continue
                    rels.append([pk[s-1], r[1], pk[ob-1], r[3]]); got += 1
                except Exception: pass
            stats["pairs_with_relation"] += got > 0; stats["added_relations"] += got
        allids = sorted({x for r in rels for x in (r[0], r[2]) if x >= 0} | set(kept)); ix = {v: i + 1 for i, v in enumerate(allids)}
        out = {"relationships": [[ix.get(r[0], -1) if r[0] >= 0 else -1, r[1], ix.get(r[2], -1) if r[2] >= 0 else -1, r[3]] for r in rels]}
        o = dict(o); o["kept_obj_ids"] = allids; o["text"] = json.dumps(out); fo.write(json.dumps(o) + "\n")
print(dict(stats), file=sys.stderr)
