import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Apply per-pair decodes of TraSeR-mentioned pairs to full graphs (GT id space).
  verify : drop the full graph's relations on a pair whose pair decode is empty
  replace: the full graph's relations on a decoded pair are replaced by the pair decode's relations
Usage: pv_apply.py {verify|replace} FULL_JSONL PAIRS_JSONL DST   (also importable: apply(mode, base_rows, pair_rows))"""
import sys, json, collections

def to_gt(text, kept):
    out = []
    for r in parse_rels(text):
        try:
            s, o = int(r[0]), int(r[2])
            out.append([kept[s - 1] if 1 <= s <= len(kept) else -1, r[1], kept[o - 1] if 1 <= o <= len(kept) else -1, r[3]])
        except Exception: pass
    return out

def apply(mode, base, pair_rows):
    dec = collections.defaultdict(dict)
    for r in pair_rows: dec[r["video_id"]][frozenset(r["kept_obj_ids"])] = to_gt(r["text"], r["kept_obj_ids"])
    out, st = [], collections.Counter()
    for o in base:
        kept = list(o["kept_obj_ids"]); rels = to_gt(o["text"], kept); D = dec.get(o["video_id"], {}); new = []
        for r in rels:
            k = frozenset((r[0], r[2]))
            if k in D and (mode == "replace" or not D[k]): st["dropped"] += 1; continue
            new.append(r)
        if mode == "replace":
            for k, rr in D.items(): new += [r for r in rr if r[0] >= 0 and r[2] >= 0 and r[0] != r[2]]; st["added"] += len(rr)
        st["base"] += len(rels); st["out"] += len(new)
        ids = sorted({x for r in new for x in (r[0], r[2]) if x >= 0} | set(kept)); ix = {v: i + 1 for i, v in enumerate(ids)}
        o = dict(o); o["kept_obj_ids"] = ids
        o["text"] = json.dumps({"relationships": [[ix.get(r[0], -1), r[1], ix.get(r[2], -1), r[3]] for r in new]}); out.append(o)
    return out, dict(st)

if __name__ == "__main__":
    mode, fb, fp, dst = sys.argv[1:5]
    rows, st = apply(mode, [json.loads(l) for l in open(fb)], [json.loads(l) for l in open(fp)])
    open(dst, "w").write("".join(json.dumps(o) + "\n" for o in rows)); print(mode, st, file=sys.stderr)
