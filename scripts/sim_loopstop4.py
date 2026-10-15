"""Offline replay of LoopStop4 = LoopStop3 rules + novelty stall (among the last `win` relation tuples, fewer than `min_new` are
(s,p,o) triples not seen earlier). Truncates saved outputs where it would fire; reports catches. Usage: src dst win min_new"""
import json, re, sys
src, dst, WIN, MINNEW = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
RX = re.compile(r'\[\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,')
def fires(t):
    lo = t.rfind('"attributes": [')
    if lo >= 0 and "]" not in t[lo:] and t[lo:].count('", "') > 30: return "attr"
    r = t.find('"relationships"')
    if r < 0: return None
    rel = t[r:]
    if len(rel) > 300 and rel[:-300].count(rel[-300:]) >= 2: return "repeat"
    tup = RX.findall(rel)
    if len(tup) >= WIN:
        seen = set(tup[:-WIN]); new = 0
        for x in tup[-WIN:]:
            if x not in seen: new += 1; seen.add(x)
        if new < MINNEW: return "stall"
    return None
stops = {}; n = 0
with open(dst, "w") as fo:
    for o in map(json.loads, open(src)):
        if "error" not in o:
            t = o["text"]; n += 1
            for end in range(250, len(t) + 250, 250):
                r = fires(t[:end])
                if r: stops[o["video_id"]] = (r, end, len(t)); o = dict(o); o["text"] = t[:end]; break
        fo.write(json.dumps(o) + "\n")
print(json.dumps({"n": n, "stopped": len(stops), "chars_saved": sum(c - e for _, e, c in stops.values()), "by_rule": {k: sum(1 for v in stops.values() if v[0] == k) for k in ("attr", "repeat", "stall")}}), file=sys.stderr)
