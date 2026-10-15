"""Simulate the section-aware loop stop on saved outputs: truncate text where it would fire; write a new jsonl."""
import json, sys
src, dst, L, k = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); n = cut = saved = 0
def fires(w):
    lo = w.rfind('"attributes": [')
    if lo >= 0 and "]" not in w[lo:] and w[lo:].count('", "') > 30: return True
    r = w.find('"relationships"')
    if r >= 0:
        rel = w[r:]
        if len(rel) > L and rel[:-L].count(rel[-L:]) >= k: return True
    return False
with open(dst, "w") as fo:
    for o in map(json.loads, open(src)):
        if "error" not in o:
            t = o["text"]; n += 1
            for end in range(250, len(t) + 250, 200):
                if fires(t[:end]): o = dict(o); o["text"] = t[:end]; cut += 1; saved += len(t) - end; break
        fo.write(json.dumps(o) + "\n")
print(f"outputs {n}, stopped early {cut}, chars saved {saved} ({100*saved/max(1,sum(1 for _ in [0])):.0f})", file=sys.stderr)
