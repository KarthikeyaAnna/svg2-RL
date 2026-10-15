"""Offline smoke test of LoopStop2's text rules: replay every saved output as growing prefixes (every ~250 chars, using the last ~10k chars like the 2500-token window)."""
import json, re, sys
def fires(t):
    last_open = t.rfind('"attributes": [')
    if last_open >= 0 and "]" not in t[last_open:] and t[last_open:].count('", "') > 25: return "attr-list"
    rels = re.findall(r'\[\s*(\d+)\s*,\s*"([^"]+)"\s*,\s*(\d+)\s*,\s*(\[\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*)', t)
    fan = {}
    for a, p, b, sp in rels:
        fan.setdefault((a, p), set()).add(b)
        if sp.count("[") - 1 > 12: return "span-splinter"
    if any(len(v) >= 20 for v in fan.values()): return "fan-out"
    return None
res = {"clean_fired": [], "clean_ok": 0, "loop_fired": [], "loop_missed": []}
for f in sys.argv[1:]:
    for o in map(json.loads, open(f)):
        if "error" in o: continue
        t = o["text"]
        try: json.loads(t); clean = True
        except Exception: clean = False
        hit = None
        for end in range(250, len(t) + 250, 250):
            w = t[max(0, end - 10000):end]; r = fires(w)
            if r: hit = (r, end); break
        if clean:
            if hit: res["clean_fired"].append((o["video_id"], hit, len(t)))
            else: res["clean_ok"] += 1
        else:
            (res["loop_fired"] if hit else res["loop_missed"]).append((o["video_id"], hit, len(t)))
print("clean outputs never stopped:", res["clean_ok"], "| clean outputs WRONGLY stopped:", len(res["clean_fired"]), res["clean_fired"][:6])
print("looping/truncated outputs caught:", len(res["loop_fired"]), [(v, h) for v, h, _ in res["loop_fired"]])
print("looping/truncated outputs missed:", [(v, n) for v, h, n in res["loop_missed"]])
