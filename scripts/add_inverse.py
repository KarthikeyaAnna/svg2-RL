import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
import sys, json, re
INV = {"above": "below", "over": "below", "below": "above", "beneath": "above", "under": "above", "underneath": "above",
       "in front of": "behind", "behind": "in front of", "on": "under", "on top of": "under", "inside": "contains", "in": "contains", "next to": "next to", "near": "near", "beside": "beside", "beneath": "above", "behind": "in front of"}
REL_RX = re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
with open(sys.argv[2], "w") as fo:
    for o in map(json.loads, open(sys.argv[1])):
        if "error" not in o:
            rels = parse_rels(o["text"])
            have = {(r[0], r[1].lower().strip(), r[2]) for r in rels}
            add = [[r[2], INV[r[1].lower().strip()], r[0], r[3]] for r in rels if r[1].lower().strip() in INV and (r[2], INV[r[1].lower().strip()], r[0]) not in have]
            o = dict(o); o["text"] = json.dumps({"relationships": rels + add})
        fo.write(json.dumps(o) + "\n")
