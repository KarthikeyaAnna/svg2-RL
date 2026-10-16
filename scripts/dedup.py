"""Merge exact-duplicate (s,p,o) relations into one with the union of spans."""
import sys, json, collections, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from relparse import parse_rels
with open(sys.argv[2], "w") as fo:
    for o in map(json.loads, open(sys.argv[1])):
        if "error" not in o:
            M = collections.OrderedDict()
            for r in parse_rels(o["text"]):
                try: key = (int(r[0]), r[1].strip().lower(), int(r[2]))
                except Exception: continue
                sp = r[3] if isinstance(r[3], (list, tuple)) else []
                M.setdefault(key, []).extend([list(x) for x in sp if isinstance(x, (list, tuple)) and len(x) == 2])
            o = dict(o); o["text"] = json.dumps({"relationships": [[a, p, b, sorted(map(list, {tuple(x) for x in sp}))] for (a, p, b), sp in M.items()]})
        fo.write(json.dumps(o) + "\n")
