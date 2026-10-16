"""Sample-and-aggregate decoding (RESEARCH_LOG §63). Input: SC/SCB *_samples.jsonl (per video: k=0 greedy, k=1..K sampled graphs).
Modes: greedy | voteM (keep (s, norm predicate, o) keys present in >= M graphs; spans = medoid instance by mean temporal IoU, ties ->
greedy then lowest k; object names from the greedy graph) | mbr (graph with highest mean relation F1 against the others as pseudo-labels).
Usage: sc_agg.py samples.jsonl MODE out.jsonl"""
import sys, os, json, re, collections
W = "/home/karthik/svg2-killtest"; sys.path[:0] = [f"{W}/scripts", f"{W}/engine"]
from relparse import parse_rels
from reward import reward, norm, tiou, pred_objects

def load(path):
    V = collections.defaultdict(dict)
    for l in open(path):
        o = json.loads(l); V[(o.get("source"), o["video_id"])][o["k"]] = o
    return V

def rels(text, n):
    out = []
    for r in parse_rels(text):
        try: s, o = int(r[0]), int(r[2])
        except Exception: continue
        if not (1 <= s <= n and 1 <= o <= n): continue
        sp = [[float(x[0]), float(x[1])] for x in (r[3] if isinstance(r[3], (list, tuple)) else []) if isinstance(x, (list, tuple)) and len(x) == 2]
        out.append((s, norm(r[1]), o, sp))
    return out

def objects(text):
    try:
        ob = json.loads(text).get("objects", [])
        if isinstance(ob, list): return ob
    except Exception: pass
    return [{f"object {k}": v} for k, v in sorted(pred_objects(text).items())]

def vote(G, n, m):
    """G: {k: [(s, p, o, spans)]}. One count per graph per key."""
    inst = collections.defaultdict(list)
    for k in sorted(G):
        seen = {}
        for s, p, o, sp in G[k]:
            seen.setdefault((s, p, o), sp)                            # first instance of the key in this graph
        for key, sp in seen.items(): inst[key].append((k, sp))
    out = []
    for key, I in inst.items():
        if len(I) < m: continue
        if len(I) == 1: best = I[0][1]
        else:
            sc = [(sum(tiou(sp, sq) for kj, sq in I if kj != ki) / (len(I) - 1), -ki, sp) for ki, sp in I]
            best = max(sc, key=lambda x: (x[0], x[1]))[2]            # ties -> lowest k (greedy is k=0)
        out.append([key[0], key[1], key[2], best])
    return sorted(out, key=lambda r: (r[0], r[2], r[1]))

def mbr(G, kept):
    lab = {k: [[kept[s - 1], p, kept[o - 1], sp] for s, p, o, sp in R] for k, R in G.items()}
    best = None
    for i in sorted(G):
        txt = json.dumps({"objects": [], "relationships": [[s, p, o, sp] for s, p, o, sp in G[i]]})
        u = sum(reward(txt, kept, [], lab[j])[1]["f1"] for j in G if j != i) / max(1, len(G) - 1)
        if best is None or u > best[0]: best = (u, i)                 # strict > keeps the lowest k on ties (greedy first)
    return best[1]

def aggregate(samples, mode):
    out = []
    for (src, v), S in samples.items():
        g = S[0]; kept = g["kept_obj_ids"]; n = len(kept)
        assert all(S[k]["kept_obj_ids"] == kept for k in S), (v, "kept differs across samples")
        G = {k: rels(S[k]["text"], n) for k in S}
        if mode == "greedy": text = g["text"]
        elif mode == "mbr": text = S[mbr(G, kept)]["text"]
        elif mode.startswith("vote"):
            text = json.dumps({"objects": objects(g["text"]), "relationships": vote(G, n, int(mode[4:]))})
        else: raise ValueError(mode)
        row = {"video_id": v, "kept_obj_ids": kept, "text": text}
        if src is not None: row["source"] = src
        out.append(row)
    return out

if __name__ == "__main__":
    smp, mode, dst = sys.argv[1:4]
    rows = aggregate(load(smp), mode)
    with open(dst, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(mode, len(rows), "videos ->", dst)
