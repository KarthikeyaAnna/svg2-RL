"""Graph-matching reward against TraSeR's own training labels (fixed a priori, benchmark-agnostic).

R = F1_rel + 0.25 * object_accuracy - 0.2 * [output not a complete JSON object]
- Relations: each predicted relation [s, p, o, spans] is matched one-to-one (greedy by temporal IoU) to a label relation with the same
  ordered object pair and the same normalized predicate. A match earns its temporal IoU (soft credit, 0..1). Duplicates and repeats
  cannot match twice, so they count as false positives. F1 = 2PR/(P+R) with P = credit/#pred, R = credit/#label.
- Objects: predicted "object k" label equals the label of kept[k-1] after normalization.
Prediction ids are 1-based over `kept` (the prompt's object order); label ids are the original object ids."""
import json, re
from relparse import parse_rels

def norm(s):
    return re.sub(r"\s+", " ", str(s).lower().replace("_", " ").strip())

def secs(spans):
    out = set()
    for ab in spans or []:
        try: a, b = float(ab[0]), float(ab[1])
        except Exception: continue
        if b < a: a, b = b, a
        x = a
        while x <= b + 1e-6:
            out.add(round(x * 2) / 2); x += 0.5
    return out

def tiou(a, b):
    A, B = secs(a), secs(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0

def pred_objects(text):
    out = {}
    for m in re.finditer(r'\{\s*"object (\d+)"\s*:\s*"([^"]*)"', text):
        out[int(m.group(1))] = m.group(2)
    return out

def reward(text, kept, label_objects, label_rels, w_obj=0.25, w_fmt=0.2):
    """Returns (R, details). label_objects: [{"object_<id>": name, ...}]; label_rels: [[s, p, o, spans, ...]]."""
    idx = {oid: i + 1 for i, oid in enumerate(kept)}
    G = []
    for r in label_rels:
        s, o = int(r[0]), int(r[2])
        if s in idx and o in idx: G.append((idx[s], norm(r[1]), idx[o], r[3]))
    P = []
    for r in parse_rels(text):
        try: P.append((int(r[0]), norm(r[1]), int(r[2]), r[3]))
        except Exception: continue
    cand = []
    for i, (s, p, o, sp) in enumerate(P):
        for j, (gs, gp, go, gsp) in enumerate(G):
            if s == gs and o == go and p == gp:
                cand.append((tiou(sp, gsp), i, j))
    used_p, used_g, credit = set(), set(), 0.0
    for c, i, j in sorted(cand, reverse=True):
        if i in used_p or j in used_g or c <= 0: continue
        used_p.add(i); used_g.add(j); credit += c
    prec = credit / len(P) if P else 0.0
    rec = credit / len(G) if G else (1.0 if not P else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    lab = {}
    for ob in label_objects:
        k = next((k for k in ob if k.startswith("object_")), None)
        if k is not None: lab[int(k.split("_")[1])] = norm(ob[k])
    po = pred_objects(text)
    obj_acc = sum(1 for k, oid in enumerate(kept) if norm(po.get(k + 1, "")) == lab.get(oid, "\0")) / max(1, len(kept))
    complete = text.rstrip().endswith("}")
    try: json.loads(text); valid = True
    except Exception: valid = False
    R = f1 + w_obj * obj_acc - (0.0 if (complete and valid) else w_fmt)
    return R, dict(f1=f1, p=prec, r=rec, obj=obj_acc, n_pred=len(P), n_label=len(G), valid=valid)


def reward_g4(text, kept, label_objects, label_rels, touching_pairs=None,
              w_obj=0.25, w_fmt=0.2, lam_fp=0.15):
    """G4 reward: base F1 + geometry-gated false-positive penalty.
    
    touching_pairs: set of frozenset({obj_id_a, obj_id_b}) for pairs
                    whose masks make dilated contact on any frame.
    """
    R_base, d = reward(text, kept, label_objects, label_rels, w_obj, w_fmt)
    if touching_pairs is None:
        touching_pairs = set()
    idx = {oid: i + 1 for i, oid in enumerate(kept)}
    labelled_pairs = set()
    for r in label_rels:
        try:
            s, o = int(r[0]), int(r[2])
            if s in idx and o in idx:
                labelled_pairs.add(frozenset((s, o)))
        except Exception:
            pass
    
    n_geo_fp = 0
    for r in parse_rels(text):
        try:
            s, o = int(r[0]), int(r[2])
            s_oid = kept[s - 1] if 1 <= s <= len(kept) else -1
            o_oid = kept[o - 1] if 1 <= o <= len(kept) else -1
            pair = frozenset((s_oid, o_oid))
            if pair not in labelled_pairs and pair not in touching_pairs:
                n_geo_fp += 1
        except Exception:
            pass
    
    n_pred = d["n_pred"]
    penalty = lam_fp * n_geo_fp / max(1, n_pred)
    R = R_base - penalty
    d["n_geo_fp"] = n_geo_fp
    d["penalty"] = round(penalty, 4)
    return R, d

