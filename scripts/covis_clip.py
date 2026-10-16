"""Co-visibility span constraint: a relation between two objects can only hold while both are visible. Each predicted span is
intersected with the union of windows [t_f - h, t_f + h] (h = half the mask sampling interval) over mask frames where BOTH objects
have non-empty masks; relations with no remaining span are dropped. Benchmark-agnostic (uses only the given masks and their fps).
apply(rels_gt_ids, vis, fps) -> rels; vis: {obj_id: bool array over mask frames}."""
import numpy as np
def windows(va, vb, fps):
    f = np.nonzero(va & vb)[0]; h = 0.5 / fps; W = []
    for x in f:
        a, b = x / fps - h, x / fps + h
        if W and a <= W[-1][1] + 1e-6: W[-1][1] = b
        else: W.append([a, b])
    return W
def clip_spans(spans, W):
    out = []
    for s in spans:
        try: a, b = float(s[0]), float(s[1])
        except Exception: continue
        if b < a: a, b = b, a
        for wa, wb in W:
            lo, hi = max(a, wa), min(b, wb)
            if hi >= lo: out.append([round(lo, 2), round(hi, 2)])
    return out
def apply(rels, vis, fps):
    out = []
    for s, p, o, sp in rels:
        if s not in vis or o not in vis: out.append([s, p, o, sp]); continue
        c = clip_spans(sp, windows(vis[s], vis[o], fps))
        if c: out.append([s, p, o, c])
    return out
