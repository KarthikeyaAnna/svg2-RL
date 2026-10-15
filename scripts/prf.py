import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Precision/recall/F1 at relation level with judge: a prediction is TP if it matches some GT relation (same ordered pair, judge-lenient predicate, tIoU>=.5)."""
import sys, json, re, collections
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, sys.argv[3]); import importlib
outs=[json.loads(l) for l in open(sys.argv[1])]; gt={r["video_id"]:r for r in pq.read_table(sys.argv[2]).to_pylist()}; J=json.load(open(sys.argv[4]))
RX=re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
def ok(g,p):
    g,p=g.lower().strip(),p.lower().strip()
    if g==p: return True
    c=J.get(f"relation predicate|||{g}|||{p}"); return bool(c) and not c.startswith("mismatch")
def secs(sp):
    s=set()
    for ab in sp:
        try: a,b=float(ab[0]),float(ab[1])
        except Exception: continue
        s.update(np.round(np.arange(a,b+1e-6,.5),1).tolist())
    return s
def tiou(a,b):

    A,B=secs(a),secs(b); return len(A&B)/len(A|B) if A|B else 0
import os as _os2
def _iou1(a,b):
    i=max(0,min(float(a[1]),float(b[1]))-max(float(a[0]),float(b[0]))); u=max(float(a[1]),float(b[1]))-min(float(a[0]),float(b[0])); return i/u if u>0 else float(list(a)==list(b))
if _os2.environ.get("TIOU")=="perspan":
    def tiou(a,b): return max((_iou1(x,y) for x in a for y in b if len(x)==2 and len(y)==2), default=0)
TP=NP=NG=0; GH=0
for o in outs:
    if "error" in o: continue
    k=o["kept_obj_ids"]
    rels=parse_rels(o["text"])
    G=json.loads(gt[o["video_id"]]["relationships"]); NG+=len(G); hitg=set()
    for r in rels:
        try: s,ob=(k[int(r[0])-1] if int(r[0])>0 else -1),(k[int(r[2])-1] if int(r[2])>0 else -1)
        except Exception: NP+=1; continue
        NP+=1; m=[i for i,(gs,gp,go,gsp,*_) in enumerate(G) if gs==s and go==ob and ok(gp,r[1]) and tiou(r[3],gsp)>=.5]
        if m: TP+=1; hitg.update(m)
    GH+=len(hitg)
P=TP/NP; R=GH/NG; print(f"P {100*P:5.1f}  R {100*R:5.1f}  F1 {100*2*P*R/(P+R):5.1f}  rels {NP}")
