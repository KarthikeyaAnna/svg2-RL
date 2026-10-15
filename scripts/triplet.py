import sys as _s, os as _o; _s.path.insert(0, _o.path.dirname(_o.path.abspath(__file__))); from relparse import parse_rels
"""Relation + triplet recall with judge. Labels per GT object id from label sources (full outputs, pair outputs). Usage: merged.jsonl gt.parquet judge.json label_src1.jsonl [label_src2.jsonl ...]"""
import sys, json, re, collections, numpy as np, pyarrow.parquet as pq
outs=[json.loads(l) for l in open(sys.argv[1])]; gt={r["video_id"]:r for r in pq.read_table(sys.argv[2]).to_pylist()}; J=json.load(open(sys.argv[3]))
LAB=collections.defaultdict(dict)
for src in sys.argv[4:]:
    for l in open(src):
        o=json.loads(l)
        if "error" in o: continue
        k=o["kept_obj_ids"]
        try: objs=json.loads(o["text"]).get("objects",[])
        except Exception: continue
        for ob in objs:
            for kk,v in ob.items():
                m=re.match(r"object[ _](\d+)",kk)
                if m and int(m.group(1))-1<len(k): LAB[o["video_id"]].setdefault(k[int(m.group(1))-1], re.sub(r"\s*\(uncertain\)\s*$","",str(v).lower().strip()))
def ok(kind,g,p):
    g,p=g.lower().strip(),p.lower().strip()
    if g==p: return True
    c=J.get(f"{kind}|||{g}|||{p}"); return bool(c) and not c.startswith("mismatch")
def secs(sp):
    s=set()
    for ab in sp:
        try: a,b=float(ab[0]),float(ab[1])
        except Exception: continue
        s.update(np.round(np.arange(a,b+1e-6,.5),1).tolist())
    return s
def tiou(a,b): A,B=secs(a),secs(b); return len(A&B)/len(A|B) if A|B else 0

RX=re.compile(r'\[\s*(-?\d+)\s*,\s*"([^"]*)"\s*,\s*(-?\d+)\s*,\s*(\[\s*\[[^\]]*\](?:\s*,\s*\[[^\]]*\])*\s*\])')
import os as _os2
def _iou1(a,b):
    i=max(0,min(float(a[1]),float(b[1]))-max(float(a[0]),float(b[0]))); u=max(float(a[1]),float(b[1]))-min(float(a[0]),float(b[0])); return i/u if u>0 else float(list(a)==list(b))
if _os2.environ.get("TIOU")=="perspan":
    def tiou(a,b): return max((_iou1(x,y) for x in a for y in b if len(x)==2 and len(y)==2), default=0)
NG=RH=TH=0
for o in outs:
    if "error" in o: continue
    k=o["kept_obj_ids"]; g=gt[o["video_id"]]
    gl={int(kk.split("_")[1]):str(v) for ob in json.loads(g["objects"]) for kk,v in ob.items() if kk.startswith("object_")}
    rels=parse_rels(o["text"])
    P=collections.defaultdict(list)
    for r in rels:
        try: P[(k[int(r[0])-1],k[int(r[2])-1])].append((r[1],r[3]))
        except Exception: pass
    for s,p,ob,sp,*_ in json.loads(g["relationships"]):
        NG+=1
        if any(ok("relation predicate",p,q) and tiou(qs,sp)>=.5 for q,qs in P.get((s,ob),[])):
            RH+=1; L=LAB[o["video_id"]]
            if s in L and ob in L and ok("object",gl.get(s,""),L[s]) and ok("object",gl.get(ob,""),L[ob]): TH+=1
print(f"relation recall {100*RH/NG:5.1f} | triplet recall {100*TH/NG:5.1f}")
