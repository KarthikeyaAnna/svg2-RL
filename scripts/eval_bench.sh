#!/bin/bash
# Usage: eval_bench.sh <bench_out_dir> <gt.parquet> <masks.parquet>
W=~/svg2-killtest; O=$1; G=$2; MK=$3; J=$O/judge_cache.json; PY="$W/venv/bin/python -I"
cd $W
$PY - "$O" <<'PYEOF'
import json, sys, os
O=sys.argv[1]; best={}
for fn in ["full_cap40","full_cap100"]:
    p=f"{O}/{fn}.jsonl"
    if os.path.exists(p):
        for l in open(p):
            r=json.loads(l)
            if "error" not in r or r["video_id"] not in best: best[r["video_id"]]=r
open(f"{O}/allobj.jsonl","w").write("".join(json.dumps(best[k])+"\n" for k in sorted(best)))
print("videos", len(best))
PYEOF
$PY scripts/filter_contact.py $O/allobj.jsonl $MK $O/allobj_filt.jsonl 2>/dev/null
row(){ printf "%-28s " "$1"; $PY scripts/prf.py $2 $G $W/scripts $J; }
row "TraSeR (released, cap40)" $O/full_cap40.jsonl
row "TraSeR all objects" $O/allobj.jsonl
row "contact filter" $O/allobj_filt.jsonl
for N in 10 20 30 40; do $PY scripts/merge_pairs.py $O/allobj.jsonl $O/pairs_sal.jsonl $N $O/m.jsonl 2>/dev/null; row "+$N salient" $O/m.jsonl; $PY scripts/merge_pairs.py $O/allobj_filt.jsonl $O/pairs_sal.jsonl $N $O/m.jsonl 2>/dev/null; row "filter +$N salient" $O/m.jsonl; done
$PY scripts/merge_pairs.py $O/allobj.jsonl $O/pairs_rnd.jsonl 20 $O/m.jsonl 2>/dev/null; row "+20 RANDOM (control)" $O/m.jsonl
$PY scripts/merge_pairs.py $O/allobj.jsonl $O/pairs_sal.jsonl 20 $O/m.jsonl 2>/dev/null; echo "--- breakdown +20 salient:"; $PY scripts/kill_eval.py $O/m.jsonl $G $W/scripts $J | tail -8
echo "--- breakdown TraSeR:"; $PY scripts/kill_eval.py $O/full_cap40.jsonl $G $W/scripts $J | tail -8
echo "--- triplet recall (labels: full graph, then pair decodes)"
for spec in "TraSeR:$O/full_cap40.jsonl" "all-objects:$O/allobj.jsonl"; do printf "%-24s " ${spec%%:*}; $PY scripts/triplet.py ${spec#*:} $G $J $O/allobj.jsonl $O/pairs_sal.jsonl; done
for N in 20 40; do $PY scripts/merge_pairs.py $O/allobj.jsonl $O/pairs_sal.jsonl $N $O/m.jsonl 2>/dev/null; printf "%-24s " "+$N salient"; $PY scripts/triplet.py $O/m.jsonl $G $J $O/allobj.jsonl $O/pairs_sal.jsonl; done
$PY scripts/merge_pairs.py $O/allobj.jsonl $O/pairs_rnd.jsonl 20 $O/m.jsonl 2>/dev/null; printf "%-24s " "+20 RANDOM"; $PY scripts/triplet.py $O/m.jsonl $G $J $O/allobj.jsonl $O/pairs_rnd.jsonl
