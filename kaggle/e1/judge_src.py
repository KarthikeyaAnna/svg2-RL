# Lenient label judge (stand-in for SVG2's GPT-4o-mini aligner): classify (GT label, predicted label) pairs.
import glob, json, os, re, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
CATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]
def build_pairs(outs, gt):
    pairs = set()
    for o in outs:
        if "error" in o or o["video_id"] not in gt: continue
        kept = o["kept_obj_ids"]; g = gt[o["video_id"]]
        gobj = {int(k.split("_")[1]): str(v) for ob in json.loads(g["objects"]) for k, v in ob.items() if k.startswith("object_")}
        try: pred = json.loads(o["text"])
        except Exception: continue
        for ob in pred.get("objects", []):
            for k, v in ob.items():
                m = re.match(r"object[ _](\d+)", k)
                if m and int(m.group(1)) - 1 < len(kept):
                    gl = gobj.get(kept[int(m.group(1)) - 1])
                    if gl: pairs.add(("object", gl.lower().strip(), str(v).lower().strip()))
        P = {}
        for r in pred.get("relationships", []):
            try: P.setdefault((kept[int(r[0]) - 1] if int(r[0]) > 0 else -1, kept[int(r[2]) - 1] if int(r[2]) > 0 else -1), []).append(str(r[1]))
            except Exception: pass
        for s, p, ob, *_ in json.loads(g["relationships"]):
            for q in P.get((s, ob), []):
                if q.lower().strip() != p.lower().strip(): pairs.add(("relation predicate", p.lower().strip(), q.lower().strip()))
    return sorted(pairs)
def run_judge(pairs, model_dir, out_path, bs=64):
    tok = AutoTokenizer.from_pretrained(model_dir); tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    res = {}
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i + bs]; prompts = []
        for kind, g, p in batch:
            msg = (f"You are a strict lexical matcher for video scene graph evaluation. Compare a ground-truth {kind} label with a predicted {kind} label "
                   f"and classify their relationship into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\n"
                   f"Ground truth: \"{g}\"\nPrediction: \"{p}\"\nAnswer with the category name only.")
            prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(): gen = model.generate(**enc, max_new_tokens=8, do_sample=False)
        for (kind, g, p), t in zip(batch, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            t = t.strip().lower(); res[f"{kind}|||{g}|||{p}"] = next((c for c in CATS if t.startswith(c.split("/")[0])), "mismatch:" + t[:20])
        print("judged", min(i + bs, len(pairs)), "/", len(pairs), flush=True)
    json.dump(res, open(out_path, "w"))
