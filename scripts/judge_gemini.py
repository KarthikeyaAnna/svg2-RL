"""Predicate judge via the local Gemini gateway (gemini-3.8-flash-high), same 5-category prompt as the old Qwen3-8B judge
(kaggle/bench-*/bench.py), batched: one request = up to 50 numbered (GT, predicted) pairs, answer = JSON list of categories.
Usage: judge_gemini.py pairs.json out.json   (pairs = [[gt, pred], ...], normalised strings). Resumable: skips judged keys."""
import sys, json, os, re, time, urllib.request, concurrent.futures as cf
URL = "http://127.0.0.1:8766/api/generate"; CATS = ["identical", "synonym", "hypernym/hyponym", "semantic overlap", "mismatch"]
pairs = [tuple(p) for p in json.load(open(sys.argv[1]))]; OUT = sys.argv[2]
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
key = lambda g, p: f"relation predicate|||{g}|||{p}"
todo = [p for p in pairs if key(*p) not in res]; B = int(os.environ.get("B", 20))
def ask(batch):
    lines = "\n".join(f'{i + 1}. Ground truth: "{g}" | Prediction: "{p}"' for i, (g, p) in enumerate(batch))
    prompt = ("You are a strict lexical matcher for video scene graph evaluation. For each numbered pair below, compare the "
              "ground-truth relation predicate label with the predicted relation predicate label and classify their relationship "
              "into exactly one category: Identical, Synonym, Hypernym/Hyponym, Semantic Overlap, Mismatch.\n\n" + lines +
              f"\n\nAnswer with a JSON list of exactly {len(batch)} category names, in order, and nothing else.")
    for attempt in range(4):
        try:
            r = urllib.request.Request(URL, data=json.dumps({"prompt": prompt, "model": "gemini-3.8-flash-high", "effort": os.environ.get("EFFORT", "low"), "cache": False}).encode(),
                                       headers={"Content-Type": "application/json"})
            t = json.loads(urllib.request.urlopen(r, timeout=700).read())["text"]
            ans = json.loads(t[t.index("["):t.rindex("]") + 1])
            if len(ans) != len(batch): raise ValueError(f"got {len(ans)} answers for {len(batch)}")
            return [(b, next((c for c in CATS if str(a).strip().lower().startswith(c.split("/")[0])), "mismatch:" + str(a)[:20])) for b, a in zip(batch, ans)]
        except Exception as e:
            print("retry", attempt, repr(e)[:120], flush=True); time.sleep(10 * (attempt + 1))
    return []
batches = [todo[i:i + B] for i in range(0, len(todo), B)]
print("pairs", len(pairs), "todo", len(todo), "requests", len(batches), flush=True)
with cf.ThreadPoolExecutor(int(os.environ.get("PAR", 8))) as ex:
    for n, out in enumerate(ex.map(ask, batches), 1):
        for (g, p), c in out: res[key(g, p)] = c
        json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
        print("done", n, "/", len(batches), "judged", len(res), flush=True)
