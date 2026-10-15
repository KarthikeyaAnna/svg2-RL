import json, sys, os
from remotezip import RemoteZip
d = json.load(open(sys.argv[1])); val = set(sum([v["val"] for v in d["split"].values()], []))
for z in ["VidOR/vidor_videos.zip", "EpicKitchen/epic_kitchen_videos.zip", "Ego4D/ego4d_videos.zip"]:
    with RemoteZip("https://huggingface.co/datasets/Jingkang/PVSG/resolve/main/" + z) as Z:
        for i in Z.infolist():
            b = os.path.basename(i.filename)
            if b.lower().endswith(".mp4") and b[:-4] in val and not os.path.exists("videos/" + b[:-4] + ".mp4"):
                Z.extract(i, "tmpx"); os.makedirs("videos", exist_ok=True); os.replace("tmpx/" + i.filename, "videos/" + b[:-4] + ".mp4"); print("got", b, flush=True)
print("VIDEOS DONE", len(os.listdir("videos")))
