ARM = "A1"; BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 40, 20
STYLE_PROMPT_TRAIN = {"vidor": "scene_graph", "pvd": "scene_graph"}; TARGET_VIEW = {"vidor": "orig", "pvd": "orig"}
STYLE_PROMPT_TEST = {"svg2": "scene_graph", "vidor": "scene_graph", "pvsg": "scene_graph"}
# control for G1: same sources, same 40-video dev hold-out per source, same number of video visits (150 steps x 4 = 600, 300 per source)
MAX_TRAIN_PREFILL = 16000
MAX_PER_SOURCE, ACCUM, LR, LORA_R, N_DEV, LIMIT_VIDS = 300, 8, 1e-4, 64, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
