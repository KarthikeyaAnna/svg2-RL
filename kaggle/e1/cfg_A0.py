ARM = "A0"; BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 40, 20
STYLE_PROMPT_TEST = {"svg2": "scene_graph", "vidor": "scene_graph", "pvsg": "scene_graph"}
MAX_PER_SOURCE, ACCUM, LR, LORA_R, LIMIT_VIDS = 1200, 8, 1e-4, 64, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))} if ARM != "A0" else {}
