ARM = "G3"; BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 40, 20
STYLE_PROMPT_TEST = {"svg2": "scene_graph", "vidor": "scene_graph", "pvsg": "scene_graph"}
G, TEMP, TOP_P, BETA, LR, LORA_R, NORM_LEN = 8, 1.0, 0.95, 0.02, 3e-5, 64, 4096
MAX_TRAIN_PREFILL = 16000
BATCH_PROMPTS, NSTEPS, TRAIN_SECONDS, EVAL_EVERY, N_DEV, LIMIT_VIDS = 4, 300, 7 * 3600, 50, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
