ARM = "PS"; BENCHES = []; PAIR_BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 0, 0
PAIRSFT_FILE = find("**/pairsft_train.json"); PAIRS_FILE = find("**/pairlists_eval.json")
MAX_PAIR_SAMPLES, ACCUM, LR, LORA_R, N_DEV, LIMIT_VIDS = 10**9, 8, 1e-4, 64, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
