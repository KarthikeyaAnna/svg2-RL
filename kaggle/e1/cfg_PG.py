ARM = "PG"; BENCHES = []; PAIR_BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 0, 0
PAIRSFT_FILE = find("**/pairsft_train.json"); PAIRS_FILE = find("**/pairlists_eval.json")
G, TEMP, TOP_P, LR, LORA_R, NORM_LEN, W_NONE = 8, 1.0, 0.95, 2e-5, 64, 512, 0.5
BATCH_PAIRS, NSTEPS, TRAIN_SECONDS, N_DEV, LIMIT_VIDS = 16, 120, 4 * 3600, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
