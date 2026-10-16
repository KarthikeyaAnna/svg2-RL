ARM = "PV"; BENCHES = []; PAIR_BENCHES = ["svg2", "vidor", "pvsg"]; K_SAL, K_RND = 0, 0
PAIRS_FILE = find("**/pairlists_mentioned.json"); PG_LORA = find("**/lora_PG.pt")
LORA_R, N_DEV, LIMIT_VIDS = 64, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
