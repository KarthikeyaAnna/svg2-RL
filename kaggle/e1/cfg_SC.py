ARM = "SC"; BENCHES = []; PAIR_BENCHES = []; K_SAL, K_RND = 0, 0
SC_BENCHES = ["svg2", "vidor", "pvsg"]; SC_K, TEMP, TOP_P = 8, 1.0, 0.95
SC_LORA = find("**/lora_G3.pt"); LORA_R, N_DEV, LIMIT_VIDS = 64, 40, 10**9
TRAIN_DIRS = {"vidor": os.path.dirname(find("**/train_vidor/**/meta_train.json")), "pvd": os.path.dirname(find("**/train_pvd/**/meta_train.json"))}
