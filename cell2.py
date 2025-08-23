import os, urllib.request, zipfile, glob, json, torch

BASE = "/content/indic_tts_ckpts"; ASSET = "hi.zip"
URL = f"https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/{ASSET}"
os.makedirs(BASE, exist_ok=True)
ZIP = f"{BASE}/{ASSET}"; LANG_DIR = f"{BASE}/{ASSET[:-4]}"
if not os.path.exists(LANG_DIR):
    urllib.request.urlretrieve(URL, ZIP)
    with zipfile.ZipFile(ZIP, "r") as z: z.extractall(BASE)
    os.remove(ZIP)

FP_CFG = f"{LANG_DIR}/fastpitch/config.json"
with open(FP_CFG) as f: cfg = json.load(f)
def f1(d):
    if isinstance(d, dict):
        for k in list(d.keys()):
            v = d[k]
            if k in ("num_speakers","n_speakers","speaker_num") and isinstance(v,int): d[k]=1
            if k in ("use_speaker_embedding","use_speaker_emb","speaker_embedding"): d[k]=False
            if k in ("speakers_file","speaker_ids_file","speaker_id_file","speaker_manager","speaker_mapping"): d.pop(k,None)
            f1(v)
    elif isinstance(d,list):
        for x in d: f1(x)
f1(cfg); cfg.setdefault("model_args",{}); cfg["model_args"]["num_speakers"]=1; cfg["model_args"]["use_speaker_embedding"]=False
with open(FP_CFG,"w") as f: json.dump(cfg,f,ensure_ascii=False,indent=2)

fp_src = glob.glob(os.path.join(LANG_DIR,"fastpitch","*best*.pth"))[0]
ckpt = torch.load(fp_src, map_location="cpu"); state = ckpt["model"] if isinstance(ckpt,dict) and "model" in ckpt else ckpt
for k in list(state.keys()):
    kl = k.lower()
    if ("emb_g" in kl) or ("speaker" in kl and "emb" in kl): state.pop(k,None)
fp_clean = os.path.join(LANG_DIR,"fastpitch","best_model_1spk_clean.pth")
torch.save(ckpt, fp_clean)
print("ready:", LANG_DIR)
