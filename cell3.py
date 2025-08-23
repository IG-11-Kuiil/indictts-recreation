import os, uuid, glob, tempfile, subprocess, IPython.display as ipd

VENV = "/content/indictts311"
LANG_DIR = "/content/indic_tts_ckpts/hi"
FP_MODEL  = os.path.join(LANG_DIR,"fastpitch","best_model_1spk_clean.pth")
FP_CONFIG = os.path.join(LANG_DIR,"fastpitch","config.json")
VG_MODEL  = glob.glob(os.path.join(LANG_DIR,"hifigan","*best*.pth"))[0]
VG_CONFIG = os.path.join(LANG_DIR,"hifigan","config.json")
OUT_DIR = "/content/tts_out"; os.makedirs(OUT_DIR, exist_ok=True)
OUT_WAV = os.path.join(OUT_DIR, f"{uuid.uuid4().hex}.wav")

script = f"""
import os
os.environ['COQUI_TOS_AGREED']='1'
os.environ['PHONEMIZER_ESPEAK_LIBRARY']='/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1'
os.environ['MPLBACKEND']='Agg'
from TTS.utils.synthesizer import Synthesizer

s = Synthesizer(
    tts_checkpoint=r"{FP_MODEL}",
    tts_config_path=r"{FP_CONFIG}",
    vocoder_checkpoint=r"{VG_MODEL}",
    vocoder_config=r"{VG_CONFIG}",
    use_cuda=False
)

text = "नमस्ते, यह इंडिक टीटीएस का डेमो है।".replace("।"," ")
wav = s.tts(text)
s.save_wav(wav, r"{OUT_WAV}")
print("WROTE:", r"{OUT_WAV}")
"""
tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".py")
tmp.write(script); tmp.close()
res = subprocess.run([f"{VENV}/bin/python", tmp.name], capture_output=True, text=True)
print(res.stdout); print(res.stderr)
ipd.Audio(OUT_WAV)
