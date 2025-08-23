!apt-get -y update >/dev/null
!apt-get -y install python3.11 python3.11-venv espeak-ng >/dev/null

import os, subprocess
VENV = "/content/indictts311"
if not os.path.exists(VENV):
    subprocess.run(["python3.11", "-m", "venv", VENV], check=True)
    subprocess.run([f"{VENV}/bin/python", "-m", "pip", "install", "-U", "pip", "wheel", "setuptools"], check=True)
    subprocess.run([f"{VENV}/bin/pip", "install", "--no-cache-dir",
                    "torch==2.4.0",
                    "numpy==1.26.4", "pandas==1.5.3", "networkx==2.8.8",
                    "TTS==0.22.0", "soundfile", "librosa==0.10.2.post1", "phonemizer==3.2.1"], check=True)
print("venv ready:", VENV)
