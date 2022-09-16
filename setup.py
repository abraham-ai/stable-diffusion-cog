import os
import sys

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.huggingface/'

os.system('git clone https://github.com/abraham-ai/stable-diffusion-dev')
os.system('git clone https://github.com/deforum/k-diffusion')
os.system('git clone https://github.com/MSFTserver/pytorch3d-lite.git')
os.system('git clone https://github.com/genekogan/AdaBins')
os.system('git clone https://github.com/isl-org/MiDaS.git')
os.system('mv MiDaS/utils.py MiDaS/midas_utils.py')

sys.path.extend([
    "stable-diffusion-dev",
    "stable-diffusion-dev/eden",
])

from sd import get_model
import lpips

config_path = "stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
ckpt_path = "./sd-v1-4.ckpt"

get_model(config_path, ckpt_path, True)
lpips.LPIPS(net='alex')

