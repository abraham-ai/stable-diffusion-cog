import os
import sys

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.huggingface/'

sys.path.extend([
    "/stable-diffusion-dev",
    "/stable-diffusion-dev/eden",
])

from sd import get_model
import lpips

config_path = "/stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
ckpt_path = "./sd-v1-4.ckpt"

get_model(config_path, ckpt_path, True)

lpips_perceptor = lpips.LPIPS(net='alex')
