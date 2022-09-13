# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os
os.environ["TORCH_HOME"]="/src/.torch"

import sys
sys.path.extend([
    "/CLIP"
    "/taming-transformers",
    "/stable-diffusion-dev",
    "/stable-diffusion-dev/eden",
    "/k-diffusion",
    "/pytorch3d-lite",
    "/MiDaS",
    "/AdaBins"
])


import importlib
import lpips
#from ldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf


import lpips
from ldm.util import instantiate_from_config
import torch
from omegaconf import OmegaConf

config_file = "/stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
config_yaml = OmegaConf.load(config_file)
print(config_yaml)



instantiate_from_config(config_yaml.model)


lpips_perceptor = lpips.LPIPS(net='alex')
