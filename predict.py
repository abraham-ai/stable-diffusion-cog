import os
import sys
import tempfile
import random
import hashlib
import moviepy.editor as mpy
import numpy as np

os.environ["TORCH_HOME"] = "/src/.torch"
os.environ['TRANSFORMERS_CACHE'] = '/src/.huggingface/'

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


from settings import StableDiffusionSettings
from sd import get_model
from utils import get_file_sha256
import depth
import generation

from cog import BasePredictor, Input, Path




class Predictor(BasePredictor):
    def setup(self):
        self.config_path = "/stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
        self.ckpt_path = "./sd-v1-4.ckpt"
        self.model = get_model(self.config_path, self.ckpt_path, True)
        depth.setup_depth_models('.')

    def predict(
        self,
        text_input: str = Input(
            description="Text input"
        ),
        seed: int = Input(
            description="Width", ge=0, le=1e8, default=13
        ),
        width: int = Input(
            description="Width", ge=64, le=768, default=512
        ),
        height: int = Input(
            description="Height", ge=64, le=768, default=512
        ),
        scale: float = Input(
            description="Factor to scale image by", ge=0, le=20, default=8.0
        ),
        sampler: str = Input(
            description="Which sampler to use", default="klms"
        ),
        steps: int = Input(
            description="Diffusion steps", ge=10, le=200, default=50
        ),
        init_image_strength: float = Input(
            description="Strength of initial image", ge=0, le=1, default=0.0
        ),
        init_image_file: Path = Input(
            description="Load initial image from file", default=None
        ),
        init_image_b64: str = Input(
            description="Load initial image from base64 string", default=None
        ),
        mask_image_file: Path = Input(
            description="Load mask image from file", default=None
        ),
        mask_image_b64: str = Input(
            description="Load mask image from base64 string", default=None
        )
    ) -> Path:

        settings = StableDiffusionSettings(
            mode = "generate",
            config = self.config_path,
            ckpt = self.ckpt_path,
            sampler = sampler,
            text_input = text_input,
            seed = seed,
            steps = steps,
            scale = scale,
            H = height - (height % 64),
            W = width - (width % 64),
            init_image_file = init_image_file,
            init_image_b64 = init_image_b64,
            strength = init_image_strength,
            mask_image_file = mask_image_file,
            mask_image_b64 = mask_image_b64,
            invert_mask = True
        )

        print(settings)

        final_images = generation.make_images(settings, callback=None)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        final_images[0].save(str(out_path))

        # get_file_sha256(out_path)

        return out_path
        