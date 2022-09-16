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

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        import generation
        self.config_path = "/stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
        self.ckpt_path = "./sd-v1-4.ckpt"
        self.model = get_model(self.config_path, self.ckpt_path, True)
        depth.setup_depth_models('.')

    def predict(
        self,

        # Universal
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "interpolate", "animate"]
        ),
        width: int = Input(
            description="Width", 
            ge=64, le=768, default=512
        ),
        height: int = Input(
            description="Height", 
            ge=64, le=768, default=512
        ),
        sampler: str = Input(
            description="Which sampler to use", 
            efault="klms", choices=["ddim", "plms", "klms" "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]
        ),
        steps: int = Input(
            description="Diffusion steps", 
            ge=10, le=200, default=50
        ),
        scale: float = Input(
            description="Text conditioning scale", 
            ge=0, le=20, default=8.0
        ),
        # ddim_eta: float = 0.0
        # C: int = 4
        # f: int = 8   
        # dynamic_threshold: float = None
        # static_threshold: float = None

        # Init image and mask
        init_image_strength: float = Input(
            description="Strength of initial image", 
            ge=0.0, le=1.0, default=0.0
        ),
        init_image_file: Path = Input(
            description="Load initial image from file", 
            default=None
        ),
        init_image_b64: str = Input(
            description="Load initial image from base64 string", 
            default=None
        ),
        strength : float = Input(
            description="Strength of initial image"
            ge=0.0, le=20.0, default=7.0,
        ),
        mask_image_file: Path = Input(
            description="Load mask image from file", 
            default=None
        ),
        mask_image_b64: str = Input(
            description="Load mask image from base64 string", 
            default=None
        ),
        invert_mask: bool = Input(
            description="Invert mask", 
            default=False
        ),
        # mask_brightness_adjust: float = 1.0
        # mask_contrast_adjust: float = 1.0

        # Generate mode
        text_input: str = Input(
            description="Text input (mode=generate)",
        ),
        seed: int = Input(
            description="random seed (mode==generate)", 
            ge=0, le=1e8, default=13
        ),
        n_samples: int = Input(
            description="batch size (mode==generate)",
            ge=1, le=4, default=1
        ),

        # Interpolate mode
        interpolation_texts: str = Input(
            description="Interpolation texts (mode==interpolate)"
            default=None
        ),
        interpolation_seeds: str = Input(
            description="Seeds for interpolated texts (mode==interpolate)",
            default=None
        ),
        n_interpolate: int = Input(
            description="Number of frames between each interpolated video (mode==interpolate)"
            ge=0, le=32, default=16
        ),
        loop: bool = Input(
            description="Loops (mode==interpolate)",
            default=True
        ),
        smooth: bool = Input(
            description="Smooth (mode==interpolate)",
            default=False
        ),
        
        # Animation mode
        animation_mode: str = Input(
            description="Interpolation texts (mode==interpolate)",
            default='2D', choices= ['2D', '3D', 'Video Input']
        ),
        init_video: Path = Input(
            description="Initial video file (mode==animate)"
            default=None
        ),
        extract_nth_frame: int = Input(
            description="Extract each frame of init_video (mode==animate)",
            ge=1, le=10, default=1
        ),
        turbo_steps: int = Input(
            description="Turbo steps (mode==animate)"
            ge=1, le=8, default=3
        ),
        previous_frame_strength: float = Input(
            description="Strength of previous frame (mode==animate)"
            ge=0.0, le=1.0, default=0.65
        ),
        previous_frame_noise: float = Input(
            description="How much to noise previous frame (mode==animate)"
            ge=0.0, le=0.2, default=0.02
        ),
        color_coherence: str = Input(
            description="Color coherence strategy (mode==animate)", 
            default='Match Frame 0 LAB', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB']
        ),
        contrast: float = Input(
            description="Contrast (mode==animation)"
            ge=0.0, le=2.0, default=1.0
        ),
        angle: float = Input(
            description="Rotation angle (animation_mode==2D)"
            ge=-2.0, le=2.0, default=0.0
        ),
        zoom: float = Input(
            description="Zoom (animation_mode==2D)"
            ge=0.91, le=1.12, default=1.0
        ),
        translation_x: float = Input(description="Translation X (animation_mode==3D)", ge=-5, le=5, default=0),
        translation_y: float = Input(description="Translation U (animation_mode==3D)", ge=-5, le=5, default=0),
        translation_z: float = Input(description="Translation Z (animation_mode==3D)", ge=-5, le=5, default=0),
        rotation_x: float = Input(description="Rotation X (animation_mode==3D)", ge=-1, le=1, default=0),
        rotation_y: float = Input(description="Rotation U (animation_mode==3D)", ge=-1, le=1, default=0),
        rotation_z: float = Input(description="Rotation Z (animation_mode==3D)", ge=-1, le=1, default=0)

    ) -> Path:

        settings = StableDiffusionSettings(
            config = self.config_path,
            ckpt = self.ckpt_path,
            precision= 'autocast',
            half_precision = True,

            mode = mode,

            W = width - (width % 64),
            H = height - (height % 64),
            sampler = sampler,
            steps = steps,
            scale = scale,

            init_image_file = init_image_file,
            init_image_b64 = init_image_b64,
            init_image_strength = init_image_strength,
            mask_image_file = mask_image_file,
            mask_image_b64 = mask_image_b64,
            mask_intert = mask_intert,

            text_input = text_input,
            seed = seed,
            n_samples = n_samples,

            interpolation_texts = interpolation_texts.split('|'),
            interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')],
            n_interpolate = n_interpolate,
            loop = loop,
            smooth = smooth,

            animation_mode = animation_mode,
            color_coherence = None if color_coherence=='None' else color_coherence,
            init_video = init_video,
            extract_nth_frame = extract_nth_frame,
            turbo_steps = turbo_steps,
            previous_frame_strength = previous_frame_strength,
            previous_frame_noise = previous_frame_noise,
            contrast = contrast,
            angle = angle,
            zoom = zoom,
            translation = [translation_x, translation_y, translation_z],
            rotation = [rotation_x, rotation_y, rotation_z]
        )

        print(settings)

        final_images = generation.make_images(settings, callback=None)
        out_path = Path(tempfile.mkdtemp()) / "out.png"
        final_images[0].save(str(out_path))

        # get_file_sha256(out_path)

        return out_path
        