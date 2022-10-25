import os
import sys
import tempfile
import random
import hashlib
from typing import Iterator
import moviepy.editor as mpy
import numpy as np
from dotenv import load_dotenv

load_dotenv()
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
from sd import get_model, get_prompt_conditioning
from utils import get_file_sha256
import depth

from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):

    def setup(self):
        import generation
        self.config_path = "/stable-diffusion-dev/configs/stable-diffusion/v1-inference.yaml"
        self.ckpt_path = "./v1-5-pruned-emaonly.ckpt" #"./sd-v1-4.ckpt"
        self.model = get_model(self.config_path, self.ckpt_path, True)
        depth.setup_depth_models('.')

    def predict(
        self,
        
        # Universal args
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "interpolate", "animate"]
        ),
        stream: bool = Input(
            description="yield individual results if True", default=False
        ),
        stream_every: int = Input(
            description="for mode generate, how many steps per update to stream (steam must be set to True)", 
            default=1, ge=1, le=25
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
            default="klms", choices=["ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]
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
        init_image_file: Path = Input(
            description="Load initial image from file", 
            default=None
        ),
        init_image_b64: str = Input(
            description="Load initial image from base64 string", 
            default=None
        ),
        init_image_strength: float = Input(
            description="Strength of initial image", 
            ge=0.0, le=1.0, default=0.0
        ),
        init_image_inpaint_mode: str = Input(
            description="Inpainting method for pre-processing init_image when it's masked", 
            default="cv2_telea", choices=["mean_fill", "edge_pad", "cv2_telea", "cv2_ns"]
        ),
        mask_image_file: Path = Input(
            description="Load mask image from file", 
            default=None
        ),
        mask_image_b64: str = Input(
            description="Load mask image from base64 string", 
            default=None
        ),
        mask_invert: bool = Input(
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

        # Interpolate / Animate mode
        n_frames: int = Input(
            description="Total number of frames (mode==interpolate/animate)",
            ge=0, le=1000, default=50
        ),

        # Interpolate mode
        interpolation_texts: str = Input(
            description="Interpolation texts (mode==interpolate)",
            default=None
        ),
        interpolation_seeds: str = Input(
            description="Seeds for interpolated texts (mode==interpolate)",
            default=None
        ),
        interpolation_init_images: str = Input(
            description="Interpolation init images, file paths or urls (mode==interpolate)",
            default=None
        ),
        interpolation_init_images_top_k: int = Input(
            description="Top K for interpolation_init_images prompts (mode==interpolate)",
            ge=1, le=10, default=2
        ),
        interpolation_init_images_power: float = Input(
            description="Power for interpolation_init_images prompts (mode==interpolate)",
            ge=0.0, le=8.0, default=2.5
        ),
        interpolation_init_images_min_strength: float = Input(
            description="Minimum init image strength for interpolation_init_images prompts (mode==interpolate)",
            ge=0, le=1, default=0.2
        ),
        scale_modulation: float = Input(
            description="Scale modulation amplitude for interpolation (mode==interpolate)",
            ge=0.0, le=1/0, default=0.2
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
            description="Initial video file (mode==animate)",
            default=None
        ),
        extract_nth_frame: int = Input(
            description="Extract each frame of init_video (mode==animate)",
            ge=1, le=10, default=1
        ),
        turbo_steps: int = Input(
            description="Turbo steps (mode==animate)",
            ge=1, le=8, default=3
        ),
        previous_frame_strength: float = Input(
            description="Strength of previous frame (mode==animate)",
            ge=0.0, le=1.0, default=0.65
        ),
        previous_frame_noise: float = Input(
            description="How much to noise previous frame (mode==animate)",
            ge=0.0, le=0.2, default=0.02
        ),
        color_coherence: str = Input(
            description="Color coherence strategy (mode==animate)", 
            default='Match Frame 0 LAB', choices=['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB']
        ),
        contrast: float = Input(
            description="Contrast (mode==animation)",
            ge=0.0, le=2.0, default=1.0
        ),
        angle: float = Input(
            description="Rotation angle (animation_mode==2D)",
            ge=-2.0, le=2.0, default=0.0
        ),
        zoom: float = Input(
            description="Zoom (animation_mode==2D)",
            ge=0.91, le=1.12, default=1.0
        ),
        translation_x: float = Input(description="Translation X (animation_mode==3D)", ge=-5, le=5, default=0),
        translation_y: float = Input(description="Translation U (animation_mode==3D)", ge=-5, le=5, default=0),
        translation_z: float = Input(description="Translation Z (animation_mode==3D)", ge=-5, le=5, default=0),
        rotation_x: float = Input(description="Rotation X (animation_mode==3D)", ge=-1, le=1, default=0),
        rotation_y: float = Input(description="Rotation U (animation_mode==3D)", ge=-1, le=1, default=0),
        rotation_z: float = Input(description="Rotation Z (animation_mode==3D)", ge=-1, le=1, default=0)

    ) -> Iterator[Path]:

        import generation

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
            init_image_inpaint_mode = init_image_inpaint_mode,
            mask_image_file = mask_image_file,
            mask_image_b64 = mask_image_b64,
            mask_invert = mask_invert,

            text_input = text_input,
            seed = seed,
            n_samples = n_samples,

            n_frames = n_frames,
            interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None,
            interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None,
            interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None,
            interpolation_init_images_top_k = interpolation_init_images_top_k,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,

            scale_modulation = scale_modulation,
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

        if mode == "generate":

            steps_per_update = stream_every if stream else None

            generator = generation.make_images(settings, steps_per_update=steps_per_update)
            
            for frame, t in generator:
                out_path = Path(tempfile.mkdtemp()) / "frame.jpg"
                frame[0].save(out_path, format='JPEG', subsampling=0, quality=95)
                yield out_path
                
        else:

            if mode == "interpolate":
                generator = generation.make_interpolation(settings)
                
            elif mode == "animate":
                generator = generation.make_animation(settings)

            frames = []
            for frame, t in generator:
                out_path = Path(tempfile.mkdtemp()) / "frame.jpg"
                frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                frames.append(np.array(frame))
                if stream:
                    yield out_path

            out_path = Path(tempfile.mkdtemp()) / "out.mp4"
            clip = mpy.ImageSequenceClip(frames, fps=8)
            clip.write_videofile(str(out_path))

            yield out_path
