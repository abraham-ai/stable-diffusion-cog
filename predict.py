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
os.environ["TRANSFORMERS_CACHE"] = "/src/.huggingface/"

sys.path.extend([
    "/CLIP"
    "/taming-transformers",
    "eden-stable-diffusion",
    "eden-stable-diffusion/eden",
    "/k-diffusion",
    "/pytorch3d-lite",
    "/MiDaS",
    "/AdaBins",
    "/frame-interpolation"
    "/clip-interrogator"
])

from settings import StableDiffusionSettings
from sd import get_model, get_prompt_conditioning
#import depth
import film
import eden_utils

from cog import BasePredictor, BaseModel, File, Input, Path

film.FILM_MODEL_PATH = "/src/models/film/film_net/Style/saved_model"

CONFIG_PATH = "eden-stable-diffusion/configs/stable-diffusion/v1-inference.yaml"
CKPT_PATH = "/src/models/v1-5-pruned-emaonly.ckpt"
HALF_PRECISION = True


class CogOutput(BaseModel):
    file: Path
    thumbnail: Path
    attributes: dict


class Predictor(BasePredictor):

    def setup(self):
        import generation
        generation.MODELS_PATH = '/src/models'
        generation.CLIP_INTERROGATOR_MODEL_PATH = '/src/cache'
        self.config_path = CONFIG_PATH
        self.ckpt_path = CKPT_PATH
        self.half_precision = HALF_PRECISION
        self.model = get_model(self.config_path, self.ckpt_path, self.half_precision)
        #depth.setup_depth_models(".")

    def predict(
        self,
        
        # Universal args
        mode: str = Input(
            description="Mode", default="generate",
            choices=["generate", "remix", "interpolate", "real2real", "interrogate"]
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
            ge=64, le=1600, default=512
        ),
        height: int = Input(
            description="Height", 
            ge=64, le=1600, default=512
        ),
        sampler: str = Input(
            description="Which sampler to use", 
            default="klms", choices=["ddim", "plms", "klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]
        ),
        steps: int = Input(
            description="Diffusion steps", 
            ge=0, le=200, default=60
        ),
        scale: float = Input(
            description="Text conditioning scale", 
            ge=0, le=32, default=8.0
        ),
        # ddim_eta: float = 0.0
        # C: int = 4
        # f: int = 8   
        # dynamic_threshold: float = None
        # static_threshold: float = None
        upscale_f: int = Input(
            description="Upscaling resolution",
            default = 1, choices=[1, 2]
        ),

        # Init image and mask
        init_image_data: str = Input(
            description="Load initial image from file, url, or base64 string", 
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
        mask_image_data: str = Input(
            description="Load mask image from file, url, or base64 string", 
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
            description="Text input (mode==generate)",
        ),
        uc_text: str = Input(
            description="Negative text input (mode==all)",
            default="poorly drawn face, ugly, tiling, out of frame, extra limbs, disfigured, deformed body, blurry, blurred, watermark, text, grainy, signature, cut off, draft"
        ),
        seed: int = Input(
            description="random seed (mode==generate)", 
            ge=0, le=1e10, default=13
        ),
        n_samples: int = Input(
            description="batch size (mode==generate)",
            ge=1, le=4, default=1
        ),

        # Interpolate / Animate mode
        n_frames: int = Input(
            description="Total number of frames (mode==interpolate/animate)",
            ge=0, le=100, default=50
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
        interpolation_init_images_use_img2txt: bool = Input(
            description="Use clip_search to get prompts for the init images, if false use manual interpolation_texts (mode==interpolate)",
            default=False
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
            ge=0.0, le=10, default=0.0
        ),
        loop: bool = Input(
            description="Loops (mode==interpolate)",
            default=True
        ),
        smooth: bool = Input(
            description="Smooth (mode==interpolate)",
            default=False
        ),
        n_film: int = Input(
            description="Number of times to smooth final frames with FILM (default is 0) (mode==interpolate or animate)",
            default=0, ge=0, le=2
        ),
        fps: int = Input(
            description="Frames per second (mode==interpolate or animate)",
            default=12, ge=1, le=60
        ),
        
        # Animation mode
        animation_mode: str = Input(
            description="Interpolation texts (mode==interpolate)",
            default='2D', choices=['2D', '3D', 'Video Input']
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

    ) -> Iterator[CogOutput]:

        import generation
        
        interpolation_texts = interpolation_texts.split('|') if interpolation_texts else None
        interpolation_seeds = [float(i) for i in interpolation_seeds.split('|')] if interpolation_seeds else None
        interpolation_init_images = interpolation_init_images.split('|') if interpolation_init_images else None

        args = StableDiffusionSettings(
            config = self.config_path,
            ckpt = self.ckpt_path,
            precision= 'autocast',
            half_precision = self.half_precision,

            mode = mode,

            W = width - (width % 64),
            H = height - (height % 64),
            sampler = sampler,
            steps = steps,
            scale = scale,
            upscale_f = float(upscale_f),

            init_image_data = init_image_data,
            init_image_strength = init_image_strength,
            init_image_inpaint_mode = init_image_inpaint_mode,
            mask_image_data = mask_image_data,
            mask_invert = mask_invert,

            text_input = text_input,
            uc_text = uc_text,
            seed = seed,
            n_samples = n_samples,

            interpolation_texts = interpolation_texts,
            interpolation_seeds = interpolation_seeds,
            interpolation_init_images = interpolation_init_images,
            interpolation_init_images_use_img2txt = interpolation_init_images_use_img2txt,
            interpolation_init_images_top_k = interpolation_init_images_top_k,
            interpolation_init_images_power = interpolation_init_images_power,
            interpolation_init_images_min_strength = interpolation_init_images_min_strength,

            n_frames = n_frames,
            scale_modulation = scale_modulation,
            loop = loop,
            smooth = smooth,
            n_film = n_film,
            fps = fps,

            aesthetic_target = None, # None means we'll use the init_images as target
            aesthetic_steps = 10,
            aesthetic_lr = 0.0001,
            ag_L2_normalization_constant = 0.25, # for real2real, only 

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

        print(args)

        out_dir = Path(tempfile.mkdtemp())

        if mode == "interrogate":
            interrogation = generation.interrogate(args)
            out_path = out_dir / f"interrogation.txt"
            with open(out_path, 'w') as f:
                f.write(interrogation)
            attributes = {'interrogation': interrogation}
            yield CogOutput(file=out_path, thumbnail=None, attributes=attributes)

        elif mode == "generate" or mode == "remix":

            steps_per_update = stream_every if stream else None

            generator = generation.make_images(args, steps_per_update=steps_per_update)
            
            attributes = {}

            for frames, t in generator:                
                for f, frame in enumerate(frames):
                    out_path = out_dir / f"frame_{f:02}_{t:016}.jpg"
                    frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                    yield CogOutput(file=out_path, thumbnail=out_path,attributes=attributes)
            
            if mode == "remix":
                attributes = {"interrogation": args.text_input}

            yield CogOutput(file=out_path, thumbnail=out_path, attributes=attributes)
            
        else:

            if mode == "interpolate":
                generator = generation.make_interpolation(args)

            elif mode == "real2real":
                args.interpolation_init_images_use_img2txt = True
                generator = generation.make_interpolation(args)

            # elif mode == "animate":
            #     generator = generation.make_animation(args)

            attributes = {}
            thumbnail = None

            # generate frames
            for frame, t_raw in generator:
                out_path = out_dir / ("frame_%0.16f.jpg" % t_raw)
                frame.save(out_path, format='JPEG', subsampling=0, quality=95)
                if not thumbnail:
                    thumbnail = out_path
                if stream:
                    yield CogOutput(file=out_path, thumbnail=out_path, attributes=attributes)

            # run FILM
            if args.n_film > 0:
                film.interpolate_FILM(str(out_dir), n_film)
                out_dir = out_dir / "interpolated_frames"

            # save video
            loop = (args.loop and len(args.interpolation_seeds) == 2)
            out_path = out_dir / "out.mp4"
            eden_utils.write_video(out_dir, str(out_path), loop=loop, fps=args.fps)

            if mode == "real2real":
                attributes["interrogation"] = args.interpolation_texts

            yield CogOutput(file=out_path, thumbnail=thumbnail, attributes=attributes)
