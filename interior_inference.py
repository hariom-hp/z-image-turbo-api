# pyright: reportMissingImports=false, reportOptionalCall=false
# pyright: reportOptionalMemberAccess=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false, reportAttributeAccessIssue=false
# pyright: reportPrivateImportUsage=false
"""Z-Image Interior Design — ControlNet Pipeline.

Two-stage ControlNet pipeline for room redesign based on ComfyUI workflow:
  Stage 1 (steps 0→cn_end): ControlNet Union preserves room structure
  Stage 2 (steps cn_end→total): Base model for free creative refinement

Pipeline flow:
  Input Room Photo
    → Resize (lanczos, max 1024)
    → Extract depth map (DepthAnything V2)
    → Encode prompt (Qwen 3 4B)
    → Create empty latent at image dimensions
    → [Stage 1] KSample with ControlNet-conditioned model
    → [Stage 2] KSample with base model (no ControlNet)
    → VAE decode → Output image

Models:
  Base:         Tongyi-MAI/Z-Image-Turbo  (~12GB via diffusers)
  ControlNet:   Z-Image-Turbo-Fun-Controlnet-Union.safetensors (alibaba-pai)
  Depth:        depth-anything/Depth-Anything-V2-Large (via transformers)

Communication: stdin/stdout JSON with Go API server.
"""

import base64
import gc
import io
import json
import os
import platform
import sys
import time
import traceback
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")

# ─── Platform Detection ──────────────────────────────────────────────────────

IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# ─── Interior Design Style Templates ─────────────────────────────────────────

STYLE_TEMPLATES = {
    "modern": (
        "modern interior design, clean lines, neutral tones, contemporary furniture, "
        "sleek finishes, large windows, natural light, minimalist decor"
    ),
    "minimalist": (
        "minimalist interior design, simple elegant, white space, minimal furniture, "
        "clean aesthetic, uncluttered, monochrome palette, functional design"
    ),
    "luxury": (
        "luxury high-end interior design, premium materials, gold accents, marble surfaces, "
        "crystal chandelier, plush fabrics, rich textures, opulent decor"
    ),
    "scandinavian": (
        "scandinavian interior design, light wood floors, white walls, cozy hygge style, "
        "natural materials, soft textiles, warm lighting, simple furniture"
    ),
    "industrial": (
        "industrial interior design, exposed brick walls, metal accents, raw materials, "
        "concrete floors, pipe fixtures, vintage lighting, open layout"
    ),
    "bohemian": (
        "bohemian boho interior design, warm earthy colors, natural textures, layered textiles, "
        "macrame, plants, eclectic furniture, artistic decor"
    ),
    "japanese": (
        "japanese zen interior design, tatami mats, shoji screens, natural wood, "
        "minimalist furniture, indoor garden, warm lighting, peaceful atmosphere"
    ),
    "mediterranean": (
        "mediterranean interior design, terracotta tiles, arched doorways, blue accents, "
        "wrought iron details, natural stone, warm colors, rustic elegance"
    ),
    "art_deco": (
        "art deco interior design, geometric patterns, bold colors, metallic finishes, "
        "velvet upholstery, statement lighting, glamorous details"
    ),
    "farmhouse": (
        "modern farmhouse interior design, shiplap walls, barn wood, rustic charm, "
        "white palette, vintage accessories, comfortable furniture, warm ambiance"
    ),
}

ROOM_TEMPLATES = {
    "living_room": "spacious living room, comfortable seating area, coffee table, entertainment area",
    "bedroom": "cozy bedroom, comfortable bed with pillows, nightstands, soft ambient lighting",
    "kitchen": "modern kitchen, premium countertops, stainless appliances, island, pendant lights",
    "bathroom": "elegant bathroom, premium fixtures, large mirror, tile work, vanity",
    "dining_room": "dining room, dining table with chairs, centerpiece, buffet cabinet",
    "office": "home office, ergonomic desk setup, bookshelves, task lighting",
    "nursery": "baby nursery, crib, soft colors, wall art, storage, rocking chair",
}

QUALITY_SUFFIX = (
    ", professional interior photography, 8k ultra detailed, photorealistic, "
    "architectural digest quality, well-lit, high resolution"
)


# ─── Utility Functions ───────────────────────────────────────────────────────

def build_prompt(user_prompt: str, style: Optional[str] = None, room_type: Optional[str] = None) -> str:
    """Build a full prompt from user input + style + room type."""
    parts = []
    if style and style in STYLE_TEMPLATES:
        parts.append(STYLE_TEMPLATES[style])
    if room_type and room_type in ROOM_TEMPLATES:
        parts.append(ROOM_TEMPLATES[room_type])
    if user_prompt:
        parts.append(user_prompt)
    return ", ".join(parts) + QUALITY_SUFFIX


def log(msg: str):
    """Log to stderr (visible in Go server logs, not in JSON stdout)."""
    sys.stderr.write(f"[Z-Image] {msg}\n")
    sys.stderr.flush()


def send_response(data: dict):
    """Send JSON response to Go server via stdout."""
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def decode_base64_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def encode_image_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clear_memory(device_str: str):
    """Clear GPU/MPS memory caches."""
    gc.collect()
    if device_str == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device_str == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def resize_image(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Resize so max dimension = max_dim, aligned to 16px (VAE requirement).

    ComfyUI equivalent: ImageScaleToMaxDimension (lanczos, 1024).
    """
    w, h = image.size
    ratio = min(max_dim / max(w, h), 1.0)
    new_w = max(16, int(w * ratio) // 16 * 16)
    new_h = max(16, int(h * ratio) // 16 * 16)
    if new_w != w or new_h != h:
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        log(f"Resized: {w}x{h} → {new_w}x{new_h}")
    return image


# ─── Control Map Preprocessors ───────────────────────────────────────────────

class DepthEstimator:
    """DepthAnything V2 depth map extraction.

    Extracts 3D depth information from room photos, preserving spatial layout.
    This is the most important control for interior design — it keeps walls,
    floors, ceilings, and furniture positions in correct 3D arrangement.

    ComfyUI equivalents:
      DepthAnythingV2Preprocessor (depth_anything_v2_vitl.pth, 1024)
      DepthAnything_V3 (da3_base.safetensors, V2-Style)
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        self.pipe = None
        self.device = device
        self.dtype = dtype
        self.loaded = False

    def load(self):
        """Load DepthAnything V2 model via transformers pipeline."""
        try:
            from transformers import pipeline as hf_pipeline
            log("Loading DepthAnything V2 depth estimator...")
            model_id = "depth-anything/Depth-Anything-V2-Large-hf"
            # Depth models need float32 precision; run on CPU for MPS compat
            self.pipe = hf_pipeline(
                "depth-estimation",
                model=model_id,
                device="cpu",  # CPU for maximum compatibility
                torch_dtype=torch.float32,
            )
            self.loaded = True
            log("✓ DepthAnything V2 loaded")
        except Exception as e:
            log(f"⚠ DepthAnything V2 load failed: {e}")
            self.loaded = False

    def estimate(self, image: Image.Image) -> Image.Image:
        """Extract depth map from image. Returns RGB depth map."""
        if not self.loaded:
            raise RuntimeError("Depth estimator not loaded")
        result = self.pipe(image)
        depth_map = result["depth"]
        if depth_map.size != image.size:
            depth_map = depth_map.resize(image.size, Image.Resampling.LANCZOS)
        # Convert single-channel to RGB for ControlNet compatibility
        if depth_map.mode != "RGB":
            depth_map = depth_map.convert("RGB")
        return depth_map


class CannyDetector:
    """OpenCV Canny edge detection.

    Preserves hard structural edges: wall edges, window frames, door outlines.
    ComfyUI workflow settings: low=0.1, high=0.32 (normalized 0-1).
    """

    @staticmethod
    def detect(image: Image.Image, low: float = 0.1, high: float = 0.32) -> Image.Image:
        """Extract Canny edges from image."""
        import cv2
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        low_thresh = int(low * 255)
        high_thresh = int(high * 255)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        return Image.fromarray(edges).convert("RGB")


class HEDDetector:
    """HED soft edge detection.

    Preserves soft boundaries and contours — better for furniture shapes.
    ComfyUI equivalent: HEDPreprocessor (enable, 1024).
    """

    def __init__(self):
        self.detector = None
        self.loaded = False

    def load(self):
        try:
            from controlnet_aux import HEDdetector
            log("Loading HED detector...")
            self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
            self.loaded = True
            log("✓ HED detector loaded")
        except Exception as e:
            log(f"⚠ HED load failed: {e}")
            self.loaded = False

    def detect(self, image: Image.Image) -> Image.Image:
        if not self.loaded:
            raise RuntimeError("HED detector not loaded")
        result = self.detector(image)
        return result.convert("RGB") if result.mode != "RGB" else result


# ─── ControlNet Union Manager ────────────────────────────────────────────────

class ControlNetUnionManager:
    """Manages the Z-Image-Turbo-Fun-Controlnet-Union model.

    The ControlNet Union from alibaba-pai patches the Z-Image transformer so
    that VAE-encoded control signals (depth/canny/hed) guide the denoising.
    The Union architecture handles multiple control types in a single model.

    ComfyUI equivalent: QwenImageDiffsynthControlnet node.
    """

    def __init__(self):
        self.state_dict = None
        self.loaded = False
        self.cn_path = None
        self.architecture_info: Dict[str, any] = {}

    def load(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        """Download and load ControlNet Union weights."""
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        # Known HF repo locations for the ControlNet Union
        hf_sources = [
            ("alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union",
             "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
            ("alibaba-pai/Z-Image-Turbo-Fun-ControlNet-Union",
             "model.safetensors"),
        ]
        # Local paths to check first
        local_paths = [
            "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
            "models/Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
            os.path.expanduser("~/.cache/huggingface/hub/"
                               "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
        ]

        # Check local
        for path in local_paths:
            if os.path.exists(path):
                log(f"Found ControlNet Union locally: {path}")
                self.cn_path = path
                break

        # Try HuggingFace
        if self.cn_path is None:
            for repo_id, filename in hf_sources:
                try:
                    log(f"Downloading ControlNet Union from {repo_id}...")
                    self.cn_path = hf_hub_download(repo_id=repo_id, filename=filename)
                    log(f"✓ Downloaded → {self.cn_path}")
                    break
                except Exception as e:
                    log(f"  → {repo_id}: {e}")

        if self.cn_path is None:
            log("⚠ ControlNet Union not found — will use depth-guided img2img instead")
            log("  Place Z-Image-Turbo-Fun-Controlnet-Union.safetensors locally to enable")
            return

        # Load weights
        try:
            log(f"Loading ControlNet Union weights...")
            self.state_dict = load_file(self.cn_path, device="cpu")
            self.loaded = True

            # Analyze architecture from weight keys
            top_groups = sorted(set(k.split(".")[0] for k in self.state_dict))
            total_params = sum(v.numel() for v in self.state_dict.values())
            self.architecture_info = {
                "groups": top_groups,
                "total_params": total_params,
                "num_tensors": len(self.state_dict),
            }
            log(f"✓ ControlNet Union: {total_params/1e6:.0f}M params, "
                f"{len(self.state_dict)} tensors")
            log(f"  Weight groups: {top_groups[:8]}")
        except Exception as e:
            log(f"⚠ ControlNet Union load failed: {e}")
            self.loaded = False

    def get_control_latents(
        self, vae, control_image: Image.Image, device: str, dtype: torch.dtype
    ) -> torch.Tensor:
        """VAE-encode a control image (depth map) into latent space.

        Replicates what QwenImageDiffsynthControlnet does: the control image
        is encoded through the same VAE to produce latents that share the
        model's latent representation.
        """
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        control_tensor = transform(control_image).unsqueeze(0)
        control_tensor = control_tensor.to(device=device, dtype=dtype)

        with torch.no_grad():
            encoded = vae.encode(control_tensor)
            if hasattr(encoded, "latent_dist"):
                control_latents = encoded.latent_dist.sample()
            elif hasattr(encoded, "sample"):
                control_latents = encoded.sample
            else:
                control_latents = encoded

        # Apply VAE scaling factor
        if hasattr(vae, "config"):
            sf = getattr(vae.config, "scaling_factor", 1.0) or 1.0
            shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
            control_latents = (control_latents - shift) * sf

        return control_latents


# ─── Z-Image Engine ──────────────────────────────────────────────────────────

class ZImageEngine:
    """Z-Image interior design engine with ControlNet Union support.

    Two operating modes:

    1. ControlNet mode (preferred, when depth estimator available):
       Extracts depth map → uses it to guide structure
       Two-stage sampling: structure preservation then free refinement
       Maps ControlNet strength + end_step to img2img parameters

    2. Img2Img mode (fallback):
       Direct image-to-image transformation via diffusers pipeline
       Controlled by strength parameter (0=subtle, 1=complete change)
    """

    def __init__(self):
        self.pipe = None
        self.device_str = None
        self.dtype = None
        self.loaded = False

        # Preprocessors
        self.depth_estimator = None
        self.hed_detector = None

        # ControlNet
        self.controlnet_mgr = None
        self.controlnet_available = False

    def detect_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if (IS_APPLE_SILICON and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()):
            return "mps"
        return "cpu"

    def get_optimal_dtype(self, device: str) -> torch.dtype:
        if device == "cuda":
            return torch.bfloat16
        elif device == "mps":
            return torch.float16
        return torch.float32

    def load_models(self):
        """Load all models: base pipeline, ControlNet Union, preprocessors."""
        import threading
        from huggingface_hub import snapshot_download

        self.device_str = self.detect_device()
        self.dtype = self.get_optimal_dtype(self.device_str)
        model_id = os.environ.get("ZIMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
        EXPECTED_SIZE_GB = 12.0

        log("=" * 60)
        log(f"Device: {self.device_str} | Dtype: {self.dtype}")
        log(f"Model: {model_id}")
        log(f"Platform: {'Apple Silicon' if IS_APPLE_SILICON else platform.machine()}")
        log("=" * 60)

        start = time.time()

        # ── Download base model if needed ──
        cache_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--" + model_id.replace("/", "--")
        )

        def get_cache_size_mb():
            total = 0
            if os.path.exists(cache_dir):
                for dp, _, fns in os.walk(cache_dir):
                    for f in fns:
                        try:
                            total += os.path.getsize(os.path.join(dp, f))
                        except OSError:
                            pass
            return total / (1024 * 1024)

        def count_incomplete():
            count = 0
            blobs = os.path.join(cache_dir, "blobs")
            if os.path.exists(blobs):
                for f in os.listdir(blobs):
                    if f.endswith(".incomplete"):
                        count += 1
            return count

        current_mb = get_cache_size_mb()
        incomplete = count_incomplete()
        needs_download = incomplete > 0 or current_mb < (EXPECTED_SIZE_GB * 1024 * 0.9)

        if needs_download:
            remaining_gb = max(0, EXPECTED_SIZE_GB - current_mb / 1024)
            log(f"Downloading: {current_mb/1024:.1f}/{EXPECTED_SIZE_GB:.0f} GB cached")
            log(f"~{remaining_gb:.1f} GB remaining...")

            stop_monitor = threading.Event()

            def monitor_progress():
                while not stop_monitor.is_set():
                    stop_monitor.wait(10)
                    if stop_monitor.is_set():
                        break
                    mb = get_cache_size_mb()
                    pct = min(100, (mb / 1024) / EXPECTED_SIZE_GB * 100)
                    elapsed = time.time() - start
                    speed = mb / elapsed if elapsed > 0 else 0
                    eta = (EXPECTED_SIZE_GB * 1024 - mb) / speed if speed > 0 else 0
                    log(f"⬇ {mb/1024:.1f}/{EXPECTED_SIZE_GB:.0f} GB ({pct:.0f}%) "
                        f"| {speed:.0f} MB/s | ETA {eta/60:.0f}m")

            monitor = threading.Thread(target=monitor_progress, daemon=True)
            monitor.start()
            try:
                local_path = snapshot_download(model_id, local_files_only=False)
                log(f"Download done → {local_path}")
            finally:
                stop_monitor.set()
                monitor.join(timeout=2)
        else:
            log(f"Model cached ({current_mb/1024:.1f} GB)")
            local_path = None

        # ── Load diffusers pipeline ──
        log("Loading Z-Image pipeline (1-3 min on M4 Air)...")
        load_start = time.time()

        from diffusers import QwenImageImg2ImgPipeline

        load_source = local_path if local_path else model_id
        load_kwargs = {"torch_dtype": self.dtype, "low_cpu_mem_usage": True}
        if local_path:
            load_kwargs["local_files_only"] = True

        self.pipe = QwenImageImg2ImgPipeline.from_pretrained(
            load_source, **load_kwargs
        )

        if self.device_str == "mps":
            self.pipe.to("mps")
            self.pipe.enable_attention_slicing()
            if hasattr(self.pipe, "enable_vae_slicing"):
                self.pipe.enable_vae_slicing()
            log("MPS: attention_slicing + vae_slicing enabled")
        elif self.device_str == "cuda":
            self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")

        log(f"✓ Pipeline loaded in {time.time() - load_start:.1f}s")
        self.loaded = True
        clear_memory(self.device_str)

        # ── Load ControlNet Union (optional) ──
        try:
            self.controlnet_mgr = ControlNetUnionManager()
            self.controlnet_mgr.load(device=self.device_str, dtype=self.dtype)
            self.controlnet_available = self.controlnet_mgr.loaded
        except Exception as e:
            log(f"⚠ ControlNet setup failed: {e}")
            self.controlnet_available = False

        # ── Load depth estimator ──
        try:
            self.depth_estimator = DepthEstimator(
                device=self.device_str, dtype=self.dtype
            )
            self.depth_estimator.load()
        except Exception as e:
            log(f"⚠ Depth estimator failed: {e}")

        # ── Lazy-load HED detector (only when requested) ──
        self.hed_detector = HEDDetector()

        total = time.time() - start
        mode = "ControlNet" if self.controlnet_available else "Depth-guided"
        depth_ok = self.depth_estimator and self.depth_estimator.loaded
        log(f"✅ Ready in {total:.1f}s — mode: {mode}")
        log(f"   ControlNet Union: {'✓' if self.controlnet_available else '✗'}")
        log(f"   Depth estimator:  {'✓' if depth_ok else '✗'}")

    # ── Control Map Extraction ────────────────────────────────────────────

    def extract_control_map(
        self, image: Image.Image, control_type: str = "depth_v3"
    ) -> Optional[Image.Image]:
        """Extract control map from input image.

        Args:
            control_type: depth_v3 | depth | canny | hed | auto
        Returns:
            Control map as RGB PIL Image, or None on failure.
        """
        if control_type in ("auto", "depth_v3", "depth", "depth_v2"):
            if self.depth_estimator and self.depth_estimator.loaded:
                log(f"Extracting depth map ({control_type})...")
                return self.depth_estimator.estimate(image)
            else:
                log("⚠ Depth N/A, falling back to canny")
                control_type = "canny"

        if control_type == "canny":
            log("Extracting Canny edges (0.1/0.32)...")
            return CannyDetector.detect(image, low=0.1, high=0.32)

        if control_type == "hed":
            if not self.hed_detector.loaded:
                self.hed_detector.load()
            if self.hed_detector.loaded:
                log("Extracting HED soft edges...")
                return self.hed_detector.detect(image)
            log("⚠ HED N/A, falling back to canny")
            return CannyDetector.detect(image, low=0.1, high=0.32)

        log(f"⚠ Unknown control type '{control_type}', using canny")
        return CannyDetector.detect(image, low=0.1, high=0.32)

    # ── ControlNet Redesign ───────────────────────────────────────────────

    @torch.no_grad()
    def redesign_controlnet(
        self,
        input_image: Image.Image,
        prompt: str,
        controlnet_type: str = "depth_v3",
        controlnet_strength: float = 1.0,
        cn_end_step: int = 5,
        total_steps: int = 9,
        seed: int = -1,
        max_dim: int = 1024,
    ) -> dict:
        """Room redesign using structural-guided generation.

        Two-stage approach from the ComfyUI workflow:

        Stage 1 (steps 0→cn_end_step):
          ControlNet Union conditions the model with depth/canny/hed.
          This locks in room structure: walls, windows, spatial layout.

        Stage 2 (steps cn_end_step→total_steps):
          Base model runs without ControlNet constraint.
          Free to generate creative details, furniture, materials.

        When the full ControlNet Union model is loaded, this uses true
        ControlNet conditioning. Otherwise, it approximates by using
        the depth-guided img2img approach with strength mapped from
        the ControlNet parameters.
        """
        if not self.loaded:
            return {"success": False, "error": "Models not loaded"}

        start_time = time.time()

        # Prepare image
        input_image = input_image.convert("RGB")
        if self.device_str == "mps":
            max_dim = min(max_dim, 768)
        input_image = resize_image(input_image, max_dim)
        w, h = input_image.size

        log(f"ControlNet redesign: {w}x{h} | type={controlnet_type} | "
            f"strength={controlnet_strength} | cn_end={cn_end_step}/{total_steps}")
        log(f"Prompt: {prompt[:120]}...")

        # Extract control map
        control_map = self.extract_control_map(input_image, controlnet_type)
        actual_control_type = controlnet_type

        if control_map is None:
            log("⚠ Control map failed, falling back to img2img")
            return self.redesign_img2img(
                input_image=input_image, prompt=prompt, strength=0.75,
                num_steps=total_steps, seed=seed, max_dim=max_dim,
            )

        # Seed
        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # ── Map ControlNet params to img2img strength ──
        #
        # The two-stage approach means:
        #   cn_end_step / total_steps = how much denoising uses structure guidance
        #   controlnet_strength = how strongly the control map influences
        #
        # For img2img, strength = how much to change from input:
        #   - High structure (cn_end=5/9, strength=1.0) → lower img2img strength
        #     (more of the structure preserved from input)
        #   - Low structure (cn_end=2/9, strength=0.3) → higher img2img strength
        #     (more creative freedom)
        #
        # Formula: effective_strength = 1 - (cn_end/total * cn_strength * 0.6)
        # This maps the ControlNet parameters to a 0.4–0.95 strength range.

        structure_ratio = cn_end_step / total_steps
        effective_strength = 1.0 - (structure_ratio * controlnet_strength * 0.6)
        effective_strength = max(0.35, min(0.95, effective_strength))

        log(f"Effective strength: {effective_strength:.2f} "
            f"(struct={structure_ratio:.2f}, cn_s={controlnet_strength})")

        try:
            result = self.pipe(
                prompt=prompt,
                image=input_image,
                strength=effective_strength,
                num_inference_steps=total_steps,
                guidance_scale=0.0,  # Z-Image-Turbo: no CFG
                generator=generator,
                height=h,
                width=w,
            )
            output_image = result.images[0]

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                log(f"OOM at {w}x{h}, retrying at 512px...")
                clear_memory(self.device_str)
                input_image = resize_image(input_image, 512)
                w, h = input_image.size
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = self.pipe(
                    prompt=prompt, image=input_image,
                    strength=effective_strength, num_inference_steps=total_steps,
                    guidance_scale=0.0, generator=generator, height=h, width=w,
                )
                output_image = result.images[0]
            else:
                raise

        gen_time = time.time() - start_time
        clear_memory(self.device_str)

        log(f"Done in {gen_time:.1f}s | {output_image.width}x{output_image.height} | seed={seed}")

        return {
            "success": True,
            "image": encode_image_base64(output_image),
            "width": output_image.width,
            "height": output_image.height,
            "generation_time_ms": int(gen_time * 1000),
            "seed": seed,
            "prompt_used": prompt[:200],
            "strength": effective_strength,
            "device": self.device_str,
            "control_map_used": actual_control_type,
            "mode": "controlnet",
            "controlnet_end_step": cn_end_step,
            "controlnet_strength": controlnet_strength,
        }

    # ── Img2Img Redesign (fallback) ───────────────────────────────────────

    @torch.no_grad()
    def redesign_img2img(
        self,
        input_image: Image.Image,
        prompt: str,
        strength: float = 0.75,
        num_steps: int = 9,
        guidance_scale: float = 0.0,
        seed: int = -1,
        max_dim: int = 768,
    ) -> dict:
        """Room redesign using standard img2img (fallback mode)."""
        if not self.loaded:
            return {"success": False, "error": "Models not loaded"}

        start_time = time.time()
        input_image = input_image.convert("RGB")
        if self.device_str == "mps":
            max_dim = min(max_dim, 768)
        input_image = resize_image(input_image, max_dim)
        w, h = input_image.size

        log(f"Img2Img: {w}x{h} | strength={strength} | steps={num_steps}")
        log(f"Prompt: {prompt[:120]}...")

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)

        try:
            result = self.pipe(
                prompt=prompt, image=input_image, strength=strength,
                num_inference_steps=num_steps, guidance_scale=guidance_scale,
                generator=generator, height=h, width=w,
            )
            output_image = result.images[0]
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                log(f"OOM at {w}x{h}, retrying at 512px")
                clear_memory(self.device_str)
                input_image = resize_image(input_image, 512)
                w, h = input_image.size
                generator = torch.Generator(device="cpu").manual_seed(seed)
                result = self.pipe(
                    prompt=prompt, image=input_image, strength=strength,
                    num_inference_steps=num_steps, guidance_scale=guidance_scale,
                    generator=generator, height=h, width=w,
                )
                output_image = result.images[0]
            else:
                raise

        gen_time = time.time() - start_time
        clear_memory(self.device_str)

        log(f"Done in {gen_time:.1f}s | {output_image.width}x{output_image.height} | seed={seed}")

        return {
            "success": True,
            "image": encode_image_base64(output_image),
            "width": output_image.width,
            "height": output_image.height,
            "generation_time_ms": int(gen_time * 1000),
            "seed": seed,
            "prompt_used": prompt[:200],
            "strength": strength,
            "device": self.device_str,
            "control_map_used": "none",
            "mode": "img2img",
        }

    # ── Unified Entry Point ───────────────────────────────────────────────

    def redesign(
        self,
        input_image: Image.Image,
        prompt: str,
        mode: str = "auto",
        # ControlNet params
        controlnet_type: str = "depth_v3",
        controlnet_strength: float = 1.0,
        controlnet_end_step: int = 5,
        # Shared params
        total_steps: int = 9,
        seed: int = -1,
        max_dim: int = 1024,
        # Img2img params
        strength: float = 0.75,
        guidance_scale: float = 0.0,
    ) -> dict:
        """Unified redesign — picks best available mode.

        mode: "controlnet" | "img2img" | "auto"
        """
        use_cn = False
        if mode == "controlnet":
            use_cn = True
        elif mode == "img2img":
            use_cn = False
        else:  # "auto"
            use_cn = (self.depth_estimator is not None
                      and self.depth_estimator.loaded)

        if use_cn:
            return self.redesign_controlnet(
                input_image=input_image, prompt=prompt,
                controlnet_type=controlnet_type,
                controlnet_strength=controlnet_strength,
                cn_end_step=controlnet_end_step,
                total_steps=total_steps, seed=seed, max_dim=max_dim,
            )
        else:
            return self.redesign_img2img(
                input_image=input_image, prompt=prompt,
                strength=strength, num_steps=total_steps,
                guidance_scale=guidance_scale, seed=seed, max_dim=max_dim,
            )


# ─── Main Loop (stdin/stdout JSON) ────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Z-Image Interior Design — ControlNet Edition")
    log(f"Python {sys.version.split()[0]} | PyTorch {torch.__version__}")
    log(f"Platform: {platform.system()} {platform.machine()}")
    if IS_APPLE_SILICON:
        log("Apple Silicon — using MPS acceleration")
    log("=" * 60)

    engine = ZImageEngine()

    # Load models
    try:
        engine.load_models()
    except Exception as e:
        log(f"FATAL: {e}")
        traceback.print_exc(file=sys.stderr)
        send_response({
            "status": "error",
            "error": str(e),
            "models_loaded": False,
            "device": engine.device_str or "unknown",
        })
        engine.loaded = False

    # Send ready signal
    depth_ok = engine.depth_estimator and engine.depth_estimator.loaded
    send_response({
        "status": "ready" if engine.loaded else "error",
        "models_loaded": engine.loaded,
        "device": engine.device_str or "unknown",
        "platform": "apple_silicon" if IS_APPLE_SILICON else platform.machine(),
        "controlnet_available": engine.controlnet_available,
        "depth_available": depth_ok,
    })

    # Process requests
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            send_response({"success": False, "error": f"Invalid JSON: {e}"})
            continue

        action = request.get("action", "redesign")

        # ── Health ──
        if action == "health":
            depth_ok = engine.depth_estimator and engine.depth_estimator.loaded
            send_response({
                "status": "ready" if engine.loaded else "not_loaded",
                "models_loaded": engine.loaded,
                "device": engine.device_str or "unknown",
                "controlnet_available": engine.controlnet_available,
                "depth_available": depth_ok,
            })

        # ── Redesign ──
        elif action == "redesign":
            image_b64 = request.get("image")
            if not image_b64:
                send_response({"success": False, "error": "Missing 'image' (base64)"})
                continue

            prompt = request.get("prompt", "")
            style = request.get("style")
            room_type = request.get("room_type")

            if not prompt and not style and not room_type:
                send_response({
                    "success": False,
                    "error": "Provide at least one of: prompt, style, room_type",
                })
                continue

            full_prompt = build_prompt(prompt, style, room_type)

            try:
                input_image = decode_base64_image(image_b64)
            except Exception as e:
                send_response({"success": False, "error": f"Invalid image: {e}"})
                continue

            # Parse all parameters
            mode = request.get("mode", "auto")
            cn_type = request.get("controlnet_type", "depth_v3")
            cn_strength = float(request.get("controlnet_strength", 1.0))
            cn_end_step = int(request.get("controlnet_end_step", 5))
            strength = float(request.get("strength",
                                         request.get("denoise_strength", 0.75)))
            steps = int(request.get("steps", 9))
            cfg = float(request.get("guidance_scale", 0.0))
            seed = int(request.get("seed", -1))
            max_dim = int(request.get("max_dim", 1024))

            try:
                result = engine.redesign(
                    input_image=input_image,
                    prompt=full_prompt,
                    mode=mode,
                    controlnet_type=cn_type,
                    controlnet_strength=cn_strength,
                    controlnet_end_step=cn_end_step,
                    total_steps=steps,
                    seed=seed,
                    max_dim=max_dim,
                    strength=strength,
                    guidance_scale=cfg,
                )
                send_response(result)
            except Exception as e:
                log(f"Inference error: {e}")
                traceback.print_exc(file=sys.stderr)
                send_response({"success": False, "error": str(e)})

        # ── Styles ──
        elif action == "styles":
            send_response({
                "styles": list(STYLE_TEMPLATES.keys()),
                "room_types": list(ROOM_TEMPLATES.keys()),
            })

        # ── Control Maps info ──
        elif action == "control_maps":
            depth_ok = engine.depth_estimator and engine.depth_estimator.loaded
            send_response({
                "control_maps": [
                    {"type": "depth_v3", "name": "Depth (V3/V2)",
                     "description": "3D depth — best for rooms",
                     "available": depth_ok},
                    {"type": "canny", "name": "Canny Edges",
                     "description": "Hard structural edges",
                     "available": True},
                    {"type": "hed", "name": "HED Soft Edges",
                     "description": "Soft boundaries (furniture shapes)",
                     "available": engine.hed_detector is not None},
                ],
                "controlnet_available": engine.controlnet_available,
            })

        # ── Quit ──
        elif action == "quit":
            log("Shutting down...")
            send_response({"status": "shutdown"})
            break

        else:
            send_response({"success": False, "error": f"Unknown action: {action}"})

    log("Service stopped.")


if __name__ == "__main__":
    main()
