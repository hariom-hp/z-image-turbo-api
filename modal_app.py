# pyright: reportMissingImports=false, reportAttributeAccessIssue=false
# pyright: reportArgumentType=false, reportPrivateImportUsage=false
"""Z-Image Interior Design API — Modal Labs (A10G GPU).

Image-to-Image editing using the official diffusers ZImageImg2ImgPipeline:
  1. Load input image
  2. Run ZImageImg2ImgPipeline with prompt + strength
  3. Return generated image

  Official params: steps=9, guidance_scale=0.0, strength=0.6-0.8
  Uses the official HuggingFace diffusers pipeline (no VideoX-Fun dependency).

Deploy:   modal deploy modal_app.py
Serve:    modal serve modal_app.py   (dev mode with hot reload)
"""

from __future__ import annotations

import modal

# ─── Modal App ────────────────────────────────────────────────────────────────

app = modal.App("z-image-interior-design")

# ─── Images ───────────────────────────────────────────────────────────────────
#
# TWO separate images:
#   gpu_image  — Z-Image-Turbo model (only used by ZImageInference)
#   web_image  — ~50MB, just FastAPI + pydantic (used by the web server)

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


def download_models():
    """Pre-download Z-Image-Turbo model weights (cached by Modal)."""
    from diffusers import ZImageImg2ImgPipeline
    import torch

    print(f"Downloading {MODEL_ID}...")
    pipe = ZImageImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    print(f"✓ {MODEL_ID} downloaded and verified")
    del pipe


gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "git")
    .pip_install(
        "torch>=2.5.0",
        "torchvision>=0.20.0",
        "transformers>=4.53.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "huggingface_hub>=0.25.0",
        "fastapi[standard]",
        "einops",
        "sentencepiece",
        # Latest diffusers from source (for ZImageImg2ImgPipeline)
        "git+https://github.com/huggingface/diffusers.git",
    )
    .run_function(download_models)
)

web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("fastapi[standard]", "pydantic>=2.0")
)


# ─── Static Data (no GPU needed) ─────────────────────────────────────────────

STYLE_TEMPLATES: dict[str, str] = {
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

ROOM_TEMPLATES: dict[str, str] = {
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


def build_prompt(
    user_prompt: str,
    style: str | None = None,
    room_type: str | None = None,
) -> str:
    parts: list[str] = []
    if style and style in STYLE_TEMPLATES:
        parts.append(STYLE_TEMPLATES[style])
    if room_type and room_type in ROOM_TEMPLATES:
        parts.append(ROOM_TEMPLATES[room_type])
    if user_prompt:
        parts.append(user_prompt)
    return ", ".join(parts) + QUALITY_SUFFIX


# ─── GPU Inference Class ──────────────────────────────────────────────────────
#
# Only spawned when /api/interior/redesign is called.
# Image-to-image with Z-Image-Turbo using official diffusers pipeline.


@app.cls(
    image=gpu_image,
    gpu="A10G",
    timeout=120,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=2)
class ZImageInference:
    """Image-to-image editing with Z-Image-Turbo on A10G GPU.

    Uses the official diffusers ZImageImg2ImgPipeline.
    Z-Image-Turbo fits comfortably within 24GB VRAM.
    """

    @modal.enter()
    def load_models(self):
        import torch
        import time

        start = time.time()

        print("=" * 60)
        print("Loading Z-Image-Turbo (official diffusers pipeline)")
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        print("=" * 60)

        from diffusers import ZImageImg2ImgPipeline

        self.pipe = ZImageImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")

        # Memory optimizations
        self.pipe.enable_attention_slicing("auto")

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        vram = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Ready in {time.time() - start:.1f}s — VRAM {vram:.1f}/{total:.1f} GB")

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _resize(image, max_dim=1024):
        """Resize image so max dimension ≤ max_dim, aligned to 16px."""
        from PIL import Image as PILImage

        w, h = image.size
        ratio = min(max_dim / max(w, h), 1.0)
        nw = max(16, int(w * ratio) // 16 * 16)
        nh = max(16, int(h * ratio) // 16 * 16)
        if nw != w or nh != h:
            image = image.resize((nw, nh), PILImage.LANCZOS)
        return image

    @staticmethod
    def _to_b64(image) -> str:
        import base64
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _from_b64(s: str):
        import base64
        import io
        from PIL import Image as PILImage

        return PILImage.open(io.BytesIO(base64.b64decode(s)))

    # ── The one and only inference method ─────────────────────────────────

    @modal.method()
    def redesign(
        self,
        image_b64: str,
        prompt: str,
        strength: float = 0.67,
        total_steps: int = 8,
        seed: int = -1,
        max_dim: int = 1024,
    ) -> dict:
        """Room redesign using img2img with Z-Image-Turbo.

        Uses the official diffusers ZImageImg2ImgPipeline.

        strength controls how much the image changes:
          0.4 → subtle changes, more original preserved
          0.6 → balanced style change (recommended)
          0.8 → strong style change, rough structure kept
          0.95 → almost complete regeneration
        """
        import torch
        import time
        import gc

        t0 = time.time()

        img = self._from_b64(image_b64).convert("RGB")
        img = self._resize(img, max_dim)
        w, h = img.size

        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        gen = torch.Generator(device="cuda").manual_seed(seed)

        print(f"Img2img: {w}x{h} strength={strength} steps={total_steps} seed={seed}")

        try:
            result = self.pipe(
                prompt=prompt,
                image=img,
                strength=strength,
                num_inference_steps=total_steps,
                guidance_scale=0.0,
                generator=gen,
            )
            out = result.images[0]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            print(f"OOM at {w}x{h}, retrying at 768…")
            gc.collect()
            torch.cuda.empty_cache()
            img = self._resize(img, 768)
            w, h = img.size
            gen = torch.Generator(device="cuda").manual_seed(seed)
            result = self.pipe(
                prompt=prompt,
                image=img,
                strength=strength,
                num_inference_steps=total_steps,
                guidance_scale=0.0,
                generator=gen,
            )
            out = result.images[0]

        dt = time.time() - t0
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Done {out.width}x{out.height} in {dt:.1f}s seed={seed}")

        return {
            "success": True,
            "image": self._to_b64(out),
            "width": out.width,
            "height": out.height,
            "generation_time_ms": int(dt * 1000),
            "seed": seed,
            "prompt_used": prompt[:200],
            "strength": strength,
            "device": "cuda-l4",
            "mode": "img2img",
        }


# ─── FastAPI (runs on web_image — NO GPU, NO models) ─────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

web_app = FastAPI(title="Z-Image Interior Design API", version="4.0.0")
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RedesignRequest(BaseModel):
    image: str
    prompt: str = ""
    style: str | None = None
    room_type: str | None = None
    strength: float = 0.7
    steps: int = 8
    seed: int = -1
    max_dim: int = 1024


# ── /health — instant, no GPU ─────────────────────────────────────────────────

@web_app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "z-image-turbo-img2img",
        "version": "4.0.0",
        "gpu": "NVIDIA A10G (24GB)",
        "device": "cuda-a10g",
        "mode": "image-to-image",
        "pipeline": "diffusers.ZImageImg2ImgPipeline",
    }


# ── /styles — instant, no GPU ─────────────────────────────────────────────────

@web_app.get("/api/interior/styles")
async def styles():
    return {
        "styles": list(STYLE_TEMPLATES.keys()),
        "room_types": list(ROOM_TEMPLATES.keys()),
    }


# ── /redesign — this is the ONLY endpoint that calls the GPU ──────────────────

@web_app.post("/api/interior/redesign")
async def redesign(req: RedesignRequest):
    if not req.image:
        raise HTTPException(400, "Missing 'image' (base64)")
    if not req.prompt and not req.style and not req.room_type:
        raise HTTPException(400, "Provide at least one of: prompt, style, room_type")

    prompt = build_prompt(req.prompt, req.style, req.room_type)

    inference = ZImageInference()
    return inference.redesign.remote(
        image_b64=req.image,
        prompt=prompt,
        strength=req.strength,
        total_steps=req.steps,
        seed=req.seed,
        max_dim=req.max_dim,
    )

# ── /generate — LoRA-compatible endpoint for Flutter app ──────────────────────
# This matches the EXACT same request/response format as the LoRA Go API,
# so swapping the URL in Firebase Remote Config works without Flutter changes.

class GenerateRequest(BaseModel):
    image: str                                          # Base64 encoded image
    prompt: str                                         # User prompt
    negative_prompt: str = ""                           # Ignored (Turbo doesn't use it)
    width: int | None = None                            # Ignored (auto from input)
    height: int | None = None                           # Ignored (auto from input)
    preserve_aspect_ratio: bool | None = True           # Always preserved
    seed: int | None = None                             # Optional seed
    strength: float | None = None                       # Default 0.7
    mode: str | None = None                             # "style", "refine", "redesign"
    guidance_scale: float | None = None                 # Ignored (Turbo uses 0.0)
    controlnet_scale: float | None = None               # Ignored (no ControlNet in Z-Image)
    num_inference_steps: int | None = None              # Default 8


@web_app.post("/generate")
async def generate(req: GenerateRequest):
    """LoRA-compatible endpoint — same format as the old SD1.5+LoRA API.

    Flutter app calls this via Firebase Remote Config URL swap.
    Request/response format matches go__apis/interior-api/main.go exactly.
    """
    if not req.image:
        raise HTTPException(400, "Missing 'image' (base64)")
    if not req.prompt:
        raise HTTPException(400, "Missing 'prompt'")

    # Strip data:image/... prefix if present (Flutter might send it)
    image_b64 = req.image
    if image_b64.startswith("data:"):
        image_b64 = image_b64.split(",", 1)[-1]

    # Map LoRA params → Z-Image params
    strength = 0.7
    steps = req.num_inference_steps if req.num_inference_steps is not None else 8
    seed = req.seed if req.seed is not None else -1

    # Build prompt with quality suffix
    prompt = req.prompt + QUALITY_SUFFIX

    inference = ZImageInference()
    result = inference.redesign.remote(
        image_b64=image_b64,
        prompt=prompt,
        strength=strength,
        total_steps=steps,
        seed=seed,
        max_dim=1024,
    )

    if not result.get("success"):
        return {
            "success": False,
            "error": result.get("error", "Inference failed"),
        }

    # Return in EXACT same format as LoRA Go API (with data: prefix)
    return {
        "success": True,
        "image": "data:image/png;base64," + result["image"],
        "seed": result.get("seed", 0),
        "width": result.get("width", 0),
        "height": result.get("height", 0),
        "detected_mode": "z-image-turbo",
        "mode_description": "Z-Image Turbo img2img",
    }


# ── POST / — alias for /generate (Flutter app hits root path) ─────────────────

@web_app.post("/")
async def root_generate(req: GenerateRequest):
    """Root POST alias — Flutter app sends requests to / directly."""
    return await generate(req)


# ── Mount the web app on the LIGHTWEIGHT image ───────────────────────────────

@app.function(image=web_image)
@modal.asgi_app()
def fastapi_app():
    return web_app
