import os
from uuid import uuid4

import torch
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from diffusers import FluxPipeline

# ───────── CONFIG ─────────────────────────────────────────────
API_KEY   = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
MODEL_ID  = "black-forest-labs/FLUX.1-dev"

BASE_DIR   = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "generated_flux")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROUTE_PREFIX = "/medium"                    # <── CHANGE HERE once only
IMAGE_ROUTE  = f"{ROUTE_PREFIX}/images"

# ───────── FastAPI ────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.wildmindai.com",
        "https://api.wildmindai.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve files at  https://api.wildmindai.com/medium/images/<file>.png
app.mount(IMAGE_ROUTE, StaticFiles(directory=OUTPUT_DIR), name="medium-images")

# ───────── Model ──────────────────────────────────────────────
print("🔄 Loading FLUX-Dev …")
pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()
print("✅ FLUX-Dev ready!")

# ───────── Schema ─────────────────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str
    height:   int = 512
    width:    int = 512
    steps:    int = 50
    guidance: float = 6.5
    seed:     int = 42

# ───────── Routes ─────────────────────────────────────────────
@app.get(f"{ROUTE_PREFIX}/ping")
def ping():
    return {"status": "ok"}

@app.post(ROUTE_PREFIX)
async def generate(request: Request, body: PromptRequest):
    # ---------- auth ----------
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = body.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    # ---------- generate -------
    image = pipe(
        prompt,
        height=body.height,
        width=body.width,
        num_inference_steps=body.steps,
        guidance_scale=body.guidance,
        generator=torch.manual_seed(body.seed),
    ).images[0]

    # ---------- save & reply ---
    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)
    print("🖼️  saved", filepath)

    return JSONResponse(
        {"image_url": f"https://api.wildmindai.com{IMAGE_ROUTE}/{filename}"}
    )
