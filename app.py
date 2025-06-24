from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch
import os
import time

# === CONFIG ===
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = os.path.abspath("generated")  # ✅ Make path absolute
os.makedirs(OUTPUT_DIR, exist_ok=True)
model_id = "stabilityai/stable-diffusion-3.5-medium"

# === Load Transformer with Quantization ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("🔄 Loading SD 3.5 Medium transformer...")
transformer = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

print("🔄 Loading SD 3.5 Medium pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
)

# pipe.enable_model_cpu_offload()  # Optional, comment out if not needed
pipe.to("cuda")
print("✅ SD 3.5 Medium ready!")

# === FastAPI App ===
app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.wildmindai.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static images from the absolute output directory
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# Request body schema
class PromptInput(BaseModel):
    prompt: str

# Unified image generation logic
async def generate_and_respond(request: Request, data: PromptInput):
    # API key check
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    try:
        print(f"🎨 Generating image for prompt: {prompt}")
        image = pipe(prompt=prompt, num_inference_steps=50, guidance_scale=5.5).images[0]
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)

    try:
        image.save(filepath)
        print(f"✅ Image saved at: {filepath}")
    except Exception as e:
        print(f"❌ Failed to save image: {e}")
        raise HTTPException(status_code=500, detail="Image save failed")

    if not os.path.exists(filepath):
        print(f"❌ File not found immediately after saving: {filepath}")
        raise HTTPException(status_code=500, detail="Image file not found")

    time.sleep(0.4)  # prevent frontend race condition

    return {"image_url": f"https://api.wildmindai.com/images/{filename}"}

# POST /generate
@app.post("/generate")
async def generate_image(request: Request, data: PromptInput):
    return await generate_and_respond(request, data)

# POST /medium (alias)
@app.post("/medium")
async def generate_medium(request: Request, data: PromptInput):
    return await generate_and_respond(request, data)
