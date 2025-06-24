from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch
import os
from uuid import uuid4
from io import BytesIO

# === CONFIG ===
API_KEY = "wildmind_5879fcd4a8b94743b3a7c8c1a1b4"
OUTPUT_DIR = "generated"
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
pipe.enable_model_cpu_offload()
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

# Serve static images
app.mount("/images", StaticFiles(directory=OUTPUT_DIR), name="images")

# === Input Schema ===
class PromptInput(BaseModel):
    prompt: str

# === /generate endpoint ===
@app.post("/generate")
async def generate_image(request: Request, data: PromptInput):
    # API key check
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is empty")

    image = pipe(
        prompt=prompt,
        num_inference_steps=50,
        guidance_scale=5.5,
    ).images[0]

    filename = f"{uuid4().hex}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    image.save(filepath)

    return {"image_url": f"https://api.wildmindai.com/images/{filename}"}
