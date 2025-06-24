import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch

# Create output folder if it doesn't exist
os.makedirs("static/outputs", exist_ok=True)

# --------------------------
# Model Setup (GPU optimized)
# --------------------------
model_id = "stabilityai/stable-diffusion-3.5-medium"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading transformer in 4-bit...")
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

print("Loading full pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_model_cpu_offload()

# -----------------------
# FastAPI Initialization
# -----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OR restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Request Schema
# -----------------------
class GenerateRequest(BaseModel):
    prompt: str

# -----------------------
# API Endpoint
# -----------------------
@app.post("/generate")
def generate_image(req: GenerateRequest):
    try:
        print(f"Received prompt: {req.prompt}")
        image = pipe(
            prompt=req.prompt,
            num_inference_steps=50,
            guidance_scale=5.5,
        ).images[0]

        image_id = str(uuid.uuid4())
        file_path = f"static/outputs/{image_id}.png"
        image.save(file_path)

        # Return public image URL
        return {
            "success": True,
            "image_url": f"https://api.wildmindai.com/images/{image_id}.png"
        }

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail="CUDA OOM: GPU is out of memory.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
