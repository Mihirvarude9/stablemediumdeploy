import os
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
from uuid import uuid4
from fastapi.responses import FileResponse

# Load model
model_id = "stabilityai/stable-diffusion-3.5-medium"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("⏳ Loading transformer...")
transformer = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)

print("⏳ Loading full pipeline...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_model_cpu_offload()

# FastAPI app setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(req: PromptRequest):
    output_path = f"/tmp/{uuid4().hex}.png"
    print(f"🚀 Generating: {req.prompt}")
    image = pipe(prompt=req.prompt, num_inference_steps=50, guidance_scale=5.5).images[0]
    image.save(output_path)
    return FileResponse(output_path, media_type="image/png", filename="generated.png")
