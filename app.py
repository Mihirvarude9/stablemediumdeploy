# app_medium.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from diffusers import StableDiffusion3Pipeline, BitsAndBytesConfig, SD3Transformer2DModel
import torch
from io import BytesIO
from fastapi.responses import StreamingResponse

model_id = "stabilityai/stable-diffusion-3.5-medium"

app = FastAPI()

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# Load model transformer
transformer = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Load pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

# Input model
class PromptInput(BaseModel):
    prompt: str

@app.post("/generate")
async def generate_image(data: PromptInput):
    image = pipe(
        prompt=data.prompt,
        num_inference_steps=50,
        guidance_scale=5.5,
    ).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")
