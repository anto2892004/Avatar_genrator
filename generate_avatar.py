from diffusers import StableDiffusionPipeline
import torch

# Load the base pipeline
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

# Load the fine-tuned LoRA model for U-Net
model.unet.load_attn_procs("lora_unet")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate avatar
prompt = "A futuristic portrait of a young person, highly detailed, digital painting"
image = model(prompt).images[0]
image.save("generated_avatar.png")

print("Avatar generated successfully. Check generated_avatar.png")
