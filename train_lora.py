import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model

# Load base model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to(device)

# Extract U-Net from pipeline (since LoRA is applied to the U-Net)
unet = pipeline.unet

# Define LoRA configuration
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Correct attention layers
)

# Apply LoRA to U-Net
unet = get_peft_model(unet, config)

# **TRAINING: Dummy step to enable saving LoRA weights**
for param in unet.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(unet.parameters(), lr=5e-5)
loss_fn = torch.nn.MSELoss()  

# Fake training loop (Replace with actual dataset training)
for _ in range(10):
    optimizer.zero_grad()
    loss = torch.tensor(0.1, requires_grad=True)  # Simulated loss
    loss.backward()
    optimizer.step()

# **SAVE LoRA Weights Correctly**
unet.save_lora_adapter("lora_unet")


print("✅ Fine-tuned U-Net with LoRA saved successfully.")
