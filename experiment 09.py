import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# EXPERIMENT 9
# IMAGE GENERATION USING DIFFUSION MODEL
# ---------------------------------------------------------

print("Loading Stable Diffusion model...")

# Pretrained Stable Diffusion model
model_id = "runwayml/stable-diffusion-v1-5"

# ---------------------------------------------------------
# Select available device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float16
    print("Using Apple Silicon GPU (MPS)")

elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("Using NVIDIA GPU (CUDA)")

else:
    device = "cpu"
    dtype = torch.float32
    print("Using CPU")

# ---------------------------------------------------------
# Load Stable Diffusion
# ---------------------------------------------------------

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype
)

pipe = pipe.to(device)

print("Model loaded successfully.")

# ---------------------------------------------------------
# Input Prompt
# ---------------------------------------------------------

prompt = """
A futuristic smart city with flying cars,
green buildings, robots assisting people,
highly detailed, realistic, 4K quality.
"""

print("\nGenerating image...")
print("Prompt:", prompt)

# ---------------------------------------------------------
# Generate Image
# ---------------------------------------------------------

image = pipe(
    prompt,
    num_inference_steps=20
).images[0]

# ---------------------------------------------------------
# Save Image
# ---------------------------------------------------------

image_path = "generated_image.png"

image.save(image_path)

# ---------------------------------------------------------
# Display Image
# ---------------------------------------------------------

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis("off")
plt.title("Generated Image")
plt.show()

print("\nImage successfully generated!")
print("Saved as:", image_path)