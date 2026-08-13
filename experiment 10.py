import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# ---------------------------------------------------------
# EXPERIMENT 10
# MULTIMODAL AI - VISUAL QUESTION ANSWERING
# ---------------------------------------------------------

print("Loading multimodal AI model...")

# ---------------------------------------------------------
# 1. Select available device
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
    print("Device: Apple Silicon GPU (MPS)")

elif torch.cuda.is_available():
    device = "cuda"
    print("Device: NVIDIA GPU (CUDA)")

else:
    device = "cpu"
    print("Device: CPU")

# ---------------------------------------------------------
# 2. Load pretrained BLIP model
# ---------------------------------------------------------

model_name = "Salesforce/blip-vqa-base"

processor = BlipProcessor.from_pretrained(model_name)

model = BlipForQuestionAnswering.from_pretrained(
    model_name
)

model = model.to(device)

print("Model loaded successfully.")

# ---------------------------------------------------------
# 3. Get image path
# ---------------------------------------------------------

print("\nEnter the path of the image you want to analyze.")

image_path = input("Image path: ")

# Open image
image = Image.open(image_path).convert("RGB")

print("Image loaded successfully.")

# ---------------------------------------------------------
# 4. Ask question
# ---------------------------------------------------------

question = input(
    "\nEnter a question about the image: "
)

# ---------------------------------------------------------
# 5. Process image and question
# ---------------------------------------------------------

inputs = processor(
    images=image,
    text=question,
    return_tensors="pt"
)

# Move tensors to selected device
inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

# ---------------------------------------------------------
# 6. Generate answer
# ---------------------------------------------------------

print("\nAnalyzing image...")

with torch.no_grad():

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=30
    )

# ---------------------------------------------------------
# 7. Decode answer
# ---------------------------------------------------------

answer = processor.decode(
    generated_ids[0],
    skip_special_tokens=True
)

# ---------------------------------------------------------
# 8. Display image
# ---------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.imshow(image)

plt.axis("off")

plt.title("Input Image")

plt.show()

# ---------------------------------------------------------
# 9. Display result
# ---------------------------------------------------------

print("\nMULTIMODAL AI RESULT")
print("-" * 50)

print("Question:", question)

print("Answer:", answer)