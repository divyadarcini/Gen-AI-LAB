import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from diffusers import StableDiffusionPipeline


# =========================================================
# 1. LOAD TEXT GENERATION MODEL
# =========================================================

print("Loading text generation model...")

text_model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(text_model_name)

text_model = AutoModelForSeq2SeqLM.from_pretrained(
    text_model_name
)

print("Text generation model loaded successfully.")


# =========================================================
# 2. FUNCTION FOR TEXT GENERATION
# =========================================================

def generate_text(prompt, max_new_tokens=180):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        output = text_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


# =========================================================
# 3. SELECT DEVICE
# =========================================================

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


# =========================================================
# 4. LOAD IMAGE GENERATION MODEL
# =========================================================

print("\nLoading image generation model...")
print("This may take some time during the first run...")

image_generator = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
)

image_generator = image_generator.to(device)

# Reduce memory usage
image_generator.enable_attention_slicing()

print("Image generation model loaded successfully.")


# =========================================================
# 5. GET CONTENT TOPIC
# =========================================================

topic = input("\nEnter a content topic: ")

# Keep the topic short for image generation
topic = topic.strip()


# =========================================================
# 6. GENERATE TEXT
# =========================================================

print("\nGenerating text...")

text_prompt = f"""
Write a short article of approximately 120 words about:

{topic}

Include:
1. A brief introduction
2. Its importance
3. Real-world applications

Use simple and clear language.
"""

generated_text = generate_text(
    text_prompt,
    max_new_tokens=180
)


# =========================================================
# 7. CREATE SHORT IMAGE PROMPT
# =========================================================

image_prompt = (
    f"Futuristic illustration of {topic}, "
    "artificial intelligence, neural networks, "
    "digital technology, intelligent systems, "
    "high quality digital art"
)


# =========================================================
# 8. GENERATE IMAGE
# =========================================================

print("\nGenerating image...")
print("Please wait...")

generated_image = image_generator(
    prompt=image_prompt,
    num_inference_steps=15,
    height=256,
    width=256
).images[0]


# =========================================================
# 9. SAVE IMAGE
# =========================================================

image_file = "generated_content_image.png"

generated_image.save(image_file)


# =========================================================
# 10. DISPLAY GENERATED TEXT
# =========================================================

print("\n" + "=" * 60)
print("GENERATED TEXT")
print("=" * 60)

print(generated_text)


# =========================================================
# 11. DISPLAY GENERATED IMAGE
# =========================================================

plt.figure(figsize=(6, 6))

plt.imshow(generated_image)

plt.axis("off")

plt.title("AI Generated Image")

plt.show()


# =========================================================
# 12. FINAL MESSAGE
# =========================================================

print("\nImage saved as:", image_file)
print("\nContent generation completed successfully.")