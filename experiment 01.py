from transformers import GPT2LMHeadModel, GPT2Tokenizer

print("Loading GPT-2 model...")

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input prompt
prompt = "Artificial Intelligence is"

# Tokenize input
inputs = tokenizer.encode(prompt, return_tensors="pt")

# Generate text
outputs = model.generate(
    inputs,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

# Decode generated text
generated_text = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\nGenerated Text:")
print(generated_text)