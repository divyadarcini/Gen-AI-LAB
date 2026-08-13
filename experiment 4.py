import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pretrained conversational model
model_name = "microsoft/DialoGPT-small"

print("Loading chatbot model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("\nChatbot: Hello! Type 'exit' to end the conversation.\n")

# Predefined responses for the lab demonstration
responses = {
    "hi": "Hello! How can I help you today?",
    "hello": "Hello! How can I help you today?",
    "what is artificial intelligence?":
        "Artificial Intelligence is the simulation of human intelligence by machines. It enables machines to learn, reason, and solve problems.",
    "what is machine learning?":
        "Machine Learning (ML) is a computer science term that refers to the use of computer algorithms to learn information.",
    "what are large language models?":
        "Large Language Models are AI models trained on large amounts of text data to understand and generate human language.",
    "thank you": "You're welcome! Happy learning!"
}

while True:

    user_input = input("You: ")

    # Exit condition
    if user_input.lower().strip() == "exit":
        print("Chatbot: Goodbye! Have a nice day.")
        break

    # Normalize input
    normalized_input = user_input.lower().strip()

    # Use predefined response if available
    if normalized_input in responses:
        response = responses[normalized_input]

    else:
        # Use DialoGPT for other conversations
        new_input_ids = tokenizer.encode(
            user_input + tokenizer.eos_token,
            return_tensors="pt"
        )

        response_ids = model.generate(
            new_input_ids,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

        response = tokenizer.decode(
            response_ids[:, new_input_ids.shape[-1]:][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        if not response.strip():
            response = "I'm sorry, I don't have an answer for that."

    print("Chatbot:", response)