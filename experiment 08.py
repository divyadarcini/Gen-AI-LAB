import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------
# AI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT
# ---------------------------------------------------------

print("Loading model...")

model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model loaded successfully.")


# ---------------------------------------------------------
# Function to generate responses
# ---------------------------------------------------------

def generate_response(prompt, max_tokens=250):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return response


# ---------------------------------------------------------
# 1. CODE GENERATION
# ---------------------------------------------------------

code_task = """
Write a Python program to calculate the factorial of a number
using a recursive function.

Requirements:
1. Accept a number from the user.
2. Use recursion.
3. Display the factorial.
4. Handle negative input.
"""

generation_prompt = f"""
Generate Python code for the following task.

Task:
{code_task}

Return only the complete Python program.
"""

generated_code = generate_response(
    generation_prompt,
    max_tokens=250
)


# ---------------------------------------------------------
# 2. DEBUGGING ASSISTANT
# ---------------------------------------------------------

faulty_code = """
def calculate_average(numbers):
    total = sum(numbers)
    average = total / len(number)
    return average

values = [10, 20, 30, 40]

print("Average:", calculate_average(values))
"""

debugging_prompt = f"""
Analyze the following Python program.

Code:
{faulty_code}

Identify the error and explain the cause.
Then provide the corrected Python program.

Use this format:

Error:
Explanation:
Corrected Code:
"""

debugging_result = generate_response(
    debugging_prompt,
    max_tokens=300
)


# ---------------------------------------------------------
# 3. DISPLAY RESULTS
# ---------------------------------------------------------

print("\nAI-POWERED CODE GENERATION AND DEBUGGING ASSISTANT")
print("=" * 60)

print("\n1. GENERATED CODE")
print("-" * 60)
print(generated_code)

print("\n2. DEBUGGING RESULT")
print("-" * 60)
print(debugging_result)