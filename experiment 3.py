from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering
)

# -----------------------------------------
# A. Text Summarization
# -----------------------------------------

print("Loading summarization model...")

summary_model_name = "facebook/bart-large-cnn"

summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_name)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_name)

# Input text
text = """
Artificial Intelligence is transforming many industries by enabling
machines to perform tasks that normally require human intelligence.
It is widely used in healthcare, education, manufacturing, finance,
transportation, and cybersecurity. AI systems can analyze large
amounts of data, identify patterns, make predictions, and support
intelligent decision-making. Generative AI is a branch of Artificial
Intelligence that can create new content such as text, images, audio,
video, and computer programs.
"""

# Tokenize input
inputs = summary_tokenizer(
    text,
    return_tensors="pt",
    max_length=1024,
    truncation=True
)

# Generate summary
summary_ids = summary_model.generate(
    **inputs,
    max_length=60,
    min_length=20,
    do_sample=False
)

# Decode summary
summary = summary_tokenizer.decode(
    summary_ids[0],
    skip_special_tokens=True
)

print("\n--- Text Summarization ---")
print("Summary:")
print(summary)


# -----------------------------------------
# B. Question Answering
# -----------------------------------------

print("\nLoading question-answering model...")

qa_model_name = "distilbert-base-cased-distilled-squad"

qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Input context
context = """
Generative Artificial Intelligence is a type of Artificial Intelligence
that can create new content such as text, images, audio, video, and
computer programs. Large Language Models are commonly used for text
generation, summarization, translation, and question answering.
"""

# Input question
question = "What type of content can Generative AI create?"

# Tokenize question and context
inputs = qa_tokenizer(
    question,
    context,
    return_tensors="pt"
)

# Get model predictions
outputs = qa_model(**inputs)

# Find start and end positions
start_index = outputs.start_logits.argmax()
end_index = outputs.end_logits.argmax()

# Extract answer tokens
answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]

# Decode answer
answer = qa_tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)

# Calculate confidence
start_score = outputs.start_logits.softmax(dim=-1)[0][start_index]
end_score = outputs.end_logits.softmax(dim=-1)[0][end_index]
confidence = ((start_score + end_score) / 2).item()

print("\n--- Question Answering ---")
print("Question:", question)
print("Answer:", answer)
print("Confidence Score:", round(confidence, 3))