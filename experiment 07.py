# ============================================================
# EXPERIMENT 7
# FINE-TUNING A PRETRAINED LANGUAGE MODEL
# FOR DOMAIN-SPECIFIC TEXT CLASSIFICATION
# ============================================================

import os

# Prevent tokenizer multiprocessing issues on macOS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# ============================================================
# 1. CREATE DOMAIN-SPECIFIC DATASET
# ============================================================

data = {
    "text": [
        "The transformer model achieved excellent accuracy.",
        "Large Language Models are revolutionizing AI.",
        "The football team won the championship.",
        "The cricket match was exciting.",
        "Neural networks are widely used in deep learning.",
        "The player scored a brilliant goal.",
        "Machine learning improves decision making.",
        "The tennis tournament starts tomorrow."
    ],

    "label": [
        1, 1, 0, 0,
        1, 0, 1, 0
    ]
}


dataset = Dataset.from_dict(data)

print("Dataset created successfully.")


# ============================================================
# 2. LOAD PRETRAINED TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)

print("Tokenizer loaded successfully.")


# ============================================================
# 3. TOKENIZE DATASET
# ============================================================

def tokenize(example):

    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )


dataset = dataset.map(
    tokenize
)


# ============================================================
# 4. SET DATASET FORMAT
# ============================================================

dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "label"
    ]
)

print("Dataset tokenized successfully.")


# ============================================================
# 5. LOAD PRETRAINED BERT MODEL
# ============================================================

print("\nLoading BERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

print("BERT model loaded successfully.")


# ============================================================
# 6. TRAINING CONFIGURATION
# ============================================================

training_args = TrainingArguments(
    output_dir="./fine_tuned_model",

    per_device_train_batch_size=2,

    num_train_epochs=2,

    logging_steps=1,

    save_strategy="no",

    report_to="none"
)


# ============================================================
# 7. CREATE TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)


# ============================================================
# 8. FINE-TUNE MODEL
# ============================================================

print("\nStarting fine-tuning...")
print("=" * 60)

trainer.train()

print("=" * 60)
print("Fine-tuning completed successfully.")


# ============================================================
# 9. SAVE FINE-TUNED MODEL
# ============================================================

print("\nSaving fine-tuned model...")

trainer.save_model(
    "./fine_tuned_model"
)

tokenizer.save_pretrained(
    "./fine_tuned_model"
)

print("Fine-tuned model saved successfully.")


# ============================================================
# 10. LOAD FINE-TUNED MODEL
# ============================================================

print("\nLoading fine-tuned model for prediction...")

inference_tokenizer = AutoTokenizer.from_pretrained(
    "./fine_tuned_model"
)

inference_model = AutoModelForSequenceClassification.from_pretrained(
    "./fine_tuned_model"
)

print("Fine-tuned model loaded successfully.")


# ============================================================
# 11. DEFINE CLASS LABELS
# ============================================================

labels = {
    0: "Sports",
    1: "Technology"
}


# ============================================================
# 12. PREDICTION FUNCTION
# ============================================================

def predict(text):

    inputs = inference_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = inference_model(
        **inputs
    )

    probabilities = outputs.logits.softmax(
        dim=-1
    )

    predicted_class = probabilities.argmax(
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    return (
        labels[predicted_class],
        confidence
    )


# ============================================================
# 13. TEST INPUT
# ============================================================

text = (
    "Generative AI models improve intelligent automation."
)


predicted_class, confidence = predict(
    text
)


# ============================================================
# 14. DISPLAY PREDICTION
# ============================================================

print("\nPrediction")
print("-" * 40)

print("Input :", text)

print(
    "Predicted Class :",
    predicted_class
)

print(
    "Confidence Score :",
    round(confidence, 3)
)


# ============================================================
# 15. RESULT
# ============================================================

print("\nExperiment 7 completed successfully.")