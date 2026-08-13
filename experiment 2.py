from transformers import pipeline

# -------------------------------
# A. Sentiment Analysis
# -------------------------------

print("Loading sentiment analysis model...")

sentiment_analyzer = pipeline("sentiment-analysis")

# Input text
text = "The Generative AI workshop was extremely informative and useful."

# Predict sentiment
result = sentiment_analyzer(text)

print("\n--- Sentiment Analysis ---")
print("Input:", text)
print("Label:", result[0]["label"])
print("Score:", round(result[0]["score"], 4))


# -------------------------------
# B. Document Classification
# -------------------------------

print("\nLoading document classification model...")

classifier = pipeline("zero-shot-classification")

# Input document
document = """
Artificial Intelligence and Machine Learning are transforming
industries through automation and intelligent decision-making.
"""

# Candidate labels
labels = ["Technology", "Sports", "Politics", "Entertainment"]

# Classify document
result = classifier(document, labels)

print("\n--- Document Classification ---")
print("Document:", document.strip())
print("Predicted Category:", result["labels"][0])
print("Confidence Score:", round(result["scores"][0], 4))