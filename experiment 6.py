# ============================================================
# EXPERIMENT 6
# RETRIEVAL-AUGMENTED GENERATION (RAG)
# ============================================================

import os

# Prevent tokenizer multiprocessing issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. KNOWLEDGE BASE
# ============================================================

documents = [
    """
    Generative Artificial Intelligence is a branch of AI that creates
    new content such as text, images, audio, video and computer programs.
    """,

    """
    Large Language Models are transformer-based models trained on massive
    text datasets. They are used for text generation, summarization,
    translation, question answering and conversational AI.
    """,

    """
    Retrieval-Augmented Generation combines information retrieval with
    text generation. It retrieves relevant documents from an external
    knowledge base and gives them to a language model as context.
    """,

    """
    Vector databases store high-dimensional embeddings and perform
    similarity searches. Examples of vector databases include FAISS,
    ChromaDB, Pinecone, Weaviate and Milvus.
    """,

    """
    Prompt engineering is the process of designing clear instructions
    that guide a language model to produce accurate and useful responses.
    Common techniques include zero-shot, few-shot and role-based prompting.
    """,

    """
    Fine-tuning adapts a pretrained language model to a specific domain
    or task by training it further using a smaller domain-specific dataset.
    """
]


# ============================================================
# 2. LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully.")


# ============================================================
# 3. CREATE DOCUMENT EMBEDDINGS
# ============================================================

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
)

document_embeddings = document_embeddings.astype(
    "float32"
)


# ============================================================
# 4. NORMALIZE EMBEDDINGS
# ============================================================

faiss.normalize_L2(
    document_embeddings
)


# ============================================================
# 5. CREATE FAISS DATABASE
# ============================================================

embedding_dimension = document_embeddings.shape[1]

vector_database = faiss.IndexFlatIP(
    embedding_dimension
)

vector_database.add(
    document_embeddings
)

print("FAISS vector database created successfully.")


# ============================================================
# 6. LOAD SMALL GENERATION MODEL
# ============================================================

print("Loading generation model...")

# Small model suitable for local Mac execution
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

generation_model = AutoModelForCausalLM.from_pretrained(
    model_name
)

# GPT-2 models do not have a padding token
tokenizer.pad_token = tokenizer.eos_token

print("Generation model loaded successfully.")


# ============================================================
# 7. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, top_k=2):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(
        query_embedding
    )

    similarity_scores, document_indices = (
        vector_database.search(
            query_embedding,
            top_k
        )
    )

    retrieved_documents = []

    for index, score in zip(
        document_indices[0],
        similarity_scores[0]
    ):

        retrieved_documents.append({
            "document": documents[index].strip(),
            "score": float(score)
        })

    return retrieved_documents


# ============================================================
# 8. GENERATE ANSWER
# ============================================================

def generate_answer(query, retrieved_documents):

    context = "\n\n".join(
        item["document"]
        for item in retrieved_documents
    )

    prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = generation_model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Remove prompt from generated response
    answer = generated_text[
        len(prompt):
    ].strip()

    return answer


# ============================================================
# 9. START RAG SYSTEM
# ============================================================

print()
print("=" * 60)
print("RETRIEVAL-AUGMENTED GENERATION SYSTEM")
print("=" * 60)

user_query = input(
    "\nEnter your question: "
)


# ============================================================
# 10. RETRIEVE RELEVANT DOCUMENTS
# ============================================================

retrieved_results = retrieve_documents(
    query=user_query,
    top_k=2
)


# ============================================================
# 11. DISPLAY RETRIEVED DOCUMENTS
# ============================================================

print()
print("=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for number, item in enumerate(
    retrieved_results,
    start=1
):

    print()
    print(f"Document {number}:")
    print("-" * 60)

    print(item["document"])

    print(
        f"Similarity Score: "
        f"{item['score']:.4f}"
    )


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

answer = generate_answer(
    query=user_query,
    retrieved_documents=retrieved_results
)


# ============================================================
# 13. DISPLAY ANSWER
# ============================================================

print()
print("=" * 60)
print("GENERATED ANSWER")
print("=" * 60)

print(answer)

print()
print("RAG execution completed successfully.")