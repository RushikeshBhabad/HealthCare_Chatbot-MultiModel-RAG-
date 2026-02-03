import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv

# Load HuggingFace API key from .env
load_dotenv()

# =====================================================
# QWEN 2.5 VL MODEL
# Multimodal LLM (text + image understanding)
# =====================================================
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,     # Reduce GPU memory usage
    device_map="auto"              # Auto-assign CPU/GPU
)

# Processor prepares text + images for the model
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)


def retrieve_docs(query, vectorstore, k=5):
    """
    Retrieve top-k relevant documents using vector similarity
    """
    query_emb = vectorstore.embedding_function(query)
    return vectorstore.similarity_search_by_vector(query_emb, k=k)


def build_multimodal_prompt(query, docs, image_store):
    """
    Builds a multimodal prompt with text + images
    compatible with Qwen VL
    """
    messages = []

    # User question
    messages.append({
        "role": "user",
        "content": f"Question:\n{query}\n\nContext:\n"
    })

    # Add retrieved context
    for d in docs:
        if d.metadata["type"] == "text":
            messages.append({
                "role": "user",
                "content": d.page_content
            })

        if d.metadata["type"] == "image":
            img_id = d.metadata["image_id"]
            if img_id in image_store:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_store[img_id]}
                    ]
                })

    return messages


def run_rag(query, vectorstore, image_store):
    """
    Main RAG pipeline:
    retrieve → build prompt → generate answer
    """
    docs = vectorstore.similarity_search(query, k=5)
    messages = build_multimodal_prompt(query, docs, image_store)

    # Convert messages into model-ready tensors
    inputs = processor(
        messages=messages,
        return_tensors="pt"
    ).to(model.device)

    # Generate model response
    output = model.generate(
        **inputs,
        max_new_tokens=300
    )

    return processor.decode(
        output[0],
        skip_special_tokens=True
    )
