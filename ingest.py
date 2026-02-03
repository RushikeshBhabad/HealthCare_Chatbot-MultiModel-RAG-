import fitz                 
import io                  
import base64               
import os
import numpy as np
import torch

from PIL import Image

from dotenv import load_dotenv

from transformers import CLIPModel, CLIPProcessor

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# =====================================================
# CLIP MODEL INITIALIZATION
# CLIP gives a shared embedding space for text + images
# =====================================================

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)
clip_processor = CLIPProcessor.from_pretrained(  #To put text and images into the SAME vector space so we can search both together.
    "openai/clip-vit-base-patch32"
)

# Set model to inference mode
clip_model.eval()

# =====================================================
# TEXT EMBEDDING FUNCTION
# Converts text → CLIP vector
# =====================================================
def embed_text(text):
    """
    Generate normalized CLIP embedding for text
    """
    inputs = clip_processor(
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77      # CLIP token limit
    )

    with torch.no_grad():
        emb = clip_model.get_text_features(**inputs)

        # Normalize to unit vector for cosine similarity
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.squeeze().numpy()


# =====================================================
# IMAGE EMBEDDING FUNCTION
# Converts image → CLIP vector
# =====================================================
def embed_image(image):
    """
    Generate normalized CLIP embedding for image
    """
    inputs = clip_processor(
        images=image,
        return_tensors="pt"
    )

    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)

        # Normalize embedding
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb.squeeze().numpy()


# =====================================================
# PDF INGESTION PIPELINE
# Extracts text + images → embeddings → FAISS
# =====================================================
def ingest_pdf(pdf_path):
    """
    Reads a PDF file and:
    - Extracts text chunks
    - Extracts images
    - Creates CLIP embeddings for both
    - Stores everything in a FAISS vector store
    """

    # Open PDF
    doc = fitz.open(pdf_path)

    all_docs = []          # LangChain Document objects
    all_embeddings = []    # Corresponding CLIP embeddings
    image_store = {}       # image_id → base64 image

    # Split long text into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Iterate over each page
    for page_no, page in enumerate(doc):

        # ---------------- TEXT PROCESSING ----------------
        text = page.get_text()
        if text.strip():
            temp_doc = Document(
                page_content=text,
                metadata={
                    "page": page_no,
                    "type": "text"
                }
            )

            # Split text into smaller chunks
            chunks = splitter.split_documents([temp_doc])

            for chunk in chunks:
                emb = embed_text(chunk.page_content)
                all_docs.append(chunk)
                all_embeddings.append(emb)

        # ---------------- IMAGE PROCESSING ----------------
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]

            # Extract raw image bytes
            base_img = doc.extract_image(xref)
            img_bytes = base_img["image"]

            # Convert bytes → PIL Image
            pil_img = Image.open(
                io.BytesIO(img_bytes)
            ).convert("RGB")

            # Unique identifier for image
            image_id = f"page_{page_no}_img_{img_idx}"

            # Convert image to base64 (used later by VL model)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            image_store[image_id] = base64.b64encode(
                buffer.getvalue()
            ).decode()

            # Create CLIP embedding for image
            emb = embed_image(pil_img)

            # Store image as a LangChain document
            img_doc = Document(
                page_content=f"[Image: {image_id}]",
                metadata={
                    "page": page_no,
                    "type": "image",
                    "image_id": image_id
                }
            )

            all_docs.append(img_doc)
            all_embeddings.append(emb)


    doc.close()

    # =====================================================
    # FAISS VECTOR STORE (pre-computed embeddings)
    # =====================================================
    vectorstore = FAISS.from_embeddings(
        text_embeddings=[
            (doc.page_content, emb)
            for doc, emb in zip(all_docs, all_embeddings)
        ],
        embedding=None,  # embeddings already computed
        metadatas=[doc.metadata for doc in all_docs]
    )

    return vectorstore, image_store
