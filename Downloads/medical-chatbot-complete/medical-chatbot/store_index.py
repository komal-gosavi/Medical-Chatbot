"""
store_index.py
--------------
Run this file ONCE to:
  1. Read all PDFs from the data/ folder
  2. Split them into chunks
  3. Convert chunks to embeddings (free HuggingFace model)
  4. Upload embeddings to Pinecone

You only need to run this again if you add new PDF files.

How to run:
    python store_index.py
"""

from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# ── Load .env file ────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ── Step 1: Read PDFs ─────────────────────────────────────────
print("\n📄 Step 1: Loading PDF files from data/ folder...")
extracted_data = load_pdf_file(data="data/")

# ── Step 2: Filter junk pages ─────────────────────────────────
print("\n🧹 Step 2: Filtering blank/useless pages...")
filter_data = filter_to_minimal_docs(extracted_data)

# ── Step 3: Split into chunks ─────────────────────────────────
print("\n✂️  Step 3: Splitting text into chunks...")
text_chunks = text_split(filter_data)

# ── Step 4: Load free embedding model ────────────────────────
print("\n🤖 Step 4: Loading HuggingFace embedding model (free)...")
embeddings = download_hugging_face_embeddings()

# ── Step 5: Connect to Pinecone ──────────────────────────────
print("\n🌲 Step 5: Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

INDEX_NAME = "medical-chatbot"   # ← change this name if you want

# Create index only if it doesn't already exist
if not pc.has_index(INDEX_NAME):
    print(f"   Creating new Pinecone index '{INDEX_NAME}'...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,                       # must match all-MiniLM-L6-v2 output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"   ✓ Index '{INDEX_NAME}' created")
else:
    print(f"   ✓ Index '{INDEX_NAME}' already exists — skipping creation")

# ── Step 6: Upload embeddings ─────────────────────────────────
print(f"\n⬆️  Step 6: Uploading {len(text_chunks)} chunks to Pinecone (this may take a few minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=INDEX_NAME,
    embedding=embeddings,
)

print("\n🎉 Done! Your medical data is now stored in Pinecone.")
print("   Now run:  python app.py   to start the chatbot.\n")
