"""
store_index.py
--------------
Run this file ONCE to:
  1. Read all PDFs from the data/ folder
  2. Split them into chunks
  3. Convert chunks to embeddings (Pinecone's FREE hosted model -- no torch needed)
  4. Upload embeddings to Pinecone

You only need to run this again if you add new PDF files.

How to run:
    python store_index.py
"""

from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split
from src.pinecone_embeddings import PineconeHostedEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file!")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "multilingual-e5-large")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 1024))
INDEX_NAME = os.environ.get("PINECONE_INDEX", "medical-chatbot")

print("\nStep 1: Loading PDF files from data/ folder...")
extracted_data = load_pdf_file(data="data/")

print("\nStep 2: Filtering blank/useless pages...")
filter_data = filter_to_minimal_docs(extracted_data)

print("\nStep 3: Splitting text into chunks...")
text_chunks = text_split(filter_data)

print("\nStep 4: Setting up Pinecone hosted embeddings (no local model download)...")
embeddings = PineconeHostedEmbeddings(api_key=PINECONE_API_KEY, model=EMBEDDING_MODEL)

print("\nStep 5: Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    print(f"   Creating new Pinecone index '{INDEX_NAME}' (dimension={EMBEDDING_DIMENSION})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,   # must match multilingual-e5-large = 1024
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"   Index '{INDEX_NAME}' created")
else:
    print(f"   Index '{INDEX_NAME}' already exists -- skipping creation")
    print(f"   NOTE: if this index was created earlier with dimension=384,")
    print(f"   you must delete it in the Pinecone dashboard and re-run this script,")
    print(f"   because multilingual-e5-large needs dimension=1024.")

print(f"\nStep 6: Uploading {len(text_chunks)} chunks to Pinecone (this may take a few minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=INDEX_NAME,
    embedding=embeddings,
)

print("\nDone! Your medical data is now stored in Pinecone.")
print("   Now run:  python app.py   to start the chatbot.\n")
