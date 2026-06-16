"""
helper.py
---------
Helper functions for:
  1. Loading PDF files from a folder
  2. Filtering out useless pages (blank, headers-only, etc.)
  3. Splitting text into small chunks for embeddings
  4. Downloading the free HuggingFace embedding model
"""

import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# ──────────────────────────────────────────────────────────────
# 1.  LOAD PDFs
# ──────────────────────────────────────────────────────────────

def load_pdf_file(data: str = "data/") -> list:
    """
    Read every PDF inside the `data/` folder and return a list of pages.

    Args:
        data: path to the folder that contains your PDF files

    Returns:
        List of LangChain Document objects (one per PDF page)
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",               # only pick up PDF files
        loader_cls=PyPDFLoader,     # use PyPDF to read each file
    )
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} pages from PDFs in '{data}'")
    return documents


# ──────────────────────────────────────────────────────────────
# 2.  FILTER JUNK PAGES
# ──────────────────────────────────────────────────────────────

def filter_to_minimal_docs(documents: list, min_chars: int = 100) -> list:
    """
    Remove pages that are basically empty (covers, blank pages, TOC images, etc.)
    Keeps only pages with at least `min_chars` characters of real text.

    Args:
        documents: list of Document objects from load_pdf_file()
        min_chars:  minimum character count to keep a page (default 100)

    Returns:
        Filtered list of Document objects
    """
    filtered = [
        doc for doc in documents
        if len(doc.page_content.strip()) >= min_chars
    ]
    print(f"✓ Kept {len(filtered)} / {len(documents)} pages after filtering")
    return filtered


# ──────────────────────────────────────────────────────────────
# 3.  SPLIT INTO CHUNKS
# ──────────────────────────────────────────────────────────────

def text_split(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chop pages into smaller overlapping chunks so the embedding model
    can handle them and the retriever finds precise passages.

    Args:
        documents:     list of Document objects
        chunk_size:    max characters per chunk  (500 works well for medical text)
        chunk_overlap: characters shared between neighbour chunks (helps context)

    Returns:
        List of smaller Document chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# ──────────────────────────────────────────────────────────────
# 4.  FREE EMBEDDINGS  (HuggingFace, runs on your computer)
# ──────────────────────────────────────────────────────────────

def download_hugging_face_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> HuggingFaceEmbeddings:
    """
    Load the free HuggingFace sentence-transformer model.
    First run downloads the model (~90 MB). After that it's cached locally.

    This model produces 384-dimension vectors — that's why Pinecone
    index dimension is set to 384 in store_index.py.

    Args:
        model_name: HuggingFace model ID

    Returns:
        HuggingFaceEmbeddings object ready to use with LangChain
    """
    print(f"⬇ Loading embedding model '{model_name}' (first run downloads ~90 MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},   # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✓ Embedding model ready")
    return embeddings
