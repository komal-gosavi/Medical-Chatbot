"""
helper.py
---------
Helper functions for:
  1. Loading PDF files from a folder
  2. Filtering out useless pages (blank, headers-only, etc.)
  3. Splitting text into small chunks for embeddings

NOTE: We do NOT use sentence-transformers / torch here on purpose.
Those libraries need 1-2GB of RAM just to import, which crashes
Render's free tier (512MB RAM) with a silent Out-Of-Memory kill.

Instead, embeddings are created using Pinecone's own hosted
embedding API (pc.inference.embed) -- see rag_chain.py and
store_index.py. This needs almost no RAM on our side because
the heavy model runs on Pinecone's servers, not ours.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ----------------------------------------------------------------
# 1.  LOAD PDFs
# ----------------------------------------------------------------

def load_pdf_file(data: str = "data/") -> list:
    """
    Read every PDF inside the `data/` folder and return a list of pages.
    """
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from PDFs in '{data}'")
    return documents


# ----------------------------------------------------------------
# 2.  FILTER JUNK PAGES
# ----------------------------------------------------------------

def filter_to_minimal_docs(documents: list, min_chars: int = 100) -> list:
    """
    Remove pages that are basically empty (covers, blank pages, etc.)
    """
    filtered = [
        doc for doc in documents
        if len(doc.page_content.strip()) >= min_chars
    ]
    print(f"Kept {len(filtered)} / {len(documents)} pages after filtering")
    return filtered


# ----------------------------------------------------------------
# 3.  SPLIT INTO CHUNKS
# ----------------------------------------------------------------

def text_split(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    """
    Chop pages into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
