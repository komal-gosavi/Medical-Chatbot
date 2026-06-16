"""
rag_chain.py
------------
Builds the RAG (Retrieval Augmented Generation) pipeline:

  User Question
       │
       ▼
  HuggingFace Embeddings   ← FREE, runs on your machine
       │
       ▼
  Pinecone Vector Store    ← FREE tier
  (finds top-4 matching chunks from your PDF)
       │
       ▼
  Groq LLM (llama3-8b)    ← FREE tier, replaces OpenAI
  (reads the chunks + question → writes a friendly answer)
       │
       ▼
  Answer sent back to user
"""

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.helper import download_hugging_face_embeddings


# ──────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# This is the instruction we give to the AI before every question.
# It tells the AI to act like a helpful medical assistant.
# ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable medical assistant. "
    "Use ONLY the information provided in the context below to answer the question. "
    "If the answer is not in the context, say: "
    "'I don't have enough information to answer that question. "
    "Please consult a qualified doctor.' "
    "Keep your answer clear, simple, and under 4 sentences.\n\n"
    "Context:\n{context}"
)


def create_rag_chain(config: dict):
    """
    Build and return the full RAG chain using config values.

    Args:
        config: dictionary returned by load_config()

    Returns:
        A LangChain Runnable that accepts a question string and returns an answer string.

    Usage:
        chain = create_rag_chain(config)
        answer = chain.invoke("What are the symptoms of diabetes?")
        print(answer)
    """

    # ── Step 1: Same free embeddings used when storing the index ──────────
    embeddings = download_hugging_face_embeddings(
        model_name=config["EMBEDDING_MODEL"]
    )

    # ── Step 2: Connect to existing Pinecone index (already filled by store_index.py) ──
    vector_store = PineconeVectorStore(
        index_name=config["PINECONE_INDEX"],
        embedding=embeddings,
        pinecone_api_key=config["PINECONE_API_KEY"],
    )

    # ── Step 3: Retriever — finds top 3 most relevant chunks ──────────────
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},      # return top 3 matching paragraphs
    )

    # ── Step 4: FREE Groq LLM (replaces paid OpenAI) ─────────────────────
    #   Model options (all free on Groq):
    #     "llama-3.1-8b-instant"      ← fast, good for simple queries
    #     "llama-3.1-70b-versatile"   ← better quality, recommended ✓ default
    #     "llama-3.3-70b-versatile"   ← latest, best quality
    #     "mixtral-8x7b-32768"        ← large context window
    llm = ChatGroq(
        model=config["GROQ_MODEL"],
        groq_api_key=config["GROQ_API_KEY"],
        temperature=0.4,       # 0 = very factual, 1 = creative
        max_tokens=512,        # max words in reply
    )

    # ── Step 5: Prompt template ───────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{context}\n\nQuestion: {input}"),
    ])

    # ── Step 6: Create RAG chain using new langchain API ──────────────────
    # Format retrieved documents into context
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {
            "context": retriever | (lambda docs: format_docs(docs)),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("✓ RAG chain created (Pinecone + Groq Llama-3.1 70B)")

    # ── Step 7: Wrap so it returns a plain string ───────────────────────
    class SimpleChain:
        """Thin wrapper so app.py can call chain.invoke(question) and get a string back."""

        def __init__(self, chain):
            self._chain = chain

        def invoke(self, question: str) -> str:
            try:
                result = self._chain.invoke(question)
                # result is now a string directly from the LLM
                return result if isinstance(result, str) else str(result)
            except Exception as e:
                return f"Sorry, I could not generate an answer. Error: {str(e)}"

    return SimpleChain(rag_chain)
