"""
rag_chain.py
------------
Builds the RAG (Retrieval Augmented Generation) pipeline:

  User Question
       |
       v
  Pinecone HOSTED Embeddings  <- FREE, runs on Pinecone's servers (no torch/RAM needed)
       |
       v
  Pinecone Vector Store       <- FREE tier
  (finds top-3 matching chunks from your PDF)
       |
       v
  Groq LLM (llama3-8b)       <- FREE tier, replaces OpenAI
  (reads the chunks + question -> writes a friendly answer)
       |
       v
  Answer sent back to user
"""

from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.pinecone_embeddings import PineconeHostedEmbeddings


SYSTEM_PROMPT = (
    "You are a helpful and knowledgeable medical assistant. "
    "Use ONLY the information provided in the context below to answer the question. "
    "If the answer is not in the context, say: "
    "'I don't have enough information to answer that question. "
    "Please consult a qualified doctor.' "
    "Keep your answer clear, simple, and under 4 sentences.\n\n"
    "Context:\n{context}"
)


def _format_docs(docs: list) -> str:
    """Join retrieved documents into the context string sent to the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain(config: dict):
    """
    Build and return the full RAG chain using config values.

    Args:
        config: dictionary returned by load_config()

    Returns:
        An object with .invoke(question) -> answer string
    """

    # Step 1: Pinecone-hosted embeddings (lightweight, no torch needed)
    embeddings = PineconeHostedEmbeddings(
        api_key=config["PINECONE_API_KEY"],
        model=config["EMBEDDING_MODEL"],
    )

    # Step 2: Connect to existing Pinecone index (filled by store_index.py)
    vector_store = PineconeVectorStore(
        index_name=config["PINECONE_INDEX"],
        embedding=embeddings,
        pinecone_api_key=config["PINECONE_API_KEY"],
    )

    # Step 3: Retriever -- finds top 3 most relevant chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # Step 4: FREE Groq LLM (replaces paid OpenAI)
    llm = ChatGroq(
        model=config["GROQ_MODEL"],
        groq_api_key=config["GROQ_API_KEY"],
        temperature=0.4,
        max_tokens=512,
    )

    # Step 5: Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    # Step 6: Chain that sends retrieved context + question to the LLM
    question_answer_chain = prompt | llm | StrOutputParser()

    print("RAG chain created (Pinecone hosted embeddings + Groq llama3)")

    class SimpleChain:
        """Thin wrapper so app.py can call chain.invoke(question) and get a string back."""

        def __init__(self, retriever, chain):
            self._retriever = retriever
            self._chain = chain

        def invoke(self, question: str) -> str:
            docs = self._retriever.invoke(question)
            return self._chain.invoke({
                "input": question,
                "context": _format_docs(docs),
            })

    return SimpleChain(retriever, question_answer_chain)
