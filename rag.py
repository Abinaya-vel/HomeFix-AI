import os

import chromadb
from dotenv import load_dotenv
from google import genai

from chatbot_config import SYSTEM_PROMPT


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.1-flash-lite"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


# --------------------------------------------------
# Gemini client
# --------------------------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# --------------------------------------------------
# ChromaDB configuration
# --------------------------------------------------

VECTORSTORE_FOLDER = "vectorstore"
COLLECTION_NAME = "homefix_documents"

EMBEDDING_MODEL = "gemini-embedding-001"

TOP_K = 5


# --------------------------------------------------
# ChromaDB connection
# --------------------------------------------------

def get_collection():
    """
    Connect to the persistent ChromaDB vector store
    and return the HomeFix collection.
    """

    chroma_client = chromadb.PersistentClient(
        path=VECTORSTORE_FOLDER
    )

    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


# --------------------------------------------------
# Create query embedding
# --------------------------------------------------

def create_embedding(text):
    """
    Convert the user's question into a vector embedding.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# --------------------------------------------------
# Search relevant document chunks
# --------------------------------------------------

def search_documents(question, top_k=TOP_K):
    """
    Search ChromaDB for the most relevant document chunks.
    """

    collection = get_collection()

    if collection.count() == 0:
        return []

    query_embedding = create_embedding(question)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(
            top_k,
            collection.count()
        )
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    matches = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        matches.append({
            "text": document,

            "source": metadata.get(
                "source",
                "Unknown"
            ),

            "page": metadata.get(
                "page",
                "Unknown"
            ),

            "chunk": metadata.get(
                "chunk",
                "Unknown"
            ),

            "distance": distance
        })

    return matches


# --------------------------------------------------
# Build context for Gemini
# --------------------------------------------------

def build_context(matches):
    """
    Convert retrieved chunks into a structured context
    that Gemini can use.
    """

    if not matches:
        return (
            "No relevant information was found "
            "in the uploaded documents."
        )

    context_parts = []

    for index, match in enumerate(
        matches,
        start=1
    ):

        context_parts.append(
            f"""
DOCUMENT {index}
Source: {match['source']}
Page: {match['page']}

Content:
{match['text']}
"""
        )

    return "\n".join(context_parts)


# --------------------------------------------------
# Generate RAG answer
# --------------------------------------------------

def generate_answer(question):
    """
    Complete RAG pipeline:

    Question
        ↓
    Embedding
        ↓
    ChromaDB search
        ↓
    Relevant chunks
        ↓
    Gemini
        ↓
    Grounded answer + sources
    """

    matches = search_documents(question)

    if not matches:

        return {
            "answer": (
                "I couldn't find enough information "
                "about this in the uploaded documents."
            ),
            "sources": []
        }

    context = build_context(matches)

    prompt = f"""
{SYSTEM_PROMPT}

Use ONLY the following retrieved document context
to answer the user's question.

================ DOCUMENT CONTEXT ================

{context}

====================================================

USER QUESTION:
{question}

IMPORTANT:
- Do not invent information.
- Do not use information that is not supported by the context.
- If the context does not contain enough information, say so.
- Include the relevant source PDF and page number.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    answer = response.text.strip()

    # --------------------------------------------------
    # Remove duplicate sources
    # --------------------------------------------------

    unique_sources = []
    seen = set()

    for match in matches:

        source_key = (
            match["source"],
            match["page"]
        )

        if source_key not in seen:

            seen.add(source_key)

            # IMPORTANT:
            # filename is added so the frontend
            # can create a clickable PDF URL.
            unique_sources.append({
                "filename": match["source"],
                "source": match["source"],
                "page": match["page"]
            })

    return {
        "answer": answer,
        "sources": unique_sources
    }


# --------------------------------------------------
# Simple testing
# --------------------------------------------------

if __name__ == "__main__":

    question = input(
        "Ask HomeFix AI: "
    ).strip()

    if question:

        result = generate_answer(
            question
        )

        print(
            "\n=============================="
        )

        print(
            "HOMEFIX AI"
        )

        print(
            "=============================="
        )

        print(
            "\nAnswer:"
        )

        print(
            result["answer"]
        )

        print(
            "\nSources:"
        )

        if result["sources"]:

            for source in result["sources"]:

                print(
                    f"- {source['filename']} "
                    f"(Page {source['page']})"
                )

        else:

            print(
                "No sources available."
            )