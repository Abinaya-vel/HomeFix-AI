import os
import fitz
import chromadb
from dotenv import load_dotenv
from google import genai


# --------------------------------------------------
# Environment
# --------------------------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )

client = genai.Client(api_key=GEMINI_API_KEY)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

UPLOAD_FOLDER = "data/uploads"
VECTORSTORE_FOLDER = "vectorstore"

COLLECTION_NAME = "homefix_documents"

EMBEDDING_MODEL = "gemini-embedding-001"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text):
    """
    Clean extracted PDF text.
    """

    return " ".join(text.split())


# --------------------------------------------------
# Text chunking
# --------------------------------------------------

def split_text(text):
    """
    Split text into overlapping chunks.
    """

    text = clean_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# --------------------------------------------------
# Create Gemini embedding
# --------------------------------------------------

def create_embedding(text):
    """
    Generate an embedding vector for a text chunk.
    """

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


# --------------------------------------------------
# ChromaDB
# --------------------------------------------------

def get_collection(reset=False):
    """
    Create/connect to the HomeFix ChromaDB collection.
    """

    os.makedirs(VECTORSTORE_FOLDER, exist_ok=True)

    chroma_client = chromadb.PersistentClient(
        path=VECTORSTORE_FOLDER
    )

    if reset:

        try:
            chroma_client.delete_collection(
                COLLECTION_NAME
            )
            print("✓ Existing vector collection cleared")

        except Exception:
            pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    return collection


# --------------------------------------------------
# Process one PDF
# --------------------------------------------------

def process_pdf(pdf_path, collection):

    filename = os.path.basename(pdf_path)

    print()
    print("=" * 50)
    print(f"Processing: {filename}")
    print("=" * 50)

    try:
        pdf = fitz.open(pdf_path)

    except Exception as error:

        print(f"✗ Could not open {filename}")
        print(f"Error: {error}")

        return 0

    total_chunks = 0

    for page_number, page in enumerate(
        pdf,
        start=1
    ):

        text = page.get_text("text").strip()

        if not text:
            print(
                f"  Page {page_number}: "
                "No extractable text"
            )
            continue

        chunks = split_text(text)

        print(
            f"  Page {page_number}: "
            f"{len(chunks)} chunks"
        )

        for chunk_index, chunk in enumerate(chunks):

            try:

                embedding = create_embedding(chunk)

                document_id = (
                    f"{filename}"
                    f"_page_{page_number}"
                    f"_chunk_{chunk_index}"
                )

                collection.add(
                    ids=[document_id],

                    embeddings=[embedding],

                    documents=[chunk],

                    metadatas=[{
                        "source": filename,
                        "page": page_number,
                        "chunk": chunk_index
                    }]
                )

                total_chunks += 1

            except Exception as error:

                print(
                    f"  ✗ Failed chunk "
                    f"{chunk_index} on page "
                    f"{page_number}: {error}"
                )

    pdf.close()

    print()
    print(
        f"✓ {filename}: "
        f"{total_chunks} chunks stored"
    )

    return total_chunks


# --------------------------------------------------
# Ingest all PDFs
# --------------------------------------------------

def ingest_documents():

    os.makedirs(
        UPLOAD_FOLDER,
        exist_ok=True
    )

    pdf_files = sorted(
        [
            os.path.join(
                UPLOAD_FOLDER,
                filename
            )

            for filename in os.listdir(
                UPLOAD_FOLDER
            )

            if filename.lower().endswith(".pdf")
        ]
    )

    print()
    print("=" * 50)
    print("           HOMEFIX AI")
    print("        DOCUMENT INGESTION")
    print("=" * 50)

    if not pdf_files:

        print()
        print(
            "✗ No PDF files found."
        )

        print(
            f"Add PDF files to: "
            f"{UPLOAD_FOLDER}"
        )

        return

    print()
    print(
        f"Found {len(pdf_files)} PDF file(s)"
    )

    # Rebuild vector database
    collection = get_collection(
        reset=True
    )

    total_chunks = 0

    for pdf_path in pdf_files:

        total_chunks += process_pdf(
            pdf_path,
            collection
        )

    print()
    print("=" * 50)
    print("       INGESTION COMPLETED")
    print("=" * 50)

    print(
        f"✓ PDFs processed : {len(pdf_files)}"
    )

    print(
        f"✓ Total chunks   : {total_chunks}"
    )

    print(
        f"✓ ChromaDB count : {collection.count()}"
    )

    print(
        f"✓ Vector store   : {VECTORSTORE_FOLDER}"
    )

    print("=" * 50)


# --------------------------------------------------
# Run
# --------------------------------------------------

if __name__ == "__main__":
    ingest_documents()