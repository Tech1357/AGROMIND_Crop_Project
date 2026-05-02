from dotenv import load_dotenv
load_dotenv()

import os
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# ==============================
# CONFIG
# ==============================

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "niti_setu"
COLLECTION_NAME = "scheme_chunks"

AIF_PDF_PATH = "FINALSchemeGuidelinesAIF.pdf"  # put correct path

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# ==============================
# CHUNKING FUNCTION
# ==============================

def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# ==============================
# INGEST AIF
# ==============================

def ingest_aif():

    reader = PdfReader(AIF_PDF_PATH)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue

        chunks = chunk_text(text)

        for chunk in chunks:
            embedding = model.encode(chunk).tolist()

            doc = {
                "scheme": "AIF",
                "page": page_number,
                "text": chunk,
                "embedding": embedding
            }

            collection.insert_one(doc)

    print("AIF ingestion complete.")

if __name__ == "__main__":
    ingest_aif()