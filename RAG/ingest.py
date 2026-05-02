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

if not MONGO_URI:
    raise Exception("MONGO_URI not found in .env")

DB_NAME = "niti_setu"
COLLECTION_NAME = "scheme_chunks"

PDF_PATH = "pdfs\\FINALSchemeGuidelinesAIF.pdf"

# ==============================
# INIT
# ==============================

print("Connecting to Mongo...")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

print("Mongo connected.")
print("Existing total documents:", collection.count_documents({}))

print("Loading embedding model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Model loaded.")

# ==============================
# CHUNK FUNCTION
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

    print("Deleting old AIF documents...")
    collection.delete_many({"scheme": "AIF"})

    reader = PdfReader(PDF_PATH)

    print("Total Pages in AIF:", len(reader.pages))

    inserted_count = 0

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            print(f"Skipping page {page_number} (no text)")
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
            inserted_count += 1

    print("Inserted AIF chunks:", inserted_count)
    print("Now total AIF docs:", collection.count_documents({"scheme": "AIF"}))


if __name__ == "__main__":
    ingest_aif()