from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# === CONFIG ===
import os
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["niti_setu"]
collection = db["scheme_chunks"]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def search(query):
    query_embedding = model.encode(query).tolist()

    print("Embedding length:", len(query_embedding))

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",   # <-- YOUR ACTUAL INDEX NAME
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 200,
                "limit": 5
            }
        }
    ]

    results = list(collection.aggregate(pipeline))

    if not results:
        print("No results found.")
        return

    for doc in results:
        print("\n==============================")
        print("Scheme:", doc["scheme"])
        print("Page:", doc["page"])
        print("------------------------------")
        print(doc["text"][:300])
        print("==============================\n")


if __name__ == "__main__":
    search("Eligibility criteria for small farmers with land less than 2 hectares")
