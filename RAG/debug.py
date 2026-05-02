from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import os
MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["niti_setu"]
collection = db["scheme_chunks"]

doc = collection.find_one()
print(len(doc["embedding"]))
print(type(doc["embedding"]))
print(type(doc["embedding"][0]))
