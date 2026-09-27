import os
from dotenv import load_dotenv
from google import genai
import chromadb

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="data/index")
collection = chroma_client.get_or_create_collection("documents")

print(f"Connected to database. It contains {collection.count()} chunks.\n")

question = input("Type your question: ")

question_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
).embeddings[0].values


results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,  
)


print(f"\nTop matching chunks for: \"{question}\"\n")
for i, chunk_text in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]
    print(f"--- Match {i + 1} (distance: {distance:.4f}) ---")
    print(chunk_text)
    print()
