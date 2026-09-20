import os
from dotenv import load_dotenv
from google import genai
import chromadb

# --- Step 1: Load API key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# --- Step 2: Connect to the EXISTING vector database ---
# Note: no re-extracting, re-chunking, or re-embedding the document here.
# We're just opening the database we already built in Day 5.
chroma_client = chromadb.PersistentClient(path="data/index")
collection = chroma_client.get_or_create_collection("documents")

print(f"Connected to database. It contains {collection.count()} chunks.\n")

# --- Step 3: Ask a question ---
question = input("Type your question: ")

# --- Step 4: Embed the question (same model as before) ---
question_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
).embeddings[0].values

# --- Step 5: Search for the most similar chunks ---
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,  # get the top 3 most relevant chunks
)

# --- Step 6: Show what we found ---
print(f"\nTop matching chunks for: \"{question}\"\n")
for i, chunk_text in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]
    print(f"--- Match {i + 1} (distance: {distance:.4f}) ---")
    print(chunk_text)
    print()