import os
from dotenv import load_dotenv
from google import genai
import chromadb

# --- Step 1: Load API key ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

# --- Step 2: Connect to the existing vector database ---
chroma_client = chromadb.PersistentClient(path="data/index")
collection = chroma_client.get_or_create_collection("documents")

# --- Step 3: Ask a question ---
question = input("Type your question: ")

# --- Step 4: Embed the question and search (same as Day 6) ---
question_embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
).embeddings[0].values

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3,
)

retrieved_chunks = results["documents"][0]
sources = results["metadatas"][0]

# --- Step 5: Build the RAG prompt ---
context = "\n\n".join(retrieved_chunks)

prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in the document."
Do not make up information that isn't in the context.

Context:
{context}

Question: {question}

Answer:"""

# --- Step 6: Send to Gemini for the final answer ---
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

# --- Step 7: Show the answer + sources ---
print(f"\nQuestion: {question}")
print(f"\nAnswer: {response.text}")
print(f"\nSources: chunk indexes {[s['chunk_index'] for s in sources]} from {sources[0]['source']}")
