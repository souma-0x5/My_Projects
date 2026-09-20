import os
from dotenv import load_dotenv
from google import genai
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Step 1: Load the API key from .env ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Quick sanity check before doing anything else
if not api_key:
    print("ERROR: API key not found. Check your .env file.")
else:
    print(f"Key loaded, starts with: {api_key[:6]}...")

client = genai.Client(api_key=api_key)

# --- Step 2: Extract text (same as Day 2) ---
doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

# --- Step 3: Chunk it (same as Day 3) ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)

# --- Step 4: Generate an embedding for just the FIRST chunk (test) ---
result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks[0],
)

embedding_vector = result.embeddings[0].values

print(f"\nChunk text was:\n{chunks[0][:100]}...\n")
print(f"Embedding has {len(embedding_vector)} numbers")
print("First 10 numbers of the embedding:")
print(embedding_vector[:10])
