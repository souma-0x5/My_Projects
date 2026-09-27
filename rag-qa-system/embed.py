import os
from dotenv import load_dotenv
from google import genai
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("ERROR: API key not found. Check your .env file.")
else:
    print(f"Key loaded, starts with: {api_key[:6]}...")

client = genai.Client(api_key=api_key)


doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)


result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=chunks[0],
)

embedding_vector = result.embeddings[0].values

print(f"\nChunk text was:\n{chunks[0][:100]}...\n")
print(f"Embedding has {len(embedding_vector)} numbers")
print("First 10 numbers of the embedding:")
print(embedding_vector[:10])
