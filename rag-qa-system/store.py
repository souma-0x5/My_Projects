import os
from dotenv import load_dotenv
from google import genai
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)


doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(full_text)
print(f"Total chunks: {len(chunks)}")


print("Generating embeddings for all chunks...")
embeddings = []
for i, chunk in enumerate(chunks):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=chunk,
    )
    embeddings.append(result.embeddings[0].values)
    print(f"  Chunk {i + 1}/{len(chunks)} embedded")


chroma_client = chromadb.PersistentClient(path="data/index")
collection = chroma_client.get_or_create_collection("documents")

collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[
        {"source": "Artificial_Intelligence_RAG_Project.pdf", "chunk_index": i}
        for i in range(len(chunks))
    ],
)

print(f"\nStored {collection.count()} chunks in the vector database!")
