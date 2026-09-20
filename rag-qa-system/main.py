import os
import shutil
from dotenv import load_dotenv
from google import genai
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

# --- Setup (runs once, when the server starts) ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

chroma_client = chromadb.PersistentClient(path="data/index")
collection = chroma_client.get_or_create_collection("documents")

app = FastAPI()

# --- Endpoint 1: Upload a document ---
@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    save_path = f"data/uploads/{file.filename}"
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": file.filename, "message": "Uploaded successfully"}


# --- Endpoint 2: Process a document (extract, chunk, embed, store) ---
class ProcessRequest(BaseModel):
    filename: str

@app.post("/documents/process")
def process_document(request: ProcessRequest):
    filepath = f"data/uploads/{request.filename}"

    # Extract
    doc = pymupdf.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    # Embed each chunk
    embeddings = []
    for chunk in chunks:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk,
        )
        embeddings.append(result.embeddings[0].values)

    # Store in ChromaDB
    existing_count = collection.count()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{existing_count + i}" for i in range(len(chunks))],
        metadatas=[{"source": request.filename, "chunk_index": i} for i in range(len(chunks))],
    )

    return {"filename": request.filename, "chunks_indexed": len(chunks)}


# --- Endpoint 3: Ask a question ---
class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: AskRequest):
    question_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=request.question,
    ).embeddings[0].values

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
    )

    retrieved_chunks = results["documents"][0]
    sources = results["metadatas"][0]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in the document."
Do not make up information that isn't in the context.

Context:
{context}

Question: {request.question}

Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    return {
        "answer": response.text,
        "sources": [f"{s['source']} - chunk {s['chunk_index']}" for s in sources],
    }
