import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Step 1: Extract text from PDF (same as Day 2) ---
doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()

# --- Step 2: Split into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # roughly how many characters per chunk
    chunk_overlap=100,   # overlap so we don't cut sentences awkwardly between chunks
)

chunks = splitter.split_text(full_text)

# --- Step 3: See the results ---
print(f"Total chunks created: {len(chunks)}")
print("\n---- Chunk 1 ----")
print(chunks[0])
print("\n---- Chunk 2 ----")
print(chunks[1])
