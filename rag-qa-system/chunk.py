import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")
full_text = ""
for page in doc:
    full_text += page.get_text()


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=100,   
)

chunks = splitter.split_text(full_text)


print(f"Total chunks created: {len(chunks)}")
print("\n---- Chunk 1 ----")
print(chunks[0])
print("\n---- Chunk 2 ----")
print(chunks[1])
