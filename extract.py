import pymupdf

#open the pdf file
doc = pymupdf.open("data/uploads/Artificial_Intelligence_RAG_Project.pdf")

#go through every page and collect the text
full_text = ""
for page in doc:
    full_text += page.get_text()

#print how much text we got and a preview
print(f"Extracted {len(full_text)} characters")
print("----First 500 characters----")
print(full_text[:500])