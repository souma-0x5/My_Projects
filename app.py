import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.title("📄 AI Document Q&A")
st.write("Upload a PDF and ask questions about it.")

# --- Section 1: Upload & Process ---
st.header("1. Upload a document")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file is not None:
    if st.button("Upload and Process"):
        with st.spinner("Uploading..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            upload_response = requests.post(f"{API_URL}/documents/upload", files=files)

        if upload_response.status_code == 200:
            st.success(f"Uploaded: {uploaded_file.name}")

            with st.spinner("Processing (extracting, chunking, embedding)..."):
                process_response = requests.post(
                    f"{API_URL}/documents/process",
                    json={"filename": uploaded_file.name},
                )

            if process_response.status_code == 200:
                data = process_response.json()
                st.success(f"Indexed {data['chunks_indexed']} chunks!")
            else:
                st.error("Processing failed.")
        else:
            st.error("Upload failed.")

# --- Section 2: Ask Questions ---
st.header("2. Ask a question")
question = st.text_input("Type your question here")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            ask_response = requests.post(f"{API_URL}/ask", json={"question": question})

        if ask_response.status_code == 200:
            data = ask_response.json()
            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for source in data["sources"]:
                st.write(f"- {source}")
        else:
            st.error("Something went wrong asking the question.")
    else:
        st.warning("Please type a question first.")