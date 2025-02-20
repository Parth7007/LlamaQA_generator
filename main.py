import os
import shutil
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import google.generativeai as genai
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory for uploaded PDFs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure Gemini API
GEMINI_API_KEY = "your-new-gemini-api-key"  # Replace this with your actual key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Initialize FAISS storage
index = faiss.IndexFlatL2(768)  # 768D vector space for Gemini embeddings
pdf_embeddings = {}  # Store FAISS index per file

# LangChain text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        raise ValueError("No readable text found in PDF.")

    return text.strip()

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, extracts text, splits it into chunks, and stores embeddings in FAISS."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extracted_text = extract_text_from_pdf(file_path)
        chunks = text_splitter.split_text(extracted_text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        # Generate embeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GEMINI_API_KEY)
        vectors = np.array([embeddings_model.embed_query(doc.page_content) for doc in documents], dtype=np.float32)

        # Store embeddings in FAISS
        pdf_index = faiss.IndexFlatL2(vectors.shape[1])
        pdf_index.add(vectors)
        pdf_embeddings[file.filename] = {"index": pdf_index, "documents": documents}

        return {"message": "PDF uploaded successfully!", "filename": file.filename}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ask/")
async def ask_question(filename: str, question: str):
    """Uses FAISS to retrieve relevant chunks and passes them to Gemini for better responses."""
    try:
        if filename not in pdf_embeddings:
            raise HTTPException(status_code=404, detail="File not found. Upload the PDF first.")

        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GEMINI_API_KEY)
        query_vector = np.array([embeddings_model.embed_query(question)], dtype=np.float32)

        pdf_index = pdf_embeddings[filename]["index"]
        documents = pdf_embeddings[filename]["documents"]

        # Retrieve top 3 most relevant chunks
        _, indices = pdf_index.search(query_vector, 3)
        retrieved_text = "\n\n".join([documents[idx].page_content for idx in indices[0]])

        # Create a prompt using retrieved context
        prompt = f"Context:\n{retrieved_text}\n\nQuestion: {question}\n\nAnswer:"
        response = model.generate_content(prompt)

        if hasattr(response, "text"):
            return {"answer": response.text.strip()}
        return {"error": "No valid response from Gemini."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
