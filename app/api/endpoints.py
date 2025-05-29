from fastapi import APIRouter, UploadFile, File, Form
from app.services import preprocessing, embedding, rag

router = APIRouter()

@router.post("/preprocess")
def preprocess_file(file: UploadFile = File(...)):
    return preprocessing.process_file(file)

@router.post("/embed")
def embed_documents():
    return embedding.embed_documents()

@router.post("/ask")
def ask_question(question: str = Form(...)):
    return rag.answer_question(question)
