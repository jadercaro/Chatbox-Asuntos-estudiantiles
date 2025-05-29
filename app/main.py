from fastapi import FastAPI
from app.api import endpoints

app = FastAPI()

app.include_router(endpoints.router)

@app.get("/")
def root():
    return {"message": "RAG Chatbot API running"}
