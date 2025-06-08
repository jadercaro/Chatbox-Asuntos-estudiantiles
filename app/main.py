from fastapi import FastAPI, Request, Query
from app.api import endpoints

app = FastAPI()

app.include_router(endpoints.router)

@app.get("/")
def root():
    return {"message": "RAG Chatbot API running"}


# Endpoint de verificación para Meta
@app.get("/webhook")
def verify_webhook(hub_mode: str = Query(None), hub_challenge: str = Query(None), hub_verify_token: str = Query(None)):
    VERIFY_TOKEN = "miverificacionsegura"
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"status": "error"}