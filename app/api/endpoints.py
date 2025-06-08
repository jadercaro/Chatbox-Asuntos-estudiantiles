from fastapi import APIRouter, UploadFile, File, Request, Form
from fastapi.responses import PlainTextResponse
import os
import json
from twilio.rest import Client
from app.services import preprocessing, embedding, rag

router = APIRouter()

# -----------------------------
# Procesamiento de documentos
# -----------------------------
@router.post("/preprocess")
def preprocess_file(file: UploadFile = File(...)):
    return preprocessing.process_file(file)

@router.post("/embed")
def embed_documents():
    return embedding.embed_documents()

@router.post("/ask")
def ask_question(question: str = Form(...)):
    return rag.answer_question(question)

# -----------------------------
# Enviar mensaje por WhatsApp
# -----------------------------

def send_whatsapp_message(to: str, message: str):
    account_sid = "ACd5e738a20227cca7653a00113a817802"
    auth_token = "331e068e14fb9d41c98eba0a7e537f23"
    from_whatsapp_number = "whatsapp:+16364225536"

    client = Client(account_sid, auth_token)

    try:
        message_sent = client.messages.create(
            from_=from_whatsapp_number,
            to=f"whatsapp:{to}",
            content_sid="HX2442fa9efe8bfbffebfd23927fe124d7", 
            content_variables=json.dumps({
                "1": f'{message}'
            })
        )
        print("✅ WhatsApp message sent. SID:", message_sent.sid)

    except Exception as e:
        print("❌ Error sending message:", e)

# -----------------------------
# Webhook para recibir mensajes
# -----------------------------
@router.post("/webhook", response_class=PlainTextResponse)
async def receive_whatsapp_message(request: Request):
    form = await request.form()

    incoming_msg = form.get("Body")
    from_number = form.get("From", "").replace("whatsapp:", "")  # Limpieza del número

    if incoming_msg and from_number:
        print(f"📩 Mensaje recibido de {from_number}: {incoming_msg}")

        try:

            respuesta = rag.answer_question(incoming_msg)

            # Si la respuesta es un diccionario, extrae solo el texto
            if isinstance(respuesta, dict) and "respuesta" in respuesta:
                mensaje = respuesta["respuesta"]
            else:
                mensaje = str(respuesta)

            send_whatsapp_message(from_number, mensaje)

        except Exception as e:
            print("❌ Error procesando mensaje:", e)
