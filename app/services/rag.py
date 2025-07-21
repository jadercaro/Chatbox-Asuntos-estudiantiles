import pandas as pd
import numpy as np 
import faiss
from sentence_transformers import SentenceTransformer
from app.services.utils import responder_pregunta_contexto
from app.config import API_KEY
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# LangChain: Integraciones adicionales
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

#Api Key de Groq
API_KEY = os.getenv("token_groq")

# Solo se cargan una vez
embedding_path = Path("app/services/Conformed/faq_embeddings_v5.npy")
df_path = Path("app/services/Conformed/contexto_rag_v5.csv")

# Cache global
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load(embedding_path)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

df = pd.read_csv(df_path)
#df['text'] = "Pregunta: " + df['pregunta'] + "\nRespuesta: " + df['respuesta']
texts = df['texts'].tolist()

def answer_question(question: str):
    embedding_pregunta = model.encode([question])
    distancias, indices = index.search(embedding_pregunta, k=5)

    textos_recuperados = [texts[i] for i in indices[0]]
    contexto = "\n\n".join(textos_recuperados)

    respuesta = responder_pregunta_contexto(API_KEY, contexto, question)

    #Remplazar saltos de línea y espacios al final por vacio
    respuesta = respuesta.replace("\n", "").strip()

    return {"respuesta": respuesta}
