import pandas as pd
import numpy as np 
import faiss
from sentence_transformers import SentenceTransformer
from app.services.utils import responder_pregunta_contexto
from app.config import API_KEY

def answer_question(question):
    embedding_path = Path("app/services/Conformed/faq_embeddings.npy")
    embeddings = np.load(embedding_path) 

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    pregunta_usuario = str(question)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_pregunta = model.encode([pregunta_usuario])

    distancias, indices = index.search(embedding_pregunta, k=5)

    df = pd.read_excel("app\\services\\Conformed\\df_unificado.xlsx")
    df['text'] = "Pregunta: " + df['pregunta'] + "\nRespuesta: " + df['respuesta']
    texts = df['text'].tolist()
    textos_recuperados = [texts[i] for i in indices[0]]

    contexto = "\n\n".join(textos_recuperados)
    respuesta = responder_pregunta_contexto(API_KEY, contexto, pregunta_usuario)
    
    return {"respuesta": respuesta}
