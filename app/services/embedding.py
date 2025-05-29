import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_documents():
    # Aquí va la lógica del notebook 02

    file_faq = "Conformed/df_unificado.xlsx"
    df = pd.read_excel(file_faq)

    #Juntamos la pregunta y respuesta para poderla convertir en un solo vector
    df['text'] = "Pregunta: " + df['pregunta'] + "\nRespuesta: " + df['respuesta']

    # Inicializar el modelo de Sentence-BERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    #Obtener embeddings de todas las filas del Excel
    texts = df['text'].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Guardar los embeddings en un archivo .npy
    np.save('Conformed/faq_embeddings.npy', embeddings)

    return {"status": "embeddings created"}