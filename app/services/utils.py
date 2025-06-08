# Librerías estándar de Python
import os
import re
import time

# Librerías de terceros
import faiss
import json
import openpyxl
import pandas as pd
import numpy as np
import pdfplumber
from PyPDF2 import PdfReader
from tqdm import tqdm


# LangChain: Text processing y modelos
from langchain.chains import LLMChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import HuggingFaceEndpoint, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# LangChain: Integraciones adicionales
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

#Api Key de Groq
API_KEY = os.getenv("token_groq")

# Cargar los PDF desde una carpeta
def cargar_pdfs(carpeta):
    textos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            path = os.path.join(carpeta, archivo)
            lector = PdfReader(path)
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text() or ""
            textos.append(texto)
    return "\n".join(textos)

def obtener_texto_de_pdf(path: str)->str:
        separador_paginas = "SEPARADOR"
        # Intenta abrir el archivo como PDF y extraer su texto
        doc = pymupdf.open(path)
        text = ""
        for i,page in enumerate(doc):
            text += page.get_text() # get plain text (is in UTF-8)
            text += separador_paginas  # solo agrega separador si no es la última página
        doc.close()
        return text

# Quiero ver lo que hay en un pdf
def ver_texto_pdf(carpeta):
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            path = os.path.join(carpeta, archivo)
            lector = PdfReader(path)
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text() or ""
            print(f"\n\n===== CONTENIDO DEL ARCHIVO: {archivo} =====\n")
            print(texto[:3000])  # Muestra los primeros 3000 caracteres
            break  # Solo muestra el primer PDF para revisar

def limpiar_dataframe(df):
    # Función para limpiar el cuerpo del mensaje
    def limpiar_cuerpo(texto):
        # Eliminar texto legal entre "La información aquí contenida..." y "el autor de la misma."
        texto = re.sub(r'\"La información aquí contenida.*?el autor de la misma\.\"', "", texto, flags=re.DOTALL)
        texto = re.sub(r'Universidad de Antioquia http.*?Colombia', '', texto, flags=re.DOTALL)
        # Eliminar saltos de línea y otros caracteres innecesarios
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto

    # Limpiar la columna 'Cuerpo del mensaje'
    df["Cuerpo limpio"] = df["Cuerpo del mensaje"].apply(limpiar_cuerpo)
    
    # Limpiar la columna "Archivo"
    df["Archivo"] = df["Archivo"].str.replace("Correo de Universidad de Antioquia -", "", regex=False)
    df["Archivo"] = df["Archivo"].str.replace(".pdf", "", regex=False)
    df["Archivo"] = df["Archivo"].str.strip()

    # Eliminar la columna "Encabezado" si es sensible
    df = df.drop(["Encabezado"], axis=1, errors='ignore')  # errors='ignore' evita errores si no existe

    return df

# Para eliminar "[Texto citado oculto]"
def eliminar_texto_citado(texto):
    return re.sub(r'\[Texto citado oculto\]', '', texto)

# Para eliminar el patrón de correo de Universidad de Antioquia con la URL
def eliminar_enlaces_mail(texto):
    return re.sub(r'https://mail\.google\.com/mail/u/0/[^\n]*\.{3}', '', texto)

# Para eliminar el párrafo de información confidencial
def eliminar_disclaimer(texto):
    disclaimer = r'''"La información aquí contenida es para uso exclusivo de la persona o entidad de destino. Está
estrictamente prohibida su utilización, copia, descarga, distribución, modificación y/o reproducción total
o parcial, sin el permiso expreso de Universidad de Antioquia, pues su contenido puede ser de carácter
confidencial y/o contener material privilegiado. Si usted recibió esta información por error, por favor
contacte en forma inmediata a quien la envió y borre este material de su computador. Universidad de
Antioquia no es responsable por la información contenida en esta comunicación, el directo responsable
es quien la firma o el autor de la misma."'''
    return re.sub(disclaimer, '', texto)

def eliminar_pagina_udea(texto):
    texto = re.sub(r'http://www.udea.edu.co/','', texto)
    texto = re.sub(r'www.udea.edu.co','', texto)
    return texto

# Función que aplica todas las limpiezas
def limpiar_texto(df):
    df['Cuerpo limpio'] = df['Cuerpo limpio'].apply(eliminar_texto_citado)
    df['Cuerpo limpio'] = df['Cuerpo limpio'].apply(eliminar_enlaces_mail)
    df['Cuerpo limpio'] = df['Cuerpo limpio'].apply(eliminar_disclaimer)
    df['Cuerpo limpio'] = df['Cuerpo limpio'].apply(eliminar_pagina_udea)
    return df

def extraer_respuestas_email(carpeta):
    datos = []
    for archivo in os.listdir(carpeta):
        if archivo.endswith(".pdf"):
            path = os.path.join(carpeta, archivo)
            lector = PdfReader(path)
            texto = ""
            for pagina in lector.pages:
                texto += pagina.extract_text() or ""

            # Separar los bloques por cada remitente con fecha
            bloques = re.split(
                r"\n?([A-ZÁÉÍÓÚÑ ]{3,100}<[^>]+>\s+\d{1,2} de \w+ de \d{4}, .*?)\n", 
                texto
            )

            for i in range(1, len(bloques), 2):
                encabezado = bloques[i].strip()
                cuerpo_original = bloques[i+1].strip() if i+1 < len(bloques) else ""

                # Extraer desde el nombre hasta el primer <
                nombre = encabezado.split("<")[0].strip()
                patron = re.escape(nombre) + r"\s*<"

                match = re.search(patron, cuerpo_original)
                if match:
                    extraido = match.group()
                    cuerpo_limpio = cuerpo_original.replace(extraido, "").strip()
                else:
                    extraido = ""
                    cuerpo_limpio = cuerpo_original

                datos.append({
                    "Archivo": archivo,
                    "Encabezado": encabezado,
                    "Cuerpo del mensaje": cuerpo_limpio
                })

    return pd.DataFrame(datos)

def limpiar_correo(texto):
    # Eliminar encabezados y metadatos típicos
    texto = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}.*?\n', '', texto)  # Fechas + encabezados
    texto = re.sub(r'(Para|De|Asunto):.*?\n', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\b\d+ (mensaje|mensajes)\b', '', texto, flags=re.IGNORECASE)
    texto = re.sub(r'\n+|\n', '\n', texto)  # Quitar múltiples saltos de línea

    # Eliminar firmas típicas
    texto = re.sub(r'(Saludos|Atentamente|Cordialmente|Comité de Asuntos.*?)(\n.*?)*\n+', '', texto, flags=re.IGNORECASE)

    # Eliminar líneas que contienen URLs
    texto = re.sub(r'https?://\S+', '', texto)

    # Eliminar texto citado (reenvíos, respuestas anteriores)
    texto = re.sub(r'\[Texto citado oculto\]', '', texto)
    texto = re.sub(r'El [\w\s,.:-]+ escribió:', '', texto, flags=re.IGNORECASE)

    # Eliminar avisos legales y direcciones institucionales
    texto = re.sub(r'La información aquí contenida.*?autor de la misma."\n*', '', texto, flags=re.DOTALL)
    texto = re.sub(r'(www\.udea\.edu\.co|bit\.ly/\S+)', '', texto)

    # Opcional: eliminar todo lo que está en mayúsculas sostenidas (títulos, encabezados ruidosos)
    texto = re.sub(r'^[A-ZÁÉÍÓÚÑ\s\-]{5,}$', '', texto, flags=re.MULTILINE)

    # Eliminar etiquetas tipo <<<FIN DE PÁGINA>>>
    texto = re.sub(r'<<<FIN DE PÁGINA>>>', '', texto)

    # Strip final y normalización de saltos de línea
    texto = texto.strip()
    texto = re.sub(r'\n{2,}', '\n\n', texto)
    texto = re.sub(r'\n', '', texto)  # Quitar múltiples saltos de línea
    
    #Eliminar mensajes entre mayor-menor
    texto = re.sub(r'<[^>]*>', '', texto)
    

    return texto

def configurar_modelo(api_key):
    """Configura y devuelve el modelo de Groq."""
    llm = ChatGroq(
        api_key=api_key,
        model="llama3-8b-8192",
        temperature=0.25,  # Reducir temperatura para respuestas más consistentes
        max_tokens=512,   # Limitar tokens para ahorrar recursos
    )
    return llm

def crear_cadena_prompt(llm):
    """Crea y devuelve la cadena de prompt para extraer preguntas y respuestas."""
    # Plantilla para extraer pregunta y respuesta
    prompt = PromptTemplate(
        input_variables=["Cuerpo limpio"],
        template="""
Eres un modelo que extrae información útil de correos electrónicos relacionados con temas académicos.

Dado el siguiente cuerpo de correo:
"{Cuerpo limpio}"

Sigue estas instrucciones:

1. Elimina cualquier nombre propio (de estudiantes, profesores u otros), números de teléfono, documentos de identidad, fechas específicas (como "3 de mayo", "ayer", "2024-05-01"), códigos o nombres de materias (como "Cálculo 1", "Grupo 02", etc.) u otra información personal.
2. Si el correo hace referencia a un caso específico tratado anteriormente (por ejemplo: "como les comenté antes", "mi caso anterior", "ya abrí un caso"), no extraigas detalles. En su lugar, la respuesta debe ser:  
**"Por favor escribe al correo asuntosestudiantilesingenieria@udea.edu.co para revisar tu caso específico."**
3. Tu objetivo es generalizar. No respondas con información específica del caso si no aplica a todos los estudiantes.
4. Devuelve el resultado en el siguiente formato:
Pregunta: <aquí la pregunta principal en forma general>  
Respuesta: <aquí la respuesta general o la redirección mencionada en el punto 2>

Si no hay una pregunta, responde:
Pregunta: Ninguna  
Respuesta: Ninguna
    """
    )
    # Crear la cadena LLM
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain

def respect_rate_limit(request_times, max_requests_per_minute, delay_between_requests):
    """Gestiona el límite de tasa de peticiones a la API."""
    now = time.time()
    # Eliminar timestamps antiguos (más de 60 segundos)
    while request_times and now - request_times[0] > 60:
        request_times.pop(0)
    
    # Si estamos cerca del límite, esperar
    if len(request_times) >= max_requests_per_minute:
        wait_time = 60 - (now - request_times[0]) + 1  # +1 segundo de margen
        if wait_time > 0:
            print(f"Límite de API alcanzado. Esperando {wait_time:.2f} segundos...")
            time.sleep(wait_time)
    
    # Añadir timestamp actual
    request_times.append(time.time())
    
    # Siempre esperar un mínimo entre solicitudes
    time.sleep(delay_between_requests)
    
    return request_times

def extraer_pregunta_respuesta(output):
    """Extrae la pregunta y respuesta del output del modelo."""
    if "Pregunta:" in output and "Respuesta:" in output:
        try:
            pregunta = output.split("Pregunta:")[1].split("Respuesta:")[0].strip()
            respuesta = output.split("Respuesta:")[1].strip()
        except:
            pregunta = "Error en formato"
            respuesta = "Error en formato"
    else:
        pregunta = "No detectada"
        respuesta = "No detectada"
    
    return pregunta, respuesta

def procesar_correo(chain, cuerpo, request_times, max_requests_per_minute, 
                   delay_between_requests, max_retries=3):
    """Procesa un correo individual con sistema de reintentos."""
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Respetar límites de tasa antes de cada solicitud
            request_times = respect_rate_limit(
                request_times, max_requests_per_minute, delay_between_requests
            )
            
            # Hacer la solicitud a la API
            output = chain.run({"Cuerpo limpio": cuerpo})
            #output = extraer_qa_email(API_KEY,cuerpo) #Aqui hay un cambio en prueba
            
            # Extraer pregunta y respuesta
            pregunta, respuesta = extraer_pregunta_respuesta(output)
            
            # Si llegamos aquí, todo salió bien
            return pregunta, respuesta, request_times, None
            
        except Exception as e:
            retry_count += 1
            print(f"\nError: {str(e)}")
            print(f"Reintento {retry_count}/{max_retries}")
            
            # Esperar más tiempo en caso de error
            time.sleep(2 * retry_count)  # Backoff exponencial
            
            if retry_count >= max_retries:
                error_msg = f"Error: {str(e)[:100]}"
                return "Error API", error_msg, request_times, error_msg
            
def procesar_lote(df_lote, chain, request_times, max_requests_per_minute, 
                 delay_between_requests, max_retries=3):
    """Procesa un lote de correos."""
    errors = 0
    
    # Procesar cada correo en el lote actual
    for j, row in tqdm(df_lote.iterrows(), total=len(df_lote)):
        cuerpo = row["Cuerpo limpio"]
        
        # Procesar correo
        pregunta, respuesta, request_times, error = procesar_correo(
            chain, cuerpo, request_times, max_requests_per_minute, 
            delay_between_requests, max_retries
        )
        
        # Guardar resultados
        df_lote.at[j, "pregunta"] = pregunta
        df_lote.at[j, "respuesta"] = respuesta
        
        # Actualizar contador de errores
        if error:
            errors += 1
    
    return df_lote, request_times, errors

def crear_modelo_llm(api_key, modelo="mistral-saba-24b", temperatura=0.1, max_tokens=512):
    """
    Configura y devuelve un modelo LLM de Groq con los parámetros especificados.
    
    Args:
        api_key (str): Clave API de Groq
        modelo (str): Modelo a utilizar (default: "mistral-saba-24b")
        temperatura (float): Temperatura para la generación (default: 0.25)
        max_tokens (int): Máximo de tokens a generar (default: 512)
    
    Returns:
        ChatGroq: Instancia configurada del modelo
    """
    llm = ChatGroq(
        api_key=api_key,
        model=modelo,
        temperature=temperatura,
        max_tokens=max_tokens
    )
    return llm

def crear_cadena_con_plantilla(llm, plantilla, variables_entrada):
    """
    Crea una cadena LLM con una plantilla personalizada.
    
    Args:
        llm: Modelo LLM configurado
        plantilla (str): Texto de la plantilla
        variables_entrada (list): Lista de nombres de variables para la plantilla
    
    Returns:
        LLMChain: Cadena configurada
    """
    prompt = PromptTemplate(
        input_variables=variables_entrada,
        template=plantilla
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain

def ejecutar_cadena(chain, valores_variables):
    """
    Ejecuta una cadena LLM con los valores proporcionados.
    
    Args:
        chain (LLMChain): Cadena a ejecutar
        valores_variables (dict): Diccionario con valores para las variables
    
    Returns:
        str: Respuesta generada por el modelo
    """
    respuesta = chain.run(valores_variables)
    return respuesta

PLANTILLA_CONTEXTO_PREGUNTA = """Basado en el siguiente contexto, responde esta pregunta de forma clara, precisa, sin númeraciones ni listas, en un solo parráfo de 1000 o menos caracteres:

{contexto}

Pregunta: {pregunta}
"""

PLANTILLA_EXTRACCION_EMAIL = """
Eres un modelo que extrae información útil de correos electrónicos relacionados con temas académicos.
Dado el siguiente cuerpo de correo:
"{Cuerpo limpio}"
Sigue estas instrucciones:
1. Elimina cualquier nombre propio (de estudiantes, profesores u otros), números de teléfono, documentos de identidad, fechas específicas (como "3 de mayo", "ayer", "2024-05-01"), códigos o nombres de materias (como "Cálculo 1", "Grupo 02", etc.) u otra información personal.
2. Si el correo hace referencia a un caso específico tratado anteriormente (por ejemplo: "como les comenté antes", "mi caso anterior", "ya abrí un caso"), no extraigas detalles. En su lugar, la respuesta debe ser:  
**"Por favor escribe a asuntosestudiantilesingenieria@udea.edu.co para revisar tu caso específico."**
3. Tu objetivo es generalizar. No respondas con información específica del caso si no aplica a todos los estudiantes.
4. Devuelve el resultado en el siguiente formato:
Pregunta: <aquí la pregunta principal en forma general>  
Respuesta: <aquí la respuesta general o la redirección mencionada en el punto 2>
Si no hay una pregunta, responde:
Pregunta: Ninguna  
Respuesta: Ninguna
"""

def responder_pregunta_contexto(api_key, contexto, pregunta, modelo="mistral-saba-24b"):
    """
    Función específica para responder preguntas basadas en un contexto.
    
    Args:
        api_key (str): Clave API de Groq
        contexto (str): Contexto para la pregunta
        pregunta (str): Pregunta a responder
        modelo (str): Modelo a utilizar (default: "mistral-saba-24b")
    
    Returns:
        str: Respuesta generada por el modelo
    """
    llm = crear_modelo_llm(api_key, modelo)
    chain = crear_cadena_con_plantilla(
        llm, 
        PLANTILLA_CONTEXTO_PREGUNTA, 
        ["contexto", "pregunta"]
    )
    return ejecutar_cadena(chain, {"contexto": contexto, "pregunta": pregunta})

def extraer_qa_email(cuerpo_email, api_key=API_KEY, modelo="llama3-8b-8192"):
    """
    Función específica para extraer preguntas y respuestas de emails.
    
    Args:
        api_key (str): Clave API de Groq
        cuerpo_email (str): Contenido del email
        modelo (str): Modelo a utilizar (default: "llama3-8b-8192")
    
    Returns:
        str: Pregunta y respuesta extraídas
    """
    llm = crear_modelo_llm(api_key, modelo)
    chain = crear_cadena_con_plantilla(
        llm, 
        PLANTILLA_EXTRACCION_EMAIL, 
        ["Cuerpo limpio"]
    )
    return ejecutar_cadena(chain, {"Cuerpo limpio": cuerpo_email})

def guardar_resultados(df, nombre_archivo="resultados.xlsx"):
    """Guarda los resultados en un archivo Excel."""
    df.to_excel(nombre_archivo, index=False)
    print(f"Resultados guardados en '{nombre_archivo}'")

ruta_pdfs = "/home/jovyan/PI2_Text4U/BD-CorreosEstudiantes/"