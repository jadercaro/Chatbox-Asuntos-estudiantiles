import pandas as pd
from app.services.utils import (
    extraer_respuestas_email,
    limpiar_dataframe,
    limpiar_texto,
    guardar_resultados,
    configurar_modelo,
    crear_cadena_prompt,
    procesar_lote
)

def process_file(file):
    df = extraer_respuestas_email(ruta_pdfs)

    df_limpio = limpiar_dataframe(df)
    df_limpio = limpiar_texto(df_limpio)

    # Configuración para el control de límites de API
    MAX_REQUESTS_PER_MINUTE = 45  # Groq permite hasta 30 por minuto, dejamos margen
    BATCH_SIZE = 10  # Procesar en pequeños lotes
    DELAY_BETWEEN_REQUESTS = 60 / MAX_REQUESTS_PER_MINUTE  # Calcular tiempo entre solicitudes
    MAX_RETRIES = 3

    df1 = df_limpio.copy()
    df1["pregunta"] = ""
    df1["respuesta"] = ""

    # Configurar API y cadena
    api_key = "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"
    llm = configurar_modelo(api_key)
    chain = crear_cadena_prompt(llm)

    request_times = []
    total_errors = 0
    for i in range(0, len(df1), BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, len(df1))
        print(f"\nProcesando lote {i//BATCH_SIZE + 1}: registros {i} a {end_idx-1}")
        
        # Obtener lote actual
        df_lote = df1.iloc[i:end_idx].copy()
        
        # Procesar lote
        df_actualizado, request_times, errors = procesar_lote(
            df_lote, chain, request_times, MAX_REQUESTS_PER_MINUTE, 
            DELAY_BETWEEN_REQUESTS, MAX_RETRIES
        )
        
        df1.iloc[i:end_idx] = df_actualizado
        total_errors += errors
        
        guardar_resultados(df1, "app\\services\\Conformed\\resultados_parciales--fecha--.xlsx")
    guardar_resultados(df1, "app\\services\\Conformed\\resultados_finales--fecha--.xlsx")
    return {"status": "preprocessed"}