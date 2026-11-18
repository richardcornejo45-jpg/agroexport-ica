"""
Módulo: main.py
Proyecto: AgroExport Ica - API de Predicción & Optimización de Cosecha
Empresa: AgroExportadora del Sur S.A.C. (Valle de Ica, Perú)

Backend en FastAPI para servir inferencias de Machine Learning (Regresión Lineal y Deep Learning)
junto con cubicaje logístico de contenedores refrigerados para exportación.
"""

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Desactivar avisos innecesarios de TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


# Rutas de artefactos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

RUTA_MODELO_LR = os.path.join(OUTPUTS_DIR, "modelo_regresion_lineal.pkl")
RUTA_MODELO_NN = os.path.join(OUTPUTS_DIR, "modelo_red_neuronal.keras")
RUTA_SCALER = os.path.join(OUTPUTS_DIR, "scaler.pkl")
RUTA_METRICAS = os.path.join(OUTPUTS_DIR, "metricas.json")

# Almacén de modelos en memoria
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga los modelos entrenados y artefactos al iniciar la aplicación."""
    print("Iniciando servicio AgroExport Ica API...")
    try:
        if os.path.exists(RUTA_MODELO_LR):
            ml_models["modelo_lr"] = joblib.load(RUTA_MODELO_LR)
            print("  [OK] Modelo Regresión Lineal cargado.")
        else:
            print(f"  [ALERTA] No se encontró {RUTA_MODELO_LR}")

        if os.path.exists(RUTA_SCALER):
            ml_models["scaler"] = joblib.load(RUTA_SCALER)
            print("  [OK] Escalador StandardScaler cargado.")
        else:
            print(f"  [ALERTA] No se encontró {RUTA_SCALER}")

        if os.path.exists(RUTA_MODELO_NN):
            ml_models["modelo_nn"] = tf.keras.models.load_model(RUTA_MODELO_NN)
            print("  [OK] Modelo Red Neuronal Keras cargado.")
        else:
            print(f"  [ALERTA] No se encontró {RUTA_MODELO_NN}")

        if os.path.exists(RUTA_METRICAS):
            with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
                ml_models["metricas"] = json.load(f)
            print("  [OK] Archivo metricas.json cargado.")
        else:
            print(f"  [ALERTA] No se encontró {RUTA_METRICAS}")

    except Exception as e:
        print(f"Error cargando modelos: {e}")

    yield
    print("Cerrando servicio AgroExport Ica API...")
    ml_models.clear()


app = FastAPI(
    title="AgroExport Ica - API de Predicción de Cosecha & Logística",
    description="Sistema inteligente para AgroExportadora del Sur S.A.C. (Valle de Ica, Perú). "
                "Predice rendimiento de uva de mesa y cubicaje de contenedores refrigerados.",
    version="1.0.0",
    lifespan=lifespan
)

# Habilitar CORS para integración externa
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ESQUEMAS PYDANTIC DE ENTRADA Y SALIDA
# =============================================================================
class LoteAgronomicoInput(BaseModel):
    hectareas_cultivadas: float = Field(
        ...,
        gt=0,
        le=150,
        description="Superficie cultivada del cuartel en hectáreas (ha)",
        json_schema_extra={"example": 12.5}
    )
    agua_riego_litros: float = Field(
        ...,
        gt=0,
        description="Volumen total de agua de riego aplicado por campaña (Litros)",
        json_schema_extra={"example": 135000000.0}
    )
    radiacion_solar_horas: float = Field(
        ...,
        gt=0,
        le=5000,
        description="Horas acumuladas de radiación solar en la campaña (horas)",
        json_schema_extra={"example": 2780.0}
    )
    fertilizante_kg: float = Field(
        ...,
        gt=0,
        description="Cantidad total de fertilizantes y nutrientes aplicados (kg)",
        json_schema_extra={"example": 6200.0}
    )
    antiguedad_terreno_anos: float = Field(
        ...,
        ge=0,
        le=50,
        description="Edad de las plantas en el parrón (años)",
        json_schema_extra={"example": 7.0}
    )


class PrediccionRespuesta(BaseModel):
    toneladas_red_neuronal: float
    rendimiento_ha_red_neuronal: float
    contenedores_red_neuronal: int
    capacidad_ultimo_contenedor_pct: float
    toneladas_regresion_lineal: float
    rendimiento_ha_regresion_lineal: float
    contenedores_regresion_lineal: int
    diferencia_toneladas: float
    diferencia_porcentaje: float
    capacidad_contenedor_estandar_t: float
    diagnostico_agronomico: str
    recomendacion_logistica: str


# =============================================================================
# ENDPOINTS DE LA API
# =============================================================================
@app.get("/health", tags=["Salud"])
def health_check():
    """Endpoint de verificación de estado y disponibilidad de modelos."""
    modelos_listos = (
        "modelo_lr" in ml_models
        and "modelo_nn" in ml_models
        and "scaler" in ml_models
    )
    return {
        "status": "online" if modelos_listos else "degraded",
        "empresa": "AgroExportadora del Sur S.A.C.",
        "valle": "Ica, Perú",
        "servicio": "AgroExport Ica Harvest Prediction Service",
        "modelos_cargados": {
            "regresion_lineal": "modelo_lr" in ml_models,
            "red_neuronal_keras": "modelo_nn" in ml_models,
            "standard_scaler": "scaler" in ml_models,
            "metricas_json": "metricas" in ml_models
        }
    }


@app.get("/api/metricas", tags=["Métricas"])
def obtener_metricas():
    """Retorna las métricas comparativas evaluadas en el conjunto de prueba."""
    if "metricas" in ml_models:
        return ml_models["metricas"]

    if os.path.exists(RUTA_METRICAS):
        with open(RUTA_METRICAS, "r", encoding="utf-8") as f:
            datos = json.load(f)
            ml_models["metricas"] = datos
            return datos

    raise HTTPException(status_code=404, detail="Archivo de métricas no encontrado. Ejecute entrenar_modelos.py")


@app.post("/predecir", response_model=PrediccionRespuesta, tags=["Predicción"])
def predecir_cosecha_y_logistica(datos: LoteAgronomicoInput):
    """
    Recibe las variables agronómicas del lote y estima:
    1. Toneladas cosechadas con Red Neuronal Profunda (MLP).
    2. Toneladas estimadas con Regresión Lineal Simple (Baseline).
    3. Cubicaje y requerimiento de contenedores refrigerados estándar (20 t por Reefer FEU).
    """
    if "modelo_lr" not in ml_models or "modelo_nn" not in ml_models or "scaler" not in ml_models:
        raise HTTPException(
            status_code=503,
            detail="Los modelos predictivos no están disponibles en memoria. Ejecute el pipeline de entrenamiento."
        )

    # 1. Inferencia Modelo 1: Regresión Lineal (solo hectáreas)
    df_lr = pd.DataFrame([{"hectareas_cultivadas": datos.hectareas_cultivadas}])
    pred_lr_raw = float(ml_models["modelo_lr"].predict(df_lr.values)[0])
    toneladas_lr = max(1.0, round(pred_lr_raw, 2))
    rinde_ha_lr = round(toneladas_lr / datos.hectareas_cultivadas, 2)

    # 2. Inferencia Modelo 2: Red Neuronal Profunda (5 variables normalizadas)
    df_nn = pd.DataFrame([{
        "hectareas_cultivadas": datos.hectareas_cultivadas,
        "agua_riego_litros": datos.agua_riego_litros,
        "radiacion_solar_horas": datos.radiacion_solar_horas,
        "fertilizante_kg": datos.fertilizante_kg,
        "antiguedad_terreno_anos": datos.antiguedad_terreno_anos
    }])

    vector_scaled = ml_models["scaler"].transform(df_nn)
    pred_nn_raw = float(ml_models["modelo_nn"].predict(vector_scaled, verbose=0)[0][0])
    toneladas_nn = max(1.0, round(pred_nn_raw, 2))
    rinde_ha_nn = round(toneladas_nn / datos.hectareas_cultivadas, 2)

    # 3. Logística de Contenedores Refrigerados (40 pies High Cube Reefer - 20 toneladas útiles estándar)
    CAPACIDAD_REEFER = 20.0  # toneladas métricas por contenedor
    contenedores_nn = math.ceil(toneladas_nn / CAPACIDAD_REEFER)
    contenedores_lr = math.ceil(toneladas_lr / CAPACIDAD_REEFER)

    # Ocupación del último contenedor
    remanente = toneladas_nn % CAPACIDAD_REEFER
    if remanente == 0 and toneladas_nn > 0:
        ocupacion_ultimo_pct = 100.0
    else:
        ocupacion_ultimo_pct = round((remanente / CAPACIDAD_REEFER) * 100.0, 1)

    # Diferencia entre modelos
    dif_t = round(toneladas_nn - toneladas_lr, 2)
    dif_pct = round(((toneladas_nn - toneladas_lr) / toneladas_lr) * 100.0, 2)

    # Diagnóstico agronómico y recomendación
    agua_m3_ha = (datos.agua_riego_litros / datos.hectareas_cultivadas) / 1000.0

    alertas = []
    if agua_m3_ha < 8500:
        alertas.append("Déficit hídrico detectado (< 8,500 m³/ha)")
    elif agua_m3_ha > 12500:
        alertas.append("Sobredosis hídrica (> 12,500 m³/ha) - riesgo de anoxia radicular")
    else:
        alertas.append("Manejo de riego tecnificado óptimo")

    if datos.antiguedad_terreno_anos <= 2:
        alertas.append("Parrón en formación inicial (rendimiento juvenil)")
    elif 5 <= datos.antiguedad_terreno_anos <= 11:
        alertas.append("Parrón en máxima madurez productiva")
    elif datos.antiguedad_terreno_anos > 12:
        alertas.append("Parrón maduro (ligero declive vegetativo)")

    diagnostico = " | ".join(alertas)
    recomendacion = (
        f"Programar {contenedores_nn} contenedores refrigerados (40' Reefer) con destino al "
        f"Terminal Portuario de Paracas (Pisco) / Callao. El último contenedor registra {ocupacion_ultimo_pct}% de llenado."
    )

    return PrediccionRespuesta(
        toneladas_red_neuronal=toneladas_nn,
        rendimiento_ha_red_neuronal=rinde_ha_nn,
        contenedores_red_neuronal=contenedores_nn,
        capacidad_ultimo_contenedor_pct=ocupacion_ultimo_pct,
        toneladas_regresion_lineal=toneladas_lr,
        rendimiento_ha_regresion_lineal=rinde_ha_lr,
        contenedores_regresion_lineal=contenedores_lr,
        diferencia_toneladas=dif_t,
        diferencia_porcentaje=dif_pct,
        capacidad_contenedor_estandar_t=CAPACIDAD_REEFER,
        diagnostico_agronomico=diagnostico,
        recomendacion_logistica=recomendacion
    )


# =============================================================================
# ARCHIVOS ESTÁTICOS: OUTPUTS Y FRONTEND
# =============================================================================
# Servir gráficas y artefactos en /outputs
if os.path.exists(OUTPUTS_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Servir frontend estático en /
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
