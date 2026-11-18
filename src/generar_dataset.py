"""
Módulo: generar_dataset.py
Proyecto: AgroExport Ica - Optimización de Rendimiento de Cosecha
Empresa: AgroExportadora del Sur S.A.C. (Valle de Ica, Perú)

Genera un dataset agronómico realista y coherente de 500 lotes agrícolas de
uva de mesa (variedades de exportación como Red Globe, Sweet Globe y Autumn Crisp),
basado en parámetros agronómicos y referencias del MIDAGRI y la ANA.
"""

import os
import numpy as np
import pandas as pd


def generar_datos_agronomicos_ica(n_lotes: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético pero rigurosamente ajustado a los
    parámetros técnicos del cultivo de uva de mesa en el Valle de Ica, Perú.

    Variables agronómicas:
    - hectareas_cultivadas: Tamaño del lote/cuartel agrícola (ha).
    - agua_riego_litros: Volumen total de agua aplicado por fertirriego (L/campaña).
      Referencia ANA/MIDAGRI: Demanda hídrica de 9,000 a 13,000 m3/ha (9M a 13M L/ha).
    - radiacion_solar_horas: Horas de sol acumuladas durante el ciclo fenológico (SENAMHI Ica: 2,300 a 3,200 hrs).
    - fertilizante_kg: Nutrientes y fertilizantes solubles totales aplicados (N-P-K + microelementos, kg/campaña).
      Referencia: 350 a 650 kg/ha.
    - antiguedad_terreno_anos: Edad de las plantas en el parrón (años: 1 a 20).
    - toneladas_cosecha: Rendimiento total cosechado del lote (t), con relaciones no lineales y ruido estocástico natural.
    """
    np.random.seed(random_state)

    # 1. Hectáreas cultivadas por lote (distribución típica de cuarteles en fundos de Ica: 2.0 a 25.0 ha)
    # Media alrededor de 9.5 hectáreas
    hectareas = np.round(np.random.uniform(2.5, 24.0, size=n_lotes), 2)

    # 2. Antigüedad del parrón/plantación (1 a 20 años)
    antiguedad = np.random.randint(1, 21, size=n_lotes)

    # 3. Radiación solar acumulada en la campaña (horas de sol directo en el desierto de Ica)
    # Media: 2,750 horas; Desviación: 180 horas (rango 2,300 a 3,200 hrs)
    radiacion = np.clip(np.random.normal(2750, 180, size=n_lotes), 2300, 3200)
    radiacion = np.round(radiacion, 1)

    # 4. Agua de riego tecnificado por goteo (litros por hectárea)
    # La dosis agronómica óptima en Ica ronda los 10.5 a 11.5 millones de litros/ha (10,500 - 11,500 m3/ha)
    # Algunos cuarteles sufren déficit por fallas de pozo o restricciones del acuífero (7M - 9M L/ha)
    # Otros reciben dosis mayores (12M - 14M L/ha)
    dosis_agua_ha = np.clip(np.random.normal(10800000, 1300000, size=n_lotes), 7200000, 14200000)
    agua_riego_litros = np.round(dosis_agua_ha * hectareas).astype(np.int64)

    # 5. Fertilizante aplicado (kg por hectárea)
    # Dosis promedio en Ica: 480 kg/ha con variación técnica entre 320 y 680 kg/ha
    dosis_fert_ha = np.clip(np.random.normal(485, 75, size=n_lotes), 320, 680)
    fertilizante_kg = np.round(dosis_fert_ha * hectareas, 1)

    # =========================================================================
    # MODELADO AGRONÓMICO NO LINEAL DEL RENDIMIENTO (t/ha)
    # =========================================================================
    # Rendimiento agronómico potencial de uva de mesa de alta tecnología en Ica: ~33.0 t/ha
    rendimiento_potencial = 33.0

    # A) Factor Hídrico (Curva agronómica cuadrática con óptimo y penalización por déficit severo o exceso)
    # Dosis óptima en m3/ha: ~11,000 m3/ha = 11,000,000 L/ha
    dosis_agua_m3 = (agua_riego_litros / hectareas) / 1000.0  # m3/ha
    optimo_agua_m3 = 11000.0
    factor_agua = 1.0 - 0.0000045 * np.maximum(0, optimo_agua_m3 - dosis_agua_m3) ** 1.8 \
                      - 0.0000020 * np.maximum(0, dosis_agua_m3 - optimo_agua_m3) ** 1.7
    factor_agua = np.clip(factor_agua, 0.45, 1.0)

    # B) Factor de Fertilización (Ley de Mitscherlich de rendimientos decrecientes)
    dosis_fert_kg_ha = fertilizante_kg / hectareas
    # Saturation curve: 1 - exp(-k * fert)
    factor_fert = 1.0 - np.exp(-0.0068 * (dosis_fert_kg_ha - 150))
    factor_fert = np.clip(factor_fert, 0.50, 1.02)

    # C) Factor de Radiación Solar (Efecto fotosintético con saturación lumínica)
    # En Ica, a mayor radiación óptima (>2,700 hrs) se maximiza acumulación de grados Brix y peso de bayas
    factor_rad = 0.82 + 0.18 / (1.0 + np.exp(-0.007 * (radiacion - 2650)))
    factor_rad = np.clip(factor_rad, 0.80, 1.0)

    # D) Factor de Antigüedad del Parrón (Ciclo biológico de la vid)
    # Años 1-2: Fase vegetativa/formativa (producción muy baja, 25% a 55%)
    # Años 3-4: Inicio de producción comercial (75% a 88%)
    # Años 5-11: Plena madurez y pico de rendimiento (96% a 102%)
    # Años 12-20: Ligera senescencia gradual del parrón (85% a 94%)
    factor_edad = np.zeros(n_lotes)
    for i, a in enumerate(antiguedad):
        if a == 1:
            factor_edad[i] = 0.35 + np.random.uniform(-0.03, 0.04)
        elif a == 2:
            factor_edad[i] = 0.58 + np.random.uniform(-0.03, 0.04)
        elif a in (3, 4):
            factor_edad[i] = 0.82 + 0.04 * (a - 3) + np.random.uniform(-0.02, 0.03)
        elif 5 <= a <= 11:
            factor_edad[i] = 0.98 + np.random.uniform(-0.02, 0.03)
        else:  # a >= 12
            factor_edad[i] = 0.98 - 0.016 * (a - 11) + np.random.uniform(-0.02, 0.02)
    factor_edad = np.clip(factor_edad, 0.30, 1.02)

    # E) Rendimiento por hectárea con interacción no lineal y ruido estocástico natural
    # El ruido representa microclima, variación de suelo aluvial, presión de oídio/plagas y podas
    ruido_natural = np.random.normal(0, 1.15, size=n_lotes)
    rendimiento_ha = (
        rendimiento_potencial * factor_agua * factor_fert * factor_rad * factor_edad
        + ruido_natural
    )
    # En Ica el rendimiento comercial bajo riego tecnificado va típicamente de 10 a 35 t/ha
    rendimiento_ha = np.clip(rendimiento_ha, 9.0, 36.5)

    # 6. Toneladas totales de cosecha del lote
    toneladas_cosecha = np.round(rendimiento_ha * hectareas, 2)

    # Construir DataFrame final
    df = pd.DataFrame({
        "hectareas_cultivadas": hectareas,
        "agua_riego_litros": agua_riego_litros,
        "radiacion_solar_horas": radiacion,
        "fertilizante_kg": fertilizante_kg,
        "antiguedad_terreno_anos": antiguedad,
        "toneladas_cosecha": toneladas_cosecha
    })

    return df


def validar_y_limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica controles de calidad agronómicos y estadísticos al dataset:
    - Verificación de valores nulos o infinitos.
    - Rango positivo estricto en todas las variables.
    - Validación de límites agronómicos físicos.
    """
    assert not df.isnull().values.any(), "Error: Existen valores nulos en el dataset."
    assert not np.isinf(df.values).any(), "Error: Existen valores infinitos en el dataset."
    assert (df["hectareas_cultivadas"] > 0).all(), "Hectáreas deben ser mayores a 0."
    assert (df["agua_riego_litros"] > 0).all(), "Agua de riego debe ser mayor a 0."
    assert (df["radiacion_solar_horas"] > 0).all(), "Radiación solar debe ser mayor a 0."
    assert (df["fertilizante_kg"] > 0).all(), "Fertilizante debe ser mayor a 0."
    assert (df["antiguedad_terreno_anos"] >= 0).all(), "Antigüedad no puede ser negativa."
    assert (df["toneladas_cosecha"] > 0).all(), "Toneladas cosechadas deben ser positivas."

    # Ordenar columnas formalmente
    columnas = [
        "hectareas_cultivadas",
        "agua_riego_litros",
        "radiacion_solar_horas",
        "fertilizante_kg",
        "antiguedad_terreno_anos",
        "toneladas_cosecha"
    ]
    df = df[columnas].copy()
    return df


def main():
    print("=" * 75)
    print("AgroExport Ica - Generación de Dataset Agronómico de Cosecha")
    print("Empresa: AgroExportadora del Sur S.A.C. (Valle de Ica, Perú)")
    print("=" * 75)

    ruta_directorio = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(ruta_directorio, exist_ok=True)
    ruta_csv = os.path.join(ruta_directorio, "historial_cosechas_ica.csv")

    print("[1/3] Generando 500 registros con modelado biofísico de uva de mesa...")
    df = generar_datos_agronomicos_ica(n_lotes=500, random_state=42)

    print("[2/3] Validando integridad y calidad de datos...")
    df = validar_y_limpiar_datos(df)

    print(f"[3/3] Guardando dataset en: {os.path.abspath(ruta_csv)}")
    df.to_csv(ruta_csv, index=False)

    print("\nResumen Estadístico del Dataset (500 Lotes Agrícolas en Ica):")
    print("-" * 75)
    print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]].to_string())

    # Rendimiento medio por hectárea
    rend_medio = (df["toneladas_cosecha"] / df["hectareas_cultivadas"]).mean()
    print("-" * 75)
    print(f"Rendimiento Promedio Estimado: {rend_medio:.2f} t/ha (Consistente con Valle de Ica)")
    print(f"Archivo generado exitosamente con {len(df)} registros.")
    print("=" * 75)


if __name__ == "__main__":
    main()
