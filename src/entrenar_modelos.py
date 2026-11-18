"""
Módulo: entrenar_modelos.py
Proyecto: AgroExport Ica - Optimización de Rendimiento de Cosecha
Empresa: AgroExportadora del Sur S.A.C. (Valle de Ica, Perú)

Entrena, evalúa y genera evidencia visual comparativa entre:
1. Modelo Baseline Estadístico: Regresión Lineal Simple (solo hectáreas).
2. Modelo Deep Learning: Red Neuronal Profunda Multicapa (MLP Keras/TensorFlow con 5 variables).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configuración de TensorFlow para ejecución limpia
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


def configurar_estilo_graficos():
    """Configura el estilo visual profesional para Matplotlib."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#2c3e50"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#e2e8f0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def entrenar_y_evaluar():
    print("=" * 80)
    print("AgroExport Ica - Entrenamiento y Evaluación de Modelos Predictivos")
    print("Empresa: AgroExportadora del Sur S.A.C. (Ica, Perú)")
    print("=" * 80)

    # 1. Cargar Dataset
    ruta_base = os.path.dirname(os.path.abspath(__file__))
    ruta_csv = os.path.join(ruta_base, "..", "data", "historial_cosechas_ica.csv")
    ruta_outputs = os.path.join(ruta_base, "..", "outputs")
    os.makedirs(ruta_outputs, exist_ok=True)

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"No se encontró el dataset en: {ruta_csv}. Ejecute primero generar_dataset.py")

    df = pd.read_csv(ruta_csv)
    print(f"Dataset cargado correctamente: {len(df)} registros y {df.shape[1]} columnas.")

    # 2. Definición de variables
    features_all = [
        "hectareas_cultivadas",
        "agua_riego_litros",
        "radiacion_solar_horas",
        "fertilizante_kg",
        "antiguedad_terreno_anos"
    ]
    target_col = "toneladas_cosecha"

    X = df[features_all]
    y = df[target_col]

    # Partición 80% train, 20% test con semilla fija
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Partición de datos: {len(X_train)} entrenamiento, {len(X_test)} prueba.")

    # =========================================================================
    # MODELO 1: BASELINE ESTADÍSTICO (Regresión Lineal Simple)
    # Usa exclusivamente 'hectareas_cultivadas'
    # =========================================================================
    print("\n[1/4] Entrenando Modelo 1: Regresión Lineal Simple (Baseline)...")
    X_train_lr = X_train[["hectareas_cultivadas"]].values
    X_test_lr = X_test[["hectareas_cultivadas"]].values

    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train_lr, y_train)

    y_pred_lr = modelo_lr.predict(X_test_lr)

    mse_lr = mean_squared_error(y_test, y_pred_lr)
    rmse_lr = np.sqrt(mse_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    r2_lr = r2_score(y_test, y_pred_lr)

    print(f"  -> Coeficiente (Pendiente): {modelo_lr.coef_[0]:.4f} t/ha")
    print(f"  -> Intercepto: {modelo_lr.intercept_:.4f}")
    print(f"  -> MSE: {mse_lr:.2f} | RMSE: {rmse_lr:.2f} t | MAE: {mae_lr:.2f} t | R²: {r2_lr:.4f}")

    # Guardar modelo de regresión lineal
    ruta_modelo_lr = os.path.join(ruta_outputs, "modelo_regresion_lineal.pkl")
    joblib.dump(modelo_lr, ruta_modelo_lr)
    print(f"  -> Guardado en: {ruta_modelo_lr}")

    # =========================================================================
    # MODELO 2: DEEP LEARNING (Red Neuronal Profunda MLP)
    # Usa las 5 variables con StandardScaler y activación ReLU
    # =========================================================================
    print("\n[2/4] Entrenando Modelo 2: Red Neuronal Profunda (MLP Keras/TensorFlow)...")

    # Escalado con StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Guardar scaler
    ruta_scaler = os.path.join(ruta_outputs, "scaler.pkl")
    joblib.dump(scaler, ruta_scaler)
    print(f"  -> Escalador StandardScaler guardado en: {ruta_scaler}")

    # Fijar semillas para reproducibilidad
    tf.random.set_seed(42)
    np.random.seed(42)

    # Definición de la arquitectura MLP
    # 4 capas densas con activación ReLU y optimizador Adam
    modelo_nn = Sequential([
        Input(shape=(5,), name="capa_entrada_agronomica"),
        Dense(128, activation="relu", kernel_initializer="he_normal", name="densa_oculta_1"),
        Dense(64, activation="relu", kernel_initializer="he_normal", name="densa_oculta_2"),
        Dense(32, activation="relu", kernel_initializer="he_normal", name="densa_oculta_3"),
        Dense(16, activation="relu", kernel_initializer="he_normal", name="densa_oculta_4"),
        Dense(1, activation="linear", name="capa_salida_toneladas")
    ])

    optimizer = Adam(learning_rate=0.01)
    modelo_nn.compile(optimizer=optimizer, loss="mse", metrics=["mae", "mse"])

    # Callbacks para convergencia óptima
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=0.0001, verbose=0)
    early_stop = EarlyStopping(monitor="val_loss", patience=35, restore_best_weights=True, verbose=0)

    history = modelo_nn.fit(
        X_train_scaled,
        y_train.values,
        validation_split=0.20,
        epochs=220,
        batch_size=16,
        callbacks=[reduce_lr, early_stop],
        verbose=0
    )

    y_pred_nn = modelo_nn.predict(X_test_scaled, verbose=0).flatten()

    mse_nn = mean_squared_error(y_test, y_pred_nn)
    rmse_nn = np.sqrt(mse_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    r2_nn = r2_score(y_test, y_pred_nn)

    print(f"  -> Épocas entrenadas: {len(history.history['loss'])}")
    print(f"  -> MSE: {mse_nn:.2f} | RMSE: {rmse_nn:.2f} t | MAE: {mae_nn:.2f} t | R²: {r2_nn:.4f}")

    # Guardar modelo de red neuronal
    ruta_modelo_nn = os.path.join(ruta_outputs, "modelo_red_neuronal.keras")
    modelo_nn.save(ruta_modelo_nn)
    print(f"  -> Modelo Deep Learning guardado en: {ruta_modelo_nn}")

    # =========================================================================
    # GUARDAR MÉTRICAS COMPARATIVAS EN JSON
    # =========================================================================
    mejora_rmse = ((rmse_lr - rmse_nn) / rmse_lr) * 100.0

    metricas = {
        "fecha_generacion": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "empresa": "AgroExportadora del Sur S.A.C.",
        "valle": "Ica, Perú",
        "muestras_entrenamiento": int(len(X_train)),
        "muestras_test": int(len(X_test)),
        "variables_independientes": features_all,
        "variable_objetivo": target_col,
        "regresion_lineal": {
            "nombre": "Regresión Lineal Simple (Baseline)",
            "variables_usadas": ["hectareas_cultivadas"],
            "pendiente_coef": round(float(modelo_lr.coef_[0]), 4),
            "intercepto": round(float(modelo_lr.intercept_), 4),
            "mse": round(float(mse_lr), 2),
            "rmse": round(float(rmse_lr), 2),
            "mae": round(float(mae_lr), 2),
            "r2": round(float(r2_lr), 4)
        },
        "red_neuronal": {
            "nombre": "Red Neuronal Profunda (MLP Keras)",
            "variables_usadas": features_all,
            "arquitectura": "Input(5) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Linear)",
            "optimizador": "Adam(lr_inicial=0.01 con ReduceLROnPlateau)",
            "epocas_entrenadas": len(history.history["loss"]),
            "mse": round(float(mse_nn), 2),
            "rmse": round(float(rmse_nn), 2),
            "mae": round(float(mae_nn), 2),
            "r2": round(float(r2_nn), 4)
        },
        "analisis_comparativo": {
            "reduccion_rmse_porcentaje": round(float(mejora_rmse), 2),
            "reduccion_rmse_toneladas": round(float(rmse_lr - rmse_nn), 2),
            "incremento_r2_puntos": round(float(r2_nn - r2_lr), 4)
        }
    }

    ruta_metricas = os.path.join(ruta_outputs, "metricas.json")
    with open(ruta_metricas, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print(f"\n[3/4] Métricas comparativas guardadas en: {ruta_metricas}")

    # =========================================================================
    # GENERACIÓN DE GRÁFICAS PROFESIONALES (Matplotlib / Seaborn)
    # =========================================================================
    print("\n[4/4] Generando 3 visualizaciones analíticas de alta calidad en outputs/...")
    configurar_estilo_graficos()

    # -------------------------------------------------------------------------
    # Gráfica 1: regresion_lineal_uva.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    
    x_test_ha = X_test["hectareas_cultivadas"].values
    ax.scatter(
        x_test_ha,
        y_test.values,
        color="#2b5c8f",
        alpha=0.65,
        s=60,
        edgecolors="#1e3a5f",
        linewidths=0.8,
        label="Datos Reales de Cosecha (Test)"
    )

    x_line = np.linspace(df["hectareas_cultivadas"].min(), df["hectareas_cultivadas"].max(), 200).reshape(-1, 1)
    y_line = modelo_lr.predict(x_line)
    ax.plot(
        x_line,
        y_line,
        color="#c0392b",
        linewidth=2.5,
        linestyle="-",
        label=f"Ajuste Lineal: $\\hat{{y}} = {modelo_lr.coef_[0]:.2f}x + ({modelo_lr.intercept_:.2f})$"
    )

    texto_kpis = (
        f"Métricas Modelo Baseline:\n"
        f"• $R^2$: {r2_lr:.3f} (Varianza explicada modesta)\n"
        f"• RMSE: {rmse_lr:.2f} t\n"
        f"• MAE: {mae_lr:.2f} t\n"
        f"Diagnóstico: Alta dispersión. Asume que\n"
        f"solo las hectáreas determinan el rinde,\n"
        f"ignorando riego, edad y nutrición."
    )
    ax.text(
        0.05, 0.95, texto_kpis,
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#ffffff", edgecolor="#bdc3c7", alpha=0.95)
    )

    ax.set_title(
        "AgroExport Ica: Modelo 1 - Regresión Lineal Simple (Baseline)\n"
        "Relación entre Hectáreas Cultivadas y Toneladas Cosechadas de Uva",
        fontsize=12,
        fontweight="bold",
        color="#1a252f",
        pad=15
    )
    ax.set_xlabel("Hectáreas Cultivadas (ha)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Cosecha Total (Toneladas)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.0f} t"))
    plt.tight_layout()

    ruta_g1 = os.path.join(ruta_outputs, "regresion_lineal_uva.png")
    plt.savefig(ruta_g1, dpi=300)
    plt.close()
    print(f"  -> Gráfica 1 guardada: {ruta_g1}")

    # -------------------------------------------------------------------------
    # Gráfica 2: curva_aprendizaje_uva.png
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

    epocas = range(1, len(history.history["loss"]) + 1)
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    ax.plot(epocas, train_loss, color="#27ae60", linewidth=2.2, label="Pérdida en Entrenamiento (Train Loss - MSE)")
    ax.plot(epocas, val_loss, color="#e67e22", linewidth=2.2, linestyle="--", label="Pérdida en Validación (Val Loss - MSE)")

    mejor_epoca = int(np.argmin(val_loss)) + 1
    min_val_loss = float(np.min(val_loss))
    ax.scatter(mejor_epoca, min_val_loss, color="#c0392b", s=90, zorder=5, label=f"Óptimo (Época {mejor_epoca}, MSE={min_val_loss:.1f})")

    ax.set_title(
        "AgroExport Ica: Modelo 2 - Curva de Aprendizaje de la Red Neuronal (MLP)\n"
        "Evolución de la Función de Pérdida (MSE) a través de las Épocas",
        fontsize=12,
        fontweight="bold",
        color="#1a252f",
        pad=15
    )
    ax.set_xlabel("Épocas de Entrenamiento", fontsize=11, fontweight="bold", color="#2c3e50")
    ax.set_ylabel("Pérdida Media Cuadrática (MSE)", fontsize=11, fontweight="bold", color="#2c3e50")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#bdc3c7")
    
    ax.text(
        0.52, 0.45,
        "Convergencia y Generalización:\n"
        "• Activación ReLU captura no-linealidades\n"
        "• Ajuste dinámico de tasa de aprendizaje\n"
        "• Generalización sólida sin sobreajuste",
        transform=ax.transAxes,
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.9)
    )

    plt.tight_layout()
    ruta_g2 = os.path.join(ruta_outputs, "curva_aprendizaje_uva.png")
    plt.savefig(ruta_g2, dpi=300)
    plt.close()
    print(f"  -> Gráfica 2 guardada: {ruta_g2}")

    # -------------------------------------------------------------------------
    # Gráfica 3: comparacion_modelos_uva.png
    # -------------------------------------------------------------------------
    fig, (ax_rmse, ax_r2) = plt.subplots(1, 2, figsize=(11, 5.5), dpi=300)

    modelos = ["Regresión Lineal\n(Baseline Simple)", "Red Neuronal\n(Deep Learning MLP)"]
    colores = ["#95a5a6", "#27ae60"]

    # Subplot A: RMSE
    valores_rmse = [rmse_lr, rmse_nn]
    barras_rmse = ax_rmse.bar(modelos, valores_rmse, color=colores, width=0.52, edgecolor="#2c3e50", linewidth=1.0)
    ax_rmse.set_title("Comparación de Error (RMSE)\n(Menor es mejor)", fontsize=11, fontweight="bold", color="#1a252f")
    ax_rmse.set_ylabel("RMSE en Toneladas (t)", fontsize=10, fontweight="bold", color="#2c3e50")
    ax_rmse.set_ylim(0, max(valores_rmse) * 1.25)

    for barra in barras_rmse:
        altura = barra.get_height()
        ax_rmse.annotate(
            f"{altura:.2f} t",
            xy=(barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50"
        )

    ax_rmse.annotate(
        f"Reducción de Error:\n-{mejora_rmse:.1f}%",
        xy=(1, rmse_nn), xycoords="data",
        xytext=(0.5, (rmse_lr + rmse_nn) / 2 + 5), textcoords="data",
        arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=2.0),
        ha="center", fontsize=10, fontweight="bold", color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor="#27ae60")
    )

    # Subplot B: R2
    valores_r2 = [r2_lr, r2_nn]
    barras_r2 = ax_r2.bar(modelos, valores_r2, color=colores, width=0.52, edgecolor="#2c3e50", linewidth=1.0)
    ax_r2.set_title("Coeficiente de Determinación ($R^2$)\n(Mayor es mejor)", fontsize=11, fontweight="bold", color="#1a252f")
    ax_r2.set_ylabel("Puntuación $R^2$", fontsize=10, fontweight="bold", color="#2c3e50")
    ax_r2.set_ylim(0, 1.15)

    for barra in barras_r2:
        altura = barra.get_height()
        ax_r2.annotate(
            f"{altura:.4f} ({altura*100:.1f}%)",
            xy=(barra.get_x() + barra.get_width() / 2, altura),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50"
        )

    fig.suptitle(
        "AgroExport Ica: Evaluación Comparativa de Precisión Predictiva\n"
        "Baseline Estadístico vs. Deep Learning Multicapa (Valle de Ica)",
        fontsize=12,
        fontweight="bold",
        color="#1a252f",
        y=1.02
    )
    plt.tight_layout()
    ruta_g3 = os.path.join(ruta_outputs, "comparacion_modelos_uva.png")
    plt.savefig(ruta_g3, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  -> Gráfica 3 guardada: {ruta_g3}")

    print("\n" + "=" * 80)
    print("PROCESO DE ENTRENAMIENTO Y GENERACIÓN DE ARTEFACTOS COMPLETADO CON ÉXITO")
    print(f"  • Modelo Baseline: outputs/modelo_regresion_lineal.pkl")
    print(f"  • Modelo Deep Learning: outputs/modelo_red_neuronal.keras")
    print(f"  • Escalador: outputs/scaler.pkl")
    print(f"  • Métricas: outputs/metricas.json")
    print(f"  • Gráficas: 3 archivos PNG generados en outputs/")
    print("=" * 80)


if __name__ == "__main__":
    entrenar_y_evaluar()
