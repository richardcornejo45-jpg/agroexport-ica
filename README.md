# AgroExport Ica - Optimización de Rendimiento de Cosecha & Logística de Exportación

Sistema predictivo de Inteligencia Artificial y Deep Learning para la estimación de cosecha de uva de mesa y planificación logística portuaria de contenedores refrigerados para **AgroExportadora del Sur S.A.C.** (Valle de Ica, Perú).

---

## 🍇 Arquitectura del Proyecto

```
agroexport-ica/
├── data/
│   └── historial_cosechas_ica.csv       # Dataset agronómico de 500 lotes (MIDAGRI / ANA)
├── outputs/
│   ├── modelo_regresion_lineal.pkl      # Modelo Baseline Estadístico (Scikit-Learn)
│   ├── modelo_red_neuronal.keras        # Modelo Deep Learning MLP (TensorFlow / Keras)
│   ├── scaler.pkl                       # Escalador StandardScaler
│   ├── metricas.json                    # Métricas comparativas (MSE, RMSE, MAE, R²)
│   ├── regresion_lineal_uva.png         # Gráfica 1: Dispersión y ajuste lineal
│   ├── curva_aprendizaje_uva.png        # Gráfica 2: Curva de convergencia MSE Keras
│   └── comparacion_modelos_uva.png      # Gráfica 3: Comparativa de RMSE y R²
├── src/
│   ├── generar_dataset.py               # Generación y validación del dataset agronómico
│   ├── entrenar_modelos.py              # Entrenamiento, evaluación y graficación
│   └── generar_informe_word.py          # Generador de informe ejecutivo Word corporativo
├── app/
│   └── main.py                          # Backend API en FastAPI con endpoints REST
├── frontend/
│   ├── index.html                       # Interfaz web interactiva (Tailwind CSS CDN)
│   ├── styles.css                       # Estilos personalizados para controles y badges
│   └── app.js                           # Sincronización de sliders, presets y llamadas API
├── INFORME_TECNICO_AGROEXPORT.docx      # Informe técnico ejecutivo para Gerencia de Operaciones
├── requirements.txt                     # Dependencias del proyecto
└── README.md                            # Guía de instalación y uso
```

---

## 📊 Comparativa de Rendimiento de Modelos

Evaluación sobre el **conjunto de prueba ciego (100 lotes agrícolas, 20% test)**:

| Métrica Evaluada | Modelo 1: Regresión Lineal (Baseline) | Modelo 2: Red Neuronal Profunda (MLP Keras) | Impacto / Mejora |
|---|---|---|---|
| **Variables de Entrada** | Solo `hectareas_cultivadas` | 5 variables (Hectáreas, Agua, Sol, Nutrientes, Edad) | Modelo biofísico multivariable |
| **Error Cuadrático Medio (MSE)** | 7,814.64 t² | **1,816.27 t²** | -76.8% en varianza de error |
| **Error Estándar (RMSE)** | 88.40 toneladas | **42.62 toneladas** | **-51.8% de reducción de error** |
| **Error Medio Absoluto (MAE)** | 68.20 toneladas | **30.09 toneladas** | **-55.9% de reducción de error** |
| **Coeficiente $R^2$** | 0.5826 (58.3%) | **0.9030 (90.3%)** | **+32.0 puntos porcentuales** |

---

## 🚀 Guía de Instalación y Ejecución

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Generación del Dataset Agronómico
```bash
python src/generar_dataset.py
```
Genera `data/historial_cosechas_ica.csv` con 500 registros calibrados con los parámetros biofísicos del Valle de Ica (demanda hídrica de 9,000 a 13,000 m³/ha según ANA, fertilización N-P-K y curvas de edad del parrón).

### 3. Entrenamiento de Modelos y Generación de Gráficos
```bash
python src/entrenar_modelos.py
```
Entrena ambos modelos, almacena los pesos en `outputs/` y crea las 3 gráficas analíticas a 300 DPI.

### 4. Generación del Informe Técnico en Word
```bash
python src/generar_informe_word.py
```
Genera `INFORME_TECNICO_AGROEXPORT.docx`, documento corporativo con portada ejecutiva, tablas estilizadas, formulaciones matemáticas e incrustación de las figuras en alta resolución.

### 5. Puesta en Marcha del Servidor Web & API
```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
- **Simulador Web Interactivo:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Documentación Swagger API:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Endpoint de Métricas:** [http://127.0.0.1:8000/api/metricas](http://127.0.0.1:8000/api/metricas)
- **Endpoint de Predicción:** `POST /predecir`

---

## 🚢 Optimización Logística de Frío (40' Reefer)
El sistema integra una regla de cubicaje estándar de **20 toneladas métricas útiles por contenedor refrigerado de 40 pies (FEU)**. Permite a la Gerencia de Operaciones anticipar la reserva de reefers en el **Terminal Portuario de Paracas (Pisco)** y el **Puerto del Callao**, evitando penalidades por falso flete o cuellos de botella en la planta de empaque.
