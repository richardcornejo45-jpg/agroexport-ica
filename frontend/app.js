/**
 * AgroExport Ica - Frontend Logic
 * Empresa: AgroExportadora del Sur S.A.C. (Valle de Ica, Perú)
 * 
 * Gestiona la sincronización bidireccional de sliders/inputs, llamada a la API
 * FastAPI (/predecir y /api/metricas), presets agronómicos y visualizaciones.
 */

// Elementos del DOM para las 5 variables
const inputs = {
  hectareas: {
    range: document.getElementById('range-hectareas'),
    num: document.getElementById('num-hectareas')
  },
  agua: {
    range: document.getElementById('range-agua'),
    num: document.getElementById('num-agua')
  },
  radiacion: {
    range: document.getElementById('range-radiacion'),
    num: document.getElementById('num-radiacion')
  },
  fertilizante: {
    range: document.getElementById('range-fertilizante'),
    num: document.getElementById('num-fertilizante')
  },
  antiguedad: {
    range: document.getElementById('range-antiguedad'),
    num: document.getElementById('num-antiguedad')
  }
};

// Badges de retroalimentación agronómica dinámica
const badges = {
  agua: document.getElementById('badge-dosis-agua'),
  fert: document.getElementById('badge-dosis-fert'),
  edad: document.getElementById('badge-estado-parron')
};

// Presets agronómicos típicos del Valle de Ica
const PRESETS = {
  optimo: {
    hectareas: 12.0,
    agua: 132000000,      // 11,000 m³/ha
    radiacion: 2820,
    fertilizante: 5800,   // ~483 kg/ha
    antiguedad: 7
  },
  deficit: {
    hectareas: 14.0,
    agua: 98000000,       // 7,000 m³/ha (Estrés hídrico severo)
    radiacion: 2900,
    fertilizante: 5600,
    antiguedad: 6
  },
  joven: {
    hectareas: 8.5,
    agua: 76500000,       // 9,000 m³/ha
    radiacion: 2750,
    fertilizante: 3400,
    antiguedad: 2         // Parrón formativo (cosecha baja)
  },
  antiguo: {
    hectareas: 16.0,
    agua: 168000000,      // 10,500 m³/ha
    radiacion: 2700,
    fertilizante: 7200,
    antiguedad: 16        // Parronera madura con declive
  }
};

/**
 * Inicialización al cargar el documento
 */
document.addEventListener('DOMContentLoaded', () => {
  vincularEventosSliders();
  actualizarBadgesAgronomicos();
  cargarMetricasAPI();
  verificarSaludAPI();

  // Ejecutar primera predicción automática con valores por defecto
  ejecutarPrediccion();

  // Escuchar envío del formulario
  document.getElementById('formulario-simulador').addEventListener('submit', (e) => {
    e.preventDefault();
    ejecutarPrediccion();
  });
});

/**
 * Vincula sliders e inputs numéricos en forma bidireccional
 */
function vincularEventosSliders() {
  for (const [key, elem] of Object.entries(inputs)) {
    // Cuando se mueve el slider
    elem.range.addEventListener('input', (e) => {
      elem.num.value = e.target.value;
      actualizarBadgesAgronomicos();
      ejecutarPrediccionDebounced();
    });

    // Cuando se edita el input numérico
    elem.num.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      if (!isNaN(val)) {
        elem.range.value = val;
        actualizarBadgesAgronomicos();
        ejecutarPrediccionDebounced();
      }
    });
  }
}

/**
 * Actualiza los badges de dosis agronómica (m³/ha, kg/ha, estado del parrón)
 */
function actualizarBadgesAgronomicos() {
  const ha = parseFloat(inputs.hectareas.num.value) || 1.0;
  const aguaL = parseFloat(inputs.agua.num.value) || 0;
  const fertKg = parseFloat(inputs.fertilizante.num.value) || 0;
  const edadAnos = parseInt(inputs.antiguedad.num.value) || 1;

  // Dosis de agua en m³/ha
  const m3Ha = (aguaL / ha) / 1000.0;
  let textoAgua = `Dosis: ${Math.round(m3Ha).toLocaleString('es-PE')} m³/ha`;
  if (m3Ha < 8500) {
    textoAgua += " (⚠️ Estrés Hídrico)";
    badges.agua.className = "font-semibold text-rose-700 bg-rose-50 px-2 py-0.5 rounded border border-rose-200";
  } else if (m3Ha > 12500) {
    textoAgua += " (⚠️ Sobredosis/Riesgo Anoxia)";
    badges.agua.className = "font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200";
  } else {
    textoAgua += " (✅ Rango Óptimo ANA)";
    badges.agua.className = "font-semibold text-sky-700 bg-sky-50 px-2 py-0.5 rounded border border-sky-200";
  }
  badges.agua.textContent = textoAgua;

  // Dosis de fertilizante en kg/ha
  const kgHa = (fertKg / ha);
  badges.fert.textContent = `Dosis: ${kgHa.toFixed(1)} kg/ha (Nutrición N-P-K)`;

  // Estado del parrón
  let textoEdad = "";
  let claseEdad = "";
  if (edadAnos <= 2) {
    textoEdad = `Estado: Parrón Joven en Formación (Año ${edadAnos}) — Rinde reducido`;
    claseEdad = "font-semibold text-amber-700 bg-amber-50 px-2 py-0.5 rounded border border-amber-200";
  } else if (edadAnos <= 4) {
    textoEdad = `Estado: Inicio de Producción Comercial (Año ${edadAnos})`;
    claseEdad = "font-semibold text-blue-700 bg-blue-50 px-2 py-0.5 rounded border border-blue-200";
  } else if (edadAnos <= 11) {
    textoEdad = `Estado: Plena Madurez y Máximo Vigor (Año ${edadAnos} • Pico)`;
    claseEdad = "font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200";
  } else {
    textoEdad = `Estado: Parronera Tradicional Maduro (Año ${edadAnos} • Declive leve)`;
    claseEdad = "font-semibold text-purple-700 bg-purple-50 px-2 py-0.5 rounded border border-purple-200";
  }
  badges.edad.textContent = textoEdad;
  badges.edad.className = claseEdad;
}

/**
 * Aplica un escenario agronómico predeterminado
 */
function aplicarPreset(nombre) {
  const preset = PRESETS[nombre];
  if (!preset) return;

  for (const [key, val] of Object.entries(preset)) {
    if (inputs[key]) {
      inputs[key].num.value = val;
      inputs[key].range.value = val;
    }
  }

  actualizarBadgesAgronomicos();
  ejecutarPrediccion();
}

/**
 * Control debounce para inferencias en tiempo real al mover sliders
 */
let debounceTimeout = null;
function ejecutarPrediccionDebounced() {
  clearTimeout(debounceTimeout);
  debounceTimeout = setTimeout(() => {
    ejecutarPrediccion();
  }, 120);
}

/**
 * Envía la petición POST /predecir a FastAPI y actualiza la interfaz
 */
async function ejecutarPrediccion() {
  const btn = document.getElementById('btn-predecir');
  const payload = {
    hectareas_cultivadas: parseFloat(inputs.hectareas.num.value),
    agua_riego_litros: parseFloat(inputs.agua.num.value),
    radiacion_solar_horas: parseFloat(inputs.radiacion.num.value),
    fertilizante_kg: parseFloat(inputs.fertilizante.num.value),
    antiguedad_terreno_anos: parseFloat(inputs.antiguedad.num.value)
  };

  try {
    const res = await fetch('/predecir', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      throw new Error(`Error en API: ${res.statusText}`);
    }

    const data = await res.json();
    renderizarResultados(data);

  } catch (error) {
    console.error('Error al predecir:', error);
  }
}

/**
 * Actualiza los elementos del DOM con la respuesta de la API
 */
function renderizarResultados(data) {
  // 1. Red Neuronal (Deep Learning)
  document.getElementById('res-toneladas-nn').textContent = `${data.toneladas_red_neuronal.toLocaleString('es-PE', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} t`;
  document.getElementById('res-rinde-ha-nn').textContent = `Rendimiento: ${data.rendimiento_ha_red_neuronal.toFixed(2)} t/ha`;
  document.getElementById('res-contenedores-nn').textContent = data.contenedores_red_neuronal;
  document.getElementById('res-remanente-nn').textContent = `Último contenedor: ${data.capacidad_ultimo_contenedor_pct}% lleno`;
  
  // Barra de progreso del último contenedor
  document.getElementById('res-pct-text').textContent = `${data.capacidad_ultimo_contenedor_pct}%`;
  document.getElementById('res-progress-reefer').style.width = `${data.capacidad_ultimo_contenedor_pct}%`;
  
  // Diagnóstico
  document.getElementById('res-diagnostico').textContent = data.diagnostico_agronomico;

  // 2. Regresión Lineal (Baseline)
  document.getElementById('res-toneladas-lr').textContent = `${data.toneladas_regresion_lineal.toLocaleString('es-PE', { minimumFractionDigits: 1, maximumFractionDigits: 1 })} t`;
  document.getElementById('res-rinde-ha-lr').textContent = `Rinde: ${data.rendimiento_ha_regresion_lineal.toFixed(2)} t/ha`;
  document.getElementById('res-contenedores-lr').textContent = `${data.contenedores_regresion_lineal} FEU`;

  // Brecha y Alerta Logística
  const difT = data.diferencia_toneladas;
  const difContenedores = data.contenedores_red_neuronal - data.contenedores_regresion_lineal;
  const alertaBox = document.getElementById('res-alerta-gap');
  const gapText = document.getElementById('res-gap-text');

  if (Math.abs(difContenedores) > 0) {
    gapText.innerHTML = `
      El modelo tradicional simple se desvía en <strong>${Math.abs(difT).toFixed(1)} toneladas (${data.diferencia_porcentaje > 0 ? '+' : ''}${data.diferencia_porcentaje.toFixed(1)}%)</strong>, 
      lo que provocaría un error de <strong>${Math.abs(difContenedores)} contenedor(es) reefer</strong> ${difContenedores > 0 ? 'faltantes' : 'sobrantes vacíos'} en el puerto.
    `;
    alertaBox.className = "p-3 rounded-xl bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2";
  } else {
    gapText.innerHTML = `
      Diferencia de volumen: ${Math.abs(difT).toFixed(1)} t. Ambos modelos coinciden en la cantidad entera de contenedores para esta configuración específica.
    `;
    alertaBox.className = "p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 flex items-start gap-2";
  }

  // Recomendación
  document.getElementById('res-recomendacion').textContent = data.recomendacion_logistica;
}

/**
 * Consume GET /api/metricas para llenar la tabla y KPIs
 */
async function cargarMetricasAPI() {
  try {
    const res = await fetch('/api/metricas');
    if (!res.ok) return;

    const data = await res.json();
    
    // Actualizar tabla comparativa
    if (data.regresion_lineal) {
      document.getElementById('table-lr-mse').textContent = data.regresion_lineal.mse.toLocaleString('es-PE', { minimumFractionDigits: 2 });
      document.getElementById('table-lr-rmse').textContent = `${data.regresion_lineal.rmse.toFixed(2)} t`;
      document.getElementById('table-lr-mae').textContent = `${data.regresion_lineal.mae.toFixed(2)} t`;
      document.getElementById('table-lr-r2').textContent = `${data.regresion_lineal.r2.toFixed(4)} (${(data.regresion_lineal.r2 * 100).toFixed(1)}%)`;
    }

    if (data.red_neuronal) {
      document.getElementById('table-nn-mse').textContent = data.red_neuronal.mse.toLocaleString('es-PE', { minimumFractionDigits: 2 });
      document.getElementById('table-nn-rmse').textContent = `${data.red_neuronal.rmse.toFixed(2)} t`;
      document.getElementById('table-nn-mae').textContent = `${data.red_neuronal.mae.toFixed(2)} t`;
      document.getElementById('table-nn-r2').textContent = `${data.red_neuronal.r2.toFixed(4)} (${(data.red_neuronal.r2 * 100).toFixed(1)}%)`;
      
      // Actualizar KPIs del header
      document.getElementById('kpi-r2').textContent = `${(data.red_neuronal.r2 * 100).toFixed(1)}%`;
    }

    if (data.analisis_comparativo) {
      document.getElementById('kpi-rmse').textContent = `-${data.analisis_comparativo.reduccion_rmse_porcentaje.toFixed(1)}%`;
    }

  } catch (error) {
    console.warn('No se pudieron cargar métricas desde la API:', error);
  }
}

/**
 * Verifica el estado del backend (/health)
 */
async function verificarSaludAPI() {
  try {
    const res = await fetch('/health');
    const data = await res.json();
    const badge = document.getElementById('api-status-badge');
    if (data.status === 'online') {
      badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Red Neuronal Activa (Keras)`;
    } else {
      badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span> Modo Degradado`;
    }
  } catch (e) {
    const badge = document.getElementById('api-status-badge');
    badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-400"></span> Desconectado`;
  }
}

/**
 * Funciones para el Lightbox modal de imágenes
 */
function abrirModalImagen(src, titulo) {
  const modal = document.getElementById('modal-imagen');
  const img = document.getElementById('modal-img-tag');
  const titleElem = document.getElementById('modal-titulo');

  img.src = src;
  titleElem.textContent = titulo;
  modal.classList.remove('hidden');
}

function cerrarModalImagen() {
  const modal = document.getElementById('modal-imagen');
  modal.classList.add('hidden');
}

// Cerrar modal al presionar Escape o hacer click fuera
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') cerrarModalImagen();
});

document.getElementById('modal-imagen')?.addEventListener('click', (e) => {
  if (e.target.id === 'modal-imagen') cerrarModalImagen();
});
