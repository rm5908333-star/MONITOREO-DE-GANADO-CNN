#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interfaz gráfica (Streamlit) para el Módulo de Percepción -- Inciso d)
Sistema inteligente de monitoreo de ganado con CNN.

Se conecta a cnn_monitoreo_ganado.py sin alterar su lógica.
Ejecutar (con ambos archivos en la misma carpeta):

    pip install -r requirements.txt
    streamlit run app_monitoreo_ganado.py

Notas de diseño:
  * Todo lo costoso (generar datos, entrenar CNN 1 y CNN 2, procesar el "corral
    nuevo") se hace UNA vez dentro de inicializar_sistema() con cache_resource.
    Mover un slider solo recalcula UNA imagen.
  * La CNN 1 tiene un generador aleatorio interno (imprecisión del regresor de
    cajas). Para que la interfaz sea estable, cada render usa una COPIA del
    detector con semilla fija: mismos parámetros => mismas cajas.
  * Este módulo solo percibe: la decisión de alertar es de la Función de
    Utilidad (Pasos 7 y 8) y no se simula aquí.
"""

import copy
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")  # backend sin ventana; Streamlit muestra la figura renderizada
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

import cnn_monitoreo_ganado as core

# =============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# =============================================================================
st.set_page_config(
    page_title="Monitoreo Térmico de Ganado | CNN",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FONDO = "#0E1117"
st.markdown(
    """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    .estado-card { padding: 1rem 1.2rem; border-radius: 10px; border-left: 6px solid;
                   background-color: rgba(255,255,255,0.05); margin-bottom: 0.9rem; }
    .barra-fondo { background: rgba(255,255,255,0.10); border-radius: 6px; height: 14px; width: 100%; }
    .barra-valor { height: 14px; border-radius: 6px; }
    .barra-fila  { margin-bottom: 7px; font-size: 0.88rem; }
    div[data-testid="stMetric"] { background-color: rgba(255,255,255,0.04); padding: 0.6rem 0.8rem;
                                  border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); }
</style>
""",
    unsafe_allow_html=True,
)

# Las gráficas usan el mismo fondo oscuro que la app (ver también .streamlit/config.toml)
plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "text.color": "#FAFAFA", "axes.labelcolor": "#FAFAFA", "axes.edgecolor": "#4A4F5C",
    "xtick.color": "#C9CDD6", "ytick.color": "#C9CDD6", "axes.titlecolor": "#FAFAFA",
    "grid.color": "#FFFFFF", "legend.facecolor": "#161A23", "legend.edgecolor": "#4A4F5C",
})

COLORES_ROI = {"cuerpo_animal": "#00E5FF", "ojo": "#FFD600", "zona_vulvar": "#FF4081", "ubre": "#76FF03"}
COLORES_ESTADO = {"sano": "#2ECC71", "celo": "#F39C12", "enfermo": "#E74C3C"}

N_LOTES_FEEDBACK = 3   # el veterinario confirma los casos del corral nuevo en 3 tandas

# Casos preconfigurados (idénticos a los 3 escenarios del módulo de consola)
PRESETS = {
    "Caso 1 — Vaca Sana (MX-1001)": dict(
        id="MX-1001", base=38.5, dia=9, amb=24.0, hum=55.0, d_ojo=0.05, d_vulva=0.0, d_ubre=-0.05,
        d_cuerpo=0.0, seed=101, zona="corral_A", hist=[38.5, 38.6, 38.5]),
    "Caso 2 — Vaca en Celo (MX-1002)": dict(
        id="MX-1002", base=38.6, dia=20, amb=22.0, hum=60.0, d_ojo=0.15, d_vulva=1.80, d_ubre=0.10,
        d_cuerpo=0.15, seed=202, zona="corral_B", hist=[38.6, 38.6, 38.7]),
    "Caso 3 — Vaca Enferma (MX-1003)": dict(
        id="MX-1003", base=38.4, dia=8, amb=26.0, hum=50.0, d_ojo=1.70, d_vulva=0.10, d_ubre=1.90,
        d_cuerpo=0.70, seed=303, zona="corral_C", hist=[39.7, 39.9, 40.1]),
    "Personalizado (ajustar manualmente)": dict(
        id="MX-9999", base=38.5, dia=10, amb=22.0, hum=55.0, d_ojo=0.0, d_vulva=0.0, d_ubre=0.0,
        d_cuerpo=0.0, seed=404, zona="corral_D", hist=[38.5, 38.5, 38.5]),
}
GEOMETRIA_FIJA = {"cx": 120.0, "cy": 108.0, "a": 71.0, "b": 37.0}


# =============================================================================
# INICIALIZACIÓN Y ENTRENAMIENTO (una sola vez; compartido entre sesiones, solo lectura)
# =============================================================================
@st.cache_resource(show_spinner="Inicializando y entrenando CNN 1 y CNN 2 con el dataset simulado (≈15 s la primera vez)...")
def inicializar_sistema():
    registros = core.generar_registros(n_animales=90, obs_por_animal=8, semilla=2024)
    part = core.particionar_por_animal(registros, semilla=7)
    core.verificar_sin_fuga_de_datos(part)       # lanza AssertionError si hubiera fuga

    composicion = []
    for nombre, regs in part.items():
        cuenta = {e: sum(1 for r in regs if r["estado"] == e) for e in core.ESTADOS}
        composicion.append({"Conjunto": nombre, "Animales": len({r["animal_id"] for r in regs}),
                            "Imágenes": len(regs), **{e.capitalize(): cuenta[e] for e in core.ESTADOS}})

    # --- CNN 1: detectora (Paso 3) --------------------------------------------------
    cnn1 = core.CNNDetectorObjetos()
    afinar, gts_afinar = [], []
    for r in part["train"][:60]:
        ic, gt = core.capturar_imagen_termica(r)
        afinar.append(core.normalizar_termico(ic, r["temp_amb"], r["humedad"]))
        gts_afinar.append(gt)
    cnn1.afinar_cabeza(afinar, gts_afinar)

    muestras = {}
    for split in ("train", "val", "test"):
        muestras[split], _ = core.procesar_lote(part[split], cnn1)
    ev1 = core.evaluar_cnn1(muestras["test"])

    # --- CNN 2: clasificadora multimodal (Paso 6a) ---------------------------------
    cnn2 = core.CNNClasificadoraMultimodal()
    rec_tr, ctx_tr, y_tr = core.apilar(muestras["train"])
    rec_va, ctx_va, y_va = core.apilar(muestras["val"])
    test_arrays = core.apilar(muestras["test"])
    info_train = cnn2.entrenar(rec_tr, ctx_tr, y_tr, rec_va, ctx_va, y_va, verbose=False)
    probs_te = cnn2.predecir_proba(test_arrays[0], test_arrays[1])
    cm = core.matriz_confusion(test_arrays[2], probs_te.argmax(axis=1))

    # --- Corral nuevo (Pasos 10-11): se procesa AQUÍ una vez, no en cada interacción ---
    reg_nuevos = core.generar_registros(n_animales=30, obs_por_animal=6, semilla=777,
                                        perfil=core.PERFIL_RANCHO_NUEVO, prefijo="NUEVA")
    m_nuevas, _ = core.procesar_lote(reg_nuevos, cnn1)
    ids_nuevos = sorted({m["animal_id"] for m in m_nuevas})
    ids_feedback = ids_nuevos[:len(ids_nuevos) // 2]           # animales con confirmación veterinaria
    m_eval = [m for m in m_nuevas if m["animal_id"] not in set(ids_feedback)]   # animales NO vistos
    lotes_fb = []
    for grupo in np.array_split(np.array(ids_feedback), N_LOTES_FEEDBACK):   # los animales no se parten entre lotes
        ids_grupo = set(grupo.tolist())
        lotes_fb.append([
            {"animal_id": m["animal_id"], "recortes": m["recortes"], "contexto": m["ctx"],
             "etiqueta_confirmada": core.ESTADOS[m["y"]]}
            for m in m_nuevas if m["animal_id"] in ids_grupo])

    return {
        "cnn1": cnn1, "cnn2_base": cnn2, "ev1": ev1, "info_train": info_train, "cm": cm,
        "met2": core.metricas_clasificacion(cm),
        "loss_te": core.entropia_cruzada_ponderada(probs_te, test_arrays[2]),
        "composicion": composicion, "test_arrays": test_arrays,
        "eval_corral": core.apilar(m_eval), "lotes_fb": lotes_fb,
    }


def evaluar_modelo(cnn2, datos):
    """Desempeño de una versión del modelo: corral nuevo (animales no vistos) y TEST original."""
    rec, ctx, y = datos["eval_corral"]
    p = cnn2.predecir_proba(rec, ctx)
    m = core.metricas_clasificacion(core.matriz_confusion(y, p.argmax(axis=1)))
    rec_t, ctx_t, y_t = datos["test_arrays"]
    pt = cnn2.predecir_proba(rec_t, ctx_t)
    mt = core.metricas_clasificacion(core.matriz_confusion(y_t, pt.argmax(axis=1)))
    return {"version": cnn2.version,
            "perdida_corral": core.entropia_cruzada_ponderada(p, y),
            "exactitud_corral": m["exactitud"],
            "recall_celo_corral": m["por_clase"]["celo"]["recall"],
            "recall_enfermo_corral": m["por_clase"]["enfermo"]["recall"],
            "f1_test": mt["f1_macro"], "perdida_test": core.entropia_cruzada_ponderada(pt, y_t)}


# =============================================================================
# VISUALIZACIÓN (matplotlib)
# =============================================================================
def graficar_imagen_termica(imagen_c, salida_cnn1, gt, mostrar_gt=False, mostrar_etiquetas=True):
    fig, ax = plt.subplots(figsize=(5.6, 5.2), dpi=130)
    im = ax.imshow(imagen_c, cmap="inferno", vmin=15, vmax=41)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Temperatura aparente (°C)", fontsize=9)

    for clase in core.CLASES_ROI:
        det = salida_cnn1["detecciones"][clase]
        x1, y1, x2, y2 = det["bbox"]
        color = COLORES_ROI[clase]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color,
                                       linewidth=1.3 if clase == "cuerpo_animal" else 1.8,
                                       linestyle="--" if clase == "cuerpo_animal" else "-"))
        if mostrar_gt:
            g1, h1, g2, h2 = gt[clase]
            ax.add_patch(patches.Rectangle((g1, h1), g2 - g1, h2 - h1, fill=False, edgecolor="white",
                                           linewidth=0.9, linestyle=":"))
        if mostrar_etiquetas:
            iou = core.calcular_iou(det["bbox"], gt[clase])
            texto = f"{clase} {det['confianza']:.0%} · IoU {iou:.2f}"
            # Etiquetas ancladas para que no se salgan de la imagen ni se encimen:
            if clase in ("zona_vulvar",):                  # arriba, alineada a la derecha de su caja
                xt, yt, ha, va = x2, max(6, y1 - 2), "right", "bottom"
            elif clase == "ubre":                          # abajo, alineada a la derecha
                xt, yt, ha, va = x2, min(218, y2 + 2), "right", "top"
            else:                                          # arriba, alineada a la izquierda
                xt, yt, ha, va = x1, max(6, y1 - 2), "left", "bottom"
            ax.text(xt, yt, texto, color=color, fontsize=7, fontweight="bold", ha=ha, va=va,
                    bbox=dict(facecolor="#111111", edgecolor="none", alpha=0.75, pad=1.2))

    ax.set_xlim(0, core.TAM_IMAGEN)
    ax.set_ylim(core.TAM_IMAGEN, 0)
    ax.set_title("Pasos 1 y 3 — Cámara térmica + detección de ROI (CNN 1)", fontsize=10, pad=8)
    ax.set_xlabel("Píxeles X (224)")
    ax.set_ylabel("Píxeles Y (224)")
    fig.tight_layout()
    return fig


def graficar_recortes_roi(salida_cnn1):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), dpi=130)
    for ax, clase in zip(axes, core.CLASES_ROI):
        rt = salida_cnn1["detecciones"][clase]["recorte_termico"]
        ax.imshow(rt["matriz_norm_16x16"], cmap="inferno", vmin=0.55, vmax=0.92)
        ax.set_title(f"{clase}\n{rt['temp_promedio_c']:.1f} °C", fontsize=8,
                     color=COLORES_ROI[clase], fontweight="bold")
        ax.axis("off")
    fig.suptitle("Paso 5a — Recortes normalizados 16×16 que entran a la CNN 2", fontsize=8.5)
    fig.tight_layout()
    return fig


def graficar_curva_entrenamiento(historia, mejor_epoca):
    epocas = [h[0] for h in historia]
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=130)
    ax.plot(epocas, [h[1] for h in historia], label="Entrenamiento", color="#00E5FF", linewidth=2)
    ax.plot(epocas, [h[2] for h in historia], label="Validación", color="#FF4081", linewidth=2)
    ax.axvline(mejor_epoca, color="#FFD600", linestyle="--", linewidth=1.2, label=f"Mejor época ({mejor_epoca})")
    ax.set_xlabel("Época")
    ax.set_ylabel("Entropía cruzada ponderada")
    ax.set_title("Convergencia de la CNN 2 (detención temprana)", fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    return fig


def graficar_matriz_confusion(cm):
    fig, ax = plt.subplots(figsize=(5.0, 3.8), dpi=130)
    ax.imshow(cm, cmap="Blues")
    etiquetas = [e.upper() for e in core.ESTADOS]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(etiquetas, fontsize=8.5)
    ax.set_yticklabels(etiquetas, fontsize=8.5)
    ax.set_xlabel("Predicción de la CNN 2")
    ax.set_ylabel("Estado real (veterinario)")
    ax.set_title("Matriz de confusión (conjunto Test)", fontsize=10)
    umbral = cm.max() / 2.0
    totales = cm.sum(axis=1)
    for i in range(3):
        for j in range(3):
            pct = cm[i, j] / totales[i] if totales[i] else 0.0
            ax.text(j, i, f"{cm[i, j]}\n{pct:.0%}", ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if cm[i, j] > umbral else "#111111")
    fig.tight_layout()
    return fig


def graficar_evolucion_versiones(historial):
    versiones = [h["version"] for h in historial]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=130)
    ax1.plot(versiones, [h["perdida_corral"] for h in historial], marker="o", color="#FF4081", linewidth=2)
    ax1.set_title("Pérdida ponderada · corral nuevo", fontsize=9)
    ax1.set_xlabel("Versión del modelo")
    ax2.plot(versiones, [h["recall_enfermo_corral"] for h in historial], marker="o", color="#E74C3C",
             linewidth=2, label="Recall enfermo (corral nuevo)")
    ax2.plot(versiones, [h["recall_celo_corral"] for h in historial], marker="o", color="#F39C12",
             linewidth=2, label="Recall celo (corral nuevo)")
    ax2.plot(versiones, [h["f1_test"] for h in historial], marker="s", color="#00E5FF", linewidth=1.5,
             linestyle="--", label="F1 macro TEST original")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Sensibilidad y control de olvido", fontsize=9)
    ax2.set_xlabel("Versión del modelo")
    ax2.legend(fontsize=7, loc="lower right")
    for ax in (ax1, ax2):
        ax.set_xticks(versiones)
        ax.grid(True, alpha=0.15)
    fig.tight_layout()
    return fig


def mostrar_figura(fig):
    st.pyplot(fig)
    plt.close(fig)


def barras_probabilidad_html(probs):
    filas = []
    for i, e in enumerate(core.ESTADOS):
        p = float(np.clip(probs[i], 0.0, 1.0))
        filas.append(
            f'<div class="barra-fila"><div style="display:flex;justify-content:space-between;">'
            f'<span>{e.capitalize()}</span><span><b>{p:.4f}</b></span></div>'
            f'<div class="barra-fondo"><div class="barra-valor" '
            f'style="width:{p * 100:.1f}%;background:{COLORES_ESTADO[e]};"></div></div></div>')
    return "".join(filas)


# =============================================================================
# ESTADO DE SESIÓN Y CALLBACKS
# =============================================================================
datos = inicializar_sistema()


def aplicar_preset():
    p = PRESETS[st.session_state["w_preset"]]
    st.session_state["w_base"] = float(p["base"])
    st.session_state["w_dia"] = int(p["dia"])
    st.session_state["w_amb"] = float(p["amb"])
    st.session_state["w_hum"] = float(p["hum"])
    st.session_state["w_d_ojo"] = float(p["d_ojo"])
    st.session_state["w_d_vulva"] = float(p["d_vulva"])
    st.session_state["w_d_ubre"] = float(p["d_ubre"])
    st.session_state["w_d_cuerpo"] = float(p["d_cuerpo"])
    st.session_state["w_seed"] = int(p["seed"])
    st.session_state["w_sesgo"] = 0.0
    st.session_state["w_ocl_roi"] = "Ninguna"
    st.session_state["w_ocl_frac"] = 0.2


def nueva_captura():
    st.session_state["w_seed"] = int(st.session_state["w_seed"]) + 1


def aplicar_siguiente_lote():
    k = st.session_state["lotes_aplicados"]
    if k >= len(datos["lotes_fb"]):
        return
    cnn2_act = st.session_state["cnn2_activa"]
    reporte = core.reentrenar_cnn_incremental(datos["lotes_fb"][k], cnn2=cnn2_act)   # Paso 11
    reporte["lote"] = k + 1
    st.session_state["reportes"].append(reporte)
    st.session_state["lotes_aplicados"] = k + 1
    st.session_state["historial_versiones"].append(evaluar_modelo(cnn2_act, datos))


def restaurar_modelo():
    st.session_state["cnn2_activa"] = copy.deepcopy(datos["cnn2_base"])
    st.session_state["lotes_aplicados"] = 0
    st.session_state["reportes"] = []
    st.session_state["historial_versiones"] = [evaluar_modelo(st.session_state["cnn2_activa"], datos)]


if "w_preset" not in st.session_state:
    st.session_state["w_preset"] = list(PRESETS)[0]
if "w_base" not in st.session_state:      # los sliders se inicializan desde el preset elegido
    aplicar_preset()
if "cnn2_activa" not in st.session_state:
    restaurar_modelo()

cnn2 = st.session_state["cnn2_activa"]

# =============================================================================
# ENCABEZADO
# =============================================================================
st.title("🐄 Sistema inteligente de monitoreo de ganado con CNN")
st.caption(
    f"Inciso d) Módulo de Percepción (CNN 1 detectora + CNN 2 clasificadora multimodal) · "
    f"Versión activa del modelo: **v{cnn2.version}** · Este módulo percibe y entrega probabilidades; "
    f"la Función de Utilidad (Pasos 7–8) decide las alertas. Todos los datos son simulados."
)

tab1, tab2, tab3 = st.tabs([
    "🔬 1. Simulador en vivo (Pasos 1 → 6a)",
    "📊 2. Entrenamiento y métricas",
    "🔄 3. Mejora continua (Pasos 10 y 11)",
])

# =============================================================================
# PESTAÑA 1: SIMULADOR INTERACTIVO
# =============================================================================
with tab1:
    col_ctrl, col_vis, col_res = st.columns([1.05, 1.35, 1.15], gap="large")

    with col_ctrl:
        st.subheader("🎛️ Parámetros del escenario")
        st.selectbox("Cargar caso preconfigurado:", list(PRESETS), key="w_preset", on_change=aplicar_preset)
        preset = PRESETS[st.session_state["w_preset"]]

        st.markdown("**Contexto del animal (estado interno · Paso 4)**")
        base_c = st.slider("Temp. base habitual (°C)", 38.0, 39.2, step=0.1, key="w_base")
        dia_ciclo = st.slider("Día del ciclo reproductivo (0–20)", 0, 20, step=1, key="w_dia",
                              help="Los días 19, 20, 0 y 1 son la ventana fértil.")
        if dia_ciclo in (19, 20, 0, 1):
            st.info("📅 Día dentro de la **ventana fértil** del ciclo estral.")

        st.markdown("**Condiciones ambientales (sensores · Paso 2)**")
        c_a1, c_a2 = st.columns(2)
        temp_amb = c_a1.slider("T° ambiente (°C)", 6.0, 32.0, step=0.5, key="w_amb",
                               help="El modelo se entrenó con 6–30 °C.")
        humedad = c_a2.slider("Humedad (%)", 30.0, 95.0, step=1.0, key="w_hum")
        if temp_amb > 30.0:
            st.warning("T° ambiente fuera del rango de entrenamiento (6–30 °C): la detección pierde contraste.")

        st.markdown("**Elevación térmica por región (°C sobre su valor normal)**")
        d_ojo = st.slider("Δ Ojo (°C)", -0.3, 2.5, step=0.05, key="w_d_ojo")
        d_vulva = st.slider("Δ Zona vulvar (°C)", -0.3, 2.5, step=0.05, key="w_d_vulva")
        d_ubre = st.slider("Δ Ubre (°C)", -0.3, 2.5, step=0.05, key="w_d_ubre")

        with st.expander("Opciones avanzadas"):
            d_cuerpo = st.slider("Δ Cuerpo (°C)", -0.3, 1.5, step=0.05, key="w_d_cuerpo")
            sesgo_sensor = st.slider("Deriva de la cámara (°C)", -0.6, 0.6, step=0.05, key="w_sesgo",
                                     help="El Paso 2 corrige el clima, no la deriva del sensor.")
            ocl_roi = st.selectbox("Región parcialmente tapada", ["Ninguna", "ojo", "zona_vulvar", "ubre"],
                                   key="w_ocl_roi")
            ocl_frac = st.slider("Fracción visible de esa región", 0.0, 0.6, step=0.05, key="w_ocl_frac")
            c_s1, c_s2 = st.columns([2, 1])
            semilla = c_s1.number_input("Semilla del ruido del sensor", min_value=0, max_value=999999,
                                        step=1, key="w_seed")
            c_s2.markdown("&nbsp;", unsafe_allow_html=True)
            c_s2.button("🎲 Nueva captura", on_click=nueva_captura)
            mostrar_gt = st.checkbox("Mostrar cajas verdaderas (punteadas)", value=False)
            mostrar_etiquetas = st.checkbox("Mostrar etiquetas sobre la imagen", value=True)

    # ---- Pipeline Paso 1 -> 6a (una sola imagen por interacción) ------------------------
    registro = {
        "animal_id": preset["id"], "rfid": f"RFID-{preset['id']}", "estado": "demo",
        "base_c": float(base_c), "dia_ciclo": int(dia_ciclo), "temp_amb": float(temp_amb),
        "humedad": float(humedad), "offsets": {"ojo": 0.0, "vulva": 0.0, "ubre": 0.0},
        "deltas": {"cuerpo": float(d_cuerpo), "ojo": float(d_ojo), "vulva": float(d_vulva), "ubre": float(d_ubre)},
        "geometria": dict(GEOMETRIA_FIJA),
        "oclusion": None if ocl_roi == "Ninguna" else (ocl_roi, float(ocl_frac)),
        "sesgo_sensor_c": float(sesgo_sensor), "semilla_ruido": int(semilla),
    }
    estado_interno = core.EstadoInternoGanado()
    estado_interno.registrar(registro["rfid"], preset["id"], base_c, dia_ciclo, preset["zona"], preset["hist"])

    detector = copy.deepcopy(datos["cnn1"])                      # copia por render: no se comparte estado
    detector._rng = np.random.default_rng(int(semilla))          # ruido del regresor fijado => cajas estables

    imagen_c, gt = core.capturar_imagen_termica(registro)                                     # Paso 1
    img_norm = core.normalizar_termico(imagen_c, temp_amb, humedad)                           # Paso 2
    salida_cnn1 = detector.detectar(img_norm)                                                 # Paso 3
    detectado = salida_cnn1["animal_detectado"]

    if detectado:
        est = core.identificar_animal(registro["rfid"], estado_interno)                       # Paso 4
        recortes, ctx = core.ensamblar_entrada_cnn2(salida_cnn1, est, temp_amb, humedad)      # Paso 5a
        probs = cnn2.inferir_un_animal(recortes, ctx)                                         # Paso 6a

    with col_vis:
        st.subheader("📷 Percepción visual (CNN 1)")
        if detectado:
            mostrar_figura(graficar_imagen_termica(imagen_c, salida_cnn1, gt, mostrar_gt, mostrar_etiquetas))
            mostrar_figura(graficar_recortes_roi(salida_cnn1))
        else:
            mostrar_figura(graficar_imagen_termica(imagen_c, {"detecciones": {c: {"bbox": gt[c], "confianza": 0.0}
                                                                              for c in core.CLASES_ROI}},
                                                   gt, False, False))

    with col_res:
        st.subheader("🧠 Diagnóstico multimodal (CNN 2)")
        if not detectado:
            st.error(
                "**La CNN 1 no detectó al animal.** Con estas condiciones el contraste térmico entre el cuerpo y "
                "el fondo es demasiado bajo (p. ej. ambiente muy caliente con línea base baja). Sin detección no "
                "hay recortes para la CNN 2, así que el flujo se detiene en el Paso 3. Baja la T° ambiente o sube "
                "la temperatura base.")
        else:
            idx = int(probs.argmax())
            estado_pred = core.ESTADOS[idx]
            color = COLORES_ESTADO[estado_pred]
            st.markdown(
                f"""<div class="estado-card" style="border-left-color:{color};">
                <div style="font-size:0.85rem;opacity:0.8;">ESTADO MÁS PROBABLE (SALIDA SOFTMAX)</div>
                <div style="font-size:1.7rem;font-weight:800;color:{color};text-transform:uppercase;">
                    {estado_pred} ({probs[idx]:.1%})</div>
                <div style="font-size:0.85rem;margin-top:4px;">ID: <b>{est['animal_id']}</b> · RFID:
                    <code>{registro['rfid']}</code> · Zona: <b>{est['zona_asignada']}</b></div></div>""",
                unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("P(sano)", f"{probs[0] * 100:.1f}%")
            m2.metric("P(celo)", f"{probs[1] * 100:.1f}%")
            m3.metric("P(enfermo)", f"{probs[2] * 100:.1f}%")
            st.markdown(f"**Distribución de probabilidades** (suma = {float(probs.sum()):.6f})")
            st.markdown(barras_probabilidad_html(probs), unsafe_allow_html=True)
            st.caption("La decisión de alertar la toma la Función de Utilidad (Pasos 7–8), no este módulo.")

            st.markdown("**Lecturas térmicas por región (CNN 1)**")
            filas = []
            for c in core.CLASES_ROI:
                d = salida_cnn1["detecciones"][c]
                rt = d["recorte_termico"]
                filas.append({"Región": c, "T° prom (°C)": round(rt["temp_promedio_c"], 2),
                              "Gradiente (°C/px)": round(rt["gradiente_espacial_c_por_px"], 3),
                              "Confianza": f"{d['confianza']:.0%}",
                              "IoU vs real": round(core.calcular_iou(d["bbox"], gt[c]), 2)})
            st.dataframe(filas, hide_index=True)

            with st.expander("Ver fusión multimodal (Paso 5a → 6a)"):
                st.write("- **Vector visual (Flatten):** `128` características = 4 ROI × 32 "
                         "(8 filtros × mapa 2×2 por ROI, filtros compartidos).")
                st.write(f"- **Vector de contexto (normalizado):** `{[round(float(v), 3) for v in ctx]}` "
                         "→ base habitual, fase del ciclo, T° ambiente, humedad.")
                st.write("- **Concatenación:** `128 + 4 = 132` → FC(32) + Dropout → FC(16) + Dropout → Softmax(3).")
                st.write(f"- **Historial de temperatura en ojo (estado interno):** `{est['historial_ojo_c']}`")

            resultado_json = {
                "animal_id": est["animal_id"], "rfid": registro["rfid"],
                "estado_mas_probable": estado_pred,
                "probabilidades": {e: float(probs[i]) for i, e in enumerate(core.ESTADOS)},
                "suma_probabilidades": float(probs.sum()),
                "detecciones": {c: {"bbox": [float(v) for v in salida_cnn1["detecciones"][c]["bbox"]],
                                    "confianza": float(salida_cnn1["detecciones"][c]["confianza"]),
                                    "temp_promedio_c": float(salida_cnn1["detecciones"][c]["recorte_termico"]["temp_promedio_c"]),
                                    "gradiente_c_por_px": float(salida_cnn1["detecciones"][c]["recorte_termico"]["gradiente_espacial_c_por_px"])}
                                for c in core.CLASES_ROI},
                "contexto": {"temperatura_base_habitual": float(base_c), "dia_ciclo_reproductivo": int(dia_ciclo),
                             "temperatura_ambiental": float(temp_amb), "humedad": float(humedad)},
                "version_modelo": cnn2.version,
            }
            st.download_button("⬇️ Descargar resultado (JSON)", json.dumps(resultado_json, indent=2, ensure_ascii=False),
                               file_name=f"percepcion_{est['animal_id']}.json", mime="application/json")

# =============================================================================
# PESTAÑA 2: ENTRENAMIENTO Y MÉTRICAS
# =============================================================================
with tab2:
    st.subheader("Evaluación cuantitativa sobre el conjunto de prueba (Test)")
    st.success(
        "✅ **Control de fuga de datos verificado por código** (`verificar_sin_fuga_de_datos`): la partición se hace "
        "por `animal_id` y ningún animal aparece en más de un conjunto.")
    st.dataframe(datos["composicion"], hide_index=True)

    ev1, met2 = datos["ev1"], datos["met2"]
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("CNN 1 · mAP@0.5", f"{ev1['map50']:.3f}")
    k2.metric("CNN 1 · mAP@[0.5:0.95]", f"{ev1['map_50_95']:.3f}")
    k3.metric("CNN 2 · Exactitud", f"{met2['exactitud']:.1%}")
    k4.metric("CNN 2 · F1 macro", f"{met2['f1_macro']:.3f}")
    k5.metric("Recall enfermo (prioridad 1)", f"{met2['por_clase']['enfermo']['recall']:.1%}")
    k6.metric("Recall celo (prioridad 2)", f"{met2['por_clase']['celo']['recall']:.1%}")
    st.caption("Métricas medidas sobre imágenes **simuladas**; no representan el desempeño con ganado real.")

    g1, g2 = st.columns(2, gap="large")
    with g1:
        mostrar_figura(graficar_curva_entrenamiento(datos["info_train"]["historia"], datos["info_train"]["mejor_epoca"]))
        st.caption(f"Mejor época por detención temprana: **{datos['info_train']['mejor_epoca']}** "
                   f"(pérdida ponderada en validación: **{datos['info_train']['mejor_perdida_val']:.4f}**).")
    with g2:
        mostrar_figura(graficar_matriz_confusion(datos["cm"]))
        st.caption(f"Pesos de la pérdida `[sano=1, celo=3, enfermo=5]` · pérdida ponderada en Test: "
                   f"**{datos['loss_te']:.4f}**.")

    t1, t2 = st.columns(2, gap="large")
    with t1:
        st.markdown("**Desglose CNN 1 — detectora de objetos**")
        st.dataframe([{"Clase": c, "IoU medio": round(ev1["iou_medio"][c], 3), "AP@0.5": round(ev1["ap50"][c], 3)}
                      for c in core.CLASES_ROI], hide_index=True)
    with t2:
        st.markdown("**Desglose CNN 2 — clasificadora multimodal**")
        st.dataframe([{"Estado": e, "Precisión": round(met2["por_clase"][e]["precision"], 3),
                       "Recall (sensibilidad)": round(met2["por_clase"][e]["recall"], 3),
                       "F1": round(met2["por_clase"][e]["f1"], 3), "Soporte": met2["por_clase"][e]["soporte"]}
                      for e in core.ESTADOS], hide_index=True)

    with st.expander("¿Cómo pesa la entropía cruzada ponderada?"):
        p_real = st.slider("Probabilidad que el modelo asigna a la clase REAL", 0.05, 0.95, 0.20, 0.05,
                           key="w_p_real")
        st.dataframe([{"Clase real": e, "Peso w": float(core.PESOS_CLASE[i]),
                       "CE simple = −ln p": round(float(-np.log(p_real)), 3),
                       "w · CE": round(float(core.PESOS_CLASE[i] * -np.log(p_real)), 3)}
                      for i, e in enumerate(core.ESTADOS)], hide_index=True)
        st.caption("Con el mismo error de probabilidad, no detectar un enfermo cuesta 5 veces más que una falsa "
                   "alarma de sano (criterio de prioridad del inciso a).")

# =============================================================================
# PESTAÑA 3: MEJORA CONTINUA (PASOS 10 Y 11)
# =============================================================================
with tab3:
    st.subheader("Ciclo de aprendizaje continuo (confirmación del veterinario → reentrenamiento)")
    st.write(
        "Se simula un **corral nuevo** con otra raza (línea base 38.8 °C, elevaciones más sutiles) y una cámara con "
        "deriva de −0.30 °C. El veterinario confirma los casos en "
        f"**{N_LOTES_FEEDBACK} tandas**; el modelo se evalúa siempre sobre **animales del corral nuevo que nunca "
        "vio**, y sobre el Test original para detectar olvido.")

    historial = st.session_state["historial_versiones"]
    actual = historial[-1]
    inicial = historial[0]
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Pérdida ponderada (corral nuevo)", f"{actual['perdida_corral']:.3f}",
              delta=f"{actual['perdida_corral'] - inicial['perdida_corral']:.3f}" if len(historial) > 1 else None,
              delta_color="inverse")
    n2.metric("Recall enfermo (corral nuevo)", f"{actual['recall_enfermo_corral']:.1%}",
              delta=f"{(actual['recall_enfermo_corral'] - inicial['recall_enfermo_corral']) * 100:+.1f} pp"
              if len(historial) > 1 else None)
    n3.metric("Recall celo (corral nuevo)", f"{actual['recall_celo_corral']:.1%}",
              delta=f"{(actual['recall_celo_corral'] - inicial['recall_celo_corral']) * 100:+.1f} pp"
              if len(historial) > 1 else None)
    n4.metric("F1 macro Test original (olvido)", f"{actual['f1_test']:.3f}",
              delta=f"{actual['f1_test'] - inicial['f1_test']:+.3f}" if len(historial) > 1 else None)

    aplicados = st.session_state["lotes_aplicados"]
    total_lotes = len(datos["lotes_fb"])
    st.progress(aplicados / total_lotes, text=f"Tandas de confirmación aplicadas: {aplicados} de {total_lotes}")

    b1, b2 = st.columns([1.5, 1])
    with b1:
        if aplicados < total_lotes:
            n_casos = len(datos["lotes_fb"][aplicados])
            st.button(f"🩺 Confirmación veterinaria: aplicar tanda {aplicados + 1} de {total_lotes} "
                      f"({n_casos} casos) y reentrenar (Paso 11)", type="primary", on_click=aplicar_siguiente_lote)
        else:
            st.success("Se aplicaron todas las confirmaciones disponibles de este corral.")
    with b2:
        st.button("↩️ Restaurar modelo original (v1)", on_click=restaurar_modelo, disabled=cnn2.version == 1)

    if len(historial) > 1:
        mostrar_figura(graficar_evolucion_versiones(historial))

    st.markdown("**Evolución por versión**")
    st.dataframe([{"Versión": f"v{h['version']}",
                   "Pérdida corral nuevo": round(h["perdida_corral"], 3),
                   "Exactitud corral nuevo": f"{h['exactitud_corral']:.1%}",
                   "Recall celo": f"{h['recall_celo_corral']:.1%}",
                   "Recall enfermo": f"{h['recall_enfermo_corral']:.1%}",
                   "F1 macro Test": round(h["f1_test"], 3)} for h in historial], hide_index=True)

    if st.session_state["reportes"]:
        st.markdown("### Historial de actualizaciones (reporte de `reentrenar_cnn_incremental`)")
        for rep in reversed(st.session_state["reportes"]):
            with st.container(border=True):
                st.write(f"**Modelo `v{rep['version_modelo']}`** — tanda {rep['lote']}: "
                         f"{rep['n_casos_nuevos']} etiquetas confirmadas por el Crítico.")
                r1, r2, r3 = st.columns(3)
                r1.metric("Pérdida en esos casos", f"{rep['perdida_ponderada_despues']:.3f}",
                          delta=f"{rep['perdida_ponderada_despues'] - rep['perdida_ponderada_antes']:.3f}",
                          delta_color="inverse")
                r2.metric("Aciertos en esos casos", f"{rep['aciertos_despues']} / {rep['n_casos_nuevos']}",
                          delta=f"{rep['aciertos_despues'] - rep['aciertos_antes']:+d}")
                r3.write(f"**Calibración por clase:** `{rep['sesgo_calibracion']}`  \n"
                         f"FN graves previos: {rep['falsos_negativos_graves_previos']} · "
                         f"falsas alarmas previas: {rep['falsas_alarmas_previas']}")
