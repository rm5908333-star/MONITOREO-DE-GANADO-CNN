#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Sistema inteligente de monitoreo de ganado con CNN
Inciso d) Algoritmo: Red Neuronal Convolucional (CNN) -- MÓDULO DE PERCEPCIÓN
Prototipo ejecutable de simulación (mock realista) en Python + NumPy
===============================================================================

Alcance de este módulo
----------------------
Este archivo implementa únicamente la PERCEPCIÓN del agente (basado en modelo,
utilidad y aprendizaje). NO decide si se envía una alerta: eso le corresponde a
la Función de Utilidad del agente (Pasos 7 y 8 del diagrama de flujo).

Pasos del diagrama de flujo (documento base) que toca este módulo:
    Paso 1   Captura de imágenes ........ simulada con un generador térmico
    Paso 2   Preprocesamiento ........... corrección por clima + normalización
    Paso 3   CNN detector (CNN 1) ....... animal y ROI: ojo, zona vulvar, ubre
    Paso 4   Identificación (RFID) ...... consulta del estado interno
    Paso 5a  Firma térmica y contexto ... recortes ROI + vector de contexto
    Paso 6a  CNN clasificador (CNN 2) ... P(sano), P(celo), P(enfermo)
    Paso 11  Reentrenar la CNN .......... hook de mejora continua
(Pasos 5b, 6b, 7, 8, 9 y 10 pertenecen a otros módulos y no se simulan aquí;
el Paso 10 solo aparece como entrada del hook: la confirmación del veterinario.)

Honestidad del prototipo
------------------------
No hay dataset real ni GPU. Por eso:
  * Las imágenes térmicas se SINTETIZAN con un modelo físico simple.
  * La CNN 1 usa un "backbone" convolucional congelado (filtros fijos, como en
    transfer learning) y una cabeza de detección cuyos parámetros se afinan
    con datos de entrenamiento.
  * La CNN 2 tiene capas convolucionales + ReLU + pooling congeladas (filtros
    "transferidos") y la parte totalmente conectada (la cabeza) SÍ se entrena
    con retropropagación, entropía cruzada ponderada, Dropout y detención
    temprana.
Las métricas que se imprimen miden este prototipo sobre datos simulados; NO son
evidencia del desempeño que tendría el sistema con ganado real.

Dependencias: solo numpy (biblioteca estándar + numpy).
Ejecución:    python3 cnn_monitoreo_ganado.py
===============================================================================
"""

import math
import time
from collections import Counter

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# =============================================================================
# CONSTANTES GLOBALES (nomenclatura del documento base)
# =============================================================================
CLASES_ROI = ("cuerpo_animal", "ojo", "zona_vulvar", "ubre")   # CNN 1 (Paso 3)
ESTADOS = ("sano", "celo", "enfermo")                          # CNN 2 (Paso 6a)
IDX_ESTADO = {e: i for i, e in enumerate(ESTADOS)}

# Criterio de prioridad del inciso a): FN de enfermo > FN de celo > falsa alarma.
# Peso aplicado a la clase REAL de cada ejemplo en la entropía cruzada.
PESOS_CLASE = np.array([1.0, 3.0, 5.0])    # [sano, celo, enfermo]

TAM_IMAGEN = 224          # imagen térmica preprocesada 224 x 224
TAM_RECORTE = 16          # cada ROI se remuestrea a 16 x 16 para la CNN 2
T_MIN_C, T_MAX_C = 5.0, 45.0   # rango físico usado para normalizar a [0, 1]
CICLO_DIAS = 21           # ciclo estral bovino (día 0 = celo)

# Modelo de distorsión radiométrica del sensor (clima). La corrección del
# Paso 2 invierte este modelo (se asume cámara calibrada).
K_AMB, K_HUM = 0.06, 0.015       # °C por °C de ambiente / por % de humedad
T_REF, H_REF = 20.0, 60.0        # condiciones de referencia


# =============================================================================
# UTILIDADES NUMÉRICAS (convolución, pooling, remuestreo) -- solo NumPy
# =============================================================================
def _conv2d_same(img, kernel):
    """Convolución 2D (correlación) con relleno de bordes; img (H, W)."""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    relleno = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
    ventanas = sliding_window_view(relleno, (kh, kw))
    return np.einsum("ijkl,kl->ij", ventanas, kernel)


def _conv2d_valido(x, k):
    """Convolución 'valid' multicanal. x: (B, C, H, W); k: (F, C, kh, kw)."""
    kh, kw = k.shape[2], k.shape[3]
    ventanas = sliding_window_view(x, (kh, kw), axis=(2, 3))   # (B,C,H',W',kh,kw)
    return np.einsum("bcijkl,fckl->bfij", ventanas, k, optimize=True)


def _maxpool2(x):
    """Max-pooling 2x2 con paso 2. x: (B, C, H, W)."""
    b, c, h, w = x.shape
    h2, w2 = h // 2, w // 2
    x = x[:, :, :h2 * 2, :w2 * 2].reshape(b, c, h2, 2, w2, 2)
    return x.max(axis=(3, 5))


def _relu(x):
    return np.maximum(x, 0.0)


def _redimensionar_bilineal(a, alto, ancho):
    """Remuestreo bilineal de una matriz 2D a (alto, ancho)."""
    h, w = a.shape
    ys = np.linspace(0, h - 1, alto)
    xs = np.linspace(0, w - 1, ancho)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    sup = a[np.ix_(y0, x0)] * (1 - wx) + a[np.ix_(y0, x1)] * wx
    inf = a[np.ix_(y1, x0)] * (1 - wx) + a[np.ix_(y1, x1)] * wx
    return sup * (1 - wy) + inf * wy


def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _gauss_kernel(tam, sigma):
    ax = np.arange(tam) - tam // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


# Filtros fijos del backbone congelado (transfer learning simulado)
K_GAUSS5 = _gauss_kernel(5, 1.2)
K_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
K_SOBEL_Y = K_SOBEL_X.T.copy()
K_CAJA3 = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)
K_LAPLACIANO = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


# =============================================================================
# PASO 2 -- PREPROCESAMIENTO: corrección por clima + normalización térmica
# =============================================================================
def corregir_por_clima(imagen_c, temp_ambiental_c, humedad_pct):
    """
    Paso 2. Resta el sesgo radiométrico que introducen la temperatura ambiente
    y la humedad (atenuación atmosférica) respecto de las condiciones de
    referencia. Devuelve la imagen corregida en °C.
    """
    sesgo = K_AMB * (temp_ambiental_c - T_REF) + K_HUM * (humedad_pct - H_REF)
    return imagen_c - sesgo


def normalizar_termico(imagen_c, temp_ambiental_c, humedad_pct):
    """
    Paso 2. Normalización térmica respecto a condiciones ambientales:
    1) corrige por clima; 2) escala linealmente [T_MIN_C, T_MAX_C] -> [0, 1].
    Así las imágenes de distintos días/climas son comparables entre sí.
    """
    corregida = corregir_por_clima(imagen_c, temp_ambiental_c, humedad_pct)
    norm = (corregida - T_MIN_C) / (T_MAX_C - T_MIN_C)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def desnormalizar(imagen_norm):
    """Inversa de la escala lineal del Paso 2: [0, 1] -> °C (ya corregidos)."""
    return imagen_norm * (T_MAX_C - T_MIN_C) + T_MIN_C


# =============================================================================
# PASO 1 (SIMULADO) -- GENERADOR DE IMÁGENES TÉRMICAS Y DATASET SIMULADO
# =============================================================================
# Perfiles de rancho: el segundo simula un corral nuevo (otra raza, otra cámara)
# y se usa para demostrar la mejora continua (Paso 11).
PERFIL_RANCHO_ORIGINAL = {"base_mu": 38.5, "base_sd": 0.20, "escala_celo": 1.0,
                          "escala_enfermo": 1.0, "sesgo_sensor_c": 0.0}
PERFIL_RANCHO_NUEVO = {"base_mu": 38.8, "base_sd": 0.15, "escala_celo": 0.75,
                       "escala_enfermo": 0.80, "sesgo_sensor_c": -0.30}

RADIO_ROI_PX = {"ojo": 6.0, "zona_vulvar": 9.0, "ubre": 11.0}


def _muestrear_deltas(estado, rng, perfil):
    """Elevaciones térmicas (°C) por región según el estado verdadero."""
    d = {"cuerpo": 0.0, "ojo": 0.0, "vulva": 0.0, "ubre": 0.0}
    if estado == "sano":
        for k in d:
            d[k] = rng.normal(0.0, 0.15)
        if rng.random() < 0.10:                       # ejercicio reciente
            for k in d:
                d[k] += rng.uniform(0.2, 0.5)
    elif estado == "celo":
        e = perfil["escala_celo"]
        d["vulva"] = rng.uniform(1.1, 2.2) * e        # elevación LOCALIZADA
        d["ojo"] = rng.uniform(0.0, 0.4)
        d["cuerpo"] = rng.uniform(0.0, 0.3)
        d["ubre"] = rng.uniform(0.0, 0.3)
    else:  # enfermo: elevación sostenida en ojo y ubre
        e = perfil["escala_enfermo"]
        patron = rng.choice(["ojo_ubre", "ojo_ubre", "ojo_ubre", "solo_ojo", "solo_ubre"])
        d["ojo"] = rng.uniform(0.9, 2.0) * e if patron != "solo_ubre" else rng.uniform(0.0, 0.3)
        d["ubre"] = rng.uniform(0.9, 2.2) * e if patron != "solo_ojo" else rng.uniform(0.0, 0.3)
        d["cuerpo"] = rng.uniform(0.4, 1.0) * e
        d["vulva"] = rng.uniform(0.0, 0.4)
    return d


def _muestrear_dia_ciclo(estado, rng):
    """Día del ciclo reproductivo; el celo ocurre en la ventana fértil."""
    if estado == "celo":
        return int(rng.choice([19, 20, 0, 1]))
    if estado == "sano" and rng.random() < 0.12:      # sano en ventana fértil
        return int(rng.choice([19, 20, 0, 1]))
    return int(rng.integers(0, CICLO_DIAS))


def generar_registros(n_animales, obs_por_animal, semilla=0, perfil=None, prefijo="VACA"):
    """
    Genera metadatos (sin imágenes) de un dataset simulado. Cada animal tiene
    línea base y particularidades anatómicas propias y varias observaciones
    (días distintos, clima distinto, estado distinto).
    """
    perfil = perfil or PERFIL_RANCHO_ORIGINAL
    rng = np.random.default_rng(semilla)
    registros = []
    for i in range(n_animales):
        animal_id = f"{prefijo}-{i:03d}"
        base = float(np.clip(rng.normal(perfil["base_mu"], perfil["base_sd"]), 38.0, 39.2))
        offsets = {"ojo": rng.normal(0, 0.15), "vulva": rng.normal(0, 0.15),
                   "ubre": rng.normal(0, 0.15)}
        for _ in range(obs_por_animal):
            estado = str(rng.choice(ESTADOS, p=[0.65, 0.17, 0.18]))
            a = rng.uniform(66, 76)
            b = rng.uniform(34, 40)
            ocl = None
            if rng.random() < 0.06:                    # ROI parcialmente tapada
                ocl = (str(rng.choice(["ojo", "zona_vulvar", "ubre"])), rng.uniform(0.0, 0.4))
            registros.append({
                "animal_id": animal_id,
                "rfid": f"RFID-{animal_id}",
                "estado": estado,
                "base_c": base,
                "dia_ciclo": _muestrear_dia_ciclo(estado, rng),
                "temp_amb": float(rng.uniform(6, 30)),
                "humedad": float(rng.uniform(30, 95)),
                "offsets": offsets,
                "deltas": _muestrear_deltas(estado, rng, perfil),
                "geometria": {"cx": rng.uniform(112, 128), "cy": rng.uniform(100, 116),
                              "a": a, "b": b},
                "oclusion": ocl,
                "sesgo_sensor_c": perfil["sesgo_sensor_c"],
                "semilla_ruido": int(rng.integers(0, 2 ** 31 - 1)),
            })
    return registros


def _mascara_elipse(X, Y, cx, cy, a, b):
    r = np.sqrt(((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2)
    return _sigmoide(-(r - 1.0) * math.sqrt(a * b))


def _mascara_rect(X, Y, x0, x1, y0, y1):
    s = 0.9
    return (_sigmoide((X - x0) / s) * _sigmoide((x1 - X) / s)
            * _sigmoide((Y - y0) / s) * _sigmoide((y1 - Y) / s))


def _ruido_suave(rng, amplitud):
    """Campo de ruido de baja frecuencia (variaciones de sol/suelo)."""
    campo = rng.normal(0, 1, (8, 8))
    return amplitud * _redimensionar_bilineal(campo, TAM_IMAGEN, TAM_IMAGEN)


def capturar_imagen_termica(registro):
    """
    Paso 1 (simulado). Sintetiza una imagen térmica 224x224 en °C CRUDOS (con la
    distorsión del sensor por clima, que el Paso 2 deberá corregir) y las cajas
    verdaderas (ground truth) de las 4 clases. La vaca mira a la izquierda.
    """
    rng = np.random.default_rng(registro["semilla_ruido"])
    g = registro["geometria"]
    cx, cy, a, b = g["cx"], g["cy"], g["a"], g["b"]
    yy, xx = np.mgrid[0:TAM_IMAGEN, 0:TAM_IMAGEN].astype(np.float32)

    # --- Máscara del animal: cuerpo + cabeza + 4 patas + ubre ---------------
    hx, hy = cx - a - 6.0, cy - 0.35 * b
    px_ubre, py_ubre = cx + 0.52 * a, cy + 0.90 * b
    m = _mascara_elipse(xx, yy, cx, cy, a, b)
    m = np.maximum(m, _mascara_elipse(xx, yy, hx, hy, 20.0, 15.0))
    for fx in (-0.65, -0.40, 0.40, 0.65):
        m = np.maximum(m, _mascara_rect(xx, yy, cx + fx * a - 4.5, cx + fx * a + 4.5,
                                        cy + 0.7 * b, cy + b + 34.0))
    m = np.maximum(m, _mascara_elipse(xx, yy, px_ubre, py_ubre, 12.0, 9.0))

    # --- Temperaturas reales (°C) -------------------------------------------
    base, d, off, amb = registro["base_c"], registro["deltas"], registro["offsets"], registro["temp_amb"]
    t_cuerpo = base - 3.5 + 0.08 * (amb - T_REF) + d["cuerpo"]
    t_ojo = base + off["ojo"] + d["ojo"]
    t_vulva = base - 0.8 + off["vulva"] + d["vulva"]
    t_ubre = base - 1.5 + off["ubre"] + d["ubre"]

    suelo = amb + 0.8 * (yy / TAM_IMAGEN - 0.5) + _ruido_suave(rng, 0.3)
    T = suelo * (1.0 - m) + (t_cuerpo + _ruido_suave(rng, 0.15)) * m

    # --- Regiones de interés: puntos calientes gaussianos + cajas GT --------
    centros = {"ojo": (hx - 6.0, hy - 3.0), "zona_vulvar": (cx + a - 8.0, cy + 0.15 * b),
               "ubre": (px_ubre, py_ubre)}
    objetivos = {"ojo": t_ojo, "zona_vulvar": t_vulva, "ubre": t_ubre}
    gt = {}
    for clase, (rx, ry) in centros.items():
        r = RADIO_ROI_PX[clase]
        sigma = r / 1.8
        w = np.exp(-((xx - rx) ** 2 + (yy - ry) ** 2) / (2 * sigma ** 2))
        if registro["oclusion"] and registro["oclusion"][0] == clase:
            w = w * registro["oclusion"][1]
        T = T * (1.0 - w) + objetivos[clase] * w
        gt[clase] = [float(round(rx - r)), float(round(ry - r)),
                     float(round(rx + r)), float(round(ry + r))]

    cols = np.where((m > 0.5).any(axis=0))[0]
    filas = np.where((m > 0.5).any(axis=1))[0]
    gt["cuerpo_animal"] = [float(cols[0]), float(filas[0]), float(cols[-1] + 1), float(filas[-1] + 1)]

    # --- Distorsión del sensor por clima + ruido ----------------------------
    distorsion = (K_AMB * (amb - T_REF) + K_HUM * (registro["humedad"] - H_REF)
                  + registro["sesgo_sensor_c"] + rng.normal(0, 0.08))
    T = T + distorsion + rng.normal(0, 0.12, T.shape)
    return T.astype(np.float32), gt


# =============================================================================
# PARTICIÓN DE DATOS (fase fuera de línea) SIN FUGA POR animal_id
# =============================================================================
def particionar_por_animal(registros, proporciones=(0.67, 0.17, 0.16), semilla=0):
    """
    Divide el dataset en train / val / test A NIVEL DE ANIMAL: primero se
    reparten los animal_id únicos y después cada registro hereda el conjunto de
    su animal. Así un mismo animal nunca aparece en dos conjuntos.
    """
    if not math.isclose(sum(proporciones), 1.0, abs_tol=1e-9):
        raise ValueError("Las proporciones deben sumar 1.0")
    ids = sorted({r["animal_id"] for r in registros})
    rng = np.random.default_rng(semilla)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(proporciones[0] * n))
    n_val = int(round(proporciones[1] * n))
    ids_train = set(ids[:n_train])
    ids_val = set(ids[n_train:n_train + n_val])
    ids_test = set(ids[n_train + n_val:])
    particion = {
        "train": [r for r in registros if r["animal_id"] in ids_train],
        "val": [r for r in registros if r["animal_id"] in ids_val],
        "test": [r for r in registros if r["animal_id"] in ids_test],
    }
    verificar_sin_fuga_de_datos(particion)
    return particion


def verificar_sin_fuga_de_datos(particion):
    """
    Verificación explícita: la intersección de animal_id entre cada par de
    conjuntos debe ser vacía. Lanza AssertionError si hay fuga de datos.
    """
    ids = {nombre: {r["animal_id"] for r in regs} for nombre, regs in particion.items()}
    pares = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pares:
        comunes = ids[a] & ids[b]
        assert not comunes, f"FUGA DE DATOS: {sorted(comunes)} aparece(n) en '{a}' y en '{b}'"
    return True


# =============================================================================
# PASO 4 -- IDENTIFICACIÓN DEL ANIMAL (RFID) Y ESTADO INTERNO (interfaz simulada)
# =============================================================================
class EstadoInternoGanado:
    """
    Modelo del mundo del agente (solo la interfaz que consume el módulo de
    percepción). Guarda por animal: línea base térmica, etapa del ciclo
    reproductivo, zona asignada e historial reciente.
    """

    def __init__(self):
        self._animales = {}

    def registrar(self, rfid, animal_id, temperatura_base_habitual, dia_ciclo_reproductivo,
                  zona_asignada="corral_A", historial_ojo_c=None):
        self._animales[rfid] = {
            "animal_id": animal_id,
            "temperatura_base_habitual": float(temperatura_base_habitual),
            "dia_ciclo_reproductivo": int(dia_ciclo_reproductivo),
            "zona_asignada": zona_asignada,
            "historial_ojo_c": list(historial_ojo_c or []),
        }

    def consultar(self, rfid):
        if rfid not in self._animales:
            raise KeyError(f"RFID desconocido: {rfid}")
        return dict(self._animales[rfid])


def identificar_animal(rfid_leido, estado_interno):
    """Paso 4. Asocia la detección con el animal y consulta su historial."""
    return estado_interno.consultar(rfid_leido)


# =============================================================================
# PASO 3 -- CNN 1: DETECTORA DE OBJETOS (transfer learning simulado)
# =============================================================================
class CNNDetectorObjetos:
    """
    CNN 1. Entrada: imagen térmica normalizada (224x224, valores en [0, 1]).
    Salida: cajas [x1, y1, x2, y2] de 4 clases + recortes térmicos por región.

    Transfer learning simulado:
      * BACKBONE congelado: banco de filtros fijos (suavizado gaussiano y
        gradientes de Sobel) que produce un mapa de calor y un mapa de bordes.
      * CABEZA de detección: anclas anatómicas relativas al cuerpo (ventanas de
        búsqueda por ROI) + regresión de caja. Sus parámetros de escala se
        AFINAN (fine-tuning) con datos de entrenamiento: afinar_cabeza().
    Supuesto del mock: el animal aparece de perfil con la cabeza a la izquierda.
    """

    def __init__(self, ruido_localizacion_px=0.8, semilla=5):
        # Ventanas de búsqueda (fracciones de la caja del cuerpo): x0, y0, x1, y1
        self.anclas = {"ojo": (0.00, 0.00, 0.24, 0.42),
                       "zona_vulvar": (0.84, 0.24, 1.00, 0.60),
                       "ubre": (0.66, 0.54, 0.93, 0.82)}
        self.tam_defecto = {"ojo": 12.0, "zona_vulvar": 18.0, "ubre": 22.0}
        self.k_expansion = {c: (1.5, 1.5) for c in self.anclas}   # se afina
        self.umbral_contraste_c = 0.35        # °C mínimos para aceptar un punto caliente
        self.ruido_localizacion_px = ruido_localizacion_px
        self._rng = np.random.default_rng(semilla)

    # ---- backbone congelado -------------------------------------------------
    @staticmethod
    def _backbone_congelado(img):
        calor = _conv2d_same(img, K_GAUSS5)
        gx = _conv2d_same(calor, K_SOBEL_X)
        gy = _conv2d_same(calor, K_SOBEL_Y)
        return calor, np.hypot(gx, gy)

    # ---- localización del cuerpo -------------------------------------------
    def _localizar_cuerpo(self, calor):
        borde = np.concatenate([calor[:8].ravel(), calor[-8:].ravel(),
                                calor[:, :8].ravel(), calor[:, -8:].ravel()])
        fondo = float(np.median(borde))
        nivel = float(np.percentile(calor, 90))
        contraste_c = (nivel - fondo) * (T_MAX_C - T_MIN_C)
        if contraste_c < 1.0:
            return None
        fg = calor > fondo + 0.5 * (nivel - fondo)
        cols = np.where(fg.sum(axis=0) >= 3)[0]
        filas = np.where(fg.sum(axis=1) >= 3)[0]
        if len(cols) == 0 or len(filas) == 0:
            return None
        caja = [float(cols[0]), float(filas[0]), float(cols[-1] + 1), float(filas[-1] + 1)]
        frac = fg.mean()
        confianza = float(_sigmoide(1.0 * (contraste_c - 2.0))) * (1.0 if 0.08 < frac < 0.6 else 0.5)
        return caja, min(confianza, 0.99), fg

    @staticmethod
    def _extension_contigua(perfil, idx, umbral):
        """Extremos [i0, i1] del tramo contiguo alrededor de idx sobre el umbral."""
        i0 = idx
        while i0 > 0 and perfil[i0 - 1] > umbral:
            i0 -= 1
        i1 = idx
        while i1 < len(perfil) - 1 and perfil[i1 + 1] > umbral:
            i1 += 1
        return i0, i1

    # ---- localización de una ROI -------------------------------------------
    def _medir_roi(self, calor, fg, caja_cuerpo, clase):
        """
        Busca el punto caliente dentro de la ventana ancla y mide su extensión
        a media altura. Devuelve None si no hay punto caliente fiable.
        """
        x1, y1, x2, y2 = caja_cuerpo
        w, h = x2 - x1, y2 - y1
        fx0, fy0, fx1, fy1 = self.anclas[clase]
        wx0 = int(max(0, math.floor(x1 + fx0 * w)))
        wx1 = int(min(TAM_IMAGEN, math.ceil(x1 + fx1 * w)))
        wy0 = int(max(0, math.floor(y1 + fy0 * h)))
        wy1 = int(min(TAM_IMAGEN, math.ceil(y1 + fy1 * h)))
        if wx1 - wx0 < 2 or wy1 - wy0 < 2:
            return None, (wx0, wy0, wx1, wy1)
        ventana = calor[wy0:wy1, wx0:wx1]
        en_cuerpo = fg[wy0:wy1, wx0:wx1]
        ref = float(np.median(ventana[en_cuerpo])) if en_cuerpo.any() else float(np.median(ventana))
        iy, ix = np.unravel_index(np.argmax(ventana), ventana.shape)
        pico = float(ventana[iy, ix])
        contraste = pico - ref
        if contraste * (T_MAX_C - T_MIN_C) < self.umbral_contraste_c:
            return None, (wx0, wy0, wx1, wy1)
        py, px = wy0 + iy, wx0 + ix
        umbral = ref + 0.5 * contraste
        xi0, xi1 = self._extension_contigua(calor[py, :], px, umbral)
        yi0, yi1 = self._extension_contigua(calor[:, px], py, umbral)
        medida = {"cx": (xi0 + xi1 + 1) / 2.0, "cy": (yi0 + yi1 + 1) / 2.0,
                  "ancho": float(xi1 - xi0 + 1), "alto": float(yi1 - yi0 + 1),
                  "contraste_c": contraste * (T_MAX_C - T_MIN_C)}
        return medida, (wx0, wy0, wx1, wy1)

    # ---- fine-tuning de la cabeza de detección -----------------------------
    def afinar_cabeza(self, imagenes_norm, cajas_gt):
        """
        Fine-tuning de la cabeza (transfer learning): estima por clase el factor
        que convierte la extensión a media altura medida en el tamaño real de la
        caja (regresión de caja) y el tamaño por defecto (mediana de las GT).
        """
        razones = {c: ([], []) for c in self.anclas}
        tamanos = {c: ([], []) for c in self.anclas}
        for img, gt in zip(imagenes_norm, cajas_gt):
            calor, _ = self._backbone_congelado(img)
            cuerpo = self._localizar_cuerpo(calor)
            if cuerpo is None:
                continue
            caja, _, fg = cuerpo
            for clase in self.anclas:
                med, _ = self._medir_roi(calor, fg, caja, clase)
                gx0, gy0, gx1, gy1 = gt[clase]
                tamanos[clase][0].append(gx1 - gx0)
                tamanos[clase][1].append(gy1 - gy0)
                if med is not None:
                    razones[clase][0].append((gx1 - gx0) / med["ancho"])
                    razones[clase][1].append((gy1 - gy0) / med["alto"])
        for clase in self.anclas:
            if razones[clase][0]:
                self.k_expansion[clase] = (float(np.median(razones[clase][0])),
                                           float(np.median(razones[clase][1])))
            self.tam_defecto[clase] = float(np.median(tamanos[clase][0] + tamanos[clase][1]))
        return self.k_expansion

    # ---- recorte térmico de una región -------------------------------------
    @staticmethod
    def _recorte_termico(img_norm, mapa_bordes, caja):
        x1, y1, x2, y2 = [int(round(v)) for v in caja]
        x1, x2 = max(0, x1), min(TAM_IMAGEN, max(x2, x1 + 2))
        y1, y2 = max(0, y1), min(TAM_IMAGEN, max(y2, y1 + 2))
        recorte = img_norm[y1:y2, x1:x2]
        matriz = _redimensionar_bilineal(recorte, TAM_RECORTE, TAM_RECORTE).astype(np.float32)
        return {
            "temp_promedio_c": float(desnormalizar(recorte.mean())),
            "gradiente_espacial_c_por_px": float(mapa_bordes[y1:y2, x1:x2].mean() * (T_MAX_C - T_MIN_C)),
            "matriz_norm_16x16": matriz,
        }

    # ---- inferencia ---------------------------------------------------------
    def detectar(self, img_norm):
        """
        Paso 3. Devuelve un diccionario estructurado:
        {"animal_detectado": bool,
         "detecciones": {clase: {"bbox": [x1,y1,x2,y2], "confianza": float,
                                 "recorte_termico": {"temp_promedio_c", "gradiente_espacial_c_por_px",
                                                     "matriz_norm_16x16"}}}}
        """
        if img_norm.shape != (TAM_IMAGEN, TAM_IMAGEN):
            raise ValueError(f"Se esperaba una imagen {TAM_IMAGEN}x{TAM_IMAGEN}, llegó {img_norm.shape}")
        calor, bordes = self._backbone_congelado(img_norm)
        cuerpo = self._localizar_cuerpo(calor)
        if cuerpo is None:
            return {"imagen_shape": img_norm.shape, "animal_detectado": False, "detecciones": {}}
        caja_c, conf_c, fg = cuerpo
        detecciones = {"cuerpo_animal": {"bbox": caja_c, "confianza": conf_c}}

        for clase in ("ojo", "zona_vulvar", "ubre"):
            med, ventana = self._medir_roi(calor, fg, caja_c, clase)
            if med is None:   # sin punto caliente fiable: ancla anatómica, confianza baja
                cx = (ventana[0] + ventana[2]) / 2.0
                cy = (ventana[1] + ventana[3]) / 2.0
                mitad_x = mitad_y = self.tam_defecto[clase] / 2.0
                confianza = 0.10
            else:
                cx, cy = med["cx"], med["cy"]
                mitad_x = 0.5 * med["ancho"] * self.k_expansion[clase][0]
                mitad_y = 0.5 * med["alto"] * self.k_expansion[clase][1]
                confianza = min(0.98 * float(_sigmoide(1.1 * (med["contraste_c"] - 1.2))), 0.98)
            ruido = self._rng.normal(0, self.ruido_localizacion_px, 2)   # imprecisión del regresor
            cx, cy = cx + ruido[0], cy + ruido[1]
            caja = [max(0.0, cx - mitad_x), max(0.0, cy - mitad_y),
                    min(float(TAM_IMAGEN), cx + mitad_x), min(float(TAM_IMAGEN), cy + mitad_y)]
            detecciones[clase] = {"bbox": caja, "confianza": confianza}

        for clase, det in detecciones.items():
            det["bbox"] = [float(round(v, 2)) for v in det["bbox"]]
            det["confianza"] = float(round(det["confianza"], 4))
            det["recorte_termico"] = self._recorte_termico(img_norm, bordes, det["bbox"])
        return {"imagen_shape": img_norm.shape, "animal_detectado": True, "detecciones": detecciones}


# =============================================================================
# PASO 5a -- FIRMA TÉRMICA Y CONTEXTO (entrada de la CNN 2)
# =============================================================================
def construir_vector_contexto(contexto):
    """
    Paso 5a. Vector de contexto del animal (4 entradas, en este orden):
    temperatura_base_habitual, dia_ciclo_reproductivo, temperatura_ambiental, humedad.
    El día del ciclo se codifica como fase circular (1 en el día de celo, 0 a mitad
    del ciclo) para que el día 20 y el día 0 queden cercanos. Valores a ~[0, 1].
    """
    faltan = {"temperatura_base_habitual", "dia_ciclo_reproductivo",
              "temperatura_ambiental", "humedad"} - set(contexto)
    if faltan:
        raise KeyError(f"Faltan campos de contexto: {sorted(faltan)}")
    fase = 0.5 * (1.0 + math.cos(2.0 * math.pi * contexto["dia_ciclo_reproductivo"] / CICLO_DIAS))
    return np.array([(contexto["temperatura_base_habitual"] - 37.0) / 3.0,
                     fase,
                     (contexto["temperatura_ambiental"] - 5.0) / 40.0,
                     contexto["humedad"] / 100.0], dtype=np.float32)


def ensamblar_entrada_cnn2(salida_cnn1, estado_interno_animal, temp_ambiental, humedad):
    """
    Paso 5a. Toma los recortes térmicos de las 4 ROI extraídos por la CNN 1 y los
    combina con el vector de contexto del estado interno. Devuelve
    (recortes[4,16,16], contexto[4]) listos para la CNN 2.
    """
    det = salida_cnn1["detecciones"]
    recortes = np.stack([det[c]["recorte_termico"]["matriz_norm_16x16"] for c in CLASES_ROI])
    ctx = construir_vector_contexto({
        "temperatura_base_habitual": estado_interno_animal["temperatura_base_habitual"],
        "dia_ciclo_reproductivo": estado_interno_animal["dia_ciclo_reproductivo"],
        "temperatura_ambiental": temp_ambiental,
        "humedad": humedad})
    return recortes.astype(np.float32), ctx


# =============================================================================
# FUNCIÓN DE PÉRDIDA: ENTROPÍA CRUZADA PONDERADA
# =============================================================================
def entropia_cruzada_ponderada(probabilidades, etiquetas, pesos_clase=PESOS_CLASE, eps=1e-12):
    """
    Weighted Cross-Entropy:  L = - sum_i w[y_i] * log p_i[y_i] / sum_i w[y_i]
    El peso depende de la clase REAL: no detectar un 'enfermo' (peso 5) o un
    'celo' (peso 3) cuesta más que alarmar por un 'sano' (peso 1), conforme al
    criterio de prioridad del inciso a).
    """
    p = np.clip(np.asarray(probabilidades, dtype=np.float64), eps, 1.0)
    y = np.asarray(etiquetas, dtype=int)
    w = np.asarray(pesos_clase, dtype=np.float64)[y]
    nll = -np.log(p[np.arange(len(y)), y])
    return float((w * nll).sum() / w.sum())


# =============================================================================
# PASO 6a -- CNN 2: CLASIFICADORA MULTIMODAL
# =============================================================================
class CNNClasificadoraMultimodal:
    """
    CNN 2. Entradas: recortes térmicos de las 4 ROI (4 x 16 x 16) + vector de
    contexto (4). Arquitectura:

      por ROI (filtros compartidos):  Conv3x3(4) -> ReLU -> MaxPool2
                                      -> Conv3x3(8) -> ReLU -> MaxPool2 -> Flatten (32)
      vector visual = concatenación de las 4 ROI (128)
      CONCATENACIÓN MULTIMODAL: [visual(128) | contexto(4)] = 132
      FC(132->32) ReLU Dropout -> FC(32->16) ReLU Dropout -> FC(16->3) -> Softmax

    Transfer learning simulado: las capas convolucionales están congeladas y la
    parte totalmente conectada se entrena (propagación hacia atrás, Adam,
    entropía cruzada ponderada, Dropout, L2 y detención temprana).
    """

    def __init__(self, semilla=11, p_dropout=0.3, ocultas=(32, 16)):
        rng = np.random.default_rng(semilla)
        self.k1 = np.stack([K_CAJA3, K_SOBEL_X, K_SOBEL_Y, K_LAPLACIANO])[:, None].astype(np.float32)
        self.k2 = rng.normal(0, math.sqrt(2.0 / (4 * 9)), (8, 4, 3, 3)).astype(np.float32)
        self.dim_visual = len(CLASES_ROI) * 32
        self.dim_ctx = 4
        dims = [self.dim_visual + self.dim_ctx, *ocultas, len(ESTADOS)]
        self.W = [rng.normal(0, math.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.p_dropout = p_dropout
        self.pesos_clase = PESOS_CLASE.copy()
        self.sesgo_calibracion = np.zeros(len(ESTADOS))   # umbral/calibración blanda por clase
        self.media_x = None
        self.std_x = None
        self.memoria_replay = None                         # (X estandarizado, y) para Paso 11
        self.version = 1
        self.historial_reentrenamientos = []

    # ---- parte convolucional (congelada) -----------------------------------
    def extraer_vector_visual(self, recortes):
        """Capas convolucionales + ReLU + Pooling + Flatten. recortes: (N,4,16,16)."""
        n = recortes.shape[0]
        x = recortes.reshape(n * len(CLASES_ROI), 1, TAM_RECORTE, TAM_RECORTE).astype(np.float32)
        x = _maxpool2(_relu(_conv2d_valido(x, self.k1)))    # 16->14->7
        x = _maxpool2(_relu(_conv2d_valido(x, self.k2)))    # 7->5->2
        return x.reshape(n, -1)                              # (N, 4*32)

    def construir_entrada(self, recortes, contexto, ajustar_escalador=False):
        """Concatenación multimodal [visual | contexto] + estandarización."""
        visual = self.extraer_vector_visual(recortes)
        x = np.concatenate([visual, np.asarray(contexto, dtype=np.float32)], axis=1).astype(np.float64)
        if ajustar_escalador:
            self.media_x = x.mean(axis=0)
            self.std_x = x.std(axis=0) + 1e-6
        return (x - self.media_x) / self.std_x

    # ---- parte totalmente conectada (entrenable) ---------------------------
    @staticmethod
    def _softmax(logits):
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, x, entrenamiento=False, rng=None):
        entradas, preact, mascaras = [], [], []
        a = x
        for i in range(len(self.W) - 1):
            entradas.append(a)
            z = a @ self.W[i] + self.b[i]
            preact.append(z)
            a = _relu(z)
            if entrenamiento and self.p_dropout > 0:
                m = (rng.random(a.shape) >= self.p_dropout) / (1.0 - self.p_dropout)
                a = a * m
                mascaras.append(m)
            else:
                mascaras.append(None)
        entradas.append(a)
        logits = a @ self.W[-1] + self.b[-1] + self.sesgo_calibracion
        return self._softmax(logits), (entradas, preact, mascaras)

    def _perdida_y_gradientes(self, x, y, entrenamiento, rng, l2):
        probs, (entradas, preact, mascaras) = self._forward(x, entrenamiento, rng)
        w = self.pesos_clase[y]
        sw = w.sum()
        perdida = float((w * -np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0))).sum() / sw)
        dz = probs.copy()
        dz[np.arange(len(y)), y] -= 1.0
        dz *= (w / sw)[:, None]                              # d L / d logits
        gW = [None] * len(self.W)
        gb = [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = entradas[i].T @ dz + l2 * self.W[i]
            gb[i] = dz.sum(axis=0)
            if i > 0:
                da = dz @ self.W[i].T
                if mascaras[i - 1] is not None:
                    da = da * mascaras[i - 1]
                dz = da * (preact[i - 1] > 0)
        return perdida, gW, gb

    @staticmethod
    def _adam_paso(params, grads, estado, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for i, (p, g) in enumerate(zip(params, grads)):
            estado["m"][i] = b1 * estado["m"][i] + (1 - b1) * g
            estado["v"][i] = b2 * estado["v"][i] + (1 - b2) * g * g
            mh = estado["m"][i] / (1 - b1 ** t)
            vh = estado["v"][i] / (1 - b2 ** t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def _nuevo_estado_adam(self):
        params = self.W + self.b
        return {"m": [np.zeros_like(p) for p in params], "v": [np.zeros_like(p) for p in params]}

    def _paso_optimizador(self, x, y, estado, lr, t, rng, l2):
        perdida, gW, gb = self._perdida_y_gradientes(x, y, True, rng, l2)
        self._adam_paso(self.W + self.b, gW + gb, estado, lr, t)   # actualiza in situ
        return perdida

    def entrenar(self, recortes_tr, ctx_tr, y_tr, recortes_val, ctx_val, y_val,
                 epocas=300, lr=3e-3, l2=1e-3, tam_lote=48, paciencia=40, semilla=3, verbose=True):
        """
        Entrenamiento supervisado de la cabeza: entropía cruzada ponderada,
        retropropagación + Adam, Dropout y detención temprana sobre validación.
        """
        rng = np.random.default_rng(semilla)
        x_tr = self.construir_entrada(recortes_tr, ctx_tr, ajustar_escalador=True)
        x_val = self.construir_entrada(recortes_val, ctx_val)
        y_tr, y_val = np.asarray(y_tr), np.asarray(y_val)
        estado, t = self._nuevo_estado_adam(), 0
        mejor, mejor_ep, sin_mejora = np.inf, 0, 0
        mejores_params = None
        historia = []
        for ep in range(1, epocas + 1):
            orden = rng.permutation(len(y_tr))
            perd_ep = []
            for ini in range(0, len(orden), tam_lote):
                idx = orden[ini:ini + tam_lote]
                t += 1
                perd_ep.append(self._paso_optimizador(x_tr[idx], y_tr[idx], estado, lr, t, rng, l2))
            p_val, _ = self._forward(x_val)
            l_val = entropia_cruzada_ponderada(p_val, y_val, self.pesos_clase)
            historia.append((ep, float(np.mean(perd_ep)), l_val))
            if l_val < mejor - 1e-5:
                mejor, mejor_ep, sin_mejora = l_val, ep, 0
                mejores_params = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                sin_mejora += 1
            if verbose and ep % 25 == 0:
                print(f"    época {ep:3d} | pérdida train {np.mean(perd_ep):.4f} | pérdida val {l_val:.4f}")
            if sin_mejora >= paciencia:
                if verbose:
                    print(f"    Detención temprana en la época {ep} (mejor época: {mejor_ep}).")
                break
        self.W, self.b = mejores_params
        self.memoria_replay = (x_tr.copy(), y_tr.copy())
        return {"mejor_epoca": mejor_ep, "mejor_perdida_val": mejor, "historia": historia}

    # ---- inferencia ---------------------------------------------------------
    def predecir_proba(self, recortes, contexto):
        """Salida softmax: [P(sano), P(celo), P(enfermo)] por fila; suma = 1.0."""
        x = self.construir_entrada(np.asarray(recortes), np.asarray(contexto))
        probs, _ = self._forward(x)
        return probs / probs.sum(axis=1, keepdims=True)

    def predecir_estado(self, recortes, contexto):
        return self.predecir_proba(recortes, contexto).argmax(axis=1)

    def inferir_un_animal(self, recortes_4x16x16, contexto_4):
        """Atajo para un solo animal: devuelve el vector de 3 probabilidades."""
        return self.predecir_proba(recortes_4x16x16[None], contexto_4[None])[0]


def verificar_gradiente(modelo, x, y, n_comprobaciones=6, h=1e-5, semilla=0):
    """
    Prueba de sanidad: compara el gradiente analítico de la entropía cruzada
    ponderada con diferencias finitas centrales (sin Dropout). Devuelve el
    máximo error relativo encontrado.
    """
    rng = np.random.default_rng(semilla)
    _, gW, gb = modelo._perdida_y_gradientes(x, y, False, rng, 0.0)
    peor = 0.0
    for _ in range(n_comprobaciones):
        capa = int(rng.integers(0, len(modelo.W)))
        i = int(rng.integers(0, modelo.W[capa].shape[0]))
        j = int(rng.integers(0, modelo.W[capa].shape[1]))
        original = modelo.W[capa][i, j]
        modelo.W[capa][i, j] = original + h
        lp, _, _ = modelo._perdida_y_gradientes(x, y, False, rng, 0.0)
        modelo.W[capa][i, j] = original - h
        lm, _, _ = modelo._perdida_y_gradientes(x, y, False, rng, 0.0)
        modelo.W[capa][i, j] = original
        numerico = (lp - lm) / (2 * h)
        analitico = gW[capa][i, j]
        peor = max(peor, abs(numerico - analitico) / max(1e-8, abs(numerico) + abs(analitico)))
    return peor


# =============================================================================
# MÉTRICAS CNN 1: IoU y mAP
# =============================================================================
def calcular_iou(caja_a, caja_b):
    """Intersección sobre unión de dos cajas [x1, y1, x2, y2]."""
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def calcular_ap(predicciones, ground_truth, iou_umbral=0.5):
    """
    Precisión promedio (AP) de UNA clase con interpolación de todos los puntos
    (estilo PASCAL VOC). predicciones: [(id_imagen, confianza, bbox)];
    ground_truth: {id_imagen: [bbox, ...]}.
    """
    n_gt = sum(len(v) for v in ground_truth.values())
    if n_gt == 0:
        return 0.0
    orden = sorted(predicciones, key=lambda p: -p[1])
    usada = {k: [False] * len(v) for k, v in ground_truth.items()}
    tp = np.zeros(len(orden))
    fp = np.zeros(len(orden))
    for i, (img, _, caja) in enumerate(orden):
        mejor_iou, mejor_j = 0.0, -1
        for j, g in enumerate(ground_truth.get(img, [])):
            iou = calcular_iou(caja, g)
            if iou > mejor_iou:
                mejor_iou, mejor_j = iou, j
        if mejor_iou >= iou_umbral and not usada[img][mejor_j]:
            tp[i] = 1
            usada[img][mejor_j] = True
        else:
            fp[i] = 1
    tp_acum, fp_acum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_acum / n_gt
    precision = tp_acum / np.maximum(tp_acum + fp_acum, 1e-12)
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def calcular_map(detecciones_por_clase, gt_por_clase, umbrales_iou=(0.5,)):
    """
    mAP = promedio de AP entre clases (y entre umbrales de IoU si se dan varios,
    p. ej. 0.50:0.05:0.95). Devuelve AP por clase y mAP global.
    """
    ap = {c: float(np.mean([calcular_ap(detecciones_por_clase[c], gt_por_clase[c], u)
                            for u in umbrales_iou])) for c in gt_por_clase}
    return {"ap_por_clase": ap, "mAP": float(np.mean(list(ap.values())))}


# =============================================================================
# MÉTRICAS CNN 2: matriz de confusión, precisión, recall, F1
# =============================================================================
def matriz_confusion(y_real, y_pred, n_clases=len(ESTADOS)):
    """Filas = estado real, columnas = estado predicho."""
    cm = np.zeros((n_clases, n_clases), dtype=int)
    for r, p in zip(y_real, y_pred):
        cm[int(r), int(p)] += 1
    return cm


def metricas_clasificacion(cm):
    """Precisión, Recall (sensibilidad) y F1 por clase, F1 macro y exactitud."""
    n = cm.shape[0]
    por_clase = {}
    for k in range(n):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        por_clase[ESTADOS[k]] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                                 "soporte": int(cm[k, :].sum())}
    return {"por_clase": por_clase,
            "f1_macro": float(np.mean([v["f1"] for v in por_clase.values()])),
            "exactitud": float(np.trace(cm) / max(1, cm.sum()))}


def imprimir_matriz_confusion(cm):
    ancho = 11
    print("    " + " " * 10 + "".join(f"pred:{e:<{ancho - 5}}" for e in ESTADOS))
    for i, e in enumerate(ESTADOS):
        print(f"    real:{e:<8}" + "".join(f"{cm[i, j]:<{ancho}d}" for j in range(len(ESTADOS))))


def imprimir_metricas(m):
    print(f"    {'estado':<9}{'precisión':>10}{'recall':>9}{'F1':>8}{'soporte':>9}")
    for e, v in m["por_clase"].items():
        print(f"    {e:<9}{v['precision']:>10.3f}{v['recall']:>9.3f}{v['f1']:>8.3f}{v['soporte']:>9d}")
    print(f"    F1 macro = {m['f1_macro']:.3f} | exactitud = {m['exactitud']:.3f}")


# =============================================================================
# CANALIZACIÓN COMPLETA DE PERCEPCIÓN (Pasos 1-5a-6a para un registro)
# =============================================================================
def procesar_registro(registro, cnn1):
    """
    Ejecuta Paso 1 (captura simulada) -> Paso 2 (normalización) -> Paso 3 (CNN 1)
    -> Paso 5a (recortes + contexto). Devuelve la muestra lista para la CNN 2,
    junto con las cajas detectadas y verdaderas para evaluar la CNN 1.
    """
    imagen_c, gt = capturar_imagen_termica(registro)                                  # Paso 1
    img_norm = normalizar_termico(imagen_c, registro["temp_amb"], registro["humedad"])  # Paso 2
    salida = cnn1.detectar(img_norm)                                                  # Paso 3
    if not salida["animal_detectado"]:
        return None
    estado_animal = {"temperatura_base_habitual": registro["base_c"],
                     "dia_ciclo_reproductivo": registro["dia_ciclo"]}
    recortes, ctx = ensamblar_entrada_cnn2(salida, estado_animal, registro["temp_amb"], registro["humedad"])  # 5a
    return {"animal_id": registro["animal_id"], "y": IDX_ESTADO[registro["estado"]],
            "recortes": recortes, "ctx": ctx, "gt": gt, "salida_cnn1": salida}


def procesar_lote(registros, cnn1):
    muestras, descartados = [], 0
    for r in registros:
        m = procesar_registro(r, cnn1)
        if m is None:
            descartados += 1
        else:
            muestras.append(m)
    return muestras, descartados


def apilar(muestras):
    return (np.stack([m["recortes"] for m in muestras]),
            np.stack([m["ctx"] for m in muestras]),
            np.array([m["y"] for m in muestras]))


def evaluar_cnn1(muestras):
    """Evalúa la CNN 1: IoU medio por clase y mAP (@0.5 y @[0.5:0.95])."""
    dets = {c: [] for c in CLASES_ROI}
    gts = {c: {} for c in CLASES_ROI}
    ious = {c: [] for c in CLASES_ROI}
    for i, m in enumerate(muestras):
        for c in CLASES_ROI:
            d = m["salida_cnn1"]["detecciones"][c]
            dets[c].append((i, d["confianza"], d["bbox"]))
            gts[c][i] = [m["gt"][c]]
            ious[c].append(calcular_iou(d["bbox"], m["gt"][c]))
    map50 = calcular_map(dets, gts, (0.5,))
    map_rango = calcular_map(dets, gts, tuple(np.arange(0.5, 0.96, 0.05)))
    return {"iou_medio": {c: float(np.mean(v)) for c, v in ious.items()},
            "ap50": map50["ap_por_clase"], "map50": map50["mAP"], "map_50_95": map_rango["mAP"]}


# =============================================================================
# PASO 11 -- HOOK DE MEJORA CONTINUA: reentrenar la CNN con casos confirmados
# =============================================================================
_SISTEMA = {"cnn1": None, "cnn2": None}   # modelos activos (singleton del prototipo)


def reentrenar_cnn_incremental(nuevos_casos_confirmados, cnn2=None, epocas=30, lr=5e-4,
                               repeticiones_caso_nuevo=3, razon_replay=2.0, semilla=0):
    """
    Pasos 10 y 11. Recibe la retroalimentación del Crítico (confirmación del
    veterinario) y actualiza los parámetros de la CNN 2 sin olvidar lo anterior.

    nuevos_casos_confirmados: lista de dicts con
        "animal_id"            str
        "recortes"             ndarray (4, 16, 16)   (recortes ROI de la CNN 1)
        "contexto"             ndarray (4,)          (vector de contexto, Paso 5a)
        "etiqueta_confirmada"  "sano" | "celo" | "enfermo"   (verdad del veterinario)

    Qué hace:
      1. Mide pérdida y aciertos del modelo actual sobre los casos confirmados.
      2. Fine-tuning de la cabeza totalmente conectada (Adam, lr bajo) sobre una
         mezcla de casos nuevos (sobremuestreados) y ejemplos viejos (replay)
         para evitar el olvido catastrófico.
      3. Ajusta el sesgo de calibración por clase (umbral blando): sube el de la
         clase con falsos negativos confirmados y baja el de las falsas alarmas.
      4. Guarda los casos en la memoria de replay, sube la versión del modelo y
         devuelve un reporte.
    """
    cnn2 = cnn2 or _SISTEMA["cnn2"]
    if cnn2 is None or cnn2.media_x is None:
        raise RuntimeError("No hay una CNN 2 entrenada disponible para reentrenar.")
    if not nuevos_casos_confirmados:
        raise ValueError("nuevos_casos_confirmados está vacío.")
    for caso in nuevos_casos_confirmados:
        for campo in ("animal_id", "recortes", "contexto", "etiqueta_confirmada"):
            if campo not in caso:
                raise KeyError(f"Caso sin el campo obligatorio '{campo}'")
        if caso["etiqueta_confirmada"] not in IDX_ESTADO:
            raise ValueError(f"Etiqueta inválida: {caso['etiqueta_confirmada']!r}")

    rng = np.random.default_rng(semilla)
    rec = np.stack([np.asarray(c["recortes"], dtype=np.float32) for c in nuevos_casos_confirmados])
    ctx = np.stack([np.asarray(c["contexto"], dtype=np.float32) for c in nuevos_casos_confirmados])
    y_nuevo = np.array([IDX_ESTADO[c["etiqueta_confirmada"]] for c in nuevos_casos_confirmados])
    x_nuevo = cnn2.construir_entrada(rec, ctx)

    # 1) Desempeño previo sobre los casos confirmados
    p_antes, _ = cnn2._forward(x_nuevo)
    pred_antes = p_antes.argmax(axis=1)
    perdida_antes = entropia_cruzada_ponderada(p_antes, y_nuevo, cnn2.pesos_clase)
    aciertos_antes = int((pred_antes == y_nuevo).sum())
    fn_graves = int(((y_nuevo > 0) & (pred_antes != y_nuevo) & (pred_antes == 0)).sum())
    falsas_alarmas = int(((y_nuevo == 0) & (pred_antes != 0)).sum())

    # 2) Fine-tuning con replay
    x_viejo, y_viejo = cnn2.memoria_replay
    n_viejo = int(min(len(y_viejo), max(32, razon_replay * len(y_nuevo) * repeticiones_caso_nuevo)))
    estado, t = cnn2._nuevo_estado_adam(), 0
    for _ in range(epocas):
        idx_v = rng.choice(len(y_viejo), size=n_viejo, replace=False)
        x_mix = np.concatenate([np.tile(x_nuevo, (repeticiones_caso_nuevo, 1)), x_viejo[idx_v]])
        y_mix = np.concatenate([np.tile(y_nuevo, repeticiones_caso_nuevo), y_viejo[idx_v]])
        t += 1
        cnn2._paso_optimizador(x_mix, y_mix, estado, lr, t, rng, 1e-3)

    # 3) Calibración por clase según la retroalimentación del Crítico: se usan los
    #    errores que el veterinario realmente vio (predicciones emitidas ANTES del
    #    fine-tuning), no las del modelo ya actualizado.
    for conf, pred in zip(y_nuevo, pred_antes):
        if conf != pred and conf > 0:          # falso negativo de celo/enfermo
            cnn2.sesgo_calibracion[conf] += 0.05
        elif conf == 0 and pred > 0:           # falsa alarma (menos costosa)
            cnn2.sesgo_calibracion[pred] -= 0.02
    cnn2.sesgo_calibracion = np.clip(cnn2.sesgo_calibracion, -0.5, 0.5)

    # 4) Resultado
    p_despues, _ = cnn2._forward(x_nuevo)
    pred_despues = p_despues.argmax(axis=1)
    perdida_despues = entropia_cruzada_ponderada(p_despues, y_nuevo, cnn2.pesos_clase)
    cnn2.memoria_replay = (np.concatenate([x_viejo, x_nuevo]), np.concatenate([y_viejo, y_nuevo]))
    cnn2.version += 1
    reporte = {
        "version_modelo": cnn2.version,
        "n_casos_nuevos": len(y_nuevo),
        "perdida_ponderada_antes": perdida_antes, "perdida_ponderada_despues": perdida_despues,
        "aciertos_antes": aciertos_antes, "aciertos_despues": int((pred_despues == y_nuevo).sum()),
        "falsos_negativos_graves_previos": fn_graves, "falsas_alarmas_previas": falsas_alarmas,
        "sesgo_calibracion": dict(zip(ESTADOS, np.round(cnn2.sesgo_calibracion, 3).tolist())),
    }
    cnn2.historial_reentrenamientos.append(reporte)
    return reporte


# =============================================================================
# DEMOSTRACIONES (usadas por main)
# =============================================================================
def _titulo(texto):
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78)


def demostrar_entropia_cruzada_ponderada():
    """Muestra por qué el mismo error de probabilidad cuesta distinto según la clase."""
    print("Pesos por clase real:", dict(zip(ESTADOS, PESOS_CLASE.tolist())))
    casos = [
        ("Falsa alarma: real=sano, el modelo da P(sano)=0.20", [0.20, 0.10, 0.70], 0),
        ("FN de celo:   real=celo, el modelo da P(celo)=0.20", [0.70, 0.20, 0.10], 1),
        ("FN de enfermo: real=enfermo, el modelo da P(enfermo)=0.20", [0.70, 0.10, 0.20], 2),
    ]
    print(f"  {'caso':<62}{'CE simple':>10}{'w * CE':>14}")
    for desc, p, y in casos:
        simple = entropia_cruzada_ponderada([p], [y], np.ones(3))
        # Con un solo ejemplo la normalización por sum(w) cancela el peso, por eso se
        # muestra la contribución sin normalizar w[y] * CE: es la que pesa en un lote.
        print(f"  {desc:<62}{simple:>10.4f}{PESOS_CLASE[y] * simple:>14.4f}")
    print("  (w * CE = w[clase real] * -log p[clase real]; en un lote la suma se divide entre sum(w).)")


def _imprimir_deteccion(salida, gt):
    print(f"    {'clase':<14}{'bbox [x1, y1, x2, y2]':<30}{'conf':>6}{'T prom °C':>11}{'grad °C/px':>12}{'IoU':>7}")
    for c in CLASES_ROI:
        d = salida["detecciones"][c]
        rt = d["recorte_termico"]
        bbox = "[" + ", ".join(f"{v:6.1f}" for v in d["bbox"]) + "]"
        print(f"    {c:<14}{bbox:<30}{d['confianza']:>6.2f}{rt['temp_promedio_c']:>11.2f}"
              f"{rt['gradiente_espacial_c_por_px']:>12.3f}{calcular_iou(d['bbox'], gt[c]):>7.2f}")


def ejecutar_escenario(nombre, descripcion, registro, cnn1, cnn2, estado_interno):
    """Recorre Paso 1 -> 2 -> 3 -> 4 -> 5a -> 6a para un animal y lo imprime."""
    print("\n" + "-" * 78)
    print(f"{nombre}")
    print(f"  {descripcion}")
    print("-" * 78)
    imagen_c, gt = capturar_imagen_termica(registro)                                           # Paso 1
    print(f"  [Paso 1] Captura simulada {imagen_c.shape} | T ambiente {registro['temp_amb']:.1f} °C | "
          f"humedad {registro['humedad']:.0f} %")
    img_norm = normalizar_termico(imagen_c, registro["temp_amb"], registro["humedad"])         # Paso 2
    print(f"  [Paso 2] Corrección por clima y normalización: rango [{img_norm.min():.3f}, {img_norm.max():.3f}]")
    salida = cnn1.detectar(img_norm)                                                           # Paso 3
    print("  [Paso 3] CNN 1 detector (cajas, recortes térmicos por región):")
    _imprimir_deteccion(salida, gt)
    est = identificar_animal(registro["rfid"], estado_interno)                                 # Paso 4
    print(f"  [Paso 4] RFID {registro['rfid']} -> {est['animal_id']} | base habitual "
          f"{est['temperatura_base_habitual']:.2f} °C | día de ciclo {est['dia_ciclo_reproductivo']} "
          f"| zona {est['zona_asignada']}")
    if est["historial_ojo_c"]:
        print(f"           Historial reciente de temperatura en ojo (°C): {est['historial_ojo_c']}")
    recortes, ctx = ensamblar_entrada_cnn2(salida, est, registro["temp_amb"], registro["humedad"])  # 5a
    ctx_txt = "[" + ", ".join(f"{float(v):.3f}" for v in ctx) + "]"
    print(f"  [Paso 5a] Firma térmica (4 recortes {recortes.shape[1:]}) + contexto {ctx_txt} "
          f"(base, fase del ciclo, T amb, humedad; normalizados)")
    visual = cnn2.extraer_vector_visual(recortes[None])
    print(f"  [Paso 6a] CNN 2: Conv+ReLU+Pool -> Flatten visual {visual.shape[1]} + contexto {ctx.shape[0]} "
          f"= concatenación {visual.shape[1] + ctx.shape[0]} -> FC -> Softmax")
    p = cnn2.inferir_un_animal(recortes, ctx)
    suma = float(p.sum())
    assert abs(suma - 1.0) < 1e-9, "El softmax debe sumar 1.0"
    print(f"           [P(sano), P(celo), P(enfermo)] = [{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}]   suma = {suma:.10f}")
    print(f"           Estado más probable según percepción: {ESTADOS[int(p.argmax())]} "
          f"(confianza {p.max():.1%}). La decisión de alertar la toma la Función de Utilidad (Pasos 7-8).")
    return {"probs": p, "recortes": recortes, "ctx": ctx, "salida": salida}


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()
    np.set_printoptions(precision=3, suppress=True)
    print("SISTEMA INTELIGENTE DE MONITOREO DE GANADO CON CNN")
    print("Inciso d) Módulo de Percepción -- prototipo de simulación (CNN 1 + CNN 2)")

    # ------------------------------------------------------------------ A
    _titulo("A. FUNCIÓN DE PÉRDIDA: ENTROPÍA CRUZADA PONDERADA (Paso 6a, entrenamiento)")
    demostrar_entropia_cruzada_ponderada()

    # ------------------------------------------------------------------ B
    _titulo("B. FASE FUERA DE LÍNEA: dataset simulado, Paso 2 y partición sin fuga")
    registros = generar_registros(n_animales=90, obs_por_animal=8, semilla=2024)
    part = particionar_por_animal(registros, semilla=7)
    print(f"  Registros totales: {len(registros)} ({len({r['animal_id'] for r in registros})} animales)")
    for nombre, regs in part.items():
        cuenta = Counter(r["estado"] for r in regs)
        print(f"  {nombre:<5}: {len({r['animal_id'] for r in regs}):>3} animales | {len(regs):>3} imágenes | "
              + ", ".join(f"{e}={cuenta.get(e, 0)}" for e in ESTADOS))
    print("  Verificación de fuga de datos (intersección de animal_id entre conjuntos):",
          "OK, ningún animal_id se repite entre train/val/test.")
    corrupto = {k: list(v) for k, v in part.items()}
    corrupto["test"] = corrupto["test"] + [part["train"][0]]          # contaminación a propósito
    try:
        verificar_sin_fuga_de_datos(corrupto)
    except AssertionError as e:
        print(f"  Prueba negativa (se contaminó 'test' a propósito) -> el control la detecta: {e}")

    reg = part["train"][0]
    img_c, _ = capturar_imagen_termica(reg)
    img_n = normalizar_termico(img_c, reg["temp_amb"], reg["humedad"])
    print(f"  Paso 2 sobre una imagen: T cruda [{img_c.min():.1f}, {img_c.max():.1f}] °C "
          f"(ambiente {reg['temp_amb']:.1f} °C, humedad {reg['humedad']:.0f} %) -> normalizada "
          f"[{img_n.min():.3f}, {img_n.max():.3f}]")

    # ------------------------------------------------------------------ C
    _titulo("C. CNN 1 -- DETECTORA DE OBJETOS (Paso 3): fine-tuning de la cabeza y evaluación")
    cnn1 = CNNDetectorObjetos()
    afinar = []
    gts_afinar = []
    for r in part["train"][:60]:
        ic, gt = capturar_imagen_termica(r)
        afinar.append(normalizar_termico(ic, r["temp_amb"], r["humedad"]))
        gts_afinar.append(gt)
    k = cnn1.afinar_cabeza(afinar, gts_afinar)
    print("  Cabeza afinada con 60 imágenes de train. Factores de regresión de caja (x, y):",
          {c: tuple(round(v, 2) for v in kk) for c, kk in k.items()})
    print("  Procesando Pasos 1-3 y 5a para train/val/test (puede tardar unos segundos)...")
    muestras, descartes = {}, {}
    for nombre in ("train", "val", "test"):
        muestras[nombre], descartes[nombre] = procesar_lote(part[nombre], cnn1)
    print("  Imágenes sin animal detectado:", descartes)
    ev1 = evaluar_cnn1(muestras["test"])
    print(f"  Evaluación CNN 1 en TEST ({len(muestras['test'])} imágenes):")
    print(f"    {'clase':<14}{'IoU medio':>10}{'AP@0.5':>9}")
    for c in CLASES_ROI:
        print(f"    {c:<14}{ev1['iou_medio'][c]:>10.3f}{ev1['ap50'][c]:>9.3f}")
    print(f"    mAP@0.5 = {ev1['map50']:.3f} | mAP@[0.5:0.95] = {ev1['map_50_95']:.3f}")

    # ------------------------------------------------------------------ D
    _titulo("D. CNN 2 -- CLASIFICADORA MULTIMODAL (Paso 6a): entrenamiento y evaluación")
    cnn2 = CNNClasificadoraMultimodal()
    _SISTEMA["cnn1"], _SISTEMA["cnn2"] = cnn1, cnn2
    rec_tr, ctx_tr, y_tr = apilar(muestras["train"])
    rec_va, ctx_va, y_va = apilar(muestras["val"])
    rec_te, ctx_te, y_te = apilar(muestras["test"])
    x_chk = cnn2.construir_entrada(rec_tr[:40], ctx_tr[:40], ajustar_escalador=True)
    err_grad = verificar_gradiente(cnn2, x_chk, y_tr[:40])
    print(f"  Prueba de sanidad del gradiente (analítico vs diferencias finitas): error relativo máx = {err_grad:.2e}")
    assert err_grad < 1e-4, "El gradiente analítico no coincide con el numérico"
    print(f"  Pesos de la pérdida [sano, celo, enfermo] = {PESOS_CLASE.tolist()} | Dropout = {cnn2.p_dropout}")
    info = cnn2.entrenar(rec_tr, ctx_tr, y_tr, rec_va, ctx_va, y_va)
    print(f"  Mejor época: {info['mejor_epoca']} | pérdida ponderada en validación: {info['mejor_perdida_val']:.4f}")
    probs_te = cnn2.predecir_proba(rec_te, ctx_te)
    assert np.allclose(probs_te.sum(axis=1), 1.0, atol=1e-9)
    pred_te = probs_te.argmax(axis=1)
    cm = matriz_confusion(y_te, pred_te)
    met = metricas_clasificacion(cm)
    print(f"  Evaluación CNN 2 en TEST ({len(y_te)} imágenes). Pérdida ponderada: "
          f"{entropia_cruzada_ponderada(probs_te, y_te):.4f}")
    print("  Matriz de confusión:")
    imprimir_matriz_confusion(cm)
    imprimir_metricas(met)
    fn_enf = int(cm[IDX_ESTADO['enfermo'], :].sum() - cm[IDX_ESTADO['enfermo'], IDX_ESTADO['enfermo']])
    print(f"  Prioridad del inciso a): falsos negativos de enfermo = {fn_enf}, "
          f"recall(enfermo) = {met['por_clase']['enfermo']['recall']:.3f}, "
          f"recall(celo) = {met['por_clase']['celo']['recall']:.3f}.")

    # ------------------------------------------------------------------ E
    _titulo("E. DEMOSTRACIÓN: 3 ESCENARIOS COMPLETOS (Pasos 1 -> 2 -> 3 -> 4 -> 5a -> 6a)")
    estado_interno = EstadoInternoGanado()
    nuevo_perfil = PERFIL_RANCHO_ORIGINAL
    offs0 = {"ojo": 0.0, "vulva": 0.0, "ubre": 0.0}
    geo = {"cx": 120.0, "cy": 108.0, "a": 71.0, "b": 37.0}

    def registro_escenario(animal_id, base, dia, amb, hum, deltas, semilla):
        return {"animal_id": animal_id, "rfid": f"RFID-{animal_id}", "estado": "demo", "base_c": base,
                "dia_ciclo": dia, "temp_amb": amb, "humedad": hum, "offsets": dict(offs0),
                "deltas": deltas, "geometria": dict(geo), "oclusion": None,
                "sesgo_sensor_c": nuevo_perfil["sesgo_sensor_c"], "semilla_ruido": semilla}

    # Caso 1: vaca sana
    r1 = registro_escenario("MX-1001", 38.5, 9, 24.0, 55.0,
                            {"cuerpo": 0.0, "ojo": 0.05, "vulva": 0.0, "ubre": -0.05}, 101)
    estado_interno.registrar(r1["rfid"], "MX-1001", 38.5, 9, "corral_A", [38.5, 38.6, 38.5])
    # Caso 2: celo (elevación localizada en zona vulvar + etapa fértil)
    r2 = registro_escenario("MX-1002", 38.6, 20, 22.0, 60.0,
                            {"cuerpo": 0.15, "ojo": 0.15, "vulva": 1.8, "ubre": 0.10}, 202)
    estado_interno.registrar(r2["rfid"], "MX-1002", 38.6, 20, "corral_B", [38.6, 38.6, 38.7])
    # Caso 3: posiblemente enferma (elevación sostenida en ojo y ubre)
    r3 = registro_escenario("MX-1003", 38.4, 8, 26.0, 50.0,
                            {"cuerpo": 0.7, "ojo": 1.7, "vulva": 0.1, "ubre": 1.9}, 303)
    estado_interno.registrar(r3["rfid"], "MX-1003", 38.4, 8, "corral_C", [39.7, 39.9, 40.1])

    res1 = ejecutar_escenario("CASO 1 -- VACA SANA",
                              "Firma térmica normal acorde a su línea base; día 9 del ciclo (fuera de ventana fértil).",
                              r1, cnn1, cnn2, estado_interno)
    res2 = ejecutar_escenario("CASO 2 -- VACA EN CELO",
                              "Elevación térmica localizada en zona_vulvar (+1.8 °C) + día 20 del ciclo (etapa fértil).",
                              r2, cnn1, cnn2, estado_interno)
    res3 = ejecutar_escenario("CASO 3 -- VACA POSIBLEMENTE ENFERMA",
                              "Elevación sostenida en ojo (+1.7 °C) y ubre (+1.9 °C) fuera de su línea base habitual.",
                              r3, cnn1, cnn2, estado_interno)

    # ------------------------------------------------------------------ F
    _titulo("F. PASOS 10 y 11: CONFIRMACIÓN DEL VETERINARIO Y REENTRENAMIENTO INCREMENTAL")
    print("  Se simula un corral nuevo (otra raza y cámara con deriva de -0.3 °C, celo y enfermedad menos marcados).")
    reg_nuevos = generar_registros(n_animales=30, obs_por_animal=6, semilla=777,
                                   perfil=PERFIL_RANCHO_NUEVO, prefijo="NUEVA")
    m_nuevas, _ = procesar_lote(reg_nuevos, cnn1)
    ids_nuevos = sorted({m["animal_id"] for m in m_nuevas})
    ids_feedback = set(ids_nuevos[:len(ids_nuevos) // 2])      # animales con confirmación vs animales no vistos
    m_fb = [m for m in m_nuevas if m["animal_id"] in ids_feedback]
    m_eval = [m for m in m_nuevas if m["animal_id"] not in ids_feedback]
    print(f"  Casos confirmados por el veterinario (retroalimentación): {len(m_fb)} | "
          f"casos de evaluación del mismo corral (otros animales): {len(m_eval)}")

    def desempeno(muestras_, etiqueta):
        rec, ctx, y = apilar(muestras_)
        p = cnn2.predecir_proba(rec, ctx)
        m = metricas_clasificacion(matriz_confusion(y, p.argmax(axis=1)))
        print(f"    {etiqueta:<42} pérdida pond. {entropia_cruzada_ponderada(p, y):.3f} | "
              f"exactitud {m['exactitud']:.3f} | recall celo {m['por_clase']['celo']['recall']:.2f} | "
              f"recall enfermo {m['por_clase']['enfermo']['recall']:.2f}")

    print("  ANTES del reentrenamiento (modelo v%d):" % cnn2.version)
    desempeno(m_eval, "corral nuevo, animales no vistos")
    desempeno(muestras["test"], "TEST original")
    casos = [{"animal_id": m["animal_id"], "recortes": m["recortes"], "contexto": m["ctx"],
              "etiqueta_confirmada": ESTADOS[m["y"]]} for m in m_fb]
    rep = reentrenar_cnn_incremental(casos)
    print(f"  reentrenar_cnn_incremental(): {rep['n_casos_nuevos']} casos -> modelo v{rep['version_modelo']}")
    print(f"    En los casos confirmados: pérdida ponderada {rep['perdida_ponderada_antes']:.3f} -> "
          f"{rep['perdida_ponderada_despues']:.3f} | aciertos {rep['aciertos_antes']} -> {rep['aciertos_despues']} "
          f"de {rep['n_casos_nuevos']}")
    print(f"    FN graves previos (real celo/enfermo predicho sano): {rep['falsos_negativos_graves_previos']} | "
          f"falsas alarmas previas: {rep['falsas_alarmas_previas']}")
    print(f"    Sesgo de calibración por clase: {rep['sesgo_calibracion']}")
    print(f"  DESPUÉS del reentrenamiento (modelo v{cnn2.version}):")
    desempeno(m_eval, "corral nuevo, animales no vistos")
    desempeno(muestras["test"], "TEST original (control de olvido)")

    # ------------------------------------------------------------------ G
    _titulo("RESUMEN")
    for nombre, res, esperado in (("Caso 1 (sana)", res1, "sano"), ("Caso 2 (celo)", res2, "celo"),
                                  ("Caso 3 (enferma)", res3, "enfermo")):
        print(f"  {nombre:<17} -> P = {np.round(res['probs'], 3).tolist()} (suma {res['probs'].sum():.1f}) "
              f"| esperado: {esperado}")
    print(f"  CNN 1: mAP@0.5 = {ev1['map50']:.3f} | CNN 2: F1 macro = {met['f1_macro']:.3f}")
    print("  Recordatorio: métricas medidas sobre datos SIMULADOS; este módulo solo percibe, no decide alertas.")
    print(f"  Tiempo total: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
