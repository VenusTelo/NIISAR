"""Parámetros NISAR GCOV para dos categorías: agua y bosque.

Solo depende de numpy, para poder probarse sin descargar datos.

IMPORTANTE: los umbrales de abajo son valores iniciales razonables, NO valores
validados para Honduras. Se muestran en el modo técnico y deben calibrarse
contra sitios conocidos (p. ej. el embalse El Cajón para agua).
"""

import numpy as np

# ---------------------------------------------------------------- umbrales
WATER_HV_DB = -22.0         # agua abierta y lisa: HV muy bajo
WATER_HH_DB = -18.0         # ...y HH bajo (calibración inicial Honduras)
VEG_HV_DB = -20.0           # por encima: hay estructura vegetal
DENSE_HV_DB = -12.5         # por encima: vegetación densa tropical (bosque probable)
LOSS_DB = -3.0              # caída de HV: posible pérdida de cobertura
DISTURB_DB = -1.5           # caída menor: perturbación leve
RISE_DB = 3.0               # subida de HV
FLOOD_VEG_HH_RISE_DB = 3.0  # subida de HH (rebote doble) en vegetación

# Fiabilidad: z = |cambio| / ruido esperado. Corte entre baja/media/alta.
Z_MEDIUM = 2.0
Z_HIGH = 4.0
GAP_CAP_MEDIUM_DAYS = 24    # más de esto: la fiabilidad no pasa de "media"
GAP_CAP_LOW_DAYS = 60       # más de esto: se queda en "baja"

NODATA = 255
LOW, MEDIUM, HIGH = 0, 1, 2
CONF_LABELS = {LOW: "low", MEDIUM: "medium", HIGH: "high"}
CONF_ALPHA = {LOW: 80, MEDIUM: 170, HIGH: 255}

# ------------------------------------------------------------ utilidades
def to_db(linear):
    """Potencia lineal -> dB. Valores <= 0 o no finitos dan NaN."""
    lin = np.asarray(linear, dtype="float64")
    out = np.full(lin.shape, np.nan, dtype="float32")
    ok = np.isfinite(lin) & (lin > 0)
    out[ok] = 10.0 * np.log10(lin[ok])
    return out


def rvi(hh_lin, hv_lin):
    """Índice de vegetación radar (forma dual-pol HH/HV): 4·HV / (HH + HV).

    Existen varias definiciones; esta es la usada aquí. Rango aprox. 0 a 2.
    """
    hh, hv = np.asarray(hh_lin, "float64"), np.asarray(hv_lin, "float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 4.0 * hv / (hh + hv)
    out[~np.isfinite(out)] = np.nan
    return out.astype("float32")


def rfdi(hh_lin, hv_lin):
    """Índice de degradación forestal: (HH - HV) / (HH + HV). Rango -1 a 1."""
    hh, hv = np.asarray(hh_lin, "float64"), np.asarray(hv_lin, "float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (hh - hv) / (hh + hv)
    out[~np.isfinite(out)] = np.nan
    return out.astype("float32")


def noise_db(looks):
    """Desviación aproximada (dB) del ruido de moteado con N looks: 4.34/√N.

    Sin la capa de looks se asume 1 (lo más conservador). No se multiplican los
    looks por el promediado de bloques, porque los píxeles vecinos pueden estar
    correlacionados.
    """
    if looks is None:
        return np.float32(4.34)
    n = np.asarray(looks, dtype="float64")
    n = np.where(np.isfinite(n) & (n >= 1), n, 1.0)
    return (4.34 / np.sqrt(n)).astype("float32")


def z_change(delta_db, looks_a=None, looks_b=None):
    """Tamaño del cambio frente al ruido esperado de las dos fechas."""
    sigma = np.sqrt(noise_db(looks_a) ** 2 + noise_db(looks_b) ** 2)
    return np.abs(delta_db) / sigma


def confidence_from_z(z, gap_days=None):
    """z -> 0 baja / 1 media / 2 alta, limitada por el tiempo entre fechas."""
    z = np.asarray(z, dtype="float64")
    conf = np.full(z.shape, LOW, dtype="uint8")
    conf[z >= Z_MEDIUM] = MEDIUM
    conf[z >= Z_HIGH] = HIGH
    if gap_days is not None:
        if gap_days > GAP_CAP_LOW_DAYS:
            conf = np.minimum(conf, LOW)
        elif gap_days > GAP_CAP_MEDIUM_DAYS:
            conf = np.minimum(conf, MEDIUM)
    return conf


def water_mask(hh_db, hv_db):
    """True donde HV y HH están por debajo de los umbrales de agua."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(hh_db) & np.isfinite(hv_db) & \
            (hv_db < WATER_HV_DB) & (hh_db < WATER_HH_DB)


# --------------------------------------------------------- categoría AGUA
WATER_COMPARE = {
    0: ("Unchanged land", "#D9D9D9", False),
    1: ("Stable water", "#56B4E9", False),
    2: ("New water (possible flooding)", "#0072B2", True),
    3: ("Water that disappeared", "#E69F00", True),
    4: ("Possible flooded vegetation (candidate)", "#CC79A7", True),
}
WATER_SINGLE = {
    0: ("Land", "#D9D9D9", False),
    1: ("Water", "#0072B2", False),
}


def water_change(hh1, hv1, hh2, hv2, looks1=None, looks2=None, gap_days=None):
    """Cambio de agua entre dos fechas (arrays en dB). Devuelve clases y confianza."""
    valid = np.isfinite(hh1) & np.isfinite(hv1) & np.isfinite(hh2) & np.isfinite(hv2)
    w1, w2 = water_mask(hh1, hv1), water_mask(hh2, hv2)
    d_hh, d_hv = hh2 - hh1, hv2 - hv1

    flood_veg = (~w1 & ~w2 & valid & (hv1 >= VEG_HV_DB)
                 & (d_hh >= FLOOD_VEG_HH_RISE_DB)
                 & ((d_hh - d_hv) >= FLOOD_VEG_HH_RISE_DB))

    classes = np.zeros(hv1.shape, dtype="uint8")
    classes[w1 & w2] = 1
    classes[~w1 & w2] = 2
    classes[w1 & ~w2] = 3
    classes[flood_veg] = 4
    classes[~valid] = NODATA

    z_hv, z_hh = z_change(d_hv, looks1, looks2), z_change(d_hh, looks1, looks2)
    z = np.where(classes == 4, z_hh, np.minimum(z_hv, z_hh))
    conf = confidence_from_z(z, gap_days)
    changed = np.isin(classes, (2, 3, 4))
    conf = np.where(changed, conf, NODATA).astype("uint8")
    conf[classes == NODATA] = NODATA
    return classes, conf


def water_state(hh_db, hv_db, looks=None):
    """Agua en una sola fecha. La confianza mide la distancia al umbral."""
    valid = np.isfinite(hh_db) & np.isfinite(hv_db)
    classes = np.where(water_mask(hh_db, hv_db), 1, 0).astype("uint8")
    classes[~valid] = NODATA
    with np.errstate(invalid="ignore"):
        margin = np.minimum(np.abs(hv_db - WATER_HV_DB), np.abs(hh_db - WATER_HH_DB))
    conf = confidence_from_z(np.nan_to_num(margin / noise_db(looks)))
    conf[classes == NODATA] = NODATA
    return classes, conf


# ------------------------------------------------------- categoría BOSQUE
FOREST_COMPARE = {
    0: ("Dense vegetation, unchanged", "#B7E1CD", False),
    1: ("Possible canopy loss", "#D55E00", True),
    2: ("Slight disturbance (cause unidentified)", "#F0E442", True),
    3: ("Signal increase (possible regrowth or moisture)", "#0072B2", True),
    4: ("Other surfaces, unchanged", "#D9D9D9", False),
    5: ("Water (excluded)", "#BFD9EA", False),
}
FOREST_SINGLE = {
    0: ("Very low (water, smooth soil)", "#F7F7F7", False),
    1: ("Low (pasture, crops, soil)", "#E5C494", False),
    2: ("Dense vegetation", "#66C2A4", False),
    3: ("Very dense vegetation", "#006D2C", False),
}
FOREST_BIN_EDGES = (-22.0, -16.5, DENSE_HV_DB)


def forest_change(hh1, hv1, hh2, hv2, looks1=None, looks2=None, gap_days=None):
    """Cambio de estructura de la vegetación (HV) entre dos fechas."""
    valid = np.isfinite(hv1) & np.isfinite(hv2)
    w1, w2 = water_mask(hh1, hv1), water_mask(hh2, hv2)
    d_hv = hv2 - hv1
    dense1 = hv1 >= DENSE_HV_DB

    classes = np.full(hv1.shape, 4, dtype="uint8")
    classes[dense1 & (d_hv > DISTURB_DB)] = 0
    classes[dense1 & (d_hv <= DISTURB_DB) & (d_hv > LOSS_DB)] = 2
    classes[dense1 & (d_hv <= LOSS_DB)] = 1
    classes[(hv1 >= VEG_HV_DB) & (hv2 >= VEG_HV_DB) & (d_hv >= RISE_DB)] = 3
    classes[w1 | w2] = 5          # agua: no es deforestación
    classes[~valid] = NODATA

    conf = confidence_from_z(z_change(d_hv, looks1, looks2), gap_days)
    changed = np.isin(classes, (1, 2, 3))
    conf = np.where(changed, conf, NODATA).astype("uint8")
    conf[classes == NODATA] = NODATA
    return classes, conf


def forest_state(hv_db, looks=None):
    """Estructura de vegetación en una fecha: 4 niveles de HV."""
    valid = np.isfinite(hv_db)
    classes = np.digitize(np.nan_to_num(hv_db, nan=-99.0), FOREST_BIN_EDGES).astype("uint8")
    classes[~valid] = NODATA
    edges = np.array(FOREST_BIN_EDGES)
    with np.errstate(invalid="ignore"):
        margin = np.min(np.abs(hv_db[..., None] - edges), axis=-1)
    conf = confidence_from_z(np.nan_to_num(margin / noise_db(looks)))
    conf[classes == NODATA] = NODATA
    return classes, conf


# ------------------------------------------------------------ resúmenes
def class_stats(classes, conf, legend, pixel_area_km2=None):
    """Porcentaje (y km² si se conoce) por clase y fiabilidad de los cambios."""
    valid = classes != NODATA
    total = int(valid.sum())
    out = {"total_pixels": total, "classes": [], "confidence": None}
    for code, (label, color, is_change) in legend.items():
        n = int((classes == code).sum())
        row = {"code": code, "label": label, "color": color, "is_change": is_change,
               "percent": (n / total * 100.0) if total else 0.0, "pixels": n}
        if pixel_area_km2:
            row["km2"] = n * pixel_area_km2
        out["classes"].append(row)
    changed = np.isin(classes, [c for c, v in legend.items() if v[2]])
    n_changed = int(changed.sum())
    if n_changed:
        out["confidence"] = {
            CONF_LABELS[k]: float((conf[changed] == k).sum() / n_changed * 100.0)
            for k in (HIGH, MEDIUM, LOW)
        }
    return out


_ALPHA_LUT = np.full(256, CONF_ALPHA[LOW], dtype="uint8")
for _k, _a in CONF_ALPHA.items():
    _ALPHA_LUT[_k] = _a


def classes_to_rgba(classes, conf, legend, context_alpha=255):
    """RGBA uint8: color por clase; los cambios se vuelven tenues si son poco fiables."""
    h, w = classes.shape
    rgba = np.zeros((h, w, 4), dtype="uint8")
    for code, (_, color, is_change) in legend.items():
        sel = classes == code
        if not sel.any():
            continue
        rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
        rgba[sel, :3] = rgb
        if is_change:
            rgba[sel, 3] = _ALPHA_LUT[conf[sel]]
        else:
            rgba[sel, 3] = context_alpha
    return rgba


def _pct(stats, code):
    for row in stats["classes"]:
        if row["code"] == code:
            return row["percent"]
    return 0.0


def _conf_sentence(stats):
    c = stats.get("confidence")
    if not c:
        return ""
    return (f" Reliability of detections: high {c['high']:.0f}%, "
            f"medium {c['medium']:.0f}%, low {c['low']:.0f}%.")


def describe(category, mode, stats, gap_days=None):
    """Texto en lenguaje sencillo, siempre con su advertencia de fiabilidad."""
    lines = []
    if category == "water":
        if mode == "compare":
            new, lost = _pct(stats, 2), _pct(stats, 3)
            veg = _pct(stats, 4)
            lines.append(f"New water: {new:.1f}% of the area. Water that disappeared: {lost:.1f}%.")
            if veg > 0:
                lines.append(f"Possible flooded vegetation: {veg:.1f}% (indirect signal, needs confirmation).")
            if new == lost == veg == 0:
                lines.append("No water changes were detected in this comparison.")
            lines.append(_conf_sentence(stats).strip())
        else:
            lines.append(f"Water covers {_pct(stats, 1):.1f}% of the area on this date.")
        lines.append("Water is detected by its low backscatter; the threshold is preliminary and not calibrated for Honduras.")
    else:
        if mode == "compare":
            loss, dist, rise = _pct(stats, 1), _pct(stats, 2), _pct(stats, 3)
            lines.append(f"Possible canopy loss: {loss:.1f}% of the area. Slight disturbance: {dist:.1f}%. Signal increase: {rise:.1f}%.")
            lines.append(_conf_sentence(stats).strip())
            lines.append("A drop in signal can come from logging, but also from rain or moisture; it does not prove deforestation on its own. "
                         "NISAR does not identify fires: cross-check with active fire data (NASA FIRMS).")
        else:
            lines.append(f"Dense or very dense vegetation: {_pct(stats, 2) + _pct(stats, 3):.1f}% of the area on this date.")
            lines.append("The 4 classes come from preliminary HV ranges, not calibrated.")
    if gap_days is not None and gap_days > GAP_CAP_MEDIUM_DAYS:
        lines.append(f"The dates are {gap_days} days apart: reliability is capped at medium or low because of seasonal effects.")
    return [t for t in lines if t]


# ------------------------------------------------------- orquestador
def build_categories(layers, mode, gap_days=None, pixel_area_km2=None):
    """Calcula las dos categorías.

    layers: dict con 'hh' y 'hv' (lineal, fecha 1), opcional 'looks', y en modo
    compare los mismos con sufijo '2'.  Devuelve arrays y metadatos; el
    llamador los convierte en imágenes.
    """
    hh1, hv1 = to_db(layers["hh"]), to_db(layers["hv"])
    l1 = layers.get("looks")
    result = {}
    if mode == "compare":
        hh2, hv2 = to_db(layers["hh2"]), to_db(layers["hv2"])
        l2 = layers.get("looks2")
        w_cls, w_conf = water_change(hh1, hv1, hh2, hv2, l1, l2, gap_days)
        f_cls, f_conf = forest_change(hh1, hv1, hh2, hv2, l1, l2, gap_days)
        w_leg, f_leg = WATER_COMPARE, FOREST_COMPARE
        tech = {
            "water": [
                ("HH date 1 (dB)", hh1, -25, 0, "gray"),
                ("HH date 2 (dB)", hh2, -25, 0, "gray"),
                ("HV date 1 (dB)", hv1, -30, -10, "gray"),
                ("HV date 2 (dB)", hv2, -30, -10, "gray"),
                ("HH change (dB)", hh2 - hh1, -8, 8, "RdBu"),
                ("HV change (dB)", hv2 - hv1, -8, 8, "RdBu"),
            ],
            "forest": [
                ("HV date 1 (dB)", hv1, -30, -10, "gray"),
                ("HV date 2 (dB)", hv2, -30, -10, "gray"),
                ("HV change (dB)", hv2 - hv1, -5, 5, "RdBu"),
                ("RVI date 1", rvi(layers["hh"], layers["hv"]), 0, 2, "YlGn"),
                ("RVI date 2", rvi(layers["hh2"], layers["hv2"]), 0, 2, "YlGn"),
                ("RFDI date 2", rfdi(layers["hh2"], layers["hv2"]), -1, 1, "viridis"),
            ],
        }
    else:
        w_cls, w_conf = water_state(hh1, hv1, l1)
        f_cls, f_conf = forest_state(hv1, l1)
        w_leg, f_leg = WATER_SINGLE, FOREST_SINGLE
        tech = {
            "water": [
                ("HH (dB)", hh1, -25, 0, "gray"),
                ("HV (dB)", hv1, -30, -10, "gray"),
                ("HH − HV (dB)", hh1 - hv1, 0, 15, "magma"),
            ],
            "forest": [
                ("HV (dB)", hv1, -30, -10, "gray"),
                ("RVI", rvi(layers["hh"], layers["hv"]), 0, 2, "YlGn"),
                ("RFDI", rfdi(layers["hh"], layers["hv"]), -1, 1, "viridis"),
            ],
        }
    if l1 is not None:
        looks_layer = ("Number of looks (more = less noise)", np.asarray(l1, "float32"), 0, None, "cividis")
        tech["forest"].append(looks_layer)

    var_tables = {
        "water": [
            {"param": "HV polarization (dB)", "role": "Detects smooth open-water surfaces (specular reflection)", "threshold": f"< {WATER_HV_DB} dB"},
            {"param": "HH polarization (dB)", "role": "Confirms water bodies and detects double-bounce in flooded wetlands", "threshold": f"< {WATER_HH_DB} dB (o ΔHH ≥ +{FLOOD_VEG_HH_RISE_DB} dB)"},
            {"param": "Number of looks (N)", "role": "Controls speckle noise and sets the statistical reliability", "threshold": f"z ≥ {Z_MEDIUM} (medium), z ≥ {Z_HIGH} (high)"},
        ],
        "forest": [
            {"param": "Cross-polarization HV (dB)", "role": "Measures canopy volume scattering and tree biomass", "threshold": f"≥ {DENSE_HV_DB} dB (dense forest)"},
            {"param": "Temporal change ΔHV (dB)", "role": "Identifies canopy loss or slight disturbance", "threshold": f"≤ {LOSS_DB} dB (loss), ≤ {DISTURB_DB} dB (disturbance)"},
            {"param": "RVI and RFDI indices", "role": "Assess vegetation vigor (RVI) and structural degradation (RFDI)", "threshold": "RVI [0–2], RFDI [-1–1]"},
        ],
    }

    for name, cls, conf, legend in (("water", w_cls, w_conf, w_leg), ("forest", f_cls, f_conf, f_leg)):
        stats = class_stats(cls, conf, legend, pixel_area_km2)
        result[name] = {
            "classes": cls, "confidence": conf, "legend": legend, "stats": stats,
            "variables": var_tables[name],
            "summary": describe(name, mode, stats, gap_days),
            "technical": tech[name],
        }
    return result
