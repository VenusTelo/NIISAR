import os
import math
from datetime import datetime
import streamlit as st
import asf_search as asf
import earthaccess
import h5py
import numpy as np
import pandas as pd
import pydeck as pdk

st.set_page_config(page_title="Monitor NISAR - Prototipo Alpha Multi-Mapa", layout="wide")
st.title("Monitor NISAR — Inspección .h5 y Detección de Cambios")

LUGARES_PREDEFINIDOS = {
    "Gracias a Dios / Laguna de Caratasca (Trayecto 4, Marco 9)": (15.1500, -83.9000, 4, 9),
    "Roatán / Islas de la Bahía (Trayecto 105, Marco 10)": (16.3200, -86.5300, 105, 10),
}

MAPAS_NISAR = {
    "GCOV — Nivel 2 (Retrodispersión / Agua y Vegetación)": "GCOV",
    "GUNW — Nivel 2 (Interferometría / Fase y Deformación)": "GUNW",
    "SME2 — Nivel 3 (Humedad del suelo en m³/m³)": "SME2",
    "GSLC — Nivel 2 (Alta resolución compleja geocodificada)": "GSLC",
    "RSLC — Nivel 1 (Alta resolución en geometría radar)": "RSLC",
}

if "lat_punto" not in st.session_state:
    st.session_state.lat_punto = 15.1500
if "lon_punto" not in st.session_state:
    st.session_state.lon_punto = -83.9000
if "trayecto_ui" not in st.session_state:
    st.session_state.trayecto_ui = 4
if "marco_ui" not in st.session_state:
    st.session_state.marco_ui = 9


def latlon_a_proyeccion(lat_deg: float, lon_deg: float, epsg: int):
    if epsg == 4326:
        return lon_deg, lat_deg

    if epsg == 6933:
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = f * (2 - f)
        e = math.sqrt(e2)
        phi1 = math.radians(30.0)
        k0 = math.cos(phi1) / math.sqrt(1.0 - e2 * math.sin(phi1) ** 2)
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        sin_p = math.sin(phi)
        q = (1.0 - e2) * (
            sin_p / (1.0 - e2 * sin_p**2)
            - (1.0 / (2.0 * e)) * math.log((1.0 - e * sin_p) / (1.0 + e * sin_p))
        )
        return a * k0 * lam, (a * q) / (2.0 * k0)

    zona = epsg - 32600 if epsg < 32700 else epsg - 32700
    norte = epsg < 32700
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    phi = math.radians(lat_deg)
    lam = math.radians(lon_deg)
    lam0 = math.radians((zona - 1) * 6 - 180 + 3)
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T = math.tan(phi) ** 2
    C = ep2 * math.cos(phi) ** 2
    A = math.cos(phi) * (lam - lam0)
    M = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
        - (35 * e2**3 / 3072) * math.sin(6 * phi)
    )
    x = 500000.0 + k0 * N * (
        A + (1 - T + C) * A**3 / 6 + (5 - 18 * T + T**2 + 72 * C - 58 * ep2) * A**5 / 120
    )
    y = k0 * (
        M
        + N
        * math.tan(phi)
        * (
            A**2 / 2
            + (5 - T + 9 * C + 4 * C**2) * A**4 / 24
            + (61 - 58 * T + T**2 + 600 * C - 330 * ep2) * A**6 / 720
        )
    )
    if not norte:
        y += 10000000.0
    return x, y


def proyeccion_a_latlon(x: float, y: float, epsg: int):
    if epsg == 4326:
        return x, y

    if epsg == 6933:
        a = 6378137.0
        f = 1 / 298.257223563
        e2 = f * (2 - f)
        e = math.sqrt(e2)
        phi1 = math.radians(30.0)
        k0 = math.cos(phi1) / math.sqrt(1.0 - e2 * math.sin(phi1) ** 2)
        q_polar = (1.0 - e2) * (
            1.0 / (1.0 - e2) - (1.0 / (2.0 * e)) * math.log((1.0 - e) / (1.0 + e))
        )
        q = (2.0 * k0 * y) / a
        beta = math.asin(max(-1.0, min(1.0, q / q_polar)))
        lat_rad = (
            beta
            + (e2 / 3.0 + 31.0 * e2**2 / 180.0 + 517.0 * e2**3 / 5040.0) * math.sin(2.0 * beta)
            + (23.0 * e2**2 / 360.0 + 251.0 * e2**3 / 3780.0) * math.sin(4.0 * beta)
            + (761.0 * e2**3 / 45360.0) * math.sin(6.0 * beta)
        )
        lon_rad = x / (a * k0)
        return math.degrees(lon_rad), math.degrees(lat_rad)

    zona = epsg - 32600 if epsg < 32700 else epsg - 32700
    norte = epsg < 32700
    a = 6378137.0
    f = 1 / 298.257223563
    k0 = 0.9996
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    lam0 = math.radians((zona - 1) * 6 - 180 + 3)

    x_adj = x - 500000.0
    y_adj = y if norte else y - 10000000.0
    M = y_adj / k0
    mu = M / (a * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = (
        mu
        + (3 * e1 / 2 - 27 * e1**3 / 32) * math.sin(2 * mu)
        + (21 * e1**2 / 16 - 55 * e1**4 / 32) * math.sin(4 * mu)
        + (151 * e1**3 / 96) * math.sin(6 * mu)
        + (1097 * e1**4 / 512) * math.sin(8 * mu)
    )
    N1 = a / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    T1 = math.tan(phi1) ** 2
    C1 = ep2 * math.cos(phi1) ** 2
    R1 = a * (1 - e2) / ((1 - e2 * math.sin(phi1) ** 2) ** 1.5)
    D = x_adj / (N1 * k0)

    lat_rad = phi1 - (N1 * math.tan(phi1) / R1) * (
        D**2 / 2
        - (5 + 3 * T1 + 10 * C1 - 4 * C1**2 - 9 * ep2) * D**4 / 24
        + (61 + 90 * T1 + 298 * C1 + 45 * T1**2 - 252 * ep2 - 3 * C1**2) * D**6 / 720
    )
    lon_rad = lam0 + (
        D
        - (1 + 2 * T1 + C1) * D**3 / 6
        + (5 - 2 * C1 + 28 * T1 - 3 * C1**2 + 8 * ep2 + 24 * T1**2) * D**5 / 120
    ) / math.cos(phi1)
    return math.degrees(lon_rad), math.degrees(lat_rad)


def renderizar_capa(d_co: np.ndarray, d_cr: np.ndarray, modo_unidad: str, es_amplitud_slc: bool = False):
    h, w = d_co.shape
    rgb = np.full((h, w, 3), 25, dtype=np.uint8)

    if modo_unidad == "dB":
        valid = np.isfinite(d_co) & (d_co > 0)
        factor_db = 20.0 if es_amplitud_slc else 10.0
        vals = np.where(valid, factor_db * np.log10(np.maximum(d_co, 1e-6)), np.nan)
        if not np.any(valid):
            return vals, rgb, valid
        p2, p98 = np.nanpercentile(vals, (2, 98))
        r = np.nan_to_num(np.clip((vals - p2) / (p98 - p2 + 1e-6) * 255, 0, 255), nan=25).astype(np.uint8)
        if d_cr is not None:
            v_cr = np.isfinite(d_cr) & (d_cr > 0)
            db_cr = np.where(v_cr, factor_db * np.log10(np.maximum(d_cr, 1e-6)), np.nan)
            if np.any(v_cr):
                q2, q98 = np.nanpercentile(db_cr, (2, 98))
                g = np.nan_to_num(np.clip((db_cr - q2) / (q98 - q2 + 1e-6) * 255, 0, 255), nan=25).astype(np.uint8)
            else:
                g = r
        else:
            g = r
        rgb = np.dstack([r, g, r])
        return vals, rgb, valid

    if modo_unidad == "rad":
        valid = np.isfinite(d_co) & (np.abs(d_co) < 500.0) & (d_co != 0.0)
        vals = np.where(valid, d_co, np.nan)
        if not np.any(valid):
            return vals, rgb, valid
        lim = max(float(np.nanpercentile(np.abs(vals), 95)), 0.5)
        norm = np.clip(vals / lim, -1.0, 1.0)

        neg = valid & (norm < 0)
        t_neg = np.abs(norm[neg])
        rgb[neg, 0] = (242 * (1 - t_neg) + 25 * t_neg).astype(np.uint8)
        rgb[neg, 1] = (242 * (1 - t_neg) + 65 * t_neg).astype(np.uint8)
        rgb[neg, 2] = (248 * (1 - t_neg) + 175 * t_neg).astype(np.uint8)

        pos = valid & (norm >= 0)
        t_pos = norm[pos]
        rgb[pos, 0] = (248 * (1 - t_pos) + 150 * t_pos).astype(np.uint8)
        rgb[pos, 1] = (240 * (1 - t_pos) + 35 * t_pos).astype(np.uint8)
        rgb[pos, 2] = (235 * (1 - t_pos) + 25 * t_pos).astype(np.uint8)
        return vals, rgb, valid

    # Humedad del suelo SME2 (m3/m3): paleta científica YlGnBu (0.0 a 0.6 m³/m³)
    valid = np.isfinite(d_co) & (d_co >= 0.0) & (d_co <= 1.0)
    vals = np.where(valid, d_co, np.nan)
    if not np.any(valid):
        return vals, rgb, valid
    stops = [0.0, 0.15, 0.30, 0.45, 0.60]
    v_ok = vals[valid]
    rgb[valid, 0] = np.interp(v_ok, stops, [255, 199, 65, 34, 8]).astype(np.uint8)
    rgb[valid, 1] = np.interp(v_ok, stops, [255, 233, 182, 94, 29]).astype(np.uint8)
    rgb[valid, 2] = np.interp(v_ok, stops, [217, 180, 196, 168, 88]).astype(np.uint8)
    return vals, rgb, valid


def colorear_diferencia(delta: np.ndarray, rango_max: float):
    h, w = delta.shape
    rgb = np.full((h, w, 3), 25, dtype=np.uint8)
    valid = np.isfinite(delta)
    norm = np.clip(delta / max(rango_max, 1e-6), -1.0, 1.0)

    neg = valid & (norm < 0)
    intensidad_neg = np.abs(norm[neg])
    rgb[neg, 0] = (240 * (1 - intensidad_neg) + 20 * intensidad_neg).astype(np.uint8)
    rgb[neg, 1] = (240 * (1 - intensidad_neg) + 100 * intensidad_neg).astype(np.uint8)
    rgb[neg, 2] = 250

    pos = valid & (norm >= 0)
    intensidad_pos = norm[pos]
    rgb[pos, 0] = 250
    rgb[pos, 1] = (240 * (1 - intensidad_pos) + 55 * intensidad_pos).astype(np.uint8)
    rgb[pos, 2] = (240 * (1 - intensidad_pos) + 30 * intensidad_pos).astype(np.uint8)
    return rgb


def alinear_mallas(res1: dict, res2: dict):
    if res1["geocodificado"] and res2["geocodificado"] and res1["epsg"] == res2["epsg"]:
        xs1, ys1 = res1["xs"], res1["ys"]
        xs2, ys2 = res2["xs"], res2["ys"]
        x_min = max(float(xs1.min()), float(xs2.min()))
        x_max = min(float(xs1.max()), float(xs2.max()))
        y_min = max(float(ys1.min()), float(ys2.min()))
        y_max = min(float(ys1.max()), float(ys2.max()))

        ix1 = np.where((xs1 >= x_min) & (xs1 <= x_max))[0]
        iy1 = np.where((ys1 >= y_min) & (ys1 <= y_max))[0]
        if ix1.size > 4 and iy1.size > 4:
            xs_c = xs1[ix1]
            ys_c = ys1[iy1]
            ix2 = (
                np.clip(np.searchsorted(xs2, xs_c), 0, len(xs2) - 1)
                if xs2[0] <= xs2[-1]
                else len(xs2) - 1 - np.clip(np.searchsorted(xs2[::-1], xs_c), 0, len(xs2) - 1)
            )
            iy2 = (
                np.clip(np.searchsorted(ys2, ys_c), 0, len(ys2) - 1)
                if ys2[0] <= ys2[-1]
                else len(ys2) - 1 - np.clip(np.searchsorted(ys2[::-1], ys_c), 0, len(ys2) - 1)
            )
            d1_co = res1["datos_co"][np.ix_(iy1, ix1)]
            d2_co = res2["datos_co"][np.ix_(iy2, ix2)]
            d1_cr = res1["datos_cr"][np.ix_(iy1, ix1)] if res1["datos_cr"] is not None else None
            d2_cr = res2["datos_cr"][np.ix_(iy2, ix2)] if res2["datos_cr"] is not None else None
            d1_ex = res1["datos_extra"][np.ix_(iy1, ix1)] if res1["datos_extra"] is not None else None
            d2_ex = res2["datos_extra"][np.ix_(iy2, ix2)] if res2["datos_extra"] is not None else None
            return d1_co, d2_co, d1_cr, d2_cr, d1_ex, d2_ex, xs_c, ys_c

    mh = min(res1["datos_co"].shape[0], res2["datos_co"].shape[0])
    mw = min(res1["datos_co"].shape[1], res2["datos_co"].shape[1])
    return (
        res1["datos_co"][:mh, :mw],
        res2["datos_co"][:mh, :mw],
        res1["datos_cr"][:mh, :mw] if res1["datos_cr"] is not None else None,
        res2["datos_cr"][:mh, :mw] if res2["datos_cr"] is not None else None,
        res1["datos_extra"][:mh, :mw] if res1["datos_extra"] is not None else None,
        res2["datos_extra"][:mh, :mw] if res2["datos_extra"] is not None else None,
        res1["xs"][:mw],
        res1["ys"][:mh],
    )


def inspeccionar_grupo_rapido(nodo, prefijo="", profundidad=0, max_prof=3):
    datasets = {}
    tabla = []
    if isinstance(nodo, h5py.Dataset):
        datasets[prefijo] = nodo
        tabla.append({"Ruta interna": f"/{prefijo}", "Dimensiones": str(nodo.shape), "Tipo": str(nodo.dtype)})
    elif isinstance(nodo, h5py.Group) and profundidad <= max_prof:
        for k in nodo.keys():
            ruta = f"{prefijo}/{k}" if prefijo else k
            sub_d, sub_t = inspeccionar_grupo_rapido(nodo[k], ruta, profundidad + 1, max_prof)
            datasets.update(sub_d)
            tabla.extend(sub_t)
    return datasets, tabla


with st.sidebar:
    st.header("1. Cuenta NASA Earthdata")
    usuario = st.text_input("Usuario Earthdata", value="artur28")
    password = st.text_input("Contraseña Earthdata", type="password")

    st.divider()
    st.header("2. Zona y Escala de Análisis")
    preset = st.selectbox("Zona de estudio", list(LUGARES_PREDEFINIDOS.keys()))
    lat_def, lon_def, orb_def, frame_def = LUGARES_PREDEFINIDOS[preset]

    if "ultimo_preset" not in st.session_state:
        st.session_state.ultimo_preset = preset
    elif st.session_state.ultimo_preset != preset:
        st.session_state.ultimo_preset = preset
        st.session_state.lat_punto = lat_def
        st.session_state.lon_punto = lon_def
        st.session_state.trayecto_ui = orb_def
        st.session_state.marco_ui = frame_def

    lat = st.number_input("Latitud (Norte + / Sur -)", key="lat_punto", step=0.02, format="%.4f")
    lon = st.number_input("Longitud (Este + / Oeste -)", key="lon_punto", step=0.02, format="%.4f")

    ancho_km = st.slider(
        "Área de medición (2 km = detalle | 250 km = todo el mapa)",
        min_value=2,
        max_value=250,
        value=250,
        step=2,
    )

    mapa_label = st.selectbox("Tipo de mapa NISAR", list(MAPAS_NISAR.keys()), index=0)
    nivel = MAPAS_NISAR[mapa_label]

    if ancho_km >= 250:
        st.caption("Modo activo: **Todo el mapa / marco completo**")
    else:
        st.caption(f"Modo activo: **Zona específica {ancho_km} × {ancho_km} km**")

    if nivel in ("GSLC", "RSLC"):
        st.warning(
            f"**Aviso ({nivel}):** Este mapa se descarga en **alta resolución completa** sin recorte forzado. "
            "La carga tardará varios minutos dependiendo de tu conexión."
        )

    trayecto = st.number_input("Trayecto (0 = cualquiera)", min_value=0, value=st.session_state.trayecto_ui, step=1)
    marco = st.number_input("Marco (0 = cualquiera)", min_value=0, value=st.session_state.marco_ui, step=1)

if not usuario or not password:
    st.info("Ingresa tu usuario y contraseña de NASA Earthdata en el panel izquierdo.")
    st.stop()

os.environ["EARTHDATA_USERNAME"] = usuario
os.environ["EARTHDATA_PASSWORD"] = password
earthaccess.login(strategy="environment")


@st.cache_data(ttl=3600)
def buscar_escenas_punto(prod_level, lat_q, lon_q, orb_num, frame_num):
    kwargs = {
        "dataset": "NISAR",
        "processingLevel": prod_level,
        "intersectsWith": f"POINT({lon_q} {lat_q})",
        "maxResults": 60,
    }
    if prod_level != "SME2":
        if orb_num > 0:
            kwargs["relativeOrbit"] = int(orb_num)
        if frame_num > 0:
            kwargs["frame"] = int(frame_num)
    resultados = asf.geo_search(**kwargs)
    if not resultados and (orb_num > 0 or frame_num > 0):
        kwargs.pop("relativeOrbit", None)
        kwargs.pop("frame", None)
        resultados = asf.geo_search(**kwargs)
    return sorted(resultados, key=lambda r: r.properties["startTime"])


@st.cache_data(ttl=3600, show_spinner=False)
def leer_mapa_h5_rapido(url_h5, user_ed, pass_ed, prod_level, lat_c, lon_c, tam_km):
    os.environ["EARTHDATA_USERNAME"] = user_ed
    os.environ["EARTHDATA_PASSWORD"] = pass_ed
    earthaccess.login(strategy="environment")

    archivo_remoto = earthaccess.open([url_h5], pqdm_kwargs={"disable": True})[0]
    with h5py.File(archivo_remoto, "r") as h5:
        lsar = h5["/science/LSAR"]
        prod_key = prod_level if prod_level in lsar else [k for k in lsar.keys() if k != "identification"][0]
        grupo_prod = lsar[prod_key]

        geocodificado = "grids" in grupo_prod
        raiz = grupo_prod["grids"] if geocodificado else grupo_prod["swaths"]

        if (tam_km >= 60) and prod_key == "GCOV" and isinstance(raiz, h5py.Group) and ("frequencyB" in raiz) and isinstance(raiz["frequencyB"], h5py.Group):
            freq_elegida = "frequencyB"
            grupo_base = raiz["frequencyB"]
        elif isinstance(raiz, h5py.Group) and ("frequencyA" in raiz) and isinstance(raiz["frequencyA"], h5py.Group):
            freq_elegida = "frequencyA"
            grupo_base = raiz["frequencyA"]
        else:
            freq_elegida = "grids (L3)" if geocodificado else "swaths (L1)"
            grupo_base = raiz

        datasets_map, estructura_h5 = inspeccionar_grupo_rapido(
            grupo_base, f"science/LSAR/{prod_key}/{'grids' if geocodificado else 'swaths'}"
        )
        candidatos_2d = [
            (k, ds)
            for k, ds in datasets_map.items()
            if ds.ndim == 2
            and ds.shape[0] > 15
            and ds.shape[1] > 15
            and not any(ign in k.lower() for ign in ["mask", "numberoflooks", "rtcgamma", "layover", "shadow"])
        ]

        ds_extra = None
        if prod_key == "SME2":
            modo_unidad = "m3/m3"
            capa_co, ds_co = next(
                ((k, ds) for k, ds in candidatos_2d if "soilmoisture" in k.lower() and "uncertainty" not in k.lower()),
                candidatos_2d[0],
            )
            capa_cr, ds_cr = None, None
            ds_extra = next(
                (ds for k, ds in candidatos_2d if any(t in k.lower() for t in ["backscatterhv", "sigma0hv"])),
                None,
            )
        elif prod_key in ("GUNW", "RUNW"):
            modo_unidad = "rad"
            capa_co, ds_co = next(
                ((k, ds) for k, ds in candidatos_2d if "unwrappedphase" in k.lower()),
                candidatos_2d[0],
            )
            capa_cr, ds_cr = None, None
            ds_extra = next(
                (ds for k, ds in candidatos_2d if "coherencemagnitude" in k.lower()),
                None,
            )
        else:
            modo_unidad = "dB"
            capa_co, ds_co = next(
                ((k, ds) for k, ds in candidatos_2d if any(k.endswith(p) for p in ["HHHH", "VVVV", "HH", "VV"])),
                candidatos_2d[0],
            )
            if prod_key in ("GSLC", "RSLC"):
                capa_cr, ds_cr = None, None
            else:
                par_cr = next(
                    ((k, ds) for k, ds in candidatos_2d if any(k.endswith(p) for p in ["HVHV", "VHVH", "HV", "VH"])),
                    (None, None),
                )
                capa_cr, ds_cr = par_cr

        xs, ys, epsg = None, None, (6933 if prod_key == "SME2" else 32616)
        nodo_actual = ds_co.parent
        while nodo_actual is not None and nodo_actual.name != "/science/LSAR":
            if xs is None and "xCoordinates" in nodo_actual and "yCoordinates" in nodo_actual:
                xs = nodo_actual["xCoordinates"][:]
                ys = nodo_actual["yCoordinates"][:]
            if "projection" in nodo_actual:
                epsg = int(nodo_actual["projection"][()])
            nodo_actual = nodo_actual.parent

        if xs is not None and ys is not None and geocodificado:
            dx = abs(float(xs[1] - xs[0])) if len(xs) > 1 else 20.0
            dy = abs(float(ys[1] - ys[0])) if len(ys) > 1 else dx
            paso_x_m = dx * 111000.0 if epsg == 4326 else dx
            paso_y_m = dy * 111000.0 if epsg == 4326 else dy

            if tam_km >= 250:
                x0, x1 = 0, len(xs)
                y0, y1 = 0, len(ys)
            else:
                x_proj, y_proj = latlon_a_proyeccion(lat_c, lon_c, epsg)
                ix = int(np.argmin(np.abs(xs - x_proj)))
                iy = int(np.argmin(np.abs(ys - y_proj)))
                radio_px = max(15, int((tam_km * 1000.0 / 2.0) / max(paso_x_m, 1.0)))
                x0, x1 = max(0, ix - radio_px), min(len(xs), ix + radio_px)
                y0, y1 = max(0, iy - radio_px), min(len(ys), iy + radio_px)
        else:
            geocodificado = False
            alto_tot, ancho_tot = ds_co.shape
            xs = np.arange(ancho_tot, dtype=np.float64)
            ys = np.arange(alto_tot, dtype=np.float64)
            paso_x_m = 6.0
            paso_y_m = 7.0
            if tam_km >= 250:
                x0, x1, y0, y1 = 0, ancho_tot, 0, alto_tot
            else:
                radio_px = max(120, int((tam_km * 1000.0 / 2.0) / paso_x_m))
                cx, cy = ancho_tot // 2, alto_tot // 2
                x0, x1 = max(0, cx - radio_px), min(ancho_tot, cx + radio_px)
                y0, y1 = max(0, cy - radio_px), min(alto_tot, cy + radio_px)

        if x1 <= x0 or y1 <= y0:
            return None

        stride = max(1, max(x1 - x0, y1 - y0) // 950)
        if (x1 - x0) <= 1200 and (y1 - y0) <= 1200:
            datos_co = ds_co[y0:y1, x0:x1][::stride, ::stride]
            datos_cr = ds_cr[y0:y1, x0:x1][::stride, ::stride] if ds_cr is not None else None
            datos_extra = ds_extra[y0:y1, x0:x1][::stride, ::stride] if ds_extra is not None else None
        else:
            datos_co = ds_co[y0:y1:stride, x0:x1:stride]
            datos_cr = ds_cr[y0:y1:stride, x0:x1:stride] if ds_cr is not None else None
            datos_extra = ds_extra[y0:y1:stride, x0:x1:stride] if ds_extra is not None else None

        if datos_co.dtype.names and "r" in datos_co.dtype.names and "i" in datos_co.dtype.names:
            datos_co = np.hypot(datos_co["r"], datos_co["i"])
        elif np.iscomplexobj(datos_co):
            datos_co = np.abs(datos_co)

        if datos_cr is not None:
            if datos_cr.dtype.names and "r" in datos_cr.dtype.names and "i" in datos_cr.dtype.names:
                datos_cr = np.hypot(datos_cr["r"], datos_cr["i"])
            elif np.iscomplexobj(datos_cr):
                datos_cr = np.abs(datos_cr)

        ident = lsar["identification"]
        metadatos = {}
        for k in ident.keys():
            if isinstance(ident[k], h5py.Dataset) and ident[k].shape == ():
                val = ident[k][()]
                metadatos[k] = val.decode() if isinstance(val, (bytes, np.bytes_)) else str(val)

        return {
            "producto": prod_key,
            "modo_unidad": modo_unidad,
            "geocodificado": geocodificado,
            "datos_co": datos_co,
            "datos_cr": datos_cr,
            "datos_extra": datos_extra,
            "capa_co": capa_co.split("/")[-1],
            "capa_cr": capa_cr.split("/")[-1] if capa_cr else None,
            "frecuencia": freq_elegida,
            "epsg": epsg,
            "xs": xs[x0:x1:stride],
            "ys": ys[y0:y1:stride],
            "paso_x_efectivo": paso_x_m * stride,
            "paso_y_efectivo": paso_y_m * stride,
            "paso_m_efectivo": paso_x_m * stride,
            "estructura_h5": estructura_h5,
            "metadatos": metadatos,
        }


escenas = buscar_escenas_punto(nivel, lat, lon, trayecto, marco)
if not escenas:
    st.warning("No se encontraron escenas para ese punto/mapa. Prueba poniendo Trayecto = 0 y Marco = 0.")
    st.stop()

opciones = {
    f"[{e.properties.get('processingLevel', nivel)}] {e.properties['startTime'][:10]} | Órbita {e.properties.get('orbit', '?')} ({e.properties['sceneName'][-15:]})": e
    for e in escenas
}
claves_fecha = list(opciones.keys())

col_f1, col_f2 = st.columns(2)
with col_f1:
    sel_f1 = st.selectbox("Fecha 1 (Antes / Referencia):", claves_fecha, index=0)
with col_f2:
    sel_f2 = st.selectbox("Fecha 2 (Después / Comparación):", claves_fecha, index=len(claves_fecha) - 1)

escena_1 = opciones[sel_f1]
escena_2 = opciones[sel_f2]

if nivel in ("GSLC", "RSLC"):
    st.info(
        f"Modo de alta resolución activo (**{nivel}**): se descargará el área completa seleccionada "
        f"({'todo el mapa' if ancho_km >= 250 else f'{ancho_km} × {ancho_km} km'}). "
        "Este proceso tardará más que los productos de resolución media debido al volumen de datos."
    )

spinner_msg = (
    f"Descargando y procesando {nivel} en alta resolución completa (puede tardar varios minutos)..."
    if nivel in ("GSLC", "RSLC")
    else f"Renderizando {nivel} desde Earthdata Cloud..."
)

with st.spinner(spinner_msg):
    res1 = leer_mapa_h5_rapido(escena_1.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    res2 = (
        res1
        if sel_f1 == sel_f2
        else leer_mapa_h5_rapido(escena_2.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    )

if res1 is None or res2 is None:
    st.warning("Las coordenadas ingresadas quedaron fuera del área de datos de esta escena.")
    st.stop()

d1_co, d2_co, d1_cr, d2_cr, d1_ex, d2_ex, xs_comun, ys_comun = alinear_mallas(res1, res2)

unidad = res2["modo_unidad"]
es_slc = res2["producto"] in ("GSLC", "RSLC")
val1_raw, img1_raw, val1 = renderizar_capa(d1_co, d1_cr, unidad, es_amplitud_slc=es_slc)
val2_raw, img2_raw, val2 = renderizar_capa(d2_co, d2_cr, unidad, es_amplitud_slc=es_slc)

union_valida = val1 | val2
if np.any(union_valida):
    filas_v = np.where(np.any(union_valida, axis=1))[0]
    cols_v = np.where(np.any(union_valida, axis=0))[0]
    r0, r1_c = int(filas_v.min()), int(filas_v.max()) + 1
    c0, c1_c = int(cols_v.min()), int(cols_v.max()) + 1
    val1_co = val1_raw[r0:r1_c, c0:c1_c]
    val2_co = val2_raw[r0:r1_c, c0:c1_c]
    img1_igual = img1_raw[r0:r1_c, c0:c1_c]
    img2_igual = img2_raw[r0:r1_c, c0:c1_c]
    val1 = val1[r0:r1_c, c0:c1_c]
    val2 = val2[r0:r1_c, c0:c1_c]
    d1_co_c = d1_co[r0:r1_c, c0:c1_c]
    d2_co_c = d2_co[r0:r1_c, c0:c1_c]
    d1_cr_c = d1_cr[r0:r1_c, c0:c1_c] if d1_cr is not None else None
    d2_cr_c = d2_cr[r0:r1_c, c0:c1_c] if d2_cr is not None else None
    d1_ex_c = d1_ex[r0:r1_c, c0:c1_c] if d1_ex is not None else None
    d2_ex_c = d2_ex[r0:r1_c, c0:c1_c] if d2_ex is not None else None
    xs_final = xs_comun[c0:c1_c]
    ys_final = ys_comun[r0:r1_c]
else:
    val1_co, val2_co = val1_raw, val2_raw
    img1_igual, img2_igual = img1_raw, img2_raw
    d1_co_c, d2_co_c = d1_co, d2_co
    d1_cr_c, d2_cr_c = d1_cr, d2_cr
    d1_ex_c, d2_ex_c = d1_ex, d2_ex
    xs_final, ys_final = xs_comun, ys_comun

delta_vals = val2_co - val1_co
rango_color = 0.20 if unidad == "m3/m3" else (3.14 if unidad == "rad" else 8.0)
umbral_cambio = 0.05 if unidad == "m3/m3" else (1.0 if unidad == "rad" else 3.0)
img_diff = colorear_diferencia(delta_vals, rango_max=rango_color)

valid_diff = np.isfinite(delta_vals)
media_1 = float(np.nanmean(val1_co)) if np.any(val1) else float("nan")
media_2 = float(np.nanmean(val2_co)) if np.any(val2) else float("nan")
cambio_medio = media_2 - media_1
pct_cambio_fuerte = float(np.mean(np.abs(delta_vals[valid_diff]) > umbral_cambio) * 100) if np.any(valid_diff) else 0.0

fmt_v = "{:.3f}" if unidad == "m3/m3" else "{:.2f}"
fmt_d = "{:+.3f}" if unidad == "m3/m3" else "{:+.2f}"
unidad_label = "m³/m³" if unidad == "m3/m3" else unidad

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Media Fecha 1 ({res1['capa_co']})", f"{fmt_v.format(media_1)} {unidad_label}")
m2.metric(
    f"Media Fecha 2 ({res2['capa_co']})",
    f"{fmt_v.format(media_2)} {unidad_label}",
    delta=f"{fmt_d.format(cambio_medio)} {unidad_label}",
)
m3.metric(f"Área con cambio > {umbral_cambio} {unidad_label}", f"{pct_cambio_fuerte:.1f}%")
m4.metric("Resolución renderizada", f"{img1_igual.shape[1]} × {img1_igual.shape[0]} px ({res2['frecuencia']})")

c_img1, c_img2, c_diff = st.columns(3)
with c_img1:
    st.subheader("Fecha 1 (Antes)")
    st.image(img1_igual, caption=escena_1.properties["startTime"], width="stretch")
with c_img2:
    st.subheader("Fecha 2 (Después)")
    st.image(img2_igual, caption=escena_2.properties["startTime"], width="stretch")
with c_diff:
    st.subheader("Diferencia (Fecha 2 - Fecha 1)")
    st.image(
        img_diff,
        caption=f"Azul: disminuye ({unidad_label}) | Rojo: aumenta ({unidad_label})",
        width="stretch",
    )

st.divider()
col_mapa, col_h5 = st.columns([1.3, 1.2])

with col_mapa:
    st.subheader("Ubicación del Cuadro Azul y Cambios en el Mapa")
    poligono_azul = escena_2.geometry["coordinates"][0]

    if res2["geocodificado"] and len(xs_final) > 1 and len(ys_final) > 1:
        oeste, sur = proyeccion_a_latlon(float(xs_final.min()), float(ys_final.min()), res2["epsg"])
        este, norte = proyeccion_a_latlon(float(xs_final.max()), float(ys_final.max()), res2["epsg"])
    else:
        lons_p = [pt[0] for pt in poligono_azul]
        lats_p = [pt[1] for pt in poligono_azul]
        oeste, este, sur, norte = min(lons_p), max(lons_p), min(lats_p), max(lats_p)

    mh_f, mw_f = img_diff.shape[:2]
    paso_sub = max(1, max(mh_f, mw_f) // 85)
    sub_diff = img_diff[::paso_sub, ::paso_sub]
    sub_val = valid_diff[::paso_sub, ::paso_sub]
    lons_grid = np.linspace(oeste, este, mw_f)[::paso_sub]
    lats_grid = np.linspace(norte, sur, mh_f)[::paso_sub]

    puntos_cambio = []
    for r_i in range(sub_val.shape[0]):
        for c_i in range(sub_val.shape[1]):
            if sub_val[r_i, c_i]:
                rc, gc, bc = [int(v) for v in sub_diff[r_i, c_i]]
                puntos_cambio.append(
                    {
                        "lon": float(lons_grid[c_i]),
                        "lat": float(lats_grid[r_i]),
                        "color": [rc, gc, bc, 210],
                    }
                )

    capas_mapa = [
        pdk.Layer(
            "PolygonLayer",
            data=[{"poligono": poligono_azul}],
            get_polygon="poligono",
            get_line_color=[30, 144, 255],
            get_fill_color=[30, 144, 255, 20],
            line_width_min_pixels=3,
        ),
        pdk.Layer(
            "PolygonLayer",
            data=[{"poligono": [[oeste, sur], [este, sur], [este, norte], [oeste, norte], [oeste, sur]]}],
            get_polygon="poligono",
            get_line_color=[255, 60, 60],
            get_fill_color=[0, 0, 0, 0],
            line_width_min_pixels=2,
        ),
    ]
    if puntos_cambio:
        capas_mapa.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=puntos_cambio,
                get_position="[lon, lat]",
                get_fill_color="color",
                get_radius=float(res2["paso_m_efectivo"] * paso_sub * 0.72),
            )
        )
    capas_mapa.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"lon": lon, "lat": lat}],
            get_position="[lon, lat]",
            get_fill_color=[255, 215, 0, 240],
            get_line_color=[0, 0, 0, 255],
            stroked=True,
            line_width_min_pixels=2,
            radius_min_pixels=6,
            radius_max_pixels=10,
        )
    )

    zoom_mapa = 10.5 if ancho_km <= 10 else (9.2 if ancho_km <= 35 else (8.2 if ancho_km <= 90 else 7.2))
    st.pydeck_chart(
        pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            layers=capas_mapa,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom_mapa, pitch=0),
        )
    )
    st.caption("Contorno azul: marco NISAR | Recuadro rojo: área medida | Punto amarillo: coordenada elegida")

with col_h5:
    st.subheader("Contenido interno del archivo .h5")
    tab_vars, tab_meta = st.tabs(["Variables y Matrices (.h5)", "Metadatos de Identificación"])
    with tab_vars:
        st.dataframe(res2["estructura_h5"], width="stretch", height=340)
    with tab_meta:
        st.json(res2["metadatos"])

# --- TABLA DE MAGNITUDES FÍSICAS CALCULABLES DESDE EL .H5 ---
st.divider()
f1_str = escena_1.properties["startTime"][:10]
f2_str = escena_2.properties["startTime"][:10]
st.subheader(f"Tabla de Magnitudes Físicas Calculables ({res2['producto']})")

try:
    dt_1 = datetime.fromisoformat(escena_1.properties["startTime"].replace("Z", "+00:00"))
    dt_2 = datetime.fromisoformat(escena_2.properties["startTime"].replace("Z", "+00:00"))
    dias_diff = abs((dt_2 - dt_1).total_seconds()) / 86400.0
except Exception:
    dias_diff = 0.0

area_px_km2 = (res2["paso_x_efectivo"] * res2["paso_y_efectivo"]) / 1e6
filas_magnitudes = [
    {
        "Datos \\ fecha": "Tiempo de observación / Duración (días)",
        f"Fecha 1 ({f1_str})": "Día 0.0",
        f"Fecha 2 ({f2_str})": f"Día {dias_diff:.1f}",
        "Diferencia": f"{dias_diff:+.1f} días",
    }
]

if res2["producto"] == "GCOV":
    # 1. Retrodispersión co-polar (dB)
    filas_magnitudes.append(
        {
            "Datos \\ fecha": f"Retrodispersión Co-pol {res2['capa_co']} (dB)",
            f"Fecha 1 ({f1_str})": f"{media_1:.2f} dB",
            f"Fecha 2 ({f2_str})": f"{media_2:.2f} dB",
            "Diferencia": f"{cambio_medio:+.2f} dB",
        }
    )

    # 2. Extensión inundada / agua calmada (< -17 dB)
    agua_1_km2 = float(np.sum(val1 & (val1_co < -17.0))) * area_px_km2
    agua_2_km2 = float(np.sum(val2 & (val2_co < -17.0))) * area_px_km2
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Extensión inundada / agua abierta (km²)",
            f"Fecha 1 ({f1_str})": f"{agua_1_km2:.2f} km²",
            f"Fecha 2 ({f2_str})": f"{agua_2_km2:.2f} km²",
            "Diferencia": f"{(agua_2_km2 - agua_1_km2):+.2f} km²",
        }
    )

    if d1_cr_c is not None and d2_cr_c is not None:
        v_cr1 = np.isfinite(d1_cr_c) & (d1_cr_c > 0)
        v_cr2 = np.isfinite(d2_cr_c) & (d2_cr_c > 0)
        db_cr1 = np.where(v_cr1, 10.0 * np.log10(np.maximum(d1_cr_c, 1e-6)), np.nan)
        db_cr2 = np.where(v_cr2, 10.0 * np.log10(np.maximum(d2_cr_c, 1e-6)), np.nan)

        m_cr1 = float(np.nanmean(db_cr1)) if np.any(v_cr1) else float("nan")
        m_cr2 = float(np.nanmean(db_cr2)) if np.any(v_cr2) else float("nan")
        filas_magnitudes.append(
            {
                "Datos \\ fecha": f"Retrodispersión Cruzada {res2['capa_cr']} (dB)",
                f"Fecha 1 ({f1_str})": f"{m_cr1:.2f} dB",
                f"Fecha 2 ({f2_str})": f"{m_cr2:.2f} dB",
                "Diferencia": f"{(m_cr2 - m_cr1):+.2f} dB",
            }
        )

        # Cobertura forestal densa (HVHV > -16 dB) y pérdida/ganancia neta (ha)
        bosque_1_km2 = float(np.sum(v_cr1 & (db_cr1 > -16.0))) * area_px_km2
        bosque_2_km2 = float(np.sum(v_cr2 & (db_cr2 > -16.0))) * area_px_km2
        diff_bosque_km2 = bosque_2_km2 - bosque_1_km2
        filas_magnitudes.append(
            {
                "Datos \\ fecha": "Cobertura forestal densa (km²)",
                f"Fecha 1 ({f1_str})": f"{bosque_1_km2:.2f} km²",
                f"Fecha 2 ({f2_str})": f"{bosque_2_km2:.2f} km²",
                "Diferencia": f"{diff_bosque_km2:+.2f} km²",
            }
        )
        filas_magnitudes.append(
            {
                "Datos \\ fecha": "Cambio forestal neto (ha)",
                f"Fecha 1 ({f1_str})": f"{(bosque_1_km2 * 100.0):.1f} ha",
                f"Fecha 2 ({f2_str})": f"{(bosque_2_km2 * 100.0):.1f} ha",
                "Diferencia": f"{(diff_bosque_km2 * 100.0):+.1f} ha",
            }
        )

        # Superficie agrícola / vegetación baja (-21 dB <= HVHV <= -16 dB)
        agri_1_km2 = float(np.sum(v_cr1 & (db_cr1 >= -21.0) & (db_cr1 <= -16.0))) * area_px_km2
        agri_2_km2 = float(np.sum(v_cr2 & (db_cr2 >= -21.0) & (db_cr2 <= -16.0))) * area_px_km2
        filas_magnitudes.append(
            {
                "Datos \\ fecha": "Superficie agrícola / vegetación baja (km²)",
                f"Fecha 1 ({f1_str})": f"{agri_1_km2:.2f} km²",
                f"Fecha 2 ({f2_str})": f"{agri_2_km2:.2f} km²",
                "Diferencia": f"{(agri_2_km2 - agri_1_km2):+.2f} km²",
            }
        )

        # Humedad / Índice de Vegetación Radar RVI = 4*HV / (HH + HV)
        ok_rvi1 = val1 & v_cr1 & ((d1_co_c + d1_cr_c) > 0)
        ok_rvi2 = val2 & v_cr2 & ((d2_co_c + d2_cr_c) > 0)
        rvi_1 = float(np.mean(np.clip((4.0 * d1_cr_c[ok_rvi1]) / (d1_co_c[ok_rvi1] + d1_cr_c[ok_rvi1]), 0.0, 1.5))) if np.any(ok_rvi1) else float("nan")
        rvi_2 = float(np.mean(np.clip((4.0 * d2_cr_c[ok_rvi2]) / (d2_co_c[ok_rvi2] + d2_cr_c[ok_rvi2]), 0.0, 1.5))) if np.any(ok_rvi2) else float("nan")
        filas_magnitudes.append(
            {
                "Datos \\ fecha": "Humedad / Índice vegetación radar RVI (0-1)",
                f"Fecha 1 ({f1_str})": f"{rvi_1:.3f}",
                f"Fecha 2 ({f2_str})": f"{rvi_2:.3f}",
                "Diferencia": f"{(rvi_2 - rvi_1):+.3f}",
            }
        )

elif res2["producto"] == "SME2":
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Humedad volumétrica del suelo (m³/m³)",
            f"Fecha 1 ({f1_str})": f"{media_1:.3f} m³/m³",
            f"Fecha 2 ({f2_str})": f"{media_2:.3f} m³/m³",
            "Diferencia": f"{cambio_medio:+.3f} m³/m³",
        }
    )
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Humedad relativa del suelo (%)",
            f"Fecha 1 ({f1_str})": f"{(media_1 * 100.0):.1f} %",
            f"Fecha 2 ({f2_str})": f"{(media_2 * 100.0):.1f} %",
            "Diferencia": f"{(cambio_medio * 100.0):+.1f} %",
        }
    )
    sat_1_km2 = float(np.sum(val1 & (val1_co > 0.40))) * area_px_km2
    sat_2_km2 = float(np.sum(val2 & (val2_co > 0.40))) * area_px_km2
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Área de suelo saturado > 0.40 m³/m³ (km²)",
            f"Fecha 1 ({f1_str})": f"{sat_1_km2:.2f} km²",
            f"Fecha 2 ({f2_str})": f"{sat_2_km2:.2f} km²",
            "Diferencia": f"{(sat_2_km2 - sat_1_km2):+.2f} km²",
        }
    )
    if d1_ex_c is not None and d2_ex_c is not None:
        v_hv1 = np.isfinite(d1_ex_c) & (d1_ex_c > -60.0) & (d1_ex_c < 20.0)
        v_hv2 = np.isfinite(d2_ex_c) & (d2_ex_c > -60.0) & (d2_ex_c < 20.0)
        if np.any(v_hv1) and np.any(v_hv2):
            hv1_m = float(np.mean(d1_ex_c[v_hv1]))
            hv2_m = float(np.mean(d2_ex_c[v_hv2]))
            filas_magnitudes.append(
                {
                    "Datos \\ fecha": "Retrodispersión HV asociada (dB)",
                    f"Fecha 1 ({f1_str})": f"{hv1_m:.2f} dB",
                    f"Fecha 2 ({f2_str})": f"{hv2_m:.2f} dB",
                    "Diferencia": f"{(hv2_m - hv1_m):+.2f} dB",
                }
            )

elif res2["producto"] in ("GUNW", "RUNW"):
    # Longitud de onda banda L NISAR (~24 cm = 240 mm) -> d_cm = -fase * (24 / 4*pi)
    factor_cm = -24.0 / (4.0 * math.pi)
    desp1_cm = val1_co * factor_cm
    desp2_cm = val2_co * factor_cm
    m_desp1 = float(np.nanmean(desp1_cm)) if np.any(val1) else float("nan")
    m_desp2 = float(np.nanmean(desp2_cm)) if np.any(val2) else float("nan")
    diff_desp_cm = m_desp2 - m_desp1

    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Diferencia de fase interferométrica (rad)",
            f"Fecha 1 ({f1_str})": f"{media_1:.2f} rad",
            f"Fecha 2 ({f2_str})": f"{media_2:.2f} rad",
            "Diferencia": f"{cambio_medio:+.2f} rad",
        }
    )
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Desplazamiento vertical / LOS (cm)",
            f"Fecha 1 ({f1_str})": f"{m_desp1:.2f} cm",
            f"Fecha 2 ({f2_str})": f"{m_desp2:.2f} cm",
            "Diferencia": f"{diff_desp_cm:+.2f} cm",
        }
    )

    anios = max(dias_diff / 365.25, 12.0 / 365.25)
    vel_1_mm = (m_desp1 * 10.0) / (12.0 / 365.25)
    vel_2_mm = (m_desp2 * 10.0) / anios
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Velocidad de deformación estimada (mm/año)",
            f"Fecha 1 ({f1_str})": f"{vel_1_mm:.1f} mm/año",
            f"Fecha 2 ({f2_str})": f"{vel_2_mm:.1f} mm/año",
            "Diferencia": f"{(vel_2_mm - vel_1_mm):+.1f} mm/año",
        }
    )

    ext_def1 = float(np.sum(val1 & (np.abs(desp1_cm) > 2.0))) * area_px_km2
    ext_def2 = float(np.sum(val2 & (np.abs(desp2_cm) > 2.0))) * area_px_km2
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Extensión deformada > 2 cm (km²)",
            f"Fecha 1 ({f1_str})": f"{ext_def1:.2f} km²",
            f"Fecha 2 ({f2_str})": f"{ext_def2:.2f} km²",
            "Diferencia": f"{(ext_def2 - ext_def1):+.2f} km²",
        }
    )

    # Gradiente espacial de desplazamiento (mm/km)
    paso_km = max(res2["paso_x_efectivo"] / 1000.0, 0.01)
    gy1, gx1 = np.gradient(np.nan_to_num(desp1_cm * 10.0, nan=0.0), paso_km)
    gy2, gx2 = np.gradient(np.nan_to_num(desp2_cm * 10.0, nan=0.0), paso_km)
    grad1 = float(np.mean(np.hypot(gx1[val1], gy1[val1]))) if np.any(val1) else 0.0
    grad2 = float(np.mean(np.hypot(gx2[val2], gy2[val2]))) if np.any(val2) else 0.0
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Gradiente espacial de desplazamiento (mm/km)",
            f"Fecha 1 ({f1_str})": f"{grad1:.2f} mm/km",
            f"Fecha 2 ({f2_str})": f"{grad2:.2f} mm/km",
            "Diferencia": f"{(grad2 - grad1):+.2f} mm/km",
        }
    )

    if d1_ex_c is not None and d2_ex_c is not None:
        vc1 = np.isfinite(d1_ex_c) & (d1_ex_c >= 0.0) & (d1_ex_c <= 1.0)
        vc2 = np.isfinite(d2_ex_c) & (d2_ex_c >= 0.0) & (d2_ex_c <= 1.0)
        coh1 = float(np.mean(d1_ex_c[vc1])) if np.any(vc1) else float("nan")
        coh2 = float(np.mean(d2_ex_c[vc2])) if np.any(vc2) else float("nan")
        filas_magnitudes.append(
            {
                "Datos \\ fecha": "Coherencia interferométrica / temporal (0-1)",
                f"Fecha 1 ({f1_str})": f"{coh1:.3f}",
                f"Fecha 2 ({f2_str})": f"{coh2:.3f}",
                "Diferencia": f"{(coh2 - coh1):+.3f}",
            }
        )

else:
    # GSLC / RSLC
    filas_magnitudes.append(
        {
            "Datos \\ fecha": f"Retrodispersión calibrada {res2['capa_co']} (dB)",
            f"Fecha 1 ({f1_str})": f"{media_1:.2f} dB",
            f"Fecha 2 ({f2_str})": f"{media_2:.2f} dB",
            "Diferencia": f"{cambio_medio:+.2f} dB",
        }
    )
    agua_1_km2 = float(np.sum(val1 & (val1_co < -17.0))) * area_px_km2
    agua_2_km2 = float(np.sum(val2 & (val2_co < -17.0))) * area_px_km2
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Extensión de agua / baja rugosidad (km²)",
            f"Fecha 1 ({f1_str})": f"{agua_1_km2:.2f} km²",
            f"Fecha 2 ({f2_str})": f"{agua_2_km2:.2f} km²",
            "Diferencia": f"{(agua_2_km2 - agua_1_km2):+.2f} km²",
        }
    )
    denso_1_km2 = float(np.sum(val1 & (val1_co > -10.0))) * area_px_km2
    denso_2_km2 = float(np.sum(val2 & (val2_co > -10.0))) * area_px_km2
    filas_magnitudes.append(
        {
            "Datos \\ fecha": "Cobertura densa / estructuras (km²)",
            f"Fecha 1 ({f1_str})": f"{denso_1_km2:.2f} km²",
            f"Fecha 2 ({f2_str})": f"{denso_2_km2:.2f} km²",
            "Diferencia": f"{(denso_2_km2 - denso_1_km2):+.2f} km²",
        }
    )

df_magnitudes = pd.DataFrame(filas_magnitudes)
st.dataframe(df_magnitudes, width="stretch", hide_index=True)