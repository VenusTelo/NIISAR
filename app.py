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
import altair as alt

st.set_page_config(page_title="NISAR Monitor - Pre Alpha 1.2", layout="wide")
st.title("NISAR Monitor — Pre Alpha 1.2")

LUGARES_PREDEFINIDOS = {
    "Gracias a Dios / Caratasca Lagoon (Track 4, Frame 9)": (15.1500, -83.9000, 4, 9),
    "Roatán / Bay Islands (Track 105, Frame 10)": (16.3200, -86.5300, 105, 10),
}

MAPAS_NISAR = {
    "GCOV — Level 2 (Backscatter / Water & Vegetation)": "GCOV",
    "GUNW — Level 2 (Interferometry / Phase & Deformation)": "GUNW",
    "SME2 — Level 3 (Soil Moisture in m³/m³)": "SME2",
    "GSLC — Level 2 (Geocoded High-Resolution Complex)": "GSLC",
    "RSLC — Level 1 (High-Resolution Radar Geometry)": "RSLC",
}

if "lat_punto" not in st.session_state:
    st.session_state.lat_punto = 15.1500
if "lon_punto" not in st.session_state:
    st.session_state.lon_punto = -83.9000
if "trayecto_ui" not in st.session_state:
    st.session_state.trayecto_ui = 4
if "marco_ui" not in st.session_state:
    st.session_state.marco_ui = 9
if "modo_app" not in st.session_state:
    st.session_state.modo_app = "Intuitive"


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
        valid = np.isfinite(d_co) & (d_co > 0) & (d_co > -900.0)
        factor_db = 20.0 if es_amplitud_slc else 10.0
        vals = np.where(valid, factor_db * np.log10(np.maximum(d_co, 1e-6)), np.nan)
        if not np.any(valid):
            return vals, rgb, valid
        p2, p98 = np.nanpercentile(vals, (2, 98))
        r = np.nan_to_num(np.clip((vals - p2) / (p98 - p2 + 1e-6) * 255, 0, 255), nan=25).astype(np.uint8)
        if d_cr is not None:
            v_cr = np.isfinite(d_cr) & (d_cr > 0) & (d_cr > -900.0)
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
        valid = np.isfinite(d_co) & (np.abs(d_co) < 500.0) & (d_co != 0.0) & (d_co > -900.0)
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

    valid = np.isfinite(d_co) & (d_co != -9999.0) & (d_co > -900.0)
    vals = np.where(valid, d_co, np.nan)
    if not np.any(valid):
        return vals, rgb, valid
    stops = [0.0, 0.15, 0.30, 0.45, 0.60]
    v_ok = np.clip(vals[valid], 0.0, 0.60)
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


def generar_celdas_raster_mapa(
    img_rgb: np.ndarray,
    mascara_valida: np.ndarray,
    geocodificado: bool,
    xs_arr: np.ndarray,
    ys_arr: np.ndarray,
    epsg: int,
    bounds_fallback: tuple,
    max_lado: int = 75,
):
    h, w = img_rgb.shape[:2]
    paso = max(1, max(h, w) // max_lado)
    rgb_s = img_rgb[::paso, ::paso]
    val_s = mascara_valida[::paso, ::paso]
    hs, ws = rgb_s.shape[:2]

    if hs == 0 or ws == 0 or not np.any(val_s):
        return []

    if geocodificado and len(xs_arr) > 1 and len(ys_arr) > 1:
        dx = float(xs_arr[1] - xs_arr[0])
        dy = float(ys_arr[1] - ys_arr[0])
        x_ini = float(xs_arr[0]) - 0.5 * dx
        x_fin = float(xs_arr[-1]) + 0.5 * dx
        y_ini = float(ys_arr[0]) - 0.5 * dy
        y_fin = float(ys_arr[-1]) + 0.5 * dy

        x_edges = np.linspace(x_ini, x_fin, ws + 1)
        y_edges = np.linspace(y_ini, y_fin, hs + 1)

        lon_grid = np.empty((hs + 1, ws + 1), dtype=np.float64)
        lat_grid = np.empty((hs + 1, ws + 1), dtype=np.float64)
        for r in range(hs + 1):
            yv = float(y_edges[r])
            for c in range(ws + 1):
                lo, la = proyeccion_a_latlon(float(x_edges[c]), yv, epsg)
                lon_grid[r, c] = lo
                lat_grid[r, c] = la
    else:
        oeste_f, sur_f, este_f, norte_f = bounds_fallback
        lon_edges = np.linspace(oeste_f, este_f, ws + 1)
        lat_edges = np.linspace(norte_f, sur_f, hs + 1)
        lon_grid, lat_grid = np.meshgrid(lon_edges, lat_edges)

    celdas = []
    filas_ok, cols_ok = np.where(val_s)
    for r, c in zip(filas_ok.tolist(), cols_ok.tolist()):
        p0 = [round(float(lon_grid[r, c]), 6), round(float(lat_grid[r, c]), 6)]
        p1 = [round(float(lon_grid[r, c + 1]), 6), round(float(lat_grid[r, c + 1]), 6)]
        p2 = [round(float(lon_grid[r + 1, c + 1]), 6), round(float(lat_grid[r + 1, c + 1]), 6)]
        p3 = [round(float(lon_grid[r + 1, c]), 6), round(float(lat_grid[r + 1, c]), 6)]
        px = rgb_s[r, c]
        celdas.append(
            {
                "polygon": [p0, p1, p2, p3, p0],
                "color": [int(px[0]), int(px[1]), int(px[2]), 205],
            }
        )
    return celdas


def indices_mas_cercanos(arr_origen: np.ndarray, arr_objetivo: np.ndarray) -> np.ndarray:
    if len(arr_origen) == 1:
        return np.zeros(len(arr_objetivo), dtype=np.int64)
    if arr_origen[0] <= arr_origen[-1]:
        idx = np.clip(np.searchsorted(arr_origen, arr_objetivo), 1, len(arr_origen) - 1)
        izq = arr_origen[idx - 1]
        der = arr_origen[idx]
        return np.where(np.abs(arr_objetivo - izq) <= np.abs(arr_objetivo - der), idx - 1, idx)
    arr_inv = arr_origen[::-1]
    idx = np.clip(np.searchsorted(arr_inv, arr_objetivo), 1, len(arr_inv) - 1)
    izq = arr_inv[idx - 1]
    der = arr_inv[idx]
    idx_inv = np.where(np.abs(arr_objetivo - izq) <= np.abs(arr_objetivo - der), idx - 1, idx)
    return (len(arr_origen) - 1) - idx_inv


def alinear_mallas(res1: dict, res2: dict):
    if res1["geocodificado"] and res2["geocodificado"] and res1["epsg"] == res2["epsg"]:
        xs1, ys1 = res1["xs"], res1["ys"]
        xs2, ys2 = res2["xs"], res2["ys"]
        tol_x = abs(float(xs1[1] - xs1[0])) * 1e-3 if len(xs1) > 1 else 0.1
        tol_y = abs(float(ys1[1] - ys1[0])) * 1e-3 if len(ys1) > 1 else 0.1

        x_min = max(float(xs1.min()), float(xs2.min())) - tol_x
        x_max = min(float(xs1.max()), float(xs2.max())) + tol_x
        y_min = max(float(ys1.min()), float(ys2.min())) - tol_y
        y_max = min(float(ys1.max()), float(ys2.max())) + tol_y

        ix1 = np.where((xs1 >= x_min) & (xs1 <= x_max))[0]
        iy1 = np.where((ys1 >= y_min) & (ys1 <= y_max))[0]
        if ix1.size > 4 and iy1.size > 4:
            xs_c = xs1[ix1]
            ys_c = ys1[iy1]
            ix2 = indices_mas_cercanos(xs2, xs_c)
            iy2 = indices_mas_cercanos(ys2, ys_c)

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


def inspeccionar_grupo_rapido(nodo, prefijo="", profundidad=0, max_prof=6):
    datasets = {}
    tabla = []
    if isinstance(nodo, h5py.Dataset):
        datasets[prefijo] = nodo
        tabla.append({"Internal Path": f"/{prefijo}", "Dimensions": str(nodo.shape), "Type": str(nodo.dtype)})
    elif isinstance(nodo, h5py.Group) and profundidad <= max_prof:
        for k in nodo.keys():
            ruta = f"{prefijo}/{k}" if prefijo else k
            sub_d, sub_t = inspeccionar_grupo_rapido(nodo[k], ruta, profundidad + 1, max_prof)
            datasets.update(sub_d)
            tabla.extend(sub_t)
    return datasets, tabla


with st.sidebar:
    st.header("1. NASA Earthdata Account")
    usuario = st.text_input("Earthdata Username", value="", placeholder="Enter your Earthdata username")
    password = st.text_input("Earthdata Password", type="password")

    st.divider()
    st.header("Display Mode")
    st.session_state.modo_app = st.radio(
        "Select how you want to view the data:",
        ["Intuitive (for the general public)", "Scientific (for the technical community)"],
        index=0,
        help="Intuitive: simplified map for anyone. Scientific: color scales, physical ranges, and technical variables.",
    )

    st.divider()
    st.header("2. Study Area & Analysis Scale")
    preset = st.selectbox("Study area", list(LUGARES_PREDEFINIDOS.keys()))
    lat_def, lon_def, orb_def, frame_def = LUGARES_PREDEFINIDOS[preset]

    if "ultimo_preset" not in st.session_state:
        st.session_state.ultimo_preset = preset
    elif st.session_state.ultimo_preset != preset:
        st.session_state.ultimo_preset = preset
        st.session_state.lat_punto = lat_def
        st.session_state.lon_punto = lon_def
        st.session_state.trayecto_ui = orb_def
        st.session_state.marco_ui = frame_def

    lat = st.number_input("Latitude (North + / South -)", key="lat_punto", step=0.02, format="%.4f")
    lon = st.number_input("Longitude (East + / West -)", key="lon_punto", step=0.02, format="%.4f")

    ancho_km = st.slider(
        "Measurement area (2 km = detail | 250 km = full map)",
        min_value=2,
        max_value=250,
        value=62,
        step=2,
    )

    mapa_label = st.selectbox("NISAR map type", list(MAPAS_NISAR.keys()), index=0)
    nivel = MAPAS_NISAR[mapa_label]

    if ancho_km >= 250:
        st.caption("Mode: **Full map**")
    else:
        st.caption(f"Mode: **{ancho_km} × {ancho_km} km**")

    if nivel in ("GSLC", "RSLC"):
        st.warning(f"Warning: **{nivel}** product at full high resolution; download will take several minutes.")

    trayecto = st.number_input("Track (0 = any)", min_value=0, value=st.session_state.trayecto_ui, step=1)
    marco = st.number_input("Frame (0 = any)", min_value=0, value=st.session_state.marco_ui, step=1)

if not usuario or not password:
    st.info("Enter your NASA Earthdata username and password in the left panel.")
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
    if orb_num > 0:
        kwargs["relativeOrbit"] = int(orb_num)
    if prod_level != "SME2" and frame_num > 0:
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
            grupo_busqueda = raiz["frequencyB"]
            prefijo_busqueda = f"science/LSAR/{prod_key}/grids/frequencyB"
        elif prod_key != "SME2" and isinstance(raiz, h5py.Group) and ("frequencyA" in raiz) and isinstance(raiz["frequencyA"], h5py.Group):
            freq_elegida = "frequencyA"
            grupo_busqueda = raiz["frequencyA"]
            prefijo_busqueda = f"science/LSAR/{prod_key}/{'grids' if geocodificado else 'swaths'}/frequencyA"
        else:
            freq_elegida = "grids (L3)" if geocodificado else "swaths (L1)"
            grupo_busqueda = grupo_prod
            prefijo_busqueda = f"science/LSAR/{prod_key}"

        datasets_map, estructura_h5 = inspeccionar_grupo_rapido(
            grupo_busqueda, prefijo_busqueda, max_prof=6
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
        algoritmo_usado = "Standard"
        freq_hv_label = "Freq. A"

        if prod_key == "SME2":
            modo_unidad = "m3/m3"
            capa_co, ds_co = next(
                ((k, ds) for k, ds in candidatos_2d if "algorithmcandidates/dsg/soilmoisture" in k.lower()),
                next(
                    ((k, ds) for k, ds in candidatos_2d if "soilmoisture" in k.lower() and "uncertainty" not in k.lower()),
                    candidatos_2d[0],
                ),
            )
            algoritmo_usado = "DSG" if "dsg" in capa_co.lower() else "Standard"
            capa_cr, ds_cr = None, None

            par_hv = next(
                ((k, ds) for k, ds in candidatos_2d if "frequencya/sigma0hv" in k.lower()),
                next(
                    ((k, ds) for k, ds in candidatos_2d if "sigma0hv" in k.lower() or "backscatterhv" in k.lower()),
                    (None, None),
                ),
            )
            capa_hv, ds_extra = par_hv
            freq_hv_label = "Freq. B" if (capa_hv and "frequencyb" in capa_hv.lower()) else "Freq. A"

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
        dx_attr, dy_attr = None, None
        nodo_actual = ds_co.parent
        while nodo_actual is not None and nodo_actual.name != "/science/LSAR":
            if xs is None and "xCoordinates" in nodo_actual and "yCoordinates" in nodo_actual:
                xs = nodo_actual["xCoordinates"][:]
                ys = nodo_actual["yCoordinates"][:]
            if dx_attr is None and "xCoordinateSpacing" in nodo_actual:
                dx_attr = abs(float(nodo_actual["xCoordinateSpacing"][()]))
            if dy_attr is None and "yCoordinateSpacing" in nodo_actual:
                dy_attr = abs(float(nodo_actual["yCoordinateSpacing"][()]))
            if "projection" in nodo_actual:
                epsg = int(nodo_actual["projection"][()])
            nodo_actual = nodo_actual.parent

        if xs is not None and ys is not None and geocodificado:
            dx = dx_attr if dx_attr is not None else (abs(float(xs[1] - xs[0])) if len(xs) > 1 else 20.0)
            dy = dy_attr if dy_attr is not None else (abs(float(ys[1] - ys[0])) if len(ys) > 1 else dx)
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

        stride = 1 if prod_key == "SME2" else max(1, max(x1 - x0, y1 - y0) // 950)
        if (x1 - x0) <= 1600 and (y1 - y0) <= 1600:
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
                if isinstance(val, (bytes, np.bytes_)):
                    metadatos[k] = val.decode("utf-8", errors="ignore").strip("\x00").strip()
                else:
                    metadatos[k] = str(val).strip()

        return {
            "producto": prod_key,
            "algoritmo": algoritmo_usado,
            "freq_hv_label": freq_hv_label,
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
    st.warning("No scenes found. Try setting Track = 0 and Frame = 0.")
    st.stop()


def extraer_info_orbital(e):
    p = e.properties
    nombre = str(p.get("fileID") or p.get("sceneName") or "")
    partes = nombre.split("_")
    track_id = p.get("pathNumber") or p.get("relativeOrbit")
    dir_raw = str(p.get("flightDirection") or "").upper()
    if len(partes) >= 7 and partes[0] == "NISAR":
        if not track_id and partes[5].isdigit():
            track_id = int(partes[5])
        if not dir_raw and partes[6] in ("A", "D"):
            dir_raw = "ASCENDING" if partes[6] == "A" else "DESCENDING"
    dir_esp = "Ascending" if "ASC" in dir_raw else ("Descending" if "DESC" in dir_raw else "")
    return track_id or "?", dir_esp


def formatear_opcion_escena(e):
    p = e.properties
    track_id, dir_esp = extraer_info_orbital(e)
    dir_txt = f" ({dir_esp})" if dir_esp else ""
    hora_utc = p["startTime"][11:16] if len(p.get("startTime", "")) >= 16 else ""
    id_granulo = str(p.get("fileID") or p.get("sceneName") or p.get("orbit") or "")[-18:]
    return f"[{p.get('processingLevel', nivel)}] {p['startTime'][:10]} {hora_utc} UTC | Track {track_id}{dir_txt} | {id_granulo}"


opciones = {formatear_opcion_escena(e): e for e in escenas}
claves_fecha = list(opciones.keys())

col_f1, col_f2 = st.columns(2)
with col_f1:
    sel_f1 = st.selectbox("Date 1 (Before):", claves_fecha, index=0)
with col_f2:
    sel_f2 = st.selectbox("Date 2 (After):", claves_fecha, index=len(claves_fecha) - 1)

escena_1 = opciones[sel_f1]
escena_2 = opciones[sel_f2]

track_1, dir_1 = extraer_info_orbital(escena_1)
track_2, dir_2 = extraer_info_orbital(escena_2)

if (str(track_1) != "?" and str(track_2) != "?" and str(track_1) != str(track_2)) or (dir_1 and dir_2 and dir_1 != dir_2):
    st.warning(f"Warning: Different geometry between Track {track_1} ({dir_1}) and Track {track_2} ({dir_2}). Computing over the common overlap area.")

spinner_msg = (
    f"Downloading {nivel} at high resolution..."
    if nivel in ("GSLC", "RSLC")
    else f"Processing {nivel}..."
)

with st.spinner(spinner_msg):
    res1 = leer_mapa_h5_rapido(escena_1.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    res2 = (
        res1
        if sel_f1 == sel_f2
        else leer_mapa_h5_rapido(escena_2.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    )

if res1 is None or res2 is None:
    st.warning("Coordinates are outside the scene's data coverage.")
    st.stop()

sat_tot_1_km2 = float("nan")
sat_tot_2_km2 = float("nan")
if res2["producto"] == "SME2":
    raw_1 = res1["datos_co"]
    raw_2 = res2["datos_co"]
    area_px1_km2 = (res1["paso_x_efectivo"] * res1["paso_y_efectivo"]) / 1e6
    area_px2_km2 = (res2["paso_x_efectivo"] * res2["paso_y_efectivo"]) / 1e6
    val_raw_1 = np.isfinite(raw_1) & (raw_1 != -9999.0) & (raw_1 > -900.0)
    val_raw_2 = np.isfinite(raw_2) & (raw_2 != -9999.0) & (raw_2 > -900.0)
    sat_tot_1_km2 = float(np.sum(val_raw_1 & (raw_1 > 0.40))) * area_px1_km2
    sat_tot_2_km2 = float(np.sum(val_raw_2 & (raw_2 > 0.40))) * area_px2_km2

d1_co, d2_co, d1_cr, d2_cr, d1_ex, d2_ex, xs_comun, ys_comun = alinear_mallas(res1, res2)

unidad = res2["modo_unidad"]
es_slc = res2["producto"] in ("GSLC", "RSLC")
val1_raw, img1_raw, valido_F1 = renderizar_capa(d1_co, d1_cr, unidad, es_amplitud_slc=es_slc)
val2_raw, img2_raw, valido_F2 = renderizar_capa(d2_co, d2_cr, unidad, es_amplitud_slc=es_slc)

union_valida = valido_F1 | valido_F2
if np.any(union_valida):
    filas_v = np.where(np.any(union_valida, axis=1))[0]
    cols_v = np.where(np.any(union_valida, axis=0))[0]
    r0, r1_c = int(filas_v.min()), int(filas_v.max()) + 1
    c0, c1_c = int(cols_v.min()), int(cols_v.max()) + 1
    val1_co = val1_raw[r0:r1_c, c0:c1_c]
    val2_co = val2_raw[r0:r1_c, c0:c1_c]
    img1_igual = img1_raw[r0:r1_c, c0:c1_c]
    img2_igual = img2_raw[r0:r1_c, c0:c1_c]
    valido_F1 = valido_F1[r0:r1_c, c0:c1_c]
    valido_F2 = valido_F2[r0:r1_c, c0:c1_c]
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

m = valido_F1 & valido_F2
if not np.any(m):
    st.warning("No valid overlapping pixels found within this area.")
    st.stop()

hv_disponible = False
mh = m
if res2["producto"] == "SME2" and d1_ex_c is not None and d2_ex_c is not None:
    hv_valido_F1 = np.isfinite(d1_ex_c) & (d1_ex_c != -9999.0) & (d1_ex_c > 0.0)
    hv_valido_F2 = np.isfinite(d2_ex_c) & (d2_ex_c != -9999.0) & (d2_ex_c > 0.0)
    mh_cand = m & hv_valido_F1 & hv_valido_F2
    if np.any(mh_cand):
        mh = mh_cand
        hv_disponible = True

delta_vals = np.where(m, val2_co - val1_co, np.nan)
rango_color = 0.20 if unidad == "m3/m3" else (3.14 if unidad == "rad" else 8.0)
umbral_cambio = 0.05 if unidad == "m3/m3" else (1.0 if unidad == "rad" else 3.0)
img_diff = colorear_diferencia(delta_vals, rango_max=rango_color)

valid_diff = np.isfinite(delta_vals)
media_1 = float(np.mean(val1_co[m]))
media_2 = float(np.mean(val2_co[m]))
cambio_medio = media_2 - media_1
pct_cambio_fuerte = float(np.mean(np.abs(delta_vals[valid_diff]) > umbral_cambio) * 100) if np.any(valid_diff) else 0.0

fmt_v = "{:.4f}" if unidad == "m3/m3" else "{:.2f}"
fmt_d = "{:+.4f}" if unidad == "m3/m3" else "{:+.2f}"
unidad_label = "m³/m³" if unidad == "m3/m3" else unidad

etiqueta_frec = (
    res2["frecuencia"]
    .replace("frequencyA", "Frequency A")
    .replace("frequencyB", "Frequency B")
)
if res2["producto"] == "SME2":
    etiqueta_frec = f"{res2['algoritmo']} Algorithm"

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Date 1 Mean ({res1['capa_co']})", f"{fmt_v.format(media_1)} {unidad_label}")
m2.metric(
    f"Date 2 Mean ({res2['capa_co']})",
    f"{fmt_v.format(media_2)} {unidad_label}",
    delta=f"{fmt_d.format(cambio_medio)} {unidad_label}",
)
m3.metric(f"Area with change > {umbral_cambio} {unidad_label}", f"{pct_cambio_fuerte:.1f}%")
m4.metric(
    "Resolution",
    f"{img1_igual.shape[1]} × {img1_igual.shape[0]} px",
    delta=etiqueta_frec,
    delta_color="off",
)

es_modo_cientifico = st.session_state.modo_app.startswith("Scientific")

if unidad == "m3/m3":
    escala_titulo = "Soil Moisture (m³/m³)"
    escala_rango = "0.00 to 0.60 m³/m³"
    escalas = {
        "0.00": [255, 255, 217],
        "0.15": [199, 233, 180],
        "0.30": [65, 182, 196],
        "0.45": [34, 94, 168],
        "0.60": [8, 29, 88],
    }
    interpretacion = {
        "0.00 – 0.15": "Very dry soil",
        "0.15 – 0.30": "Dry soil / low vegetation",
        "0.30 – 0.45": "Moderate moisture",
        "0.45 – 0.60": "Wet soil / near saturation",
        "> 0.60": "Saturated soil or surface water",
    }
elif unidad == "rad":
    escala_titulo = "Interferometric Phase (rad)"
    escala_rango = "-3.14 to 3.14 rad"
    escalas = {
        "-π": [25, 65, 175],
        "-π/2": [138, 153, 211],
        "0": [245, 245, 245],
        "π/2": [199, 137, 130],
        "π": [150, 35, 25],
    }
    interpretacion = {
        "-π to -π/2": "Deformation toward satellite (subsidence)",
        "-π/2 to 0": "Slight deformation toward satellite",
        "0": "No detectable displacement",
        "0 to π/2": "Slight deformation away from satellite",
        "π/2 to π": "Deformation away from satellite (uplift)",
    }
else:
    escala_titulo = "Backscatter (dB)"
    escala_rango = "-25.0 to -5.0 dB"
    escalas = {
        "-25": [25, 25, 25],
        "-20": [75, 75, 75],
        "-15": [135, 135, 135],
        "-10": [195, 195, 195],
        "-5": [245, 245, 245],
    }
    interpretacion = {
        "< -20 dB": "Water / very smooth surfaces",
        "-20 to -17 dB": "Calm water / very wet soil",
        "-17 to -12 dB": "Low vegetation / wetlands",
        "-12 to -7 dB": "Forest / dense vegetation",
        "> -7 dB": "Structures / very rough surfaces",
    }

if es_modo_cientifico:
    c_img1, c_img2, c_diff, c_escala = st.columns([1, 1, 1, 0.78])
    with c_img1:
        st.subheader("Date 1 (Before)")
        st.image(img1_igual, caption=res1["metadatos"].get("zeroDopplerStartTime", escena_1.properties["startTime"]), width="stretch")
    with c_img2:
        st.subheader("Date 2 (After)")
        st.image(img2_igual, caption=res2["metadatos"].get("zeroDopplerStartTime", escena_2.properties["startTime"]), width="stretch")
    with c_diff:
        st.subheader("Difference (Common Area)")
        st.image(
            img_diff,
            caption=f"Blue: decreases ({unidad_label}) | Red: increases ({unidad_label})",
            width="stretch",
        )
    with c_escala:
        st.subheader("Color Scale")
        st.markdown(f"**{escala_titulo}**")
        st.caption(f"Range: {escala_rango}")
        colores_html = "".join(
            f'<div style="display:flex; align-items:center; margin:5px 0;">'
            f'<div style="width:95px; height:18px; background:rgb({c[0]},{c[1]},{c[2]}); border:1px solid #888; border-radius:3px;"></div>'
            f'<span style="margin-left:8px; font-family:monospace; font-size:13px;">{val}</span>'
            f"</div>"
            for val, c in escalas.items()
        )
        st.markdown(colores_html, unsafe_allow_html=True)
        st.markdown("**Difference**")
        diff_html = (
            '<div style="display:flex; align-items:center; margin:4px 0;">'
            '<div style="width:95px; height:16px; background:linear-gradient(90deg, rgb(20,100,250), rgb(240,240,245), rgb(250,55,30)); border:1px solid #888; border-radius:3px;"></div>'
            '<span style="margin-left:8px; font-size:12px;">- / +</span>'
            "</div>"
        )
        st.markdown(diff_html, unsafe_allow_html=True)

    st.subheader("Data Interpretation")
    filas_interp = [{"Range": rango, "Meaning": texto} for rango, texto in interpretacion.items()]
    st.dataframe(pd.DataFrame(filas_interp), width="stretch", hide_index=True)

    st.markdown("#### Calculated Variables")
    col_v1, col_v2, col_v3 = st.columns(3)
    with col_v1:
        st.metric("Date 1 Mean Value", f"{media_1:.3f} {unidad_label}")
        st.metric("Date 2 Mean Value", f"{media_2:.3f} {unidad_label}")
    with col_v2:
        st.metric("Mean Change", f"{cambio_medio:+.3f} {unidad_label}")
        st.metric("Date 1 Standard Deviation", f"{float(np.std(val1_co[m])):.3f} {unidad_label}")
    with col_v3:
        st.metric("Date 2 Standard Deviation", f"{float(np.std(val2_co[m])):.3f} {unidad_label}")
        st.metric("Coefficient of Variation", f"{(float(np.std(delta_vals[m])) / max(abs(cambio_medio), 1e-6)):.2f}")
else:
    c_img1, c_img2, c_diff = st.columns(3)
    with c_img1:
        st.subheader("Date 1 (Before)")
        st.image(img1_igual, caption=res1["metadatos"].get("zeroDopplerStartTime", escena_1.properties["startTime"]), width="stretch")
    with c_img2:
        st.subheader("Date 2 (After)")
        st.image(img2_igual, caption=res2["metadatos"].get("zeroDopplerStartTime", escena_2.properties["startTime"]), width="stretch")
    with c_diff:
        st.subheader("Difference (Common Area)")
        st.image(
            img_diff,
            caption=f"Blue: decreases ({unidad_label}) | Red: increases ({unidad_label})",
            width="stretch",
        )
    st.info(
        "**Intuitive Mode enabled**: You are viewing a simplified version of the map. "
        "The colors show where there is more or less water or moisture without needing technical knowledge of SAR radar."
    )

st.divider()


def renderizar_mapa():
    st.subheader("Map Location")
    poligono_azul_1 = escena_1.geometry["coordinates"][0]
    poligono_azul_2 = escena_2.geometry["coordinates"][0]

    if res2["geocodificado"] and len(xs_final) > 1 and len(ys_final) > 1:
        oeste, sur = proyeccion_a_latlon(float(xs_final.min()), float(ys_final.min()), res2["epsg"])
        este, norte = proyeccion_a_latlon(float(xs_final.max()), float(ys_final.max()), res2["epsg"])
    else:
        lons_p = [pt[0] for pt in poligono_azul_2]
        lats_p = [pt[1] for pt in poligono_azul_2]
        oeste, este, sur, norte = min(lons_p), max(lons_p), min(lats_p), max(lats_p)

    capas_mapa = [
        pdk.Layer(
            "PolygonLayer",
            data=[{"poligono": poligono_azul_1}],
            get_polygon="poligono",
            get_line_color=[100, 180, 255],
            get_fill_color=[100, 180, 255, 10],
            line_width_min_pixels=2,
        ),
        pdk.Layer(
            "PolygonLayer",
            data=[{"poligono": poligono_azul_2}],
            get_polygon="poligono",
            get_line_color=[30, 144, 255],
            get_fill_color=[30, 144, 255, 20],
            line_width_min_pixels=3,
        ),
    ]

    if np.any(valid_diff):
        celdas_raster = generar_celdas_raster_mapa(
            img_diff,
            valid_diff,
            res2["geocodificado"],
            xs_final,
            ys_final,
            res2["epsg"],
            (oeste, sur, este, norte),
            max_lado=75,
        )
        if celdas_raster:
            capas_mapa.append(
                pdk.Layer(
                    "PolygonLayer",
                    data=celdas_raster,
                    get_polygon="polygon",
                    get_fill_color="color",
                    stroked=False,
                    filled=True,
                )
            )

    capas_mapa.append(
        pdk.Layer(
            "PolygonLayer",
            data=[{"poligono": [[oeste, sur], [este, sur], [este, norte], [oeste, norte], [oeste, sur]]}],
            get_polygon="poligono",
            get_line_color=[255, 60, 60],
            get_fill_color=[0, 0, 0, 0],
            line_width_min_pixels=2,
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


if es_modo_cientifico:
    col_mapa, col_h5 = st.columns([1.3, 1.2])
    with col_mapa:
        renderizar_mapa()
    with col_h5:
        st.subheader("HDF5 (.h5) File Contents")
        tab_vars, tab_meta = st.tabs(["Variables (.h5)", "Metadata"])
        with tab_vars:
            st.dataframe(res2["estructura_h5"], width="stretch", height=340)
        with tab_meta:
            st.json(res2["metadatos"])
else:
    renderizar_mapa()

st.divider()
f1_str = escena_1.properties["startTime"][:10]
f2_str = escena_2.properties["startTime"][:10]
st.subheader(f"Results ({res2['producto']})")

t1_iso = res1["metadatos"].get("zeroDopplerStartTime") or escena_1.properties["startTime"]
t2_iso = res2["metadatos"].get("zeroDopplerStartTime") or escena_2.properties["startTime"]
try:
    dt_1 = datetime.strptime(str(t1_iso)[:19], "%Y-%m-%dT%H:%M:%S")
    dt_2 = datetime.strptime(str(t2_iso)[:19], "%Y-%m-%dT%H:%M:%S")
    dias_diff = (dt_2 - dt_1).total_seconds() / 86400.0
except Exception:
    dias_diff = 0.0

area_px_km2 = (res2["paso_x_efectivo"] * res2["paso_y_efectivo"]) / 1e6
n_px_m = int(np.sum(m))
area_m_km2 = n_px_m * area_px_km2

filas_magnitudes = [
    {
        "Metric": "Duration (days)",
        f"Date 1 ({f1_str})": f"Day 0.0 ({str(t1_iso)[11:16]} UTC)",
        f"Date 2 ({f2_str})": f"Day {dias_diff:.2f} ({str(t2_iso)[11:16]} UTC)",
        "Difference": f"{dias_diff:.2f} days",
    },
    {
        "Metric": "Common analyzed area",
        f"Date 1 ({f1_str})": f"{n_px_m:,} px",
        f"Date 2 ({f2_str})": f"{n_px_m:,} px",
        "Difference": f"{area_m_km2:.2f} km²",
    },
]

datos_graficas = []


def registrar_magnitud(etiqueta: str, unidad_mag: str, v1: float, v2: float, fmt_val: str, fmt_diff: str):
    sufijo = f" {unidad_mag}" if unidad_mag else ""
    filas_magnitudes.append(
        {
            "Metric": etiqueta,
            f"Date 1 ({f1_str})": f"{fmt_val.format(v1)}{sufijo}",
            f"Date 2 ({f2_str})": f"{fmt_val.format(v2)}{sufijo}",
            "Difference": f"{fmt_diff.format(v2 - v1)}{sufijo}",
        }
    )
    if np.isfinite(v1) and np.isfinite(v2):
        datos_graficas.append(
            {
                "Metric": etiqueta,
                "Unit": unidad_mag or "value",
                "Date 1": float(v1),
                "Date 2": float(v2),
                "Difference": float(v2 - v1),
            }
        )


if res2["producto"] == "GCOV":
    registrar_magnitud(f"Backscatter {res2['capa_co']} (dB)", "dB", media_1, media_2, "{:.2f}", "{:+.2f}")

    agua_1_km2 = float(np.sum(m & (val1_co < -17.0))) * area_px_km2
    agua_2_km2 = float(np.sum(m & (val2_co < -17.0))) * area_px_km2
    registrar_magnitud("Open water / inundation (km²)", "km²", agua_1_km2, agua_2_km2, "{:.2f}", "{:+.2f}")

    if d1_cr_c is not None and d2_cr_c is not None:
        v_cr = m & np.isfinite(d1_cr_c) & (d1_cr_c > 0) & np.isfinite(d2_cr_c) & (d2_cr_c > 0)
        db_cr1 = np.where(v_cr, 10.0 * np.log10(np.maximum(d1_cr_c, 1e-6)), np.nan)
        db_cr2 = np.where(v_cr, 10.0 * np.log10(np.maximum(d2_cr_c, 1e-6)), np.nan)

        m_cr1 = float(np.nanmean(db_cr1)) if np.any(v_cr) else float("nan")
        m_cr2 = float(np.nanmean(db_cr2)) if np.any(v_cr) else float("nan")
        registrar_magnitud(f"Backscatter {res2['capa_cr']} (dB)", "dB", m_cr1, m_cr2, "{:.2f}", "{:+.2f}")

        bosque_1_km2 = float(np.sum(v_cr & (db_cr1 > -16.0))) * area_px_km2
        bosque_2_km2 = float(np.sum(v_cr & (db_cr2 > -16.0))) * area_px_km2
        registrar_magnitud("Dense forest cover (km²)", "km²", bosque_1_km2, bosque_2_km2, "{:.2f}", "{:+.2f}")
        registrar_magnitud("Net forest change (ha)", "ha", bosque_1_km2 * 100.0, bosque_2_km2 * 100.0, "{:.1f}", "{:+.1f}")

        agri_1_km2 = float(np.sum(v_cr & (db_cr1 >= -21.0) & (db_cr1 <= -16.0))) * area_px_km2
        agri_2_km2 = float(np.sum(v_cr & (db_cr2 >= -21.0) & (db_cr2 <= -16.0))) * area_px_km2
        registrar_magnitud("Low / agricultural vegetation (km²)", "km²", agri_1_km2, agri_2_km2, "{:.2f}", "{:+.2f}")

        ok_rvi = v_cr & ((d1_co_c + d1_cr_c) > 0) & ((d2_co_c + d2_cr_c) > 0)
        rvi_1 = float(np.mean(np.clip((4.0 * d1_cr_c[ok_rvi]) / (d1_co_c[ok_rvi] + d1_cr_c[ok_rvi]), 0.0, 1.5))) if np.any(ok_rvi) else float("nan")
        rvi_2 = float(np.mean(np.clip((4.0 * d2_cr_c[ok_rvi]) / (d2_co_c[ok_rvi] + d2_cr_c[ok_rvi]), 0.0, 1.5))) if np.any(ok_rvi) else float("nan")
        registrar_magnitud("Radar Vegetation Index (RVI)", "", rvi_1, rvi_2, "{:.3f}", "{:+.3f}")

elif res2["producto"] == "SME2":
    registrar_magnitud(f"Soil moisture — {res2['algoritmo']} (m³/m³)", "m³/m³", media_1, media_2, "{:.4f}", "{:+.4f}")
    registrar_magnitud("Percentage moisture (%)", "%", media_1 * 100.0, media_2 * 100.0, "{:.2f}", "{:+.2f}")

    sat_comun_1_km2 = float(np.sum(m & (val1_co > 0.40))) * area_px_km2
    sat_comun_2_km2 = float(np.sum(m & (val2_co > 0.40))) * area_px_km2
    registrar_magnitud("Saturated area > 0.40 — common area (km²)", "km²", sat_comun_1_km2, sat_comun_2_km2, "{:.2f}", "{:+.2f}")

    corr_r = float(np.corrcoef(val1_co[m], val2_co[m])[0, 1]) if n_px_m > 1 else float("nan")
    filas_magnitudes.append(
        {
            "Metric": "Spatial correlation (r)",
            f"Date 1 ({f1_str})": "—",
            f"Date 2 ({f2_str})": f"{corr_r:.3f}",
            "Difference": f"{corr_r:.3f}",
        }
    )

    filas_magnitudes.append(
        {
            "Metric": "Saturated area > 0.40 — full scene (km²)",
            f"Date 1 ({f1_str})": f"{sat_tot_1_km2:.2f} km²",
            f"Date 2 ({f2_str})": f"{sat_tot_2_km2:.2f} km²",
            "Difference": "—",
        }
    )

    if hv_disponible:
        hv1_lin = float(np.mean(d1_ex_c[mh]))
        hv2_lin = float(np.mean(d2_ex_c[mh]))
        hv1_db = 10.0 * math.log10(hv1_lin)
        hv2_db = 10.0 * math.log10(hv2_lin)
        registrar_magnitud(
            f"HV Backscatter — {res2['freq_hv_label']} (dB)",
            "dB",
            hv1_db,
            hv2_db,
            "{:.2f}",
            "{:+.2f}",
        )

elif res2["producto"] in ("GUNW", "RUNW"):
    factor_cm = -24.0 / (4.0 * math.pi)
    desp1_cm = val1_co * factor_cm
    desp2_cm = val2_co * factor_cm
    m_desp1 = float(np.mean(desp1_cm[m]))
    m_desp2 = float(np.mean(desp2_cm[m]))

    registrar_magnitud("Interferometric phase (rad)", "rad", media_1, media_2, "{:.2f}", "{:+.2f}")
    registrar_magnitud("Vertical / LOS displacement (cm)", "cm", m_desp1, m_desp2, "{:.2f}", "{:+.2f}")

    anios = max(dias_diff / 365.25, 12.0 / 365.25)
    vel_1_mm = (m_desp1 * 10.0) / (12.0 / 365.25)
    vel_2_mm = (m_desp2 * 10.0) / anios
    registrar_magnitud("Deformation velocity (mm/yr)", "mm/yr", vel_1_mm, vel_2_mm, "{:.1f}", "{:+.1f}")

    ext_def1 = float(np.sum(m & (np.abs(desp1_cm) > 2.0))) * area_px_km2
    ext_def2 = float(np.sum(m & (np.abs(desp2_cm) > 2.0))) * area_px_km2
    registrar_magnitud("Deformed area > 2 cm (km²)", "km²", ext_def1, ext_def2, "{:.2f}", "{:+.2f}")

    paso_km = max(res2["paso_x_efectivo"] / 1000.0, 0.01)
    gy1, gx1 = np.gradient(np.nan_to_num(desp1_cm * 10.0, nan=0.0), paso_km)
    gy2, gx2 = np.gradient(np.nan_to_num(desp2_cm * 10.0, nan=0.0), paso_km)
    grad1 = float(np.mean(np.hypot(gx1[m], gy1[m])))
    grad2 = float(np.mean(np.hypot(gx2[m], gy2[m])))
    registrar_magnitud("Spatial gradient (mm/km)", "mm/km", grad1, grad2, "{:.2f}", "{:+.2f}")

    if d1_ex_c is not None and d2_ex_c is not None:
        vc = m & np.isfinite(d1_ex_c) & (d1_ex_c >= 0.0) & (d1_ex_c <= 1.0) & np.isfinite(d2_ex_c) & (d2_ex_c >= 0.0) & (d2_ex_c <= 1.0)
        if np.any(vc):
            coh1 = float(np.mean(d1_ex_c[vc]))
            coh2 = float(np.mean(d2_ex_c[vc]))
            registrar_magnitud("Temporal coherence", "", coh1, coh2, "{:.3f}", "{:+.3f}")

else:
    registrar_magnitud(f"Backscatter {res2['capa_co']} (dB)", "dB", media_1, media_2, "{:.2f}", "{:+.2f}")

    agua_1_km2 = float(np.sum(m & (val1_co < -17.0))) * area_px_km2
    agua_2_km2 = float(np.sum(m & (val2_co < -17.0))) * area_px_km2
    registrar_magnitud("Water / low roughness (km²)", "km²", agua_1_km2, agua_2_km2, "{:.2f}", "{:+.2f}")

    denso_1_km2 = float(np.sum(m & (val1_co > -10.0))) * area_px_km2
    denso_2_km2 = float(np.sum(m & (val2_co > -10.0))) * area_px_km2
    registrar_magnitud("Structures / dense cover (km²)", "km²", denso_1_km2, denso_2_km2, "{:.2f}", "{:+.2f}")

tab_tabla, tab_graficas = st.tabs(["Table", "Charts"])

with tab_tabla:
    explicaciones_filas = {
        "duration": "Exact elapsed time between the satellite's first and second acquisition.",
        "common analyzed area": "Exact surface area where both satellite images overlap and can be compared.",
        "hv backscatter": "Cross-polarized radar signal; helps interpret soil moisture and vegetation.",
        "backscatter hvhv": "Cross-polarized radar signal (HV/VH). Useful for detecting vegetation and moisture.",
        "backscatter": "How strongly the satellite radar signal bounces off the surface. Helps distinguish water, forest, or urban areas.",
        "open water": "Square kilometers covered by open water on that date.",
        "dense forest cover": "Area with thick forest (tall, dense vegetation).",
        "net forest change": "How many hectares of forest were gained or lost between the two dates.",
        "agricultural vegetation": "Land covered with crops or low grass rather than dense forest.",
        "rvi": "Index from 0 to 1 indicating the amount of green vegetation in the area.",
        "soil moisture": "Amount of water held within the soil, similar to water content in a sponge.",
        "percentage moisture": "Same soil moisture measurement expressed as a percentage (%).",
        "saturated area > 0.40 — common area": "Terrain within the common overlap area where the soil can no longer absorb more water.",
        "spatial correlation": "How similar the spatial moisture pattern is between both dates (close to 1 = very stable).",
        "full scene": "Total saturated area across the entire individual capture before cropping to the common overlap.",
        "interferometric phase": "Change in radar phase between two dates. Indicates ground movement.",
        "displacement": "How many centimeters the ground moved upward or downward.",
        "deformation velocity": "How fast the ground is subsiding or uplifting (mm per year).",
        "deformed area": "Ground surface area that moved more than 2 centimeters between dates.",
        "spatial gradient": "How rapidly ground movement changes from one point to its neighbor.",
        "temporal coherence": "How similar the terrain looks on both dates. Close to 1 = minimal change.",
        "low roughness": "Area covered by water or very smooth surfaces.",
        "structures": "Area with very dense forest or buildings/urban structures.",
    }

    def obtener_explicacion(nombre_fila: str) -> str:
        nombre_low = nombre_fila.lower()
        for clave, texto in explicaciones_filas.items():
            if clave in nombre_low:
                return texto
        return "Data measured by the NISAR radar to study changes in the area."

    if not es_modo_cientifico:
        filas_html = "".join(
            f'<tr style="background:#ffffff; color:#1a2e3a;">'
            f'<td style="padding:8px; border-bottom:1px solid #eee; font-weight:600; color:#1a2e3a;">{fila.get("Metric", "")}</td>'
            f'<td style="padding:8px; border-bottom:1px solid #eee; text-align:center;">{fila.get(f"Date 1 ({f1_str})", "")}</td>'
            f'<td style="padding:8px; border-bottom:1px solid #eee; text-align:center;">{fila.get(f"Date 2 ({f2_str})", "")}</td>'
            f'<td style="padding:8px; border-bottom:1px solid #eee; text-align:center; color:#e74c3c; font-weight:bold;">{fila.get("Difference", "")}</td>'
            f'<td style="padding:8px; border-bottom:1px solid #eee; text-align:center;">'
            f'<span title="{obtener_explicacion(fila.get("Metric", ""))}" style="display:inline-block; width:20px; height:20px; background:#1d7aad; color:white; border-radius:50%; text-align:center; line-height:20px; font-size:14px; cursor:help;">ⓘ</span>'
            f"</td></tr>"
            for fila in filas_magnitudes
        )

        tabla_html = (
            '<div style="margin-top:12px;">'
            '<table style="width:100%; border-collapse:collapse; font-family:Segoe UI, sans-serif; font-size:14px; background:#ffffff; border-radius:8px; overflow:hidden;">'
            "<thead>"
            '<tr style="background:#f0f7fa; color:#0f4c5c;">'
            '<th style="padding:10px; text-align:left; border-bottom:2px solid #1d7aad;">Metric</th>'
            f'<th style="padding:10px; text-align:center; border-bottom:2px solid #1d7aad;">Date 1 ({f1_str})</th>'
            f'<th style="padding:10px; text-align:center; border-bottom:2px solid #1d7aad;">Date 2 ({f2_str})</th>'
            '<th style="padding:10px; text-align:center; border-bottom:2px solid #1d7aad;">Difference</th>'
            '<th style="padding:10px; text-align:center; border-bottom:2px solid #1d7aad;">ⓘ Info</th>'
            "</tr>"
            "</thead>"
            f"<tbody>{filas_html}</tbody>"
            "</table>"
            '<p style="font-size:12px; color:#888; margin-top:8px;">Hover over the ⓘ icon on each row to read what that metric means.</p>'
            "</div>"
        )
        st.subheader(f"Physical Magnitudes Table ({res2['producto']})")
        st.markdown(tabla_html, unsafe_allow_html=True)
    else:
        df_magnitudes = pd.DataFrame(filas_magnitudes)
        st.dataframe(df_magnitudes, width="stretch", hide_index=True)

with tab_graficas:
    if datos_graficas:
        cols_g = st.columns(2)
        etiq_f1 = f"Date 1 ({f1_str})"
        etiq_f2 = f"Date 2 ({f2_str})"

        for idx, item in enumerate(datos_graficas):
            df_item = pd.DataFrame(
                [
                    {"Date": etiq_f1, "Value": item["Date 1"], "Label": f"{item['Date 1']:.3f}"},
                    {"Date": etiq_f2, "Value": item["Date 2"], "Label": f"{item['Date 2']:.3f}"},
                ]
            )
            color_linea = "#ff5a36" if item["Difference"] >= 0 else "#38bdf8"

            base = alt.Chart(df_item).encode(
                x=alt.X(
                    "Date:N",
                    sort=[etiq_f1, etiq_f2],
                    title=None,
                    axis=alt.Axis(labelAngle=0),
                    scale=alt.Scale(paddingInner=0.35),
                ),
                y=alt.Y("Value:Q", title=item["Unit"], scale=alt.Scale(zero=False, padding=15)),
            )

            linea = base.mark_line(color=color_linea, strokeWidth=3)
            puntos = base.mark_point(size=120, filled=True).encode(
                color=alt.Color(
                    "Date:N",
                    scale=alt.Scale(domain=[etiq_f1, etiq_f2], range=["#38bdf8", "#ff5a36"]),
                    legend=None,
                ),
                tooltip=["Date", alt.Tooltip("Value:Q", format=".4f")],
            )
            textos = base.mark_text(dy=-14, fontSize=12, fontWeight="bold", color="#e2e8f0").encode(
                text="Label:N"
            )

            chart_line = (linea + puntos + textos).properties(title=item["Metric"], height=240)
            with cols_g[idx % 2]:
                st.altair_chart(chart_line, use_container_width=True)

        v1_flat = val1_co[m]
        v2_flat = val2_co[m]
        lo = float(min(np.nanpercentile(v1_flat, 1), np.nanpercentile(v2_flat, 1)))
        hi = float(max(np.nanpercentile(v1_flat, 99), np.nanpercentile(v2_flat, 99)))
        if hi > lo:
            bins = np.linspace(lo, hi, 31)
            centros = 0.5 * (bins[:-1] + bins[1:])
            h1, _ = np.histogram(v1_flat, bins=bins, density=True)
            h2, _ = np.histogram(v2_flat, bins=bins, density=True)
            df_hist = pd.concat(
                [
                    pd.DataFrame({"Value": centros, "Density": h1, "Scene": etiq_f1}),
                    pd.DataFrame({"Value": centros, "Density": h2, "Scene": etiq_f2}),
                ],
                ignore_index=True,
            )
            chart_dist = (
                alt.Chart(df_hist)
                .mark_line(strokeWidth=2.5, point=alt.OverlayMarkDef(size=35, filled=True))
                .encode(
                    x=alt.X("Value:Q", title=f"{res2['capa_co']} ({unidad_label})"),
                    y=alt.Y("Density:Q", title="Pixel density"),
                    color=alt.Color(
                        "Scene:N",
                        scale=alt.Scale(domain=[etiq_f1, etiq_f2], range=["#38bdf8", "#ff5a36"]),
                    ),
                    tooltip=["Scene", alt.Tooltip("Value:Q", format=".3f"), alt.Tooltip("Density:Q", format=".4f")],
                )
                .properties(title="Common Pixel Distribution", height=260)
            )
            st.altair_chart(chart_dist, use_container_width=True)
    else:
        st.info("Not enough data to plot.")

st.divider()

referencias_html = (
    '<div style="background: linear-gradient(135deg, #161f2c 0%, #1b2838 100%); border: 1px solid #2b3e54; border-left: 5px solid #38bdf8; border-radius: 10px; padding: 20px; margin-top: 20px; font-family: Segoe UI, sans-serif;">'
    '<h3 style="color: #38bdf8; margin-top: 0;">References and Data Sources</h3>'
    '<p style="color: #e2e8f0; font-size: 15px; margin-bottom: 16px;">'
    "This application uses <strong>NISAR</strong> satellite radar data processed through the "
    "<strong>Alaska Satellite Facility (ASF)</strong> and <strong>NASA Earthdata</strong>. Below are the consulted sources:"
    "</p>"
    '<ul style="color: #cbd5e1; font-size: 14px; line-height: 1.8; padding-left: 20px;">'
    "<li>"
    "<strong>Alaska Satellite Facility (ASF) – NISAR Data:</strong><br>"
    '<a href="https://search.asf.alaska.edu/#/?zoom=7.917&center=-83.868,13.395&polygon=POLYGON((-84.3193%2014.5298,-83.7678%2014.5298,-83.7678%2014.8763,-84.3193%2014.8763,-84.3193%2014.5298))&resultsLoaded=true&granule=NISAR_L3_PR_SME2_028_004_A_009_4005_DHDH_A_20260813T110615_20260813T110650_P05023_N_F_J_001&start=2026-05-20T06:00:00Z&end=2026-11-07T05:59:59Z&dataset=NISAR&prodConfig=PR&sciProducts=SME2" '
    'target="_blank" style="color: #7dd3fc; text-decoration: none; font-weight: 600;">'
    "View NISAR scene on ASF ↗"
    "</a>"
    "</li>"
    "<li>"
    "<strong>NASA NISAR Mission:</strong> "
    '<a href="https://nisar.jpl.nasa.gov/" target="_blank" style="color: #7dd3fc; text-decoration: none;">'
    "Official NISAR Mission Website (NASA/JPL) ↗"
    "</a>"
    "</li>"
    "<li>"
    "<strong>NASA Earthdata:</strong> "
    '<a href="https://earthdata.nasa.gov/" target="_blank" style="color: #7dd3fc; text-decoration: none;">'
    "NASA Open Data Platform ↗"
    "</a>"
    "</li>"
    "<li>"
    "<strong>Study Area in Honduras:</strong> Area selected for the <em>Dancing with the SARs</em> challenge at the "
    "<strong>NASA Space Apps Challenge</strong>, focused on water, wetlands, soil moisture, and flooding."
    "</li>"
    "<li>"
    "<strong>Development:</strong> Interactive web interface built with <strong>Python + Streamlit</strong> for the NISAR challenge."
    "</li>"
    "</ul>"
    '<p style="color: #94a3b8; font-size: 12px; margin-top: 16px; border-top: 1px solid #2b3e54; padding-top: 12px;">'
    "<strong>Note:</strong> SAR radar data requires technical processing. This application transforms it "
    "into accessible visualizations for both the general public and the scientific community."
    "</p>"
    "</div>"
)

st.markdown(referencias_html, unsafe_allow_html=True)