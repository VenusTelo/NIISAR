import os
import math
import streamlit as st
import asf_search as asf
import earthaccess
import h5py
import numpy as np
import pydeck as pdk
from PIL import Image
from google import genai

st.set_page_config(page_title="Monitor NISAR - Prototipo Alpha + IA", layout="wide")
st.title("Monitor NISAR — Inspección .h5 y Detección de Cambios")

LUGARES_PREDEFINIDOS = {
    "Gracias a Dios / Laguna de Caratasca (Trayecto 4, Marco 9)": (15.1500, -83.9000, 4, 9),
    "Costa de Trujillo y Colón (Trayecto 105, Marco 10)": (15.8500, -85.8500, 105, 10),
    "Ubicación personalizada (Lat / Lon)": (15.1500, -83.9000, 0, 0),
}

if "lat_punto" not in st.session_state:
    st.session_state.lat_punto = 15.1500
if "lon_punto" not in st.session_state:
    st.session_state.lon_punto = -83.9000


def latlon_a_utm(lat_deg: float, lon_deg: float, epsg: int):
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


def utm_a_latlon(x: float, y: float, epsg: int):
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


def normalizar_db(arr: np.ndarray):
    valid = np.isfinite(arr) & (arr > 0)
    db = np.where(valid, 10 * np.log10(np.maximum(arr, 1e-6)), np.nan)
    if not np.any(valid):
        return db, np.full(arr.shape, 25, dtype=np.uint8), valid
    p2, p98 = np.nanpercentile(db, (2, 98))
    norm = np.nan_to_num(np.clip((db - p2) / (p98 - p2 + 1e-6) * 255, 0, 255), nan=25).astype(np.uint8)
    return db, norm, valid


def colorear_diferencia(delta_db: np.ndarray, rango_max: float = 8.0):
    h, w = delta_db.shape
    rgb = np.full((h, w, 3), 25, dtype=np.uint8)
    valid = np.isfinite(delta_db)
    norm = np.clip(delta_db / rango_max, -1.0, 1.0)

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


with st.sidebar:
    st.header("1. Cuenta NASA Earthdata")
    usuario = st.text_input("Usuario Earthdata", value="artur28")
    password = st.text_input("Contraseña Earthdata", type="password")

    st.divider()
    st.header("2. Zona y Escala de Análisis")
    preset = st.selectbox("Zona base", list(LUGARES_PREDEFINIDOS.keys()))
    lat_def, lon_def, orb_def, frame_def = LUGARES_PREDEFINIDOS[preset]

    if "ultimo_preset" not in st.session_state:
        st.session_state.ultimo_preset = preset
    elif st.session_state.ultimo_preset != preset:
        st.session_state.ultimo_preset = preset
        st.session_state.lat_punto = lat_def
        st.session_state.lon_punto = lon_def

    lat = st.number_input("Latitud (Norte + / Sur -)", key="lat_punto", step=0.02, format="%.4f")
    lon = st.number_input("Longitud (Este + / Oeste -)", key="lon_punto", step=0.02, format="%.4f")

    ancho_km = st.slider(
        "Área de medición (2 km = zona específica | 250 km = todo el mapa)",
        min_value=2,
        max_value=250,
        value=10,
        step=2,
    )
    if ancho_km >= 250:
        st.caption("Modo activo: **Todo el mapa / marco completo (malla rápida 80 m)**")
    elif ancho_km >= 80:
        st.caption(f"Modo activo: **Vista regional {ancho_km} × {ancho_km} km (malla rápida 80 m)**")
    else:
        st.caption(f"Modo activo: **Zona específica {ancho_km} × {ancho_km} km (alta resolución 20 m)**")

    trayecto = st.number_input("Trayecto (0 = cualquiera)", min_value=0, value=orb_def, step=1)
    marco = st.number_input("Marco (0 = cualquiera)", min_value=0, value=frame_def, step=1)
    nivel = st.selectbox("Producto NISAR", ["GCOV", "GSLC"])

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
    if orb_num > 0:
        kwargs["relativeOrbit"] = int(orb_num)
    if frame_num > 0:
        kwargs["frame"] = int(frame_num)
    resultados = asf.geo_search(**kwargs)
    return sorted(resultados, key=lambda r: r.properties["startTime"])


@st.cache_data(ttl=3600, show_spinner=False)
def leer_bloque_h5(url_h5, user_ed, pass_ed, prod_level, lat_c, lon_c, tam_km):
    os.environ["EARTHDATA_USERNAME"] = user_ed
    os.environ["EARTHDATA_PASSWORD"] = pass_ed
    earthaccess.login(strategy="environment")

    archivo_remoto = earthaccess.open([url_h5])[0]
    with h5py.File(archivo_remoto, "r") as h5:
        grids = h5[f"/science/LSAR/{prod_level}/grids"]
        usar_freq_b = (tam_km >= 80) and ("frequencyB" in grids)
        freq_elegida = "frequencyB" if usar_freq_b else "frequencyA"
        grupo = grids[freq_elegida]
        capas = list(grupo.keys())

        capa_co = next((c for c in ["HHHH", "VVVV", "HH", "VV"] if c in capas), capas[0])
        capa_cr = next((c for c in ["HVHV", "VHVH", "HV", "VH"] if c in capas), None)

        xs = grupo["xCoordinates"][:]
        ys = grupo["yCoordinates"][:]
        epsg = (
            int(grupo["projection"][()])
            if "projection" in grupo
            else int(grids["frequencyA"]["projection"][()])
        )
        paso_m = abs(float(xs[1] - xs[0])) if len(xs) > 1 else (80.0 if usar_freq_b else 20.0)

        if tam_km >= 250 and usar_freq_b:
            x0, x1 = 0, len(xs)
            y0, y1 = 0, len(ys)
        else:
            tam_seguro = min(tam_km, 90) if not usar_freq_b else tam_km
            x_utm, y_utm = latlon_a_utm(lat_c, lon_c, epsg)
            ix = int(np.argmin(np.abs(xs - x_utm)))
            iy = int(np.argmin(np.abs(ys - y_utm)))

            radio_px = max(15, int((tam_seguro * 1000.0 / 2.0) / paso_m))
            x0, x1 = max(0, ix - radio_px), min(len(xs), ix + radio_px)
            y0, y1 = max(0, iy - radio_px), min(len(ys), iy + radio_px)

        if x1 <= x0 or y1 <= y0:
            return None

        stride = max(1, max(x1 - x0, y1 - y0) // 450)

        datos_co = grupo[capa_co][y0:y1:stride, x0:x1:stride]
        if np.iscomplexobj(datos_co):
            datos_co = np.abs(datos_co)

        datos_cr = None
        if capa_cr is not None:
            datos_cr = grupo[capa_cr][y0:y1:stride, x0:x1:stride]
            if np.iscomplexobj(datos_cr):
                datos_cr = np.abs(datos_cr)

        estructura_h5 = []

        def recolectar(nombre, obj):
            if isinstance(obj, h5py.Dataset):
                estructura_h5.append(
                    {
                        "Ruta interna": f"/{nombre}",
                        "Dimensiones": str(obj.shape),
                        "Tipo": str(obj.dtype),
                    }
                )

        h5.visititems(recolectar)

        ident = h5["/science/LSAR/identification"]
        metadatos = {}
        for k in ident.keys():
            if isinstance(ident[k], h5py.Dataset) and ident[k].shape == ():
                val = ident[k][()]
                metadatos[k] = val.decode() if isinstance(val, (bytes, np.bytes_)) else str(val)

        oeste, sur = utm_a_latlon(float(xs[x0]), float(min(ys[y0], ys[y1 - 1])), epsg)
        este, norte = utm_a_latlon(float(xs[x1 - 1]), float(max(ys[y0], ys[y1 - 1])), epsg)

        return {
            "datos_co": datos_co,
            "datos_cr": datos_cr,
            "capa_co": capa_co,
            "capa_cr": capa_cr,
            "frecuencia": freq_elegida,
            "epsg": epsg,
            "bounds": [oeste, sur, este, norte],
            "paso_m_efectivo": paso_m * stride,
            "estructura_h5": estructura_h5,
            "metadatos": metadatos,
        }


escenas = buscar_escenas_punto(nivel, lat, lon, trayecto, marco)
if not escenas:
    st.warning("El punto salió del cuadro azul actual. Regresa el valor de Latitud/Longitud o pon Trayecto y Marco en 0.")
    st.stop()

opciones = {
    f"{e.properties['startTime'][:10]} | Órbita {e.properties.get('orbit', '?')} ({e.properties['sceneName'][-15:]})": e
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

etiqueta_carga = "todo el marco completo (frequencyB)" if ancho_km >= 250 else f"zona de {ancho_km}×{ancho_km} km"
with st.spinner(f"Leyendo {etiqueta_carga} desde Earthdata Cloud..."):
    res1 = leer_bloque_h5(escena_1.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    res2 = (
        res1
        if sel_f1 == sel_f2
        else leer_bloque_h5(escena_2.properties["url"], usuario, password, nivel, lat, lon, ancho_km)
    )

if res1 is None or res2 is None:
    st.warning("Las coordenadas ingresadas quedaron fuera del área de datos de esta escena.")
    st.stop()

db1_co, r1, val1 = normalizar_db(res1["datos_co"])
db2_co, r2, val2 = normalizar_db(res2["datos_co"])

if res1["datos_cr"] is not None:
    _, g1, _ = normalizar_db(res1["datos_cr"])
    img1_rgb = np.dstack([r1, g1, r1])
else:
    img1_rgb = np.dstack([r1, r1, r1])

if res2["datos_cr"] is not None:
    _, g2, _ = normalizar_db(res2["datos_cr"])
    img2_rgb = np.dstack([r2, g2, r2])
else:
    img2_rgb = np.dstack([r2, r2, r2])

min_h = min(db1_co.shape[0], db2_co.shape[0])
min_w = min(db1_co.shape[1], db2_co.shape[1])

img1_igual = img1_rgb[:min_h, :min_w]
img2_igual = img2_rgb[:min_h, :min_w]
delta_db = db2_co[:min_h, :min_w] - db1_co[:min_h, :min_w]
img_diff = colorear_diferencia(delta_db, rango_max=8.0)

valid_diff = np.isfinite(delta_db)
media_1 = float(np.nanmean(db1_co)) if np.any(val1) else float("nan")
media_2 = float(np.nanmean(db2_co)) if np.any(val2) else float("nan")
cambio_medio = media_2 - media_1
pct_cambio_fuerte = float(np.mean(np.abs(delta_db[valid_diff]) > 3.0) * 100) if np.any(valid_diff) else 0.0

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"Media Fecha 1 ({res1['capa_co']})", f"{media_1:.2f} dB")
m2.metric(f"Media Fecha 2 ({res2['capa_co']})", f"{media_2:.2f} dB", delta=f"{cambio_medio:+.2f} dB")
m3.metric("Área con cambio > 3 dB", f"{pct_cambio_fuerte:.1f}%")
etiqueta_escala = (
    f"Marco completo ({res2['frecuencia']})"
    if ancho_km >= 250
    else f"{ancho_km} × {ancho_km} km ({res2['frecuencia']})"
)
m4.metric("Escala activa", etiqueta_escala)

c_img1, c_img2, c_diff = st.columns(3)
with c_img1:
    st.subheader("Fecha 1 (Antes)")
    st.image(img1_igual, caption=escena_1.properties["startTime"], width='stretch')
with c_img2:
    st.subheader("Fecha 2 (Después)")
    st.image(img2_igual, caption=escena_2.properties["startTime"], width='stretch')
with c_diff:
    st.subheader("Diferencia (Fecha 2 - Fecha 1)")
    st.image(
        img_diff,
        caption="Azul: baja señal (agua/inundación) | Rojo: sube señal (vegetación/estructura)",
        width='stretch',
    )

st.divider()
col_mapa, col_h5 = st.columns([1.3, 1.2])

with col_mapa:
    st.subheader("Ubicación del Cuadro Azul y Cambios en el Mapa")
    oeste, sur, este, norte = res2["bounds"]
    poligono_azul = escena_2.geometry["coordinates"][0]

    paso_sub = max(1, max(min_h, min_w) // 85)
    sub_diff = img_diff[::paso_sub, ::paso_sub]
    sub_val = valid_diff[::paso_sub, ::paso_sub]
    lons_grid = np.linspace(oeste, este, min_w)[::paso_sub]
    lats_grid = np.linspace(norte, sur, min_h)[::paso_sub]

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
        st.dataframe(res2["estructura_h5"], width='stretch', height=340)
    with tab_meta:
        st.json(res2["metadatos"])

# --- MÓDULO DE ANÁLISIS CON AGENTE IA (GOOGLE AI STUDIO) ---
st.divider()
st.subheader("Asistente IA de Interpretación SAR (Google AI Studio)")

col_api, col_preg = st.columns([1, 2])
with col_api:
    gemini_key = st.text_input(
        "API Key de Google AI Studio",
        value=os.environ.get("GEMINI_API_KEY", ""),
        type="password",
        placeholder="AQ.Ab8...",
    )
with col_preg:
    pregunta_usuario = st.text_input(
        "Consulta para el agente",
        value="Analiza los cambios físicos entre ambas fechas, identifica posibles zonas inundadas o cambios de cobertura y resume los hallazgos.",
    )

if st.button("Analizar imágenes y datos con Gemini", type="primary"):
    if not gemini_key:
        st.warning("Pega tu API Key de Google AI Studio en el recuadro izquierdo.")
    else:
        with st.spinner("Analizando imágenes y datos NISAR con Gemini..."):
            try:
                client = genai.Client(api_key=gemini_key)

                pil_antes = Image.fromarray(img1_igual)
                pil_despues = Image.fromarray(img2_igual)
                pil_diff = Image.fromarray(img_diff)

                resumen_datos = f"""
                Eres un especialista en teledetección radar SAR (misión NASA-ISRO NISAR).
                Analiza las 3 imágenes adjuntas en orden:
                1) Fecha 1 (Antes): {escena_1.properties['startTime']}
                2) Fecha 2 (Después): {escena_2.properties['startTime']}
                3) Diferencia (Fecha 2 - Fecha 1): Azul = baja señal (agua/inundación), Rojo = sube señal (vegetación/estructura).

                Datos técnicos medidos en el archivo .h5:
                - Producto NISAR: {nivel} ({res2['frecuencia']})
                - Coordenadas centrales: Latitud {lat:.4f}, Longitud {lon:.4f} (Área: {ancho_km} km)
                - Polarización / Capa: {res2['capa_co']}
                - Media Fecha 1: {media_1:.2f} dB
                - Media Fecha 2: {media_2:.2f} dB
                - Cambio medio: {cambio_medio:+.2f} dB
                - Porcentaje de área con cambio > 3 dB: {pct_cambio_fuerte:.1f}%

                Instrucción del usuario: {pregunta_usuario}
                Responde de forma técnica, clara y estructurada en español.
                """

                respuesta = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=[resumen_datos, pil_antes, pil_despues, pil_diff],
                )
                st.markdown(respuesta.text)
            except Exception as err:
                st.error(f"Error al conectar con Google AI Studio: {err}")

