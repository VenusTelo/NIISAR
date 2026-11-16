"""WAVE NISAR Monitor - backend Flask.

Sirve la interfaz web y expone la API de análisis de datos NISAR (GCOV)
para Honduras. Los textos que ve el usuario están en inglés; los
comentarios y el resto del código, en español.
"""

import json
import logging
import os
import warnings
from datetime import datetime

import asf_search as asf
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_caching import Cache

from gcov_analysis import (
    AnalysisError, analyze_area, available_area_dates,
    fetch_area_observations, validate_area, validate_dates,
)

# Configuración general
warnings.filterwarnings("ignore", category=RuntimeWarning)
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

APP_NAME = "WAVE NISAR Monitor"
APP_VERSION = "1.0.0"

# Límites geográficos de Honduras
HONDURAS_BBOX = "POLYGON((-89.4 12.9,-83.1 12.9,-83.1 16.6,-89.4 16.6,-89.4 12.9))"

# Tamaño máximo (bytes) aceptado en las peticiones POST
MAX_REQUEST_BYTES = 100_000

# Productos NISAR con descripciones para el usuario (texto visible en inglés)
NISAR_PRODUCTS = {
    "GCOV": {
        "name": "Backscatter & Water/Vegetation",
        "level": "Level 2",
        "description": "Measures how radar bounces off water and plants. Useful for detecting changes in forests, crops and water bodies.",
        "tech": "Backscatter Radar Crossover",
        "icon": "🌳",
    },
    "GUNW": {
        "name": "Interferometry & Ground Motion",
        "level": "Level 2",
        "description": "Detects millimeter-scale ground movement. Identifies landslides, subsidence and deformation.",
        "tech": "Interferometric Phase & Coherence",
        "icon": "📍",
    },
    "SME2": {
        "name": "Soil Moisture",
        "level": "Level 3",
        "description": "Measures soil moisture (0-5 cm). Critical for agriculture, flood forecasting and drought monitoring.",
        "tech": "Soil Moisture Estimation (m³/m³)",
        "icon": "💧",
    },
    "GSLC": {
        "name": "Geocoded Complex Resolution",
        "level": "Level 2",
        "description": "High-resolution radar data ready for analysis. Basis for derived products.",
        "tech": "Complex SAR Single Look Complex - Geocoded",
        "icon": "🔬",
    },
    "RSLC": {
        "name": "Raw Radar Resolution",
        "level": "Level 1",
        "description": "Unprocessed SAR data. For radar processing experts.",
        "tech": "SAR Single Look Complex - Slant Range",
        "icon": "📡",
    },
    "GOFF": {
        "name": "Geocoded Phase Offset",
        "level": "Level 2",
        "description": "Offset between radar passes. Detects terrain changes between observations.",
        "tech": "Phase Offset Field",
        "icon": "🔄",
    },
    "ROFF": {
        "name": "Raw Phase Offset",
        "level": "Level 1",
        "description": "Offset in native radar geometry.",
        "tech": "Phase Offset Field - Slant Range",
        "icon": "📊",
    },
}

# Sitios de referencia en Honduras (nombre visible, (lat, lon))
REFERENCE_SITES = {
    "Gulf of Fonseca (Shrimp/Water)": (13.2, -87.6),
    "Aguán Valley (African Palm)": (15.4, -86.3),
    "Mosquitia (Forest Frontier)": (15.6, -84.7),
    "Olancho (Cattle/Forest)": (14.7, -86.2),
    "Sula Valley (Sugar/Banana)": (15.5, -88.0),
    "Tegucigalpa (Urban)": (14.07, -87.2),
    "El Cajón Dam": (14.9, -87.7),
}

# Diccionario de términos técnicos (explicación visible en inglés)
TECH_TRANSLATIONS = {
    "HH": "Horizontal polarization (smooth surfaces, water, double bounce in flooded vegetation)",
    "HV": "Cross polarization (responds to vegetation structure)",
    "VV": "Vertical-Vertical polarization (low incidence angles)",
    "VH": "Vertical-Horizontal cross polarization",
    "dB": "Decibels (logarithmic energy scale)",
    "phase": "Phase (wave difference between passes)",
    "coherence": "Coherence (similarity between passes)",
    "incidence_angle": "Incidence angle (angle between the radar beam and the ground)",
    "SAR": "Synthetic Aperture Radar",
    "geocoded": "Converted to lat/lon coordinates",
    "slant_range": "Native radar geometry (not converted)",
    "interferometry": "Technique that compares two radar passes to detect changes",
    "backscatter": "Radar energy reflected back to the satellite",
}

# ============================================================================
# APLICACIÓN FLASK Y CACHÉS
# ============================================================================

app = Flask(__name__)

cache = Cache(app, config={"CACHE_TYPE": "simple"})
analysis_cache = Cache(app, config={
    "CACHE_TYPE": "FileSystemCache",
    "CACHE_DIR": os.path.join(os.path.dirname(__file__), ".analysis-cache"),
    "CACHE_THRESHOLD": 100,
    "CACHE_DEFAULT_TIMEOUT": 3600,
})

# ============================================================================
# INVENTARIO NISAR
# ============================================================================

@cache.memoize(timeout=3600)
def fetch_nisar_inventory(product, max_results=100):
    """Consulta ASF y devuelve los granules de un producto sobre Honduras.

    Si ASF falla, la excepción se propaga: así el error no se guarda en caché.
    """
    results = asf.search(
        dataset="NISAR",
        processingLevel=product,
        intersectsWith=HONDURAS_BBOX,
        maxResults=max_results,
    )
    frames = []
    for result in results:
        props = result.properties

        # Tamaño del archivo .h5 principal (ignora los archivos QA_STATS)
        file_size_mb = None
        if isinstance(props.get("bytes"), dict):
            for name, info in props["bytes"].items():
                if name.endswith(".h5") and "QA_STATS" not in name:
                    file_size_mb = info.get("bytes", 0) / 1e6

        frames.append({
            "granule_id": props.get("fileID", "unknown"),
            "product_type": product,
            "path": props.get("pathNumber"),
            "frame": props.get("frameNumber"),
            "start_time": str(props.get("startTime", ""))[:10],
            "flight_direction": (props.get("flightDirection") or "N/A")[:4],
            "maturity": props.get("dataMaturity", "PROVISIONAL"),
            "file_size_mb": file_size_mb,
            "url": props.get("url"),
        })
    logger.info("Se encontraron %d granules de %s", len(frames), product)
    return frames

# ============================================================================
# RUTAS
# ============================================================================

@app.route("/")
def index():
    """Página principal."""
    return render_template("index.html", app_name=APP_NAME, app_version=APP_VERSION)


@app.route("/api/products")
@cache.cached(timeout=3600)
def get_products():
    """Lista de productos NISAR con descripciones para el usuario."""
    return jsonify([
        {
            "code": code,
            "name": info["name"],
            "description": info["description"],
            "level": info["level"],
            "icon": info["icon"],
            "technical": info["tech"],
        }
        for code, info in NISAR_PRODUCTS.items()
    ])


@app.route("/api/sites")
@cache.cached(timeout=3600)
def get_reference_sites():
    """Sitios de referencia en Honduras."""
    return jsonify([
        {"name": name, "latitude": lat, "longitude": lon}
        for name, (lat, lon) in REFERENCE_SITES.items()
    ])


@app.route("/api/inventory/<product>")
def get_inventory(product):
    """Inventario de granules NISAR de un producto, agrupado por trayecto/marco."""
    if product not in NISAR_PRODUCTS:
        return jsonify({"error": "Unknown product."}), 404
    try:
        frames = fetch_nisar_inventory(product)
    except Exception:
        logger.exception("Error al consultar el inventario de %s", product)
        return jsonify({"error": "Could not query the ASF inventory. Check your connection and try again."}), 502

    grouped = {}
    for frame in frames:
        key = f"Path {frame['path']}, Frame {frame['frame']}"
        grouped.setdefault(key, []).append(frame)
    return jsonify({"product": product, "total": len(frames), "grouped": grouped})


@analysis_cache.memoize(timeout=1800)
def cached_area_observations(coordinates_json):
    return fetch_area_observations(validate_area(json.loads(coordinates_json)))


@analysis_cache.memoize(timeout=3600)
def cached_area_analysis(coordinates_json, mode, start, end):
    return analyze_area(
        json.loads(coordinates_json), mode, start, end,
        cached_area_observations(coordinates_json),
    )


def leer_json_pequeno(mensaje_grande):
    """Devuelve (datos, error): valida el tamaño y que el cuerpo sea un objeto JSON."""
    if request.content_length is not None and request.content_length > MAX_REQUEST_BYTES:
        return None, (jsonify({"error": mensaje_grande}), 400)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "The request must contain a valid JSON object."}), 400)
    return data, None


@app.route("/api/analysis/dates", methods=["POST"])
def get_area_dates():
    """Fechas GCOV disponibles (y compatibles entre sí) para el área dibujada."""
    data, error = leer_json_pequeno("The dates request is too large.")
    if error:
        return error
    coordinates = data.get("coordinates")
    try:
        validate_area(coordinates)
    except AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        key = json.dumps(coordinates, separators=(",", ":"), allow_nan=False)
        return jsonify(available_area_dates(cached_area_observations(key)))
    except Exception:
        logger.exception("Error al consultar las fechas GCOV")
        return jsonify({"error": "Could not query the available dates from ASF. "
                                 "Check your connection and try again."}), 502


@app.route("/api/analysis", methods=["POST"])
def analyze_selected_area():
    """Análisis GCOV (agua y bosque) del área y fechas elegidas."""
    data, error = leer_json_pequeno("The analysis request is too large.")
    if error:
        return error
    coordinates = data.get("coordinates")
    mode, start = data.get("mode"), data.get("start")
    end = data.get("end") if mode == "compare" else start
    try:
        validate_area(coordinates)
        validate_dates(mode, start, end)
    except AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400
    try:
        key = json.dumps(coordinates, separators=(",", ":"), allow_nan=False)
        return jsonify(cached_area_analysis(key, mode, start, end))
    except AnalysisError as exc:
        logger.warning("No se pudo analizar el área: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception:
        logger.exception("Error al consultar o leer el área GCOV")
        return jsonify({
            "error": "Could not query or read the GCOV data. "
                     "Check your connection and your Earthdata token; see the console for details."
        }), 502


@app.route("/api/technical/raw", methods=["POST"])
def get_technical_raw_products():
    """Metadatos de todos los productos NISAR disponibles en el área y rango de fechas."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "The request must contain a valid JSON object."}), 400
    start = data.get("start")
    end = data.get("end") or start
    try:
        polygon = validate_area(data.get("coordinates"))
    except AnalysisError as exc:
        return jsonify({"error": str(exc)}), 400

    raw_items = []
    for code, info in NISAR_PRODUCTS.items():
        try:
            options = {
                "dataset": "NISAR",
                "processingLevel": code,
                "intersectsWith": polygon.wkt,
                "maxResults": 30,
            }
            if start:
                options["start"] = f"{start}T00:00:00Z"
            if end:
                options["end"] = f"{end}T23:59:59Z"
            for result in asf.search(**options):
                props = dict(result.properties)
                raw_items.append({
                    "product": code,
                    "level": info["level"],
                    "granule_id": props.get("fileID", "N/A"),
                    "date": str(props.get("startTime", ""))[:19],
                    "path": props.get("pathNumber"),
                    "frame": props.get("frameNumber"),
                    "direction": props.get("flightDirection", "N/A"),
                    "url": props.get("url"),
                })
        except Exception as exc:
            logger.warning("Error consultando el producto %s: %s", code, exc)

    return jsonify({"total": len(raw_items), "products": raw_items})


@app.route("/api/translate/<term>")
def translate_term(term):
    """Explica un término técnico de NISAR."""
    known = term in TECH_TRANSLATIONS
    return jsonify({
        "term": term,
        "translation": TECH_TRANSLATIONS.get(term, f"Technical term: {term}"),
        "category": "technical" if known else "unknown",
    })


@app.route("/api/status")
def get_status():
    """Estado del sistema."""
    return jsonify({
        "status": "operational",
        "version": APP_VERSION,
        "timestamp": datetime.now().isoformat(),
        "region": "Honduras",
        "data_source": "NASA NISAR L1/L2/L3 Archive",
        "available_products": list(NISAR_PRODUCTS.keys()),
    })

# ============================================================================
# INICIO
# ============================================================================

if __name__ == "__main__":
    logger.info("Iniciando %s", APP_NAME)
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "5000")),
            debug=False, use_reloader=False)
