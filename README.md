# 🛰️ WAVE NISAR Monitor

Prototipo de plataforma web para analizar datos SAR del satélite NISAR (producto **GCOV**) sobre Honduras.
Permite dibujar un área en el mapa, elegir una o dos fechas disponibles y ver mapas de **agua/humedales** y **bosque** con su nivel de fiabilidad.

- **Modo intuitivo** (por defecto): mapas de colores, gráficas de barras y un resumen en lenguaje sencillo.
- **Modo técnico**: capas radiométricas en crudo (HH / HV), escenas de todos los productos NISAR disponibles y tres secciones de consulta (productos, inventario y glosario) que se pueden esconder haciendo clic en su barra de título.

> La interfaz que ve el usuario está en inglés; el código y los comentarios, en español.

## Requisitos

- Python 3.9 a 3.12
- Token gratuito de NASA Earthdata: https://urs.earthdata.nasa.gov

## Instalación (Windows)

1. Ejecuta `setup.bat` (crea el entorno virtual `venv` e instala las dependencias).
2. Copia `.env.example` a `.env` y pega tu token en `EDL_TOKEN`.
3. Ejecuta `run.bat` y abre http://localhost:5000

Sin `.bat`: `python -m venv venv`, activar el entorno, `pip install -r requirements.txt` y `python app.py`.

## Pruebas

```
python -m unittest
```

`test_parameters.py` no necesita internet; `test_gcov_analysis.py` simula las consultas a ASF y los archivos HDF5.

## Estructura

| Archivo | Para qué sirve |
|---|---|
| `app.py` | Servidor Flask: rutas de la API y caché |
| `gcov_analysis.py` | Búsqueda en ASF, lectura de los GCOV y construcción de resultados |
| `parameters.py` | Umbrales y clasificación de agua/bosque (solo numpy) |
| `templates/index.html` | Interfaz completa (HTML, CSS y JavaScript) |

## Aviso científico

Los umbrales de `parameters.py` son valores **iniciales y no están calibrados para Honduras**. Un cambio de señal radar no prueba por sí solo deforestación, inundación u otro fenómeno; los resultados deben validarse con datos de campo.

## Seguridad

El archivo `.env` contiene tu token personal y está en `.gitignore`: **nunca lo subas a GitHub**.
