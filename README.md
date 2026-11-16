# NISAR Monitor - Aplicación Streamlit

Aplicación de monitoreo de datos satelitales NISAR usando datos SAR (Radar de Apertura Sintética).

## Requisitos

### Entorno virtual
La aplicación está diseñada para ejecutarse en el entorno virtual ubicado en:
- `C:\Users\bryan\nisar-monitor\.venv\`

### Dependencias instaladas
Ejecutar en el entorno activado:
```
pip install streamlit asf_search earthaccess h5py numpy pydeck pyproj
```

## Ejecución

### Verificar sintaxis (recomendado)
```bash
# Activar entorno virtual
cd C:\Users\brian\nisar-monitor
.\.venv\Scripts\activate

# Verificar compilación (no ejecuta la app, solo valida sintaxis)
python -m py_compile app.py
```

### Ejecutar la aplicación
```bash
cd C:\Users\brian\nisar-monitor
.\.venv\Scripts\streamlit run app.py
```

## Funcionalidades

### Panel lateral izquierdo
- **Cuenta NASA Earthdata**: Ingresa usuario y contraseña para acceder a los datos
- **Área de Interés (WKT)**: Muestra el polígono por defecto (no requiere edición)
- **Producto NISAR**: Seleccionar entre GCOV o GSLC
- **Filtrar órbita**: Checkbox para filtrar solo Trayecto 4 / Marco 9

### Mapa principal (columna izquierda)
- Mapa de fondo estilo Carto Voyager
- **Rectángulo amarillo**: Tu Área de Interés WKT (coordenadas por defecto)
- **Rectángulo rojo**: Bloque de datos recortado (tamaño configurable)
- Imagen en escala de grises (decibeles dB) con transparencia

### Panel lateral derecho (siempre visible)
- Métricas de retrodispersión media (HHHH y HVHV en dB)
- Porcentaje de píxeles con datos válidos
- Proyección EPSG
- Órbita absoluta
- Dirección de pase
- Polarizaciones detectadas

### Control deslizador (debajo del mapa)
- **Tamaño de bloque de píxeles**: Ajustable de 300 a 800 píxeles (valor por defecto: 500)
- Determina el tamaño del bloque contiguo leído del archivo .h5

## Cómo funciona

1. La aplicación busca escenas NISAR que intersectan con el WKT definido
2. Filtra por Trayecto 4 / Marco 9 si el checkbox está marcado
3. Para la escena seleccionada, lee un bloque único y contiguo de datos del archivo .h5
4. El bloque está centrado en el centroide de tu área de interés
5. Los datos se convierten a escala de grises (dB) y se muestran en el mapa
6. Los datos internos se muestran en el panel lateral

## Notas importantes

- La primera vez que se ejecuta, el descarga de datos puede tardar unos segundos
- Las ejecuciones subsiguientes con la misma fecha serán instantáneas gracias al cacheo (ttl=3600)
- Se recomienda tener conexión a internet estable para acceder a los datos de NASA Earthdata
- Los datos mostrados corresponden a la última escena seleccionada en el selector temporal