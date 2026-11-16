@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo WAVE NISAR Monitor - Setup Rapido
echo ================================================
echo.

REM Verificar Python
python --version >nul 2>&1
if errorlevel 1 goto no_python

REM Las versiones fijadas en requirements.txt admiten Python 3.9 a 3.12
python -c "import sys; raise SystemExit(not ((3, 9) <= sys.version_info[:2] < (3, 13)))" >nul 2>&1
if errorlevel 1 goto unsupported_python

echo [1/4] Creando entorno virtual...
python -m venv venv
if errorlevel 1 goto error

echo [2/4] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 goto error

echo [3/4] Instalando dependencias...
python -m pip install -r requirements.txt
if errorlevel 1 goto error

echo.
echo [4/4] Configuracion completada!
echo.
echo PROXIMO PASO:
echo ===============
echo 1. Copia .env.example a .env
echo 2. Edita .env con tu token de NASA Earthdata
echo    (Obtén uno en: https://urs.earthdata.nasa.gov)
echo 3. Ejecuta: python app.py
echo 4. Abre: http://localhost:5000
echo.
echo ================================================
pause
exit /b 0

:no_python
echo ERROR: Python no encontrado.
echo Instala Python 3.9 a 3.12 y agrega Python al PATH.
echo Descarga: https://www.python.org/downloads/
pause
exit /b 1

:unsupported_python
echo ERROR: Se requiere Python 3.9 a 3.12 para estas dependencias.
echo Python detectado:
python --version
pause
exit /b 1

:error
echo.
echo ERROR durante la instalacion!
echo Revisa el mensaje anterior, corrige el problema y vuelve a ejecutar setup.bat.
pause
exit /b 1
