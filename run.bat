@echo off
cd /d "%~dp0"
cls
echo ================================================
echo WAVE NISAR Monitor - Iniciando...
echo ================================================
echo.

REM Verificar si el entorno virtual existe
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Entorno virtual no encontrado.
    echo Ejecuta setup.bat primero.
    pause
    exit /b 1
)

REM Verificar si existe .env
if not exist ".env" (
    echo WARNING: Archivo .env no encontrado.
    echo Por favor, copia .env.example a .env y configura tu token.
    echo.
    pause
)

REM Activar entorno
call venv\Scripts\activate.bat

REM Ejecutar la aplicación
echo.
echo [INICIANDO] WAVE NISAR Monitor
echo [URL] http://localhost:5000
echo [PRESIONA] Ctrl+C para detener
echo.

python app.py

pause
