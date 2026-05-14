@echo off
setlocal

cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage:
    echo   run_server.bat
    echo.
    echo Starts Novel JEPA Lab at http://127.0.0.1:8501
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv was not found.
    echo Run these commands first:
    echo   python -m venv .venv
    echo   .venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Streamlit is not installed in .venv.
    echo Install dependencies first:
    echo   .venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
    pause
    exit /b 1
)

echo Starting Novel JEPA Lab...
echo URL: http://127.0.0.1:8501
start "" "http://127.0.0.1:8501"

".venv\Scripts\python.exe" -m streamlit run app.py --server.headless true --server.port 8501

pause
