@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage: run_model_server.bat
    echo Legacy: starts the Qwen3.5-4B model server at http://127.0.0.1:8765
    echo Only needed when configs/default.yaml has llm.backend: local. The default backend is Ollama ^(run_ollama.bat^).
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_model_server.ps1" -Port 8765
if errorlevel 1 pause
