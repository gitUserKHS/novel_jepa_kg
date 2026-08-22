@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage: run_model_server.bat
    echo Starts the local Qwen3.5-4B model server at http://127.0.0.1:8765
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_model_server.ps1" -Port 8765
if errorlevel 1 pause
