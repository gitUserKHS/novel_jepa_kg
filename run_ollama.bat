@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage: run_ollama.bat
    echo Checks Ollama, verifies the configured model is installed, and loads it into memory.
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_ollama.ps1"
if errorlevel 1 pause
