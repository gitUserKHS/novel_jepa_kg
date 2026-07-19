@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage: run_service.bat
    echo Starts the consumer web app and generation worker on port 8501.
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_consumer_service.ps1" -Port 8501 -OpenBrowser
if errorlevel 1 pause
