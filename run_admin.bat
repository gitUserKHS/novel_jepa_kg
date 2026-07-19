@echo off
setlocal
cd /d "%~dp0"

if /i "%~1"=="--help" (
    echo Usage: run_admin.bat
    echo Starts the private admin and training UI at http://127.0.0.1:8502
    exit /b 0
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_local_service.ps1" -AppPath "app.py" -BindAddress "127.0.0.1" -Port 8502 -OpenBrowser
if errorlevel 1 pause
