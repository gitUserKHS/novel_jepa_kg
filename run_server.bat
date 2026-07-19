@echo off
setlocal
cd /d "%~dp0"

call "%~dp0run_admin.bat" %*
exit /b %errorlevel%
