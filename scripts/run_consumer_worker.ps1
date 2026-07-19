[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv was not found. Run: py -3.11 -m venv .venv"
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUTF8 = "1"
& $Python -m src.service.worker
exit $LASTEXITCODE
