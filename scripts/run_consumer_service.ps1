[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv was not found. Run: py -3.11 -m venv .venv"
}
$Worker = Start-Process `
    -FilePath $Python `
    -ArgumentList @("-m", "src.service.worker") `
    -WorkingDirectory $ProjectRoot `
    -WindowStyle Hidden `
    -PassThru
try {
    & (Join-Path $PSScriptRoot "run_consumer_web.ps1") -Port $Port -OpenBrowser:$OpenBrowser
    exit $LASTEXITCODE
}
finally {
    if ($Worker -and -not $Worker.HasExited) {
        Stop-Process -Id $Worker.Id -Force -ErrorAction SilentlyContinue
    }
}
