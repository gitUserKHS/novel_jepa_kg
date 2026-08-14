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
# The worker below is started hidden, so a launcher closed without reaching its
# finally block leaves an invisible process behind. Those accumulate across runs
# and each one reserves gigabytes of commit charge, which eventually makes
# llama-server fail to allocate its buffers. Clear any strays before starting.
$Orphans = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -and $_.CommandLine -like "*src.service.worker*" }
foreach ($Orphan in $Orphans) {
    Write-Host "Stopping orphaned consumer worker (PID $($Orphan.ProcessId))."
    Stop-Process -Id $Orphan.ProcessId -Force -ErrorAction SilentlyContinue
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
