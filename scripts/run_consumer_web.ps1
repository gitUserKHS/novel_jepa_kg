[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "run_local_service.ps1") `
    -AppPath "consumer_app.py" `
    -BindAddress "0.0.0.0" `
    -Port $Port `
    -ConsumerMode `
    -OpenBrowser:$OpenBrowser
exit $LASTEXITCODE
