[CmdletBinding()]
param(
    [string]$AppPath = "app.py",
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$RequireAuth,
    [switch]$ConsumerMode,
    [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$EnvFile = Join-Path $ProjectRoot ".env"

function Read-DotEnvValue {
    param([string]$Path, [string]$Name)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed.Split("=", 2)
        if ($parts.Count -eq 2 -and $parts[0].Trim() -eq $Name) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }
    return $null
}

function New-AccessToken {
    $bytes = New-Object byte[] 32
    $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $random.GetBytes($bytes)
    }
    finally {
        $random.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv was not found. Run: py -3.11 -m venv .venv"
}
$ResolvedApp = (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $AppPath)).Path
if (-not $ResolvedApp.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "AppPath must stay inside the project directory."
}

& $Python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Streamlit is not installed. Run: .venv\Scripts\python.exe -m pip install -r requirements.txt"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $owners = ($listener | Select-Object -ExpandProperty OwningProcess -Unique) -join ", "
    throw "Port $Port is already in use by process $owners. Stop that service or choose another port."
}

$env:PYTHONUTF8 = "1"
if ($ConsumerMode) {
    $env:NOVEL_JEPA_CONSUMER_BIND_HOST = $BindAddress
    $env:NOVEL_JEPA_CONSUMER_PORT = $Port.ToString()
}
else {
    $env:NOVEL_JEPA_BIND_HOST = $BindAddress
    $env:NOVEL_JEPA_PORT = $Port.ToString()
    $env:NOVEL_JEPA_REQUIRE_AUTH = if ($RequireAuth) { "true" } else { "false" }
}

if ($RequireAuth) {
    $token = $env:NOVEL_JEPA_ACCESS_TOKEN
    if (-not $token) {
        $token = Read-DotEnvValue -Path $EnvFile -Name "NOVEL_JEPA_ACCESS_TOKEN"
    }
    if (-not $token) {
        $token = New-AccessToken
        $prefix = if ((Test-Path -LiteralPath $EnvFile) -and (Get-Item -LiteralPath $EnvFile).Length -gt 0) { "`r`n" } else { "" }
        Add-Content -LiteralPath $EnvFile -Value "${prefix}NOVEL_JEPA_ACCESS_TOKEN=$token" -Encoding UTF8
        Write-Host "[Novel JEPA] A new access token was saved to .env." -ForegroundColor Green
    }
    $env:NOVEL_JEPA_ACCESS_TOKEN = $token
    Write-Host "Access token: $token" -ForegroundColor Yellow
}

$LocalUrl = "http://127.0.0.1:$Port"
Write-Host "Starting $AppPath" -ForegroundColor Cyan
Write-Host "Local URL: $LocalUrl"
if ($BindAddress -eq "0.0.0.0") {
    if ($ConsumerMode) {
        Write-Host "LAN mode is enabled. Consumers can register and sign in at http://<this-PC-IP>:$Port."
    }
    elseif ($RequireAuth) {
        Write-Host "LAN mode is enabled. Share the address and admin access token only with trusted operators."
    }
}

if ($OpenBrowser) {
    Start-Process $LocalUrl
}

& $Python -m streamlit run $ResolvedApp `
    --server.headless true `
    --server.address $BindAddress `
    --server.port $Port `
    --server.runOnSave false
exit $LASTEXITCODE
