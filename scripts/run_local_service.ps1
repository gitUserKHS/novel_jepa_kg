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

# Restarting is the normal case: the previous run is almost always our own
# Streamlit still holding the port. Stop that one and take the port back, and
# refuse anything else, so an unrelated server sharing the port is never killed.
#
# Deliberately no WMI (Get-NetTCPConnection / Get-CimInstance / tasklist): after
# a force-kill the WMI process provider can wedge indefinitely, and a launcher
# that waits on it never starts anything. netstat and Get-Process are native.
# Ownership comes from a pid file this launcher writes, not from command-line
# matching, which WMI alone could provide.
function Get-PortListenerPids {
    param([int]$ProbePort)
    $found = @()
    foreach ($line in (& netstat -ano -p TCP | Select-String "LISTENING")) {
        $parts = ($line.ToString().Trim() -split "\s+")
        if ($parts.Count -ge 5 -and $parts[1] -match "[:\]]$ProbePort$") {
            $found += [int]$parts[-1]
        }
    }
    return @($found | Sort-Object -Unique)
}

$PidFile = Join-Path $ProjectRoot (".runtime\web-" + [System.IO.Path]::GetFileName($ResolvedApp) + ".pid")
$listeners = Get-PortListenerPids -ProbePort $Port
if ($listeners.Count -gt 0) {
    $known = $null
    if (Test-Path -LiteralPath $PidFile) {
        $known = [int](Get-Content -LiteralPath $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    }
    if ($known) {
        $owner = Get-Process -Id $known -ErrorAction SilentlyContinue
        if ($owner -and $owner.ProcessName -like "python*") {
            Write-Host "[Novel JEPA] Stopping the previous $AppPath (PID $known)." -ForegroundColor Yellow
            & taskkill /PID $known /T /F 2>$null | Out-Null
        }
    }
    for ($wait = 0; $wait -lt 20; $wait++) {
        Start-Sleep -Milliseconds 250
        $listeners = Get-PortListenerPids -ProbePort $Port
        if ($listeners.Count -eq 0) { break }
    }
    if ($listeners.Count -gt 0) {
        throw "Port $Port is already in use by process $($listeners -join ', '), which is not this launcher's app. Stop that service or choose another port."
    }
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

# Start-Process instead of direct invocation so the PID can be recorded. The
# pid file is what lets the next launch identify and stop this instance without
# consulting WMI. /T on cleanup also takes down the real interpreter the venv
# shim spawns.
$Web = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-m", "streamlit", "run", $ResolvedApp,
        "--server.headless", "true",
        "--server.address", $BindAddress,
        "--server.port", "$Port",
        "--server.runOnSave", "false"
    ) `
    -WorkingDirectory $ProjectRoot `
    -NoNewWindow `
    -PassThru
New-Item -ItemType Directory -Force (Split-Path -Parent $PidFile) | Out-Null
Set-Content -LiteralPath $PidFile -Value $Web.Id -Encoding ascii
try {
    Wait-Process -Id $Web.Id -ErrorAction SilentlyContinue
    exit $(if ($Web.HasExited) { $Web.ExitCode } else { 0 })
}
finally {
    if ($Web -and -not $Web.HasExited) {
        & taskkill /PID $Web.Id /T /F 2>$null | Out-Null
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}
