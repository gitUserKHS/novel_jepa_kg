[CmdletBinding()]
param(
    [string]$WebTaskName = "Novel JEPA Consumer Web",
    [string]$WorkerTaskName = "Novel JEPA Consumer Worker",
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [string]$RequirementsFile = "requirements-gpu.txt",
    [switch]$SkipSmoke,
    [switch]$SkipOllamaHealth,
    [switch]$SkipActiveJepaHealth
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot $RequirementsFile
$RequirementsStamp = Join-Path $ProjectRoot ".venv\requirements.sha256"

function Invoke-Checked {
    param([string]$Label, [scriptblock]$Command)
    Write-Host "[Deploy] $Label" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE." }
}

function Stop-TaskIfRunning {
    param([string]$Name)
    $task = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($task -and $task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $Name
    }
}

Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath $Python)) {
    & py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Python 3.11 is required. Install it, then run: py -3.11 -m venv .venv"
    }
}
if (-not (Test-Path -LiteralPath $RequirementsPath)) {
    throw "Requirements file not found: $RequirementsPath"
}

Invoke-Checked "Enter maintenance drain" { & $Python scripts\service_control.py enter }
Invoke-Checked "Wait for consumer queue to become idle" { & $Python scripts\service_control.py wait-idle --timeout 1800 }
Stop-TaskIfRunning -Name $WebTaskName
Stop-TaskIfRunning -Name $WorkerTaskName
Start-Sleep -Seconds 2

$requirementsHash = (Get-FileHash -LiteralPath $RequirementsPath -Algorithm SHA256).Hash
$installedHash = if (Test-Path -LiteralPath $RequirementsStamp) {
    (Get-Content -LiteralPath $RequirementsStamp -Raw).Trim()
}
else { "" }
if ($requirementsHash -ne $installedHash) {
    Invoke-Checked "Upgrade pip" { & $Python -m pip install --upgrade pip }
    Invoke-Checked "Install dependencies from $RequirementsFile" { & $Python -m pip install -r $RequirementsPath }
    Set-Content -LiteralPath $RequirementsStamp -Value $requirementsHash -Encoding ASCII
}

Invoke-Checked "Run unit tests" { & $Python -m unittest discover -s tests -v }
if (-not $SkipSmoke) {
    Invoke-Checked "Run restartable pipeline smoke test" { & $Python scripts\smoke_jepa.py }
}

Write-Host "[Deploy] Registering consumer web and worker tasks" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "install_local_service.ps1") `
    -WebTaskName $WebTaskName `
    -WorkerTaskName $WorkerTaskName `
    -Port $Port `
    -StartNow
if ($LASTEXITCODE -ne 0) { throw "Scheduled task registration failed." }

$ollamaUrl = if ($env:NOVEL_JEPA_OLLAMA_BASE_URL) { $env:NOVEL_JEPA_OLLAMA_BASE_URL }
elseif ($env:OLLAMA_BASE_URL) { $env:OLLAMA_BASE_URL }
else { "http://127.0.0.1:11434" }
$healthArgs = @(
    "scripts\health_check.py",
    "--app-url", "http://127.0.0.1:$Port",
    "--ollama-url", $ollamaUrl,
    "--model", "gemma4:e4b",
    "--embedding-model", "embeddinggemma:latest"
)
if ($SkipOllamaHealth) { $healthArgs += "--skip-ollama" }
if ($SkipActiveJepaHealth) { $healthArgs += "--skip-active-jepa" }
Invoke-Checked "Verify consumer web, worker, Ollama, and active JEPA" { & $Python @healthArgs }
Invoke-Checked "Resume consumer service" { & $Python scripts\service_control.py resume }
Write-Host "[Deploy] Complete: http://127.0.0.1:$Port" -ForegroundColor Green
