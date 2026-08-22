[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8765,
    [string]$BindAddress = "127.0.0.1",
    [switch]$Hidden
)

# Qwen3.5-4B model server. It needs a Python environment with torch 2.13+cu126,
# transformers 5.15, bitsandbytes 0.50, peft 0.20, fastapi and uvicorn. The
# project venv does not carry those (they are ~3GB), so the launcher uses the
# lab environment by default and lets NOVEL_QWEN_PYTHON override it.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = $env:NOVEL_QWEN_PYTHON
if (-not $Python) { $Python = "C:\연구_프로젝트\ai_아키텍처\.venv\Scripts\python.exe" }
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Model server python not found: $Python  (set NOVEL_QWEN_PYTHON; see requirements-model-server.txt)"
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUTF8 = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "backend:cudaMallocAsync"
$env:NOVEL_QWEN_HOST = $BindAddress
$env:NOVEL_QWEN_PORT = $Port.ToString()
New-Item -ItemType Directory -Force (Join-Path $ProjectRoot ".runtime") | Out-Null
$Out = Join-Path $ProjectRoot ".runtime\model_server.out.log"
$Err = Join-Path $ProjectRoot ".runtime\model_server.err.log"
$PidFile = Join-Path $ProjectRoot ".runtime\model_server.pid"

# Reuse a healthy server instead of starting a second one on the same GPU.
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
    if ($health.status -eq "ok" -or $health.status -eq "loading") {
        Write-Host "[model-server] already running on port $Port ($($health.status))."
        exit 0
    }
} catch { }

$Proc = Start-Process -FilePath $Python -ArgumentList @("model_server\server.py") `
    -WorkingDirectory $ProjectRoot -RedirectStandardOutput $Out -RedirectStandardError $Err `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath $PidFile -Value $Proc.Id -Encoding ascii
Write-Host "[model-server] starting (PID $($Proc.Id)) on http://${BindAddress}:$Port - logs in .runtime\model_server.*.log"
if ($Hidden) { exit 0 }
for ($i = 0; $i -lt 120; $i++) {
    Start-Sleep -Seconds 2
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
        if ($health.status -eq "ok") {
            Write-Host "[model-server] ready: model=$($health.model) adapters=$($health.adapters -join ',') VRAM=$($health.vram_gib) GiB"
            exit 0
        }
    } catch { }
    if ($Proc.HasExited) {
        Write-Host "[model-server] exited early; see .runtime\model_server.err.log" -ForegroundColor Red
        Get-Content -LiteralPath $Err -Tail 20
        exit 1
    }
}
Write-Host "[model-server] still loading after 240s; check .runtime\model_server.err.log" -ForegroundColor Yellow
exit 1
