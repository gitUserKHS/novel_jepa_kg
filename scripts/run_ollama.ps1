[CmdletBinding()]
param(
    [switch]$Hidden
)

# Ollama 백엔드 준비 (configs/default.yaml 의 llm 절을 읽는다):
#   1. Ollama 서버가 응답하는지 확인하고, 없으면 `ollama serve` 를 숨겨서 띄운다.
#   2. 설정된 모델이 설치되어 있는지 확인한다 (없으면 pull 명령을 알려 주고 실패).
#   3. 모델을 미리 올려 둔다 (백그라운드) — 첫 장을 쓸 때 17GB 적재를 기다리지 않게.
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw ".venv was not found. Run: py -3.11 -m venv .venv"
}
Set-Location -LiteralPath $ProjectRoot
$env:PYTHONUTF8 = "1"

$Settings = @(& $Python -c "from src.utils.config import load_config; c = load_config('configs/default.yaml').llm; print(c.ollama_base_url); print(c.model); print(c.keep_alive)")
if ($LASTEXITCODE -ne 0 -or $Settings.Count -lt 3) {
    throw "Could not read llm settings from configs/default.yaml."
}
$BaseUrl = $Settings[0].Trim().TrimEnd("/")
$Model = $Settings[1].Trim()
$KeepAlive = $Settings[2].Trim()

function Get-OllamaTags {
    try { return Invoke-RestMethod -Uri "$BaseUrl/api/tags" -TimeoutSec 3 } catch { return $null }
}

$Tags = Get-OllamaTags
if (-not $Tags) {
    $Exe = Get-Command ollama -ErrorAction SilentlyContinue
    if (-not $Exe) {
        Write-Host "[ollama] Ollama 를 찾을 수 없어. https://ollama.com 에서 설치한 뒤 다시 실행해줘." -ForegroundColor Red
        exit 1
    }
    Write-Host "[ollama] starting ollama serve ($($Exe.Source))"
    Start-Process -FilePath $Exe.Source -ArgumentList "serve" -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 2
        $Tags = Get-OllamaTags
        if ($Tags) { break }
    }
    if (-not $Tags) {
        Write-Host "[ollama] $BaseUrl 에서 응답이 없어. Ollama 앱이 다른 포트에 떠 있으면 NOVEL_LLM_OLLAMA_BASE_URL 을 맞춰줘." -ForegroundColor Red
        exit 1
    }
}

$Installed = @($Tags.models | ForEach-Object { [string]$_.name })
$Wanted = $Model.ToLowerInvariant()
if ($Wanted.EndsWith(":latest")) { $Wanted = $Wanted.Substring(0, $Wanted.Length - 7) }
$Found = $false
foreach ($name in $Installed) {
    $n = $name.ToLowerInvariant()
    if ($n.EndsWith(":latest")) { $n = $n.Substring(0, $n.Length - 7) }
    if ($n -eq $Wanted) { $Found = $true; break }
}
if (-not $Found) {
    Write-Host "[ollama] 설정된 모델이 설치되어 있지 않아: $Model" -ForegroundColor Red
    Write-Host "         ollama pull $Model"
    Write-Host "         설치된 모델: $($Installed -join ', ')"
    exit 1
}
Write-Host "[ollama] ready at $BaseUrl - model $Model"

# 미리 올려 두기: 빈 generate 요청은 모델만 적재하고 바로 돌아온다. 적재(1~2분)가 웹 기동을 막지 않게 백그라운드로.
$Warm = Start-Job -ScriptBlock {
    param($Url, $Name, $Keep)
    try {
        Invoke-RestMethod -Method Post -Uri "$Url/api/generate" -ContentType "application/json" -TimeoutSec 900 `
            -Body (@{ model = $Name; keep_alive = $Keep } | ConvertTo-Json) | Out-Null
    } catch { }
} -ArgumentList $BaseUrl, $Model, $KeepAlive
if (-not $Hidden) {
    Write-Host "[ollama] loading the model into memory..."
    Wait-Job $Warm -Timeout 900 | Out-Null
    try {
        $Loaded = Invoke-RestMethod -Uri "$BaseUrl/api/ps" -TimeoutSec 5
        foreach ($m in $Loaded.models) {
            Write-Host ("[ollama] loaded: {0}  VRAM {1:N1} GiB  ctx {2}" -f $m.name, ($m.size_vram / 1GB), $m.context_length)
        }
    } catch { }
}
exit 0
