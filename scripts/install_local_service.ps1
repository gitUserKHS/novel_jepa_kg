[CmdletBinding()]
param(
    [string]$WebTaskName = "Novel JEPA Consumer Web",
    [string]$WorkerTaskName = "Novel JEPA Consumer Worker",
    [ValidateRange(1, 65535)]
    [int]$Port = 8501,
    [switch]$StartNow
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PowerShell = "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe"
$User = "$env:USERDOMAIN\$env:USERNAME"

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
    throw ".venv was not found. Create it and install requirements before registering the service."
}
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $User
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650)

function Register-NovelTask {
    param(
        [string]$Name,
        [string]$Runner,
        [string]$ExtraArguments,
        [string]$Description
    )
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing -and $existing.State -eq "Running") {
        Stop-ScheduledTask -TaskName $Name
    }
    $arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`" $ExtraArguments".Trim()
    $action = New-ScheduledTaskAction -Execute $PowerShell -Argument $arguments -WorkingDirectory $ProjectRoot
    Register-ScheduledTask `
        -TaskName $Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description $Description `
        -User $User `
        -RunLevel Limited `
        -Force | Out-Null
    Write-Host "Registered scheduled task: $Name" -ForegroundColor Green
}

Register-NovelTask `
    -Name $WebTaskName `
    -Runner (Join-Path $PSScriptRoot "run_consumer_web.ps1") `
    -ExtraArguments "-Port $Port" `
    -Description "Novel JEPA consumer Streamlit web service"
Register-NovelTask `
    -Name $WorkerTaskName `
    -Runner (Join-Path $PSScriptRoot "run_consumer_worker.ps1") `
    -ExtraArguments "" `
    -Description "Novel JEPA single-GPU generation queue worker"

if ($StartNow) {
    Start-ScheduledTask -TaskName $WorkerTaskName
    Start-ScheduledTask -TaskName $WebTaskName
    Write-Host "Started consumer worker and web tasks on port $Port."
}
