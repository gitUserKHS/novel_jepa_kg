[CmdletBinding()]
param(
    [string]$WebTaskName = "Novel JEPA Consumer Web",
    [string]$WorkerTaskName = "Novel JEPA Consumer Worker"
)

$ErrorActionPreference = "Stop"
foreach ($TaskName in @($WebTaskName, $WorkerTaskName)) {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "Scheduled task was not installed: $TaskName"
        continue
    }
    if ($task.State -eq "Running") {
        Stop-ScheduledTask -TaskName $TaskName
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task: $TaskName" -ForegroundColor Green
}
