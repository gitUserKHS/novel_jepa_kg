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
# llama-server fail to allocate its buffers.
#
# Deliberately no WMI here (Get-CimInstance / tasklist): after a force-kill the
# WMI process provider can wedge, and a startup path that waits on it hangs the
# launcher before anything starts. The worker's singleton lock file already
# knows the truth: byte 0 is range-locked exactly while a worker lives, and the
# JSON after it records that worker's PID. Probe the lock, and kill the process
# tree it names.
$WorkerLock = Join-Path $ProjectRoot ".runtime\consumer.worker.lock"
if (Test-Path -LiteralPath $WorkerLock) {
    try {
        $Stream = [System.IO.File]::Open(
            $WorkerLock,
            [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read,
            [System.IO.FileShare]::ReadWrite
        )
        try {
            $Probe = New-Object byte[] 1
            $LockHeld = $false
            try {
                $null = $Stream.Read($Probe, 0, 1)
            }
            catch [System.IO.IOException] {
                $LockHeld = $true
            }
            if ($LockHeld) {
                $null = $Stream.Seek(1, [System.IO.SeekOrigin]::Begin)
                $Buffer = New-Object byte[] 4096
                $Count = $Stream.Read($Buffer, 0, $Buffer.Length)
                $Metadata = [System.Text.Encoding]::UTF8.GetString($Buffer, 0, $Count) | ConvertFrom-Json
                $OrphanPid = [int]$Metadata.pid
                $Owner = Get-Process -Id $OrphanPid -ErrorAction SilentlyContinue
                if ($Owner -and $Owner.ProcessName -like "python*") {
                    Write-Host "Stopping orphaned consumer worker (PID $OrphanPid)."
                    & taskkill /PID $OrphanPid /T /F 2>$null | Out-Null
                }
            }
        }
        finally {
            $Stream.Close()
        }
    }
    catch {
        Write-Host "Worker lock probe failed ($($_.Exception.Message)); continuing. A stale worker, if any, will block the new one via its singleton lock." -ForegroundColor Yellow
    }
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
        # /T also takes down the real interpreter under the venv shim.
        & taskkill /PID $Worker.Id /T /F 2>$null | Out-Null
    }
}
