# One C3 paper session: start the guard, run the session, then the evening report.
# Start any time before 09:25 ET on a trading day; the runner waits for each check itself.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File research\liquid\live\run_c3.ps1
#
# Python is launched through cmd /c so its stderr goes to the log. Under Windows
# PowerShell 5.1, `& python ... *>> log` with $ErrorActionPreference = "Stop" aborts the
# script at the first stderr line (verified 2026-11-22, G3 W7).
$ErrorActionPreference = "Continue"
$py = "C:\Users\Shachar\Desktop\venvs\ML\Scripts\python.exe"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root
$env:PYTHONPATH = $root
$logDir = Join-Path $root "data\research_liquid\live\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd"
$log = Join-Path $logDir "c3_$stamp.log"

# The guard is a separate process: if the runner dies holding a position, it flattens.
Start-Process -FilePath $py -ArgumentList "-m", "research.liquid.live.main", "--guard" `
    -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $logDir "guard_$stamp.out") `
    -RedirectStandardError (Join-Path $logDir "guard_$stamp.err")

cmd /c "`"$py`" -m research.liquid.live.main --run >> `"$log`" 2>&1"
"run exit code: $LASTEXITCODE" | Out-File -Append -Encoding ascii $log

# SIP bars and quotes are free once 16 minutes old: wait until 16:20 ET for the report
$et = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), "Eastern Standard Time")
$target = $et.Date.AddHours(16).AddMinutes(20)
if ($et -lt $target) { Start-Sleep -Seconds ([int]($target - $et).TotalSeconds) }
cmd /c "`"$py`" -m research.liquid.live.main --report >> `"$log`" 2>&1"
"report exit code: $LASTEXITCODE" | Out-File -Append -Encoding ascii $log
