# Run one C3 paper session, then the evening report. Start any time before 09:25 ET
# on a trading day; the runner waits for each check itself.
# Usage (from the repo root):  powershell -ExecutionPolicy Bypass -File research\liquid\live\run_c3.ps1
$ErrorActionPreference = "Stop"
$py = "C:\Users\Shachar\Desktop\venvs\ML\Scripts\python.exe"
$root = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
Set-Location $root
$env:PYTHONPATH = $root
$logDir = Join-Path $root "data\research_liquid\live\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd"
$log = Join-Path $logDir "c3_$stamp.log"

& $py -m research.liquid.live.main --run *>> $log
# SIP bars are free once 15 minutes old: wait until 16:20 ET before the report
$et = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), "Eastern Standard Time")
$target = $et.Date.AddHours(16).AddMinutes(20)
if ($et -lt $target) { Start-Sleep -Seconds ([int]($target - $et).TotalSeconds) }
& $py -m research.liquid.live.main --report *>> $log
