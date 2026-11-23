#!/usr/bin/env bash
# One C3 paper session on the Linux server: the twin of run_c3.ps1.
# Starts the guard, runs the session, waits until 16:20 ET, then writes the evening report.
# systemd starts it on weekdays at 08:00 ET (deploy/c3.timer); a holiday exits on its own.
# Manual use, from anywhere:  bash research/liquid/live/run_c3.sh
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
export PYTHONPATH="$root" PYTHONIOENCODING=utf-8
py="${C3_PYTHON:-$root/.venv/bin/python}"
logdir="$root/data/research_liquid/live/logs"
mkdir -p "$logdir"
stamp="$(TZ=America/New_York date +%F)"
log="$logdir/c3_$stamp.log"
echo "$(date -Is)  run_c3.sh start on $(hostname), code $(git rev-parse --short HEAD 2>/dev/null)" >>"$log"

# The guard is a separate process: if the runner dies holding a position, it flattens.
"$py" -m research.liquid.live.main --guard >"$logdir/guard_$stamp.out" 2>"$logdir/guard_$stamp.err" &
guard=$!

"$py" -m research.liquid.live.main --run >>"$log" 2>&1
echo "run exit code: $?" >>"$log"

# SIP bars and quotes are free once 16 minutes old: wait until 16:20 ET for the report.
target="$(TZ=America/New_York date -d 'today 16:20' +%s)"
now="$(date +%s)"
if [ "$now" -lt "$target" ]; then sleep $((target - now)); fi
"$py" -m research.liquid.live.main --report >>"$log" 2>&1
echo "report exit code: $?" >>"$log"

# The guard leaves two minutes after the close-out; one still alive now is logged, and systemd
# stops it with the service when this script exits.
if kill -0 "$guard" 2>/dev/null; then
    echo "guard still running at report time (pid $guard)" >>"$log"
else
    wait "$guard"
    echo "guard exit code: $?" >>"$log"
fi

# Publish the dashboard data to Drive (only where publish.json names the two files; a failure
# here never touches trading, it only leaves the dashboard a day behind).
if [ -f "$root/data/research_liquid/live/publish.json" ]; then
    "$py" -m research.liquid.live.dashboard --live --publish >>"$log" 2>&1
    echo "dashboard publish exit code: $?" >>"$log"
fi
