# C3 paper trading — how to run it

The rules and gates are in `research/liquid/PAPER_C3.md` (amendment A1).

## One-time setup (you)

1. Create a **new Alpaca paper account**. Do not use KELLY or LINEAR: the live CI
   system trades those.
2. Add its keys to `QuantCode/.env`:
   ```
   ALPACA_API_KEY_C3=...
   ALPACA_SECRET_KEY_C3=...
   ```
3. Check it, read-only. This sends no orders:
   ```
   python -m research.liquid.live.main --check
   ```
   It prints:
   - the account, the market clock, and whether shorting is enabled;
   - the SPY position and open orders, the KILL file and the heartbeat;
   - the next session's plan.
4. Run the drill once:
   ```
   python -m research.liquid.live.main --drill
   ```
   It turns shorting off on this account, then checks that Alpaca rejects a
   duplicate client order id. To do that it sends one 1-share buy limit at
   $1.00, which cannot fill, and cancels it. It exits 0 when both checks hold.
5. **Keep the PC awake on trading days until about 16:25 ET.** Sleep kills both
   the runner and the guard. The GTC stop then still protects the position, and
   the next morning's run flattens it and stops on the ledger check.

## Where it runs: the Google Cloud server (since 2026-11-24)

The paper runner runs on a small Linux server, not on the PC. The Windows task
"QuantCode C3 SPY paper" is **disabled**: never run both against one account.

- **Machine:** GCP project `tradingviewapi-467921`, VM `c3-paper` (e2-micro,
  us-east1-b, Debian 13, Python 3.13, 20 GB standard disk: the free-tier shape),
  time zone America/New_York.
- **Access:** SSH only through IAP:
  `gcloud compute ssh c3-paper --zone=us-east1-b --tunnel-through-iap`.
- **Code:** `/home/c3/QuantCode`, owned by user `c3`. Its `.env` holds only the
  three `ALPACA_*_C3` lines, mode 600.
- **Schedule:** systemd `c3.timer`, Mon–Fri 08:00 ET, `Persistent=true`. It
  starts `c3.service`, which runs `run_c3.sh`: guard, session, 16:20 ET
  report, then the dashboard publish.
- **Deploy or update:** from a checkout of `research/live-r15`,
  ```
  git bundle create c3.bundle research/live-r15
  gcloud compute scp c3.bundle research/liquid/live/deploy/provision.sh c3-paper:/tmp/ --zone=us-east1-b --tunnel-through-iap
  gcloud compute ssh c3-paper --zone=us-east1-b --tunnel-through-iap --command="sudo bash /tmp/provision.sh /tmp/c3.bundle"
  ```
  `provision.sh` is idempotent: packages, the `c3` user, a 1 GB swap file, the
  clone (fast-forward only), the venv from `deploy/requirements-c3.txt`, and the
  systemd units.
- **Logs:** `/home/c3/QuantCode/data/research_liquid/live/logs/c3_<day>.log`,
  and `journalctl -u c3.service`.
- **Pause:** create the KILL file (stops trading now), or
  `sudo systemctl disable --now c3.timer` (stops the schedule).
- **Cost:** a budget with e-mail alerts at 25/50/100% of about $5 a month.
  Google Cloud has no hard spending cap for virtual machines.

## The dashboard

`dashboard.py` builds the data behind the claude.ai dashboard
(artifact `5ELZiTW9YF2qpXieq3yCFA`):

- `--dev OUT_DIR`, on a workstation: the research baseline files, dev window
  2016-01-04..2022-12-30 only. It asserts that it reproduces `l1_dev.json`, the
  saved daily returns, the by-year Sharpes and the 1.66 bp break-even.
- `--live --publish`, on the server after each report: two gzip+base64 JSON
  files written to Google Drive with the server's own service account (no key
  file; a metadata-server token). `data/research_liquid/live/publish.json`
  (server only, not committed) names the two Drive files. A publish failure
  never touches trading.

## Every trading day (the old PC route)

```
powershell -ExecutionPolicy Bypass -File research\liquid\live\run_c3.ps1
```

- **When to start it:** any time before 09:25 ET.
- **What it does:**
  - starts the guard, a separate process that flattens if the runner dies
    holding a position;
  - trades the session;
  - waits until 16:20 ET, then writes the evening report.
- **Where it writes** (under `data/research_liquid/live/`):
  - logs: `logs/`;
  - ledger: `c3_ledger.sqlite`;
  - heartbeat: `heartbeat.json`;
  - one report per day: `reports/<day>.json`;
  - the gates: `summary.json`.

## KILL file

`data/research_liquid/live/KILL` stops trading and keeps it stopped.

- **To stop now:** create the file, with any content. Within a few seconds the
  runner cancels every order, closes the SPY position and stops.
- **When it appears by itself:** the runner, the guard or the evening report
  writes it after any automatic kill, guard flatten or failed G-safety day. The
  reason is inside.
- **To restart:** log the cause in `research/liquid/LOG.md`, then delete the file.
  `--run` refuses to start while it exists.

## Gates

| gate | rule | when |
|---|---|---|
| G-cost (decides C3) | mean execution + fees per round trip against the 1.66 bp break-even, with day-clustered CIs | interim kill-only look at about 20 round trips (~6 weeks); decision at about 40 (~3 months); one extension to 80 |
| G-fidelity | pooled two-way Jaccard of entries against the SIP re-run ≥ 0.85; the live feed scored 0.97 on 2022 | same window as G-cost |
| G-safety | no kill; flat at the end; ledger equals broker; no position without a working stop for more than 60 s | every evening |
