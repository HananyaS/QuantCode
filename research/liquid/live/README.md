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

## Every trading day

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
