# C3 paper trading — how to run it

The rules and gates are in `research/liquid/PAPER_C3.md`.

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
   It prints the account, the market clock, SPY position and open orders, and the
   next session's plan: 14-day volatility, weight and notional.

## Every trading day

```
powershell -ExecutionPolicy Bypass -File research\liquid\live\run_c3.ps1
```

- **When to start it:** any time before 09:25 ET.
- **What it does:** trades the session, waits until 16:20 ET, then writes the
  evening report.
- **The machine must stay awake** until about 16:25 ET.
- **Where it writes:** the log goes to `data/research_liquid/live/logs/`. The
  ledger is `data/research_liquid/live/c3_ledger.sqlite`. Each day's report goes
  to `data/research_liquid/live/reports/<day>.json`, and the running gates to
  `data/research_liquid/live/summary.json`.

## Kill switch

Create `data/research_liquid/live/KILL` (any content). Within a few seconds the
runner cancels every order, closes the SPY position and stops. Delete the file
before the next session.

## Gates (from 20 round trips on)

- **G-cost:** median realised cost ≤ 2× modelled. A FAIL kills C3.
- **G-fidelity:** ≥ 90% of live trades match the SIP research re-run.
- **G-safety:** any reconciliation break or stop loss triggers the kill switch,
  and the runner records it in the ledger.

Measured before the first session (25 sessions, Sep–Nov 2026): IEX and SIP gave
the same entry decision at 99.3% of 300 checks.
