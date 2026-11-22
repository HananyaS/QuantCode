# Liquid intraday track — log

## 1. L1 — published SPY/QQQ day-trading rules, long-only, on dev: no cell passes

The plan was pre-registered in `PREREGISTRATION_L1.md` (commit `36bf880`, pushed
before any cell ran on market data). The data was dev only: 2016-01-04..2022-12-30,
1,762 sessions. **No 2023+ bar was read.** Trial count 374.

### Result (net of Alpaca costs; paired stationary bootstrap, block 20, 2,000 reps)

| series | Sharpe | 95% CI | gross | 2× cost | CAGR | maxDD | alpha vs SPY /yr (NW 95%) | Holm p | trades | net bp/trade |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 last half-hour, SPY | **−0.81** | [−2.09, 0.27] | −0.55 | −0.96 | −2.8% | −20.1% | −4.5% [−8.0, −1.0] | 0.99 | 963 | −2.1 |
| C2 5-min ORB, QQQ | **0.52** | [−0.14, 1.14] | 0.81 | 0.30 | +8.7% | −24.7% | +5.7% [−7.0, +18.3] | 0.38 | 881 | +2.7 |
| C3 noise area, SPY | **1.09** | [0.49, 1.68] | 1.76 | **0.62** | +7.6% | −8.6% | +5.5% [+1.2, +9.8] | 0.018 | 1,137 | +1.9 |
| SPY (total return) | 0.68 | | | | +11.5% | −33.8% | | | | |
| QQQ | 0.70 | | | | +14.5% | −35.0% | | | | |
| IWM | 0.45 | | | | +8.2% | −41.3% | | | | |

### Pre-registered rules

| cell | (a) Sharpe > SPY and QQQ | (b) Holm alpha p < 0.05 | (c) 2× cost Sharpe > SPY | dev |
|---|---|---|---|---|
| C1 | no | no | no | **fail** |
| C2 | no | no | no | **fail** |
| C3 | yes (1.09) | yes (0.018) | **no (0.62 < 0.68)** | **fail** |

**No cell passes.** Under rule 3, the long-only published SPY/QQQ rules are
closed and **no confirmation read follows**. C3 fails only the cost-stress
guard. That guard was fixed in advance and is not relaxed here.

### C3 in detail
- **Market exposure.** Beta to SPY is 0.08 (0.06 to QQQ, 0.06 to IWM). It is in
  the market on 32% of sessions, with 1.99 trades on each of those days (1,137
  trades on 571 days; corrected from "1.0" after the G3 review). Exits are
  83% stops and 17% at the close. The net win rate is 25%. Mean leverage is 2.7×,
  and 28% of trades sit at the 4× cap.
- **Edge per trade.** Gross is 2.9 bp a trade and net 1.9 bp, at about 1.0 bp of
  cost. The edge is a few basis points a trade, so the cost assumption decides
  the verdict.
- **Sharpe vs SPY.** +0.42 above, but not significantly: Ledoit-Wolf p 0.35,
  bootstrap p 0.20. The alpha *is* significant (t 2.51) because the strategy is
  nearly market-neutral and has low volatility.
- **By year:** 1.20, 0.02, 1.02, 0.65, −0.38, 2.58, 1.86 (2016..2022).
- **Deflated Sharpe** at N = 374: 0.46.

### Diagnostics (labelled, not selectable)
- **D1, does the implementation replicate the papers?** Long/short, with each
  paper's own costs, on dev:

  | cell | our Sharpe | our return /yr | paper |
  |---|---|---|---|
  | C2 | 1.02 | +29.9% | Sharpe 1.12, 31%/yr on 2016-01..2023-02 |
  | C3 | 1.12 | | 1.33 on 2007-2024 |
  | C1 | −0.59 | | Gao's sample ended in 2013; this agrees with Rosa (2022) that the effect disappeared |

  The simulator reproduces the papers. **The gap between the published numbers
  and ours is the long-only restriction plus realistic costs, not the code.**
- **D3, unlevered (cap 1×):** C2 0.46, C3 1.05. Leverage does not create C3's Sharpe.
- **D4, C3 with stops checked only at half-hours** (the paper's p.9 reading):
  0.86, against 1.09 for the resting stop.
  - The first run showed this diagnostic with no stop exits at all. The
    simulator treated a +inf stop (a market exit at the next open) as "no
    stop". It now treats only NaN as no stop; tests were added.
  - The re-run left all three cells and D1/D3 byte-identical.
- **D5, gross Sharpe:** C1 −0.55, C2 0.81, C3 1.76.

### What this means for the live system
- **C1 is dead**, as Rosa (2022) said.
- **C2's edge is real on long/short** (it replicates the paper), but the long
  half alone does not beat buy-and-hold.
- **C3 is the only rule with a market-beating, market-neutral return.** But its
  margin over SPY is a few basis points a trade, inside our cost uncertainty.
  Two unrefereed 2026 papers report that it weakened after publication.
- Taking C3 further would be a **deviation** from L1's rule 3, and the decision
  belongs to the user. The cleanest test of the cost question does not spend the
  confirmation window: forward paper trading C3 to measure real stop and
  closing-auction slippage on SPY. That needs the Phase 2 broker layer.

## 2. User decision: forward paper trading of C3 (labelled deviation)

2026-11-22. Asked how to proceed after L1, the user chose **"Paper-trade C3"**
over reading the confirmation window or accepting the kill.

C3 failed only L1's cost-stress rule (c). So the question that decides it is
**what C3 really costs to trade**, and that can be measured live without
spending the confirmation window. The rules, measurements and gates are fixed
in `PAPER_C3.md`, before any order is sent. The confirmation window stays sealed.

## 3. G3 review of the paper runner, and amendment A1 (before any order)

2026-11-22.
- **Verdict.** An independent G3 review
  (`research/intraday/docs/critiques/G3-C3-paper.md`) returned **"ready with
  fixes"**, with four blocking items.
- **What it confirmed.**
  - `cells.c3` implements the pre-registration with no look-ahead, and the fills
    are as registered.
  - The L1 verdict reproduces to the last digit: net Sharpe 1.0909, 2×-cost
    0.6179 < SPY 0.6759.
- **The fixes.** All are in `PAPER_C3.md` amendment A1, made before any order.

**Evidence produced for A1.** Each item is a diagnostic of the closed cell, or a
test of the live code on dev data L1 already read. None is a new trial.

| item | result | file |
|---|---|---|
| Break-even cost | C3's dev Sharpe equals SPY's at a uniform 1.66 bp per round trip. The model charges 0.99 bp at dev prices; gross is 2.9 bp per trade. | `results/c3_breakeven.json` |
| Runner parity | The real runner, filled by the research rules, was replayed on all 1,741 valid dev sessions. With prices floored to the cent, entries and exit bars equal `cells.c3` everywhere; 3 sessions label a last-minute exit "time" where research says "sl" (by design). Rounding to the nearest cent, the pre-A1 code, diverges on 98 sessions, reproducing the review's count. | `results/c3_live_parity.json` |
| Feed fidelity, 2022 | Pooled Jaccard of entries against research: IEX only 0.86 [0.80, 0.91]; hybrid SIP ≥ 16 min + IEX 0.97 [0.94, 0.99]. Skipping or carrying a missing decision bar changed nothing. The hybrid feed is adopted, and G-fidelity's bar is 0.85. | `results/c3_iex_fidelity.json` |
| Launcher (W7) | Under Windows PowerShell 5.1, `$ErrorActionPreference = "Stop"` with `& python ... *>> log` aborted at the first stderr line; the log held one line. The `cmd /c` launcher keeps every line. Smoke-tested with a harmless one-liner. | — |
| SIP quotes | The basic plan serves historical SIP NBBO quotes (2,521 quotes in 3 s at 10:00 on 2026-11-20), so the cost gate can measure execution against the mid at each fill. | — |

**Corrections.**
- §1 said C3 trades "1.0" times per in-market day. It is 1.99 (1,137 trades on
  571 days).
- So 20 round trips take about 31 sessions, and 40 about 62.

**Holdout bookkeeping (W9).**
- The pre-registration's holdout starts 2025-08-07. For the liquid track it is
  now bounded as **2025-08-07..2026-11-20**.
- These reads touched that window; each computed **entry decisions, not
  returns**:
  1. 2026-11-22, before A1: an ad-hoc IEX-vs-SIP comparison of C3's entry flags
     on the 25 sessions 2026-10-16..2026-11-20. The flags agreed on 99.3% of 300
     checks. The script was not kept; the trade-level 2022 calibration above
     replaces it.
  2. `main --check` and `plan_for` read the previous 20 sessions of SIP bars to
     build each day's plan.
- No return in that window has been computed. Its sealing for any future
  L-track test stands.
- **The forward paper window** starts with the first paper session. It is new
  data, not the holdout.

**Keys used for market-data reads before the C3 account existed.** The
IEX-vs-SIP comparisons and the SIP quote probe read market data with the KELLY
key pair. No order was sent and no account endpoint was called. The committed
scripts default to the C3 pair; `--env-pair` records any other.
