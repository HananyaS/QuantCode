# L1 Pre-Registration — published SPY/QQQ intraday rules as long-only brackets

**Status:** FIXED 2026-11-22, before any cell ran on real market data. Results go
in `research/liquid/LOG.md` and an outcome table added above §0 afterwards; the
body below is not edited once a run exists.

## Outcome, recorded after the fact (2026-11-22)

| cell | net Sharpe | 2× cost | alpha vs SPY (NW 95%) | Holm p | dev |
|---|---|---|---|---|---|
| C1 | −0.81 | −0.96 | −4.5% [−8.0, −1.0] | 0.99 | fail |
| C2 | 0.52 | 0.30 | +5.7% [−7.0, +18.3] | 0.38 | fail |
| C3 | 1.09 | 0.62 | +5.5% [+1.2, +9.8] | 0.018 | **fail on rule (c) only** |
| SPY / QQQ | 0.68 / 0.70 | | | | |

**No cell passes; no confirmation read follows (rule 3).** Details are in
`research/liquid/LOG.md` §1. After the run, a bug in diagnostic D4 only was
fixed: the cells and the other diagnostics were byte-identical on re-run.

**What existed when this was written:**
- the code: `research/liquid/` (calendar, data, costs, bracket, cells);
- 21 unit tests on synthetic bars (`tests/test_liquid.py`);
- the dev minute cache;
- the benchmark daily bars for 2016-2022.

The cache build read only prices. Its checks were: the early-close list against
volume, and session coverage (1 invalid session per symbol). No strategy return
had been computed on market data.

**Sources:** `research/intraday/docs/research/2026-11-liquid-intraday-literature.md`,
which has verified citations and the papers' own rules with page references.

---

## 0. Why this round exists

The goal is a live, minute-driven system that sends a **buy limit with a take-profit
limit and a stop** to Alpaca (paper) and beats SPY and QQQ buy-and-hold. The only
penny strategy that had passed earlier tests (R15) was killed in R17 (`research/penny/LOG.md` §101). The
user then chose the liquid intraday track. It takes **published, verified** day-trading
rules on SPY/QQQ, with **the papers' own parameters**, and asks whether their
**long-only** versions, at Alpaca costs, beat buy-and-hold.

Cumulative trial count on the intraday track: **371 → 374**. These are the
only selectable cells.

## 1. Data and windows

- **Bars.** One-minute SIP bars for SPY and QQQ, 09:30-15:59 ET (390 bars), or
  09:30-12:59 on the 13 dev early closes (`calendar.EARLY_CLOSES`, checked against
  volume). Source: `data/research_poolpoc/etf/minute`.
- **Prices.** Dividend-adjusted: exact within a session, and a total-return path
  across sessions.
- **Gaps.** A minute with no trade carries the previous close forward. A session
  without its 09:30 bar, or with under 95% of its bars, is flat for every cell
  (SPY 2019-08-12, QQQ 2016-02-22).
- **Prior close** = the close of the previous session's last regular bar.
- **Dev** = 2016-01-04..2022-12-30, 1,762 sessions. This run reads nothing later.
- **Confirmation window** = 2023-01-03..2025-08-06. It is read **once**, only on
  the user's explicit go (§5).
- **Holdout** = 2025-08-07 onward. It has not been fetched and stays sealed.
- **Overlap with the papers' own samples.** Dev lies inside the samples of C2
  (QQQ 2016-01..2023-02) and C3 (SPY 2007-05..2024-04), and after C1's (1993-2013).
  So for C2 and C3, dev tests whether the rule replicates, long-only and at our
  costs. It is not out-of-sample evidence. Post-publication data begins:
  - C1: 2018;
  - C2: 2023-04;
  - C3: 2024-05.

## 2. Execution model (all cells)

### 2.1 Orders
- An order is decided at the **close of bar `d`**.
- **Entry: a marketable buy limit** at `close[d] × 1.001`, working only in bar
  `d+1`:
  - fills at `open[d+1]` if `open[d+1] ≤ limit`;
  - otherwise fills at the limit if `low[d+1] < limit`;
  - otherwise it is cancelled (no trade).
- Take-profit and stop are attached to the entry (a bracket) and are live from the
  entry bar.
- The sizing and the bracket prices are fixed when the order is sent. They use
  `close[d]`, not the fill price, because a bracket is submitted before it fills.

### 2.2 Fills
- **Take-profit** (resting sell limit): fills at the TP when `high > TP`. Strictly
  above, because the queue ahead must trade first. On the entry bar it fills at
  `max(TP, entry)`.
- **Stop** (sell stop): triggers when `low ≤ stop` and fills at
  `min(stop, open)`, so a gap fills at the open. On the entry bar it fills at
  `min(stop, entry)`.
- **Both in one bar:** the stop.
- **Time exit:** at the close of the session's last bar, as a market-on-close.
- **No position is held overnight.**

### 2.3 Returns
- `daily return = Σ over the day's trades of weight × net return`, where `weight`
  is notional divided by the prior-day equity.
- A day with no trade returns 0; idle cash earns nothing.
- Sharpe = `mean / sd × √252`.

### 2.4 Leverage
- Capped at **4×**, the papers' own cap.
- Alpaca retired the PDT rule on 2026-06-04
  (`2026-11-alpaca-order-facts.md` item 9). The cap the new intraday-margin
  framework gives a paper account is **not verified**, so the run also reports an
  unlevered version (§6, D3).

### 2.5 Costs (Alpaca, $0 commission) — `research/liquid/costs.py`

| fill | cost per share |
|---|---|
| marketable entry | half-spread $0.005 + CAT $0.000003 |
| take-profit (resting limit) | CAT + TAF $0.000195 |
| stop | half-spread + one tick $0.01 + CAT + TAF |
| time exit (MOC) | half-spread + CAT + TAF |
| every sell | plus the SEC fee, 2.78e-5 × notional |

- The per-share amounts are divided by the **adjusted** price. That price is below
  the raw price, so costs are slightly overstated.
- **Cost stress (part of the rule, §5):** the half-spread and the stop slippage
  doubled (`costs.DOUBLE`).

## 3. Cells — exactly three, all long-only

Each paper is long/short and none reports a long-only result. **Each cell is a
derivative of its paper, not a replication.** Long-only is the user's order
structure, fixed in advance. It is not chosen from results.

### C1 — last-half-hour momentum, SPY (Gao, Han, Li & Zhou 2018)
- **Signal:** `r1 = close(09:59 bar) / prior close − 1`, the price at 10:00
  against the prior 16:00 close.
- **Entry:** if `r1 > 0`, the order is decided at the close of the bar 31 minutes
  before the close (15:29; 12:29 on half days) and enters in the 15:30 bar.
- **Exit:** the time exit at the close. No take-profit and no stop: the paper has
  none.
- **Size:** weight 1.0. The paper gives no sizing.

### C2 — 5-minute opening-range breakout, QQQ (Zarattini & Aziz 2023)
- **Opening range:** bars 09:30-09:34. `O = open(09:30)`, `C = close(09:34)`,
  `L = min low`.
- **Trade iff `C > O`.** `C = O` is the paper's doji (no trade); a down bar is
  flat long-only.
- **Bracket:** entry decided at 09:34's close. Stop at `L`, with `R = C − L`.
  Take-profit at `C + 10R`. Otherwise the time exit.
- **Size:** `weight = min(0.01 × C / R, 4)`, the paper's 1% of equity at risk with
  a 4× cap.
- **Excluded:** the paper's post-hoc ATR-stop/TQQQ variant.

### C3 — noise-area momentum, SPY (Zarattini, Aziz & Barbon 2024)
- **Move:** `move(t, m) = |close(t, m) / open(t, 09:30) − 1|`. `σ(t, m)` is its
  mean over the **previous 14 sessions** (at least 10 observations).
- **Upper boundary:** `UB(t, m) = max(open(t, 09:30), prior close) × (1 + σ(t, m))`,
  with volatility multiplier 1.
- **VWAP:** the session VWAP through bar `m`, computed as Σ(bar vwap × volume) /
  Σ volume.
- **Checks:** at 10:00, 10:30, …, 15:30, each decided at the close of the bar
  ending at that time (09:59, …, 15:29), and only before the last bar.
- **Entry:** while flat, enter if `price > UB` **and** `price > VWAP`.
  - The VWAP condition is fixed here. Without it, the paper's own stop
    `max(UB, VWAP)` would sit above the entry and close it on the entry bar.
  - The authors' code reportedly adds the same condition (literature A4, not
    verified).
- **Stop:** `max(UB, VWAP)` as of the entry check. It is **re-priced at each
  later check** and live from the bar after it. It is a **resting stop**, the
  paper's "as soon as the price crosses" (p.13).
  - The live system must support replacing a stop order.
  - The paper's other reading (p.9: stops checked only at half-hours) is
    diagnostic D4.
- **Exit and re-entry:** no take-profit; the time exit at the close. After an
  exit, a later check may enter again (the paper averages 1.8 trades a day).
- **Size:** `weight = min(4, 0.02 / σ_SPY)`. `σ_SPY` is the standard deviation
  (ddof 1) of the previous 14 daily returns, regular close to regular close. It
  is fixed for every trade that day.

## 4. Statistics

- **Series:** each cell's dev daily net returns, against SPY and QQQ daily total
  returns on the same sessions (`research/penny/scripts/fetch_benchmarks.py`,
  Alpaca `adjustment=all`). IWM is a diagnostic.
- **Function:** `research.penny.eval.benchmark.compare(..., mean_block=20,
  n_boot=2000, seed=17, alpha_lags=20, flat_ok=True)`.
  - `flat_ok` is used because the strategies are flat overnight, so a flat day is
    genuinely flat.
  - Paired stationary bootstrap, Ledoit-Wolf Sharpe difference.
- **Alpha:** Newey-West alpha vs SPY on excess returns over DTB3, with
  `p_one = Φ(−alpha_t)`.
- **Also reported:**
  - gross and net Sharpe, CAGR, maximum drawdown, total return;
  - trades, days in the market, exit mix, mean weight;
  - net and gross bps per trade, cost per trade;
  - deflated Sharpe at N = 374.

## 5. Decision rules (fixed now)

1. **Dev pass for a cell** requires all of:
   - (a) net Sharpe above SPY's **and** above QQQ's dev Sharpe (point estimates);
   - (b) the one-sided NW alpha p-value vs SPY, **Holm-adjusted over the 3
     cells**, below 0.05;
   - (c) with costs doubled, the net Sharpe is still above SPY's dev Sharpe.
2. **Cells that pass** go to the confirmation window (2023-01-03..2025-08-06) once,
   **only on the user's explicit go**, with every setting unchanged.
   - **Confirmation pass:** (a) and (b) on that window, with Holm over the cells
     taken.
   - **"Unconfirmed":** Sharpe above both benchmarks but the alpha test fails.
   - **Fail:** anything else.
   - A confirmation pass makes the cell the live system's first strategy (forward
     paper trading, plan Phase 3).
3. **No cell passes dev:** the long-only published SPY/QQQ day-trading rules are
   closed. No confirmation read follows.

## 6. Diagnostics (labelled, never selectable)

- **D1 — replication check of the implementation.** Each paper's long/short rule
  with the paper's own costs, on dev:
  - C2 with $0.0005/share;
  - C3 with $0.0035 + $0.001/share;
  - C1 at our costs.
- **D2:** Sharpe by calendar year (stability).
- **D3:** C2 and C3 unlevered (cap 1×).
- **D4:** C3 with stops checked only at half-hours (the paper's p.9 reading).
- **D5:** the gross (zero-cost) Sharpe for each cell.

## 7. What this round cannot tell us

- **Fills.**
  - A one-minute bar has no queue, so a limit that trades through by a cent is
    assumed filled.
  - Market-on-close is modelled as the 15:59 bar's close plus a half-spread, not
    the official closing-auction price.
- **No quotes.** Spreads are assumed to be one cent; SPY and QQQ are almost always
  one tick wide.
- **Overlap.** Dev overlaps the C2 and C3 papers' samples (§1). Two unrefereed
  2026 SSRN papers report that C3 weakened after publication (literature A3 and
  A4). Rosa (2022) reports that C1's predictability disappears out of sample (A1).
- **Leverage.** Alpaca's intraday cap is unverified (§2.4).
