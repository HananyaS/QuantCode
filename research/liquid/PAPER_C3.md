# C3 forward paper trading — a labelled deviation from L1, with its gates fixed in advance

**Status:**
- FIXED 2026-11-22, before any order was sent.
- **Amended A1 the same day, still before any order,** after the G3 review
  (`research/intraday/docs/critiques/G3-C3-paper.md`, verdict "ready with fixes").
  §6 lists what A1 changed and why.

**User decision (2026-11-22):** "Paper-trade C3".
- L1 cell C3 failed only rule (c), the 2×-cost stress (`LOG.md` §1). L1 rule 3
  would close the track.
- The user chose to measure C3's **real execution costs** on Alpaca paper instead.
  This is a **deviation from L1, recorded as such**.
- The 2023-01-03..2025-08-06 confirmation window stays sealed; reading it still
  needs the user's go.

**What this phase answers:** whether C3's realised trading costs on SPY sit inside
its dev break-even. A few months cannot measure its Sharpe, so performance is
reported, not gated.

**C3's break-even.** C3 earns 2.9 bp gross per trade on dev. Charged a uniform x bp
per round trip, its dev Sharpe equals SPY's (0.676) at **x = 1.66 bp**
(`results/c3_breakeven.json`). The cost model charges 0.99 bp at dev prices and
about 0.5 bp at today's.

## 1. Account and operation

- **Account.**
  - A **dedicated** Alpaca paper account, with keys `ALPACA_API_KEY_C3` /
    `ALPACA_SECRET_KEY_C3` in `.env`.
  - The code refuses the KELLY and LINEAR pairs, which the live CI system
    trades, and refuses any non-paper endpoint.
  - Shorting is disabled on the account, so an oversell can only be rejected.
- **Daily launch.** `run_c3.ps1` starts two processes:
  - the guard (`main --guard`), separate from the runner;
  - one session of the runner (`main --run`).
  It then runs the evening report (`main --report`).
- **Single instance.** A second runner or guard refuses to start (an OS file lock).
- **One-time drill (`main --drill`).** It confirms that shorting is off and that
  Alpaca rejects a duplicate client order id. It uses a 1-share buy limit at $1,
  which is then cancelled.

## 2. The rule (exactly `cells.c3`, with these live adaptations)

1. **Bars.**
   - Today's one-minute SPY bars: **SIP for minutes at least 16 minutes old**
     (free on the basic plan), **IEX for the most recent 16** (A1).
   - σ(minute) and the 14-day volatility come from SIP bars of the previous
     sessions, **dividend-adjusted as the research cache is** (A1). The raw
     previous close is logged too.
2. **Check at HH:00 / HH:30 (10:00..15:30; to 12:30 on a 13:00 close).** About
   5 s after the bar ending at the check closes, the runner reads today's bars
   and applies the registered decision:
   - enter iff flat and close > UB and close > VWAP;
   - the stop is max(UB, VWAP).
   If the decision bar has not been published, the runner retries for up to 25 s
   after the bar's end, then **skips the check** and logs it. A check more than
   30 s late (late start) is skipped and logged.
3. **Entry.**
   - Whole-share **buy limit, DAY**, at **floor(close × 1.001)** to the cent (A1).
   - Size: floor(last_equity × min(4, 0.02/σ_SPY) / close), capped by buying
     power. If Alpaca rejects for buying power, the runner resends once sized at
     99% of buying power at the limit price (A1). A second rejection skips the
     entry.
   - On the **first fill, the unfilled rest is cancelled at once** and the stop
     placed (A1). Unfilled orders are cancelled at the end of bar d+1, as in
     research.
4. **Stop.**
   - After the fill: a **sell stop, GTC**, at **floor(S)** to the cent, for the
     filled shares (A1: GTC so a dead runner leaves the position protected).
   - At each later check it is **replaced** (PATCH) at the new level. The
     replacement must read back open, at the new price, for the full position.
   - If the new level is at or above the price, or the replace is refused, the
     runner cancels the stop and sells at market (research: the stop triggers on
     the next bar).
   - A partly filled stop is never patched: it is cancelled and a new stop placed
     for the remainder (A1).
5. **Close exit.** At the start of the session's last minute (15:59:00; 12:59:00
   on a 13:00 close): cancel the stop, confirm, then sell at market. A paper
   market-on-close fills at the quote, not the auction (fact sheet item 4), so it
   is not used. Where research's stop triggers inside that last minute, live
   books a close exit instead: same entry and same exit bar, different label.
   That happened on 3 of 1,741 dev sessions.
6. **Re-entry.** A later check may enter again once flat, as in research.

**Measured parity (A1).** The real runner, filled by the research rules, was
replayed on all 1,741 valid dev sessions (`results/c3_live_parity.json`).
- **Floored prices:** entries and exit bars equal `cells.c3` on every session.
  The exit labels differ on the 3 sessions of item 5.
- **Control, rounding to the nearest cent (the pre-A1 code):** 98 sessions
  diverge, which reproduces the G3 count.

**Measured feed fidelity (A1).** The live feed was replayed on 2022 dev sessions,
with fills from SIP (`results/c3_iex_fidelity.json`). It was compared with
`cells.c3` on SIP by the pooled Jaccard share of (day, entry check):

| feed | Jaccard | 95% CI |
|---|---|---|
| IEX only | 0.86 | 0.80–0.91 |
| hybrid (SIP ≥ 16 min old, IEX after) | **0.97** | 0.94–0.99 |

## 3. Fees and measurements (A1)

- **Fees.** Paper charges none (fact sheet item 8). The report books what a real
  account would owe:
  - SEC $20.60 per million on sells (rate effective 2026-04-04);
  - CAT $0.000003/share on both sides;
  - TAF $0.000195/share on sells from 2027-01-01.
- **Per round trip**, all in bp of the research entry reference, against that
  day's SIP data. Each fill has a research reference: research's own fill price
  before costs.

  | fill | research reference | execution | latency |
  |---|---|---|---|
  | entry | SIP open of bar d+1 | fill − SIP NBBO mid at the broker's fill time | mid − reference |
  | resting stop | min(stop sent, SIP open of the fill's minute) | reference − fill | 0 |
  | forced market exit at check d' | SIP open of bar d'+1 | mid − fill | reference − mid |
  | close exit | SIP close of the last bar | mid − fill | reference − mid |

- **cost = execution + fees.** It is the part the cost model represents.
  **Latency** is the timing difference between live and research fills. It is
  mostly noise: about 2 bp per entry and 9 bp per close exit, one standard
  deviation. It is reported with its own CI and enters the pass bar (§4).
- **Every evening:** `cells.c3` is re-run on that day's SIP bars, and its
  entries are compared with the live entries. Every session without a report
  is processed, not only the latest.

## 4. Gates (A1)

**G-cost (decides C3).** The unit is the mean cost per round trip over complete
days, with bootstrap CIs that resample whole days (2,000 draws, seed 17).
B = 1.66 bp.

- **Interim look,** at the first days holding ≥ 20 round trips (about 6 weeks):
  **FAIL** if the 99% CI lower bound > B. The interim look never passes.
- **Decision,** at the first days holding ≥ 40 round trips (about 3 months):
  - **FAIL** if the 95% lower bound > B;
  - **PASS** if the 95% upper bound < B − max(0, latency's 95% lower bound);
  - otherwise, extend once to the first days holding ≥ 80 round trips and apply
    the same rule. Still neither → **INCONCLUSIVE**.
- The decision is taken once, on those days; later trips do not change it.
  Kill and guard exits are excluded from the cost and counted.
- **Power.** With a per-trip standard deviation up to 2 bp and about 2 trips per
  in-market day, the 95% half-width at 40 trips is about 0.6–0.9 bp. That is
  enough to tell a cost near the model's 0.5 bp from 1.66 bp.
- **What each outcome means:**
  - **FAIL:** C3 is killed.
  - **PASS:** C3's costs sit inside its dev break-even. It goes to the user for a
    decision on reading the confirmation window or a real-money pilot. It is not
    "C3 confirmed".
  - **INCONCLUSIVE:** the user decides. The default is to stop.

**G-fidelity (is the measurement about C3?).** Pooled Jaccard share of (day,
entry check), live against the SIP re-run, over the G-cost decision days.
- Live trades research missed count against it, and so do research trades live
  missed.
- **PASS at ≥ 0.85**, against 0.97 measured for the live feed on 2022 (§2).
- A FAIL halts paper trading until the mismatches are explained in the LOG. A
  cost measured on a runner that does not trade C3 is not evidence about C3.

**G-safety (operating rule, checked every evening).** A day is clean only if:
- no automatic kill or guard flatten happened;
- the account ended flat with no open order;
- every broker fill is in the ledger and every ledger fill is at the broker;
- no position was without a working stop for more than 60 s, outside the final
  minute. This is measured from the broker's entry fill time to the stop's
  placement, and from a stop cancel to the exit fill.

**Enforcement:**
- Any automatic kill, guard flatten or unclean day writes the **KILL** file.
- The next session will not start until the cause is logged in `LOG.md` and the
  file is removed.
- A trade left open in the ledger by a crashed run also blocks the next session.

**Reported, not gated:** daily P&L, and Sharpe against SPY/QQQ weekly.

## 5. What paper cannot tell us

Paper fills at the NBBO quote with no queue, no depth and no market impact (fact
sheet item 8). For a few hundred SPY shares the quote is close to real, but stop
slippage in a fast market and the closing-auction price are only tested with
real money. A small real-money pilot is the user's decision.

## 6. Amendment A1 (2026-11-22, before any order): what changed and why

| G3 item | was | now |
|---|---|---|
| B1 | median cost ≤ 2× model after 20 trips, weekly looks, no CI | absolute-bp mean against break-even 1.66 bp, cost split from latency, day-clustered CIs, one interim and one decision look (§4) |
| B2 | share of live trades matching research, live-only, uncalibrated 90% | pooled two-way Jaccard ≥ 0.85, calibrated at 0.97 (§2, §4) |
| B3 | flatten sent close right after cancel-all; no HTTP timeout; DAY stop; no watchdog | flatten waits for cancels and for flat; 10 s timeouts; GTC stop; separate guard process; shorting off (§1, §2.4) |
| B4 | G-safety stated, not enforced | KILL file on every kill and unclean day; ledger vs broker reconciliation; unprotected seconds measured; crashed-run block (§4) |
| W1 | stop and limit rounded to nearest cent (98/1,741 dev sessions diverged) | floored (0 diverge on entries and exit bars) |
| W2 | live history raw, L1 adjusted | adjusted history, raw previous close logged |
| W3 | missing decision bar silently carried | wait 25 s, else skip and log |
| W4 | partial entry unprotected until the bar's end | rest cancelled on the first fill |
| W5 | cancel/replace races killed the day | re-read and act on the order's state; partly filled stops cancelled, not patched |
| W6 | alpaca-py re-POSTs orders on 429/504; no lock | no POST retry, lookup by client id after any error, OS lock, duplicate-id drill |
| W7 | `run_c3.ps1` stopped at the first stderr line (verified) | Python launched through `cmd /c` |
| W8 | size could exceed buying power at the limit | resend at 99% of buying power on rejection |
| W9 | holdout record open-ended | bounded in `LOG.md` §3 |
| W10 | no code SHA | SHA in L1 output, ledger plan events and daily reports |
| W11 | report read only the latest session | every unreported session |

**The superseded A0 gates, for the record:**
- G-cost: median realised cost ≤ 2× modelled from 20 round trips, then weekly.
- G-fidelity: ≥ 90% of live trades matching.
