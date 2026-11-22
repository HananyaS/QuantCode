---
gate: G3 (before paper trading)
subject: C3 "Beat the Market" noise-area momentum, SPY long-only - forward paper trading (PAPER_C3.md) and the live runner (research/liquid/live/), branch research/live-r15 at 02264fb as stated by the caller (git not run, on the caller's instruction)
verdict: block
---

# G3-C3: before paper trading

Severity key: **[block]** = blocking, must be fixed before the first order; **[warn]** = should-fix;
**[note]** = record only.

**What I ran (read-only, nothing written except this file):**
- `pytest tests/test_liquid.py tests/test_liquid_live.py`: 35 passed.
- I read `data/research_liquid/l1_dev.json`.
- In-memory diagnostics on the dev SPY cache, which L1 had already read. They are labelled
  diagnostics of a closed cell, not new trials:
  - `cells.c3` re-run at several cost levels;
  - a **replay of the real `Runner`** on every valid dev session from index 20 (1,741 sessions).
    It used `FakeFeed`, a bar-driven `FakeBroker` subclass (the limit fills by the research
    rule, the stop fills when `low <= stop_price`, a market order fills at the current bar's
    open), a scripted clock, and an in-memory ledger.
- No broker was constructed, no order sent, no `.env` read and no git run.

**Overall:**
- **Research.** The research cell is implemented as registered, and the L1 verdict, including
  the rule (c) fail, reproduces exactly.
- **Decision logic.** The runner's decision logic matches the research cell **exactly once
  prices are floored to the cent**. As written, it rounds to the nearest cent, and 98 of 1,741
  dev sessions diverge (W1).
- **What blocks.**
  - The decisive gate G-cost has the wrong units and rationale, and its references are
    noise-dominated.
  - G-fidelity counts in one direction only.
  - The kill/flatten path can leave a position with no stop, and overnight.
  - G-safety, as PAPER_C3 states it, is neither enforced nor measured.

---

## 1. Research

**[note] `cells.c3` implements PREREGISTRATION_L1 §3 C3 exactly. I found no look-ahead in the
noise area, the VWAP, the volatility, the check bars or the stop arrays.**
- evidence:
  - σ sums rows i-14..i-1 with n >= 10 (cells.py:128-131); `move[~valid] = NaN` (cells.py:122).
  - vol is the ddof-1 rolling std of close-to-close returns, `.shift(1)` (cells.py:132-134).
  - UB = max(open0, prior close)·(1+σ) (cells.py:159).
  - VWAP is the cumulative Σ(bar vwap·vol)/Σvol through bar d inclusive (cells.py:138-144).
  - The checks are the decide bars 29..359 with d < last, so 12:30 is the last on a 13:00 close
    (cells.py:33-34, 162).
  - Entry needs c > UB and c > VWAP (cells.py:168).
  - The stop is max(UB, VWAP) at each check, live from bar a+1 (cells.py:176-179).
  - Re-entry is allowed once exit_bar <= d (cells.py:164-166, 184-185).
  - The weight is min(4, 0.02/vol) (cells.py:163).
  - `test_c3_noise_area_reads_only_the_past` passes.
- Only exception: `valid` uses the whole session's bar coverage, >= 95% (data.py:84-85,
  cells.py:155). That is day-D information, but it affects 1 dev session (SPY 2019-08-12).
- resolve: nothing required. Note it in the LOG.

**[note] The fills in `bracket.py` are as registered, with no hidden optimism.**
- evidence:
  - The entry fills at open if open <= limit, else at the limit if low < limit, strictly
    (bracket.py:55-60).
  - The stop fills at min(stop, open), or at min(stop, entry) on the entry bar (bracket.py:64-68).
  - C3 has no take-profit, so intrabar ordering does not matter.
  - No fill occurs inside a halt: a halted minute is a carried bar with low = prior close, and a
    gap after it fills at the open.
- House-rule deviation: the house rule is "SL never fills at the stop level". Here a stop fills
  at its level plus a modelled $0.015/share (half-spread + 1 tick). For a resting broker stop on
  SPY that is a defensible model, and it is exactly the assumption G-cost exists to test. The
  limit filling on a 1-cent trade-through and the MOC modelled as the 15:59 bar's close are
  disclosed in prereg §7.
- resolve: none.

**[note] The cost model is conservative in its fees and its price divisor, and may understate in
fast markets.**
- evidence:
  - Per-share costs are divided by *adjusted* prices: C3's mean entry is $282, range $158-$445,
    below raw.
  - SEC 2.78e-5 and TAF $0.000195 are at or above the dev-era rates (costs.py:17-21).
  - But spread is a constant 1 cent and stop slippage 1 tick. Neither widens in fast markets
    (2020), which is where breakouts and stop-outs cluster.
  - Latency is not modelled. Research fills at the open of bar d+1; live fills about 5-8 s later.
    The mean drift is negligible: dev SPY's 1-minute return after C3 entry bars is +0.20 bp.
- resolve: none for L1. Paper measures the normal-market part (PAPER_C3 §5).

**[note] The L1 verdict is correctly computed from the saved outputs.**
- evidence (`l1_dev.json`):
  - C3 net Sharpe 1.0909, above SPY 0.6759 and QQQ 0.6988, so (a) holds.
  - One-sided p = 0.006077; Holm ×3 gives 0.0182 < 0.05, so (b) holds. C2 is ×2 → 0.381 and C1
    0.994.
  - 2×-cost Sharpe 0.6179 < 0.6759, so (c) fails.
  - Rule (c) compares like with like: both are `bm.sharpe` on raw returns over the same
    `idx[1:]` (benchmark.py:40-45, 191-214; l1_dev.py:88-104).
  - My in-memory re-run of `cells.c3` gives BASE 1.090851241721676 and DOUBLE
    0.6179001088804802, identical to the JSON. DSR 0.46 at N = 374.
- resolve: none.

**[note] LOG §1 misstates C3's trade frequency.**
- It says "1.0 trades on each of those days". The JSON has 1,137 trades on 571 days, which is
  **1.99 per in-market day**. Per day: 1 trade on 265 days, 2 on 148, 3 on 88, 4 on 48, 5+ on 22.
- So the 20-round-trip gate point arrives after about 31 sessions (6 weeks), not 3 months.
- evidence: LOG.md:34-35 vs `l1_dev.json` cells.C3.trades.
- resolve: correct the LOG line.

## 2. Live vs research parity

**[warn] W1 - The runner rounds the stop and the limit to the nearest cent. On a penny grid that
diverges from research in 5.6% of sessions. Flooring gives exact parity.**
- evidence:
  - `signal.tick` uses `round(price, 2)` (signal.py:92-94). It is applied to the stop
    (runner.py:204, 233), the limit (runner.py:178) and the forced-exit test (runner.py:234).
  - The dev cache is 100% on the cent grid (checked: open, low and close over 684,840 bars).
    Live raw prints are cent-grid too.
  - A sell stop rounded *up* triggers on a print exactly at that cent, where research's
    unrounded stop does not.
- Replay of the real Runner on 1,741 dev sessions:
  - 1,643 have identical (entry check, exit kind, exit bar) lists; 98 diverge.
  - Of the 98, the first divergence is the stop rounded up in 91 cases, the forced-exit test in 5
    and other rounding in 2.
  - Live makes 1,153 trades against research's 1,131: earlier stop-outs lead to extra
    re-entries.
  - Rounding alone puts G-fidelity's live_matched_share at **0.981** on perfect data.
- With `tick` = floor to the cent for the stop, the limit and the forced test, **1,741 of 1,741
  sessions are identical**. That holds because on a cent grid, p <= L ⇔ p <= floor(L).
- resolve: floor the sell stop, the buy limit and the forced-exit comparison to the cent. Store
  the price actually sent as `stop_at_exit`; report.py:60 now uses the unrounded level. Add the
  replay as a test.

**[warn] W2 - The live history is on a different adjustment basis from L1, and PAPER_C3 does not
record it.**
- evidence:
  - `StockBarsRequest` is built with no `adjustment` (feed.py:72-73), so Alpaca returns raw bars.
  - L1's cache is dividend-adjusted (data.py:4-6; PREREG §1).
  - On an SPY ex-dividend day, the live prior close is about the dividend (~0.3%) above the
    adjusted one. That raises UB by up to ~0.3% against σ(10:00) of a similar size, so there are
    fewer entries.
  - The 14-day vol differs on the 14 sessions after each ex-date, about 22% of sessions.
  - The evening re-run also uses raw SIP (main.py:106-108), so G-fidelity is measured against a
    raw-basis re-run, not the L1 basis.
- resolve: record it in PAPER_C3 §2. Alternatively, request `adjustment=dividend` for history and
  adjust today's prior close for an ex-date.

**[warn] W3 - A stale decision bar is used silently.**
- evidence: `today_from_rows` carries *every* missing bar from 1..upto, including the decision
  bar `upto` itself (feed.py:48-53). The check runs at bar_end + 5 s (runner.py:142).
  - If Alpaca has not yet published (or has since revised) the IEX bar for minute d, the runner
    decides on close[d-1], and `complete` is still True.
  - The test covers interior gaps only (test_liquid_live.py:51-57).
- resolve:
  - Require bar `upto` to be present in the response; retry for a few seconds, else log
    `check_skipped`.
  - Log `n_carried` per check.

**Parity table: every place live differs from `cells.c3`, and whether PAPER_C3 records it.**

| difference | code | in PAPER_C3? |
|---|---|---|
| today's bars from IEX (open0, close, VWAP inputs) | feed.py:83-86 | yes, §2.1 |
| raw, not dividend-adjusted, history | feed.py:72-73 | **no** (W2) |
| stop and limit rounded to the nearest cent | signal.py:92-94 | the rounding yes; its 5.6% effect no (W1) |
| new stop live ~5-8 s into bar a+1, not at its open | runner.py:142 | implicitly ("about 5 s") |
| forced market exit if level >= price | runner.py:234-238 | yes, §2.4. Its equivalence claim holds on dev once floored. |
| entry works ~54 s, not the whole bar | runner.py:181-190 | yes, §2.3 |
| partial entry kept | runner.py:191-198 | yes, §2.3 |
| whole shares, `last_equity`, buying-power cap | signal.py:86-89 | yes. The cap is applied on the decision price, not the limit: no (W8) |
| stop cancelled at 15:59:00 (research's stop is live through bar `last`) | runner.py:148-149 | yes, §2.5 |
| check skipped if > 30 s late, or if IEX has no 09:30 bar | runner.py:32, 143-145, 159-161 | **no** |
| `valid` (whole-session coverage) vs `complete` (09:30 bar) | data.py:85; feed.py:48 | **no** (note) |
| the stop's VWAP from IEX volume moves the exit level | signal.py:67-72 | IEX yes; exit divergence not gated (note) |

## 3. Order safety

**[block] B3 - The kill switch can leave a position with no stop and no runner, and then
overnight.**
- **The flatten race.** `flatten` calls `cancel_orders()` and immediately afterwards
  `close_position()` (broker.py:150-153).
  - Alpaca's DELETE /orders is asynchronous; it is a bare DELETE in alpaca-py, client.py:203-215.
  - A stop still in `pending_cancel` holds the shares. Fact sheet item 5: a second exit is
    rejected with "insufficient qty". So `close_position` can raise.
  - `_kill` then records the trade as closed (runner.py:298-302) and the process exits.
  - The result is a position whose stop may already be cancelled, and nothing will flatten it.
  - `FakeBroker.flatten` cancels and zeroes the position at once (broker.py:221-226), so no test
    can see this.
- **Every other runner death** is a hard crash, a network loss during flatten, the PC sleeping,
  or a hung request. alpaca-py sets **no HTTP timeout**: `self._session.request(method, url,
  **opts)` (site-packages/alpaca/common/rest.py:195). After any of these:
  - the position keeps only its DAY stop, which expires at 16:00;
  - there is no 15:59 exit, so the position is held overnight at up to 4×;
  - nothing flattens it until the next morning's `_reconcile("start")`, if the scheduler runs.
- **A hard crash between the entry fill and `_place_stop`** (runner.py:194-200) leaves no stop
  at all.
- resolve:
  1. `flatten`: cancel all, then poll `open_orders` until it is empty (with a timeout), then
     `close_position`, then poll `position_qty` until it is 0, retrying with backoff. Make the
     fake model `pending_cancel`, and add a test.
  2. Set a request timeout on the trading and data clients.
  3. Add an independent end-of-day guard: a separate scheduled task at about 15:57 that flattens
     if the runner's heartbeat is stale. Optionally, use an OTO entry so that Alpaca, not the
     runner, owns the stop.

**[warn] W4 - A partial entry fill has no stop for up to about 54-74 s, beyond the 60 s G-safety
limit.**
- evidence:
  - The entry loop only breaks on `status == "filled"` or at the deadline, bar_end(d+1)
    (runner.py:181-187). A `partially_filled` order therefore keeps polling with the filled
    shares unprotected.
  - Only after cancel + `_settle` (up to 20 s, runner.py:188-190) is the stop placed. A partial
    at +6 s leaves the shares unprotected for about 54 s plus the cancel settle.
  - Paper partial-fills 10% of orders (fact sheet item 8).
  - Research protects from the fill: the stop is live on the entry bar.
- resolve: on the first `filled_qty > 0`, cancel the remainder at once (or stop the filled
  shares immediately), then place the stop.

**[warn] W5 - The cancel and replace races raise or kill, and the real exit price is lost.**
- **The cancel races.** `b.cancel` raises when the order has just filled: alpaca-py raises
  `APIError` on any non-429/504 error (rest.py:199-207), and DELETE on a filled order is a 422.
  - At the entry deadline (runner.py:189), this kills the day, and the ledger never opens the
    trade.
  - At a forced exit or at 15:59 (runner.py:261), the stop's real fill is booked as
    `exit_kind="kill"`, `exit_price=None`. G-cost then drops it (main.py:111).
- **A replace rejected for a benign reason** (for example a 6-s-stale price leaves the level
  above the market) kills the day instead of exiting (runner.py:244-249).
- **The replace read-back does not check quantity** (runner.py:250-252). `_poll_stop` ignores a
  `partially_filled` stop (runner.py:224), so a later check can PATCH a partly filled stop. What
  Alpaca does with the replacement's quantity is not in the fact sheet.
- resolve:
  - Wrap cancel and replace: on error, re-read the order and act on its state. A filled order
    closes the trade with its price; a rejected replace means cancel + market exit.
  - Assert that `got.qty - got.filled_qty` equals the position.
  - Never PATCH a partially filled stop: cancel it, then place a new stop for the remainder.

**[warn] W6 - Double entry is possible if two instances run, or if an order POST is retried.**
- evidence:
  - There is no single-instance lock (main.py:80-96).
  - alpaca-py retries a POST after 429 or 504 (constants.py:13; rest.py:129-135). A 504 on an
    order that was in fact accepted is re-sent with the same `client_order_id`.
  - Duplicate rejection is "forum-only" (fact sheet item 11). If Alpaca does not reject it, the
    position doubles, the stop covers half of it, and the 15:59 exit sells half.
- resolve:
  - Add a lock file.
  - Before day 1, verify duplicate-coid rejection on the paper account with one far-from-market
    limit.
  - After any submit error, look the order up with `GET /v2/orders:by_client_order_id`.

**[warn] W7 - run_c3.ps1 may end at the first log line.**
- Under Windows PowerShell 5.1 (`powershell -File`), `$ErrorActionPreference = "Stop"` plus
  `*>>` on a native command turns stderr lines into a terminating NativeCommandError.
- `logging.basicConfig` writes to stderr (main.py:37), first at main.py:92, before the session
  starts.
- Depending on how PowerShell stops the pipeline, either the runner is killed (no trading) or it
  is orphaned (it trades with no log and the `--report` step never runs). A Python warning
  printed mid-session is the dangerous variant.
- evidence: run_c3.ps1:4,14. **Unverified.**
- resolve:
  - Smoke-test it with a harmless Python one-liner that writes to stderr.
  - Then use `$ErrorActionPreference = "Continue"` around the native call, or `pwsh`, or a
    logging FileHandler.

**[warn] W8 - The buying-power cap uses the decision price, not the limit.**
- `shares` floors `min(equity·w, BP)/price` (signal.py:86-89), but the order is a limit at
  price × 1.001 (runner.py:178).
- When the cap binds, qty × limit exceeds buying power by about 0.1%. On dev, 28% of trades sit at
  the 4× cap, which is exactly where the cap binds against a 4× intraday buying power.
- If Alpaca rejects the order, the runner raises and kills the day (runner.py:153-155). The fake
  never checks buying power.
- resolve:
  - Size on `limit` with a margin, for example ≤ 0.99 × BP.
  - Treat a buying-power rejection as `entry_skipped`.

**[note] Paths checked and found sound:**
- **Replace semantics.** The new order id is tracked and read back for open state and price
  (runner.py:250-253).
- **Rejected stops.**
  - A synchronous rejection leads to a market exit (runner.py:211-214).
  - An asynchronous rejection triggers a kill (runner.py:224-226).
- **Stop filled at a check.** A stop that fills before a replace or forced exit is detected first
  (runner.py:230-232, 246-248).
- **Partial fills at exit.**
  - A partly filled stop that is then cancelled is netted before the market exit
    (runner.py:266-271).
  - A market exit that is not filled kills (runner.py:276-278).
- **Session timing.**
  - The 15:59:00 and 12:59:00 close-outs come from the broker calendar (main.py:41-47;
    runner.py:148). DST is handled.
  - A late start skips past checks (test passes).
- **Restart mid-session.** A restart with a leftover position or orders flattens and stops
  (runner.py:289-294; test passes). The crashed run's ledger row stays open, though (B4).
- **Kill switch.** The kill file is polled every 2 s and inside the entry loop. It is not polled
  during `_settle` (up to 20 s).
- **Rate limits.** The peak is about 120 req/min in `_settle`, 60/min in the entry loop and
  30/min in a position, all under the 200/min limit.
- **client_order_id.** It is unique per day, check, trade and kind, under 128 characters
  (runner.py:179, 210, 243, 274).
- **Oversell and short.** I found no path to an oversell or a short, given that Alpaca rejects a
  sell beyond `qty_available` (fact sheet item 5) and cancels a replacement whose original
  filled. The second assumption is not in the fact sheet.

## 4. Measurement and gates

**[block] B1 - G-cost, the gate that "decides C3", can pass or fail for the wrong reason.**
1. **Wrong rationale.** PAPER_C3:73-75 sets the bar at "≤ 2× modelled … because its 2×-cost
   Sharpe is below SPY's". But at dev prices, 2× is **already** below SPY: 0.618 < 0.676.
   - The dev break-even is 1.876× on spread+slip, or 1.62× on total cost.
   - In absolute terms (my re-run), C3's dev Sharpe equals SPY's at a uniform **1.66 bp per
     round trip**: 1.10 at 1.0 bp, 1.03 at 1.11 bp, 0.91 at 1.3 bp, 0.78 at 1.5 bp.
2. **Wrong units.** The gate is a ratio to a price-dependent model.
   - At today's SPY price (P ≈ $600-700), the model is about 0.28 bp SEC plus 1.83 c/P of
     spread+slip, so 0.54-0.58 bp. The kill line is therefore about **1.1 bp**, well below the
     1.66 bp break-even.
   - So the gate kills C3 in the 1.1-1.66 bp band, where C3 would still beat SPY on its dev
     edge. At a lower SPY price it would loosen instead.
   - Fees also differ: realised uses SEC 20.6/M with TAF $0 until 2027 (report.py:29-40), the
     model 27.8/M plus TAF (costs.py:19-20).
3. **Noise-dominated references.**
   - **Entry.** The reference is the SIP open of bar d+1 (report.py:52), but the live fill is
     about 6-8 s later. Dev SPY's 1-minute sd at C3 entry bars is 5.7 bp (robust 2.8). Scaled to
     7 s, that is about 2.0 bp of drift per trade (robust ~1.0).
   - **Time exits.** These are filled at 15:59:0x but referenced to the 15:59 close
     (report.py:54-55). The last minute's open→close sd is 9.4 bp (robust 5.5).
   - **Stop references.** The exit bar is taken from the runner's *notice* time (runner.py:126;
     report.py:43-45, 59). A fill noticed in the next minute takes that bar's lower open,
     biasing the cost down.
   - The quantity to resolve is about 0.56 bp; 20 trips give a median SE of about 0.35 bp or
     more.
4. **Statistics.**
   - The gate uses the **median** (report.py:86-89). Sharpe depends on the **mean**, and stop
     slippage is right-skewed.
   - There is no CI and no day clustering: re-entries make 2+ trades on 54% of in-market days
     (306 of 571).
   - The gate is re-evaluated weekly with "kill on FAIL" and no adjustment for repeated looks.
   - No terminal pass rule or sample size is defined.
5. **The doc and the code disagree.** PAPER_C3 §3 (lines 61-65) uses "the check bar's close"
   for the entry and "the stop level" for the stop. report.py uses the SIP open of bar d+1 and
   min(level, open[k]).
- resolve (before the first order, so the gate stays pre-registered):
  - Gate on the **mean** realised cost per round trip, in **absolute bp**, with a day-clustered
    bootstrap CI.
  - Kill if the CI's lower bound exceeds the dev break-even (1.66 bp at the model's fees;
    restate it at the current fees). Pass if the upper bound is below it.
  - Fix N from a power calculation, with one look (or an alpha-spending plan).
  - Decompose each fill into latency and execution. Latency is SIP price at `filled_at` minus
    the research reference. Execution is the fill minus SIP trades or quotes at `filled_at`
    (free once 15 minutes old). This removes most of the noise.
  - Add `filled_at` to `OrderState`.
  - Align PAPER_C3 §3 with report.py.
  - Say what a pass leads to: a pilot decision, not "C3 confirmed".

**[block] B2 - G-fidelity counts in one direction, and its 90% bar is not calibrated.**
- evidence:
  - `live_matched_share` = matched / live (report.py:73-79). A runner that misses research
    trades still scores 100%: kills, buying-power rejects, missed checks and skipped checks all
    go unnoticed.
  - `cmd_report` averages *daily* shares over days with live trades only (main.py:123-126), so
    days on which live missed research's trade drop out entirely.
- The README's "99.3% of 300 checks" (README.md:68-69) is unconditional agreement. About 95% of
  checks are "no entry" for both feeds; C3 enters on roughly 5% of checks (1,137 trades over
  about 21,000 dev checks). So 2 disagreements in 300 checks is about 10% of the entry signals.
  Rounding (W1) takes another ~2 points.
- resolve:
  - Pool the counts, and require both live-matched and research-matched shares (or a Jaccard
    share).
  - Calibrate the bar to the measured IEX/SIP disagreement rate *given an entry signal*.
  - Define what a FAIL does.
  - Commit the IEX/SIP script and its output.

**[block] B4 - G-safety (PAPER_C3:79-84) is neither enforced nor measured.**
- **No persistent halt.** "Trading restarts only after the cause is logged": an automatic kill
  writes no KILL file (runner.py:151-155, 296-302). The next morning's scheduled run trades
  normally.
- **No ledger reconciliation.** "Orders and positions reconcile with the ledger every day":
  `_reconcile` only checks that the broker is flat at start and end (runner.py:289-294). No code
  compares the ledger with Alpaca's fills.
- **Kills lose their fills.** Kill exits are booked with `exit_price=None`, and the flatten's
  fill is never fetched.
- **The 60 s rule is never computed.** "No open position without a working stop for more than
  60 s" is computed nowhere: `report.gates` has only g_cost and g_fidelity (report.py:82-92).
- resolve:
  - Write the KILL file on every automatic kill, with the reason inside.
  - In the evening report, reconcile the ledger with Alpaca's closed orders and fills for the
    day.
  - Compute the unprotected seconds per trade from the `entry_filled` to `stop_placed` events
    (and from any stop loss to the exit).
  - Book kill exits at their real fill.
  - Report g_safety next to the other two gates.

**[warn] W9 - The holdout bookkeeping contradicts the pre-registration.**
- PREREG §1 (line 61) defines the holdout as "2025-08-07 onward … sealed". All of these read
  inside it:
  - the README's IEX/SIP measurement on Sep-Nov 2026 bars, which has no script or artifact in
    the tree, so the 99.3% figure cannot be reproduced;
  - `--check` and `plan_for`, which fetch 20 SIP sessions (main.py:50-54);
  - paper trading itself.
- The reads computed decisions, not returns, as far as the README says. But the record is open
  ended.
- resolve:
  - In the LOG, bound the holdout (2025-08-07..2026-11-20 stays sealed) and declare the forward
    paper window separately.
  - Log the IEX/SIP read, with what it computed, and commit its script and output.

**[warn] W10 - Provenance is missing.**
- `l1_dev.json` carries no git SHA or dirty flag (l1_dev.py:76-77), and `data/research_liquid/`
  is untracked.
- The live ledger's `plan` event records no code SHA (runner.py:136-137). A live trade cannot be
  tied to a code version.
- resolve: write the SHA and dirty state into the L1 JSON, the ledger's plan event and each daily
  report.

**[warn] W11 - The report handles only the latest finished session.**
- It reads only `done[-1]` (main.py:103-104). An evening the PC slept loses that day's cost rows
  for good.
- Kill exits are dropped from G-cost (main.py:111).
- resolve: process every session that has no report yet, and count kills explicitly.

**[note] Smaller items:**
- **Controls (checklist 4).** No random-entry arm exists: same checks, stop and size, random
  entry. For a published rule with no selection this is not required. A cheap dev diagnostic
  would separate the noise-area signal from the vol-targeted trailing-stop mechanics.
- **G-fidelity noise at n = 20.** If the true match rate is 93%, the chance of 3 or more misses in
  20 is about 16%, so 20 trades is a noisy point at which to judge.
- **Ledger rows of crashed runs.** A run that crashes leaves its trade row open forever
  (`exit_price` NULL).

## Checklist (fixed order)

1. **Adjustment basis.**
   - L1's levels (UB, prior close) are dividend-adjusted; within a session the ratios are exact,
     and the ex-date ratio is known in advance. There is no look-ahead.
   - Costs are divided by adjusted prices, which is conservative.
   - Live pairs raw history with today's raw bars, a consistent pair, but not L1's basis (W2).
2. **Decision-time leakage.** None in σ, vol, VWAP, the checks or the stops. `valid` uses day-D
   coverage (1 session; note). The live path could use a stale bar, not a future one (W3).
3. **Bar timestamps.**
   - Bars are stamped at their open (calendar.py:43-48; feed.py:42-45). Decisions use the close
     of bar d, with entry and the new stop at d+1.
   - The live fetch drops bars past `upto`, so a half-built 10:00 bar cannot leak in.
4. **Controls.** SPY, QQQ and IWM benchmarks, plus D1 (replication), D3 and D4. There is no
   random-entry arm (note).
5. **The account.**
   - L1 is a daily account series: Σ weight × net on prior-day equity, paired stationary
     bootstrap, block 20. Same-day trades are sequential, not overlapping.
   - G-cost is per trade, with no clustering (B1).
6. **Halt fills.**
   - Nothing fills inside a carried (halted) minute, and gaps fill at the open.
   - The stop fills at its level plus 1.5 c, a house-rule deviation justified for a resting
     broker stop and tested by G-cost (note).
   - One-bar latency applies to entries and stop changes.
7. **Trial count.** 374, with all three cells reported, DSR 0.46. Paper adds no cell. The
   deviation is labelled (LOG §2).
8. **Holdout budget.** The confirmation window is unread. The prereg's open-ended holdout has
   been read (W9).
9. **Code in the commit.** I could not check this without git (caller's instruction). The run
   artifact and the ledger record no SHA (W10).

---

**READY WITH FIXES** - blocking:
- **B1:** restate G-cost as an absolute-bp, mean, day-clustered, CI-based gate with matched
  references, a fixed N and a terminal rule.
- **B2:** make G-fidelity count both directions, pooled, and calibrate its bar.
- **B3:** make `flatten` wait for cancels and verify flat, with retries; add HTTP timeouts and an
  independent end-of-day flatten guard.
- **B4:** enforce G-safety (persistent halt, ledger-vs-broker reconciliation) and measure it
  (unprotected seconds).
- W1's floor-to-cent fix is a one-line change. Make it before the first order: it changes the
  executed rule.
