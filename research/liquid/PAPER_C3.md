# C3 forward paper trading — a labelled deviation from L1, with its gates fixed in advance

**Status:** FIXED 2026-11-22, before any order was sent.

**User decision (2026-11-22):** "Paper-trade C3". L1 cell C3 failed only rule (c),
the 2×-cost stress (`LOG.md` §1). L1 rule 3 would close the track. The user chose
to measure C3's **real execution costs** on Alpaca paper instead. This is a
**deviation from L1, recorded as such**. The 2023-01-03..2025-08-06 confirmation
window stays sealed; reading it still needs the user's go.

**What this phase answers:** whether C3's realised trading costs on SPY are
within the 2× of the model that rule (c) assumed. That question decides C3. A
few months cannot measure its Sharpe, so performance is reported, not gated.

## 1. Account

- A **dedicated** Alpaca paper account, with keys `ALPACA_API_KEY_C3` /
  `ALPACA_SECRET_KEY_C3` in `.env`.
- The runner refuses the KELLY and LINEAR pairs, which the live CI system trades,
  and refuses any non-paper endpoint.

## 2. The rule (exactly `cells.c3`, with these live adaptations)

1. **Bars.**
   - Today's one-minute SPY bars come from **IEX in real time**. The free plan
     gives SIP only once it is 15 minutes old (fact sheet item 12).
   - σ(minute) and the 14-day volatility come from **SIP** bars of the previous
     14 sessions, as in research.
2. **Check at HH:00 / HH:30 (10:00..15:30; to 12:30 on a 13:00 close).** About
   5 s after the bar ending at the check closes, the runner reads today's bars and
   applies the registered decision:
   - enter iff flat and close > UB and close > VWAP;
   - the stop is max(UB, VWAP).
3. **Entry.**
   - Whole-share **buy limit, DAY**, at round(close × 1.001, 2).
   - Size: floor(last_equity × min(4, 0.02/σ_SPY) / close), capped by buying
     power.
   - **Cancelled if not filled 60 s after the check**, i.e. at the end of bar
     d+1, as in research.
   - A partial fill keeps the filled shares.
4. **Stop.**
   - After the fill: a **sell stop, DAY**, at round(S, 2) for the filled shares.
   - At each later check it is **replaced** (PATCH) at the new level, and the
     replacement is confirmed.
   - If the new level is at or above the last price, the runner cancels the stop
     and sells at market (research: the stop triggers on the next bar).
5. **Close exit.** At the start of the session's last minute (15:59:00; 12:59:00
   on a 13:00 close): cancel the stop, confirm, then sell at market. A paper
   market-on-close fills at the quote, not the auction (fact sheet item 4), so it
   is not used.
6. **Re-entry.** A later check may enter again once flat, as in research.

## 3. Fees and measurements

- **Fees.** Paper charges none (fact sheet item 8). The ledger books:
  - SEC $20.60 per million on sells (rate effective 2026-04-04);
  - CAT $0.000003/share on both sides;
  - TAF $0.000195/share on sells from 2027-01-01.
- **Per fill, actual vs modelled:**

  | fill | actual vs modelled |
  |---|---|
  | entry | fill vs the check bar's close (IEX now, SIP after the close) |
  | stop | fill vs the stop level |
  | close exit | fill vs the SIP 15:59 close |

- **Per round trip:** realised cost in bp against `costs.BASE` for the same trade.
- **Every evening:** `cells.c3` is re-run on that day's SIP bars, and its trades
  are compared with the live trades.

## 4. Gates (evaluated at 20 round trips, then weekly)

- **G-cost.** The median realised cost per round trip must be **≤ 2× the
  modelled cost**: rule (c)'s bar. Above it, **C3 is killed**, because its
  2×-cost Sharpe is below SPY's.
- **G-fidelity.** ≥ 90% of live trades must match the SIP research re-run on day,
  side and entry check. Every mismatch is explained in the log (IEX vs SIP is the
  expected cause).
- **G-safety.**
  - No open position without a working stop for more than 60 s, except in the
    final minute.
  - Orders and positions reconcile with the ledger every day.
  - Any break triggers the kill switch: cancel everything and go flat. Trading
    restarts only after the cause is logged.
- **Reported, not gated:** daily P&L and Sharpe against SPY/QQQ, weekly.

## 5. What paper cannot tell us

Paper fills at the NBBO quote with no queue, no depth and no market impact (fact
sheet item 8). For a few hundred SPY shares the quote is close to real, but stop
slippage in a fast market and the closing-auction price are only tested with
real money. A small real-money pilot is the user's decision.
