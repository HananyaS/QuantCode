# POC Plan: Small-Cap Breakout Detector vs. QQQ Timing System

## Context — what carries forward from the S&P 500 ranking system

That system was killed by real, portfolio-level evidence: a weak ranking signal (IC 0.008-0.019)
combined with aggressive rank-based entry/exit produced ~100%/day turnover and **lost to buy-and-hold
SPY** on Sharpe, return, and drawdown. The lessons that must carry into both ideas below, non-negotiably:

1. **Pre-register go/no-go criteria before evaluating**, not after. Write the bar down first.
2. **Point-in-time data is non-negotiable.** Survivorship bias inflates results; it doesn't just add noise.
3. **Test at the portfolio level** (real Sharpe/drawdown/turnover through actual position sizing and
   costs), not just raw signal IC — IC alone said "maybe fine," portfolio-level said "loses money."
4. **Walk-forward across multiple regimes**, not a single split.
5. **Report negative/mixed evidence honestly.** Both research passes below surfaced real skepticism in
   the literature — that's included deliberately, not filtered out.

---

## Idea 1: Small-Cap Breakout Detector

### Research summary

**Breakout definitions are concrete and programmable.** Three established methodologies translate
directly into features: **Darvas Box** (new high → 3-day failure-to-extend range → breakout = close
above box ceiling on volume), **Minervini's VCP** (2-4 sequential contracting pullbacks, ATR contracting
to ~1/3 of its 50-day average, breakout volume 40-50%+ above average), and **CANSLIM's pivot breakout**
(buy within 5% of the base's resistance pivot, volume confirmation, explicitly favors small/mid caps for
larger percentage moves).

**Fundamental/event overlays — separating evidenced from folklore:**
- *Evidenced*: Post-earnings-announcement drift (PEAD) — replicated since Ball & Brown 1968, concentrated
  in small caps/low-coverage names, though a 2022 study found it may have vanished outside microcaps.
  Insider cluster buying — multi-insider buys show ~2x the abnormal return of single-insider buys;
  classic studies found ~4-5% 12-month abnormal returns following heavy insider buying. Short interest —
  profits from short-interest-based signals concentrate specifically in small-cap, low-institutional-
  ownership names.
- *Folklore-leaning*: IBD-style Relative Strength Rating and sector rotation are practitioner staples
  with little rigorous academic backtesting found — treat as features to test, not assumed edges.
- *Regime filters* (VIX level, market breadth): logically sound, thinner evidence base than the above —
  treat as a risk-management prior to test empirically, not a proven overlay.

**Small-cap risks are severe and likely the deciding factor, not signal quality:**
- Liquidity: sub-$1B (especially sub-$100M) caps have materially wider spreads and far lower dollar
  volume than large caps. Implementation-shortfall estimates run **~110.8 bps** vs. ~7 bps in commissions
  for US small caps — trading costs of 2-4%/yr are cited as capable of erasing most of the small-cap
  premium. This is the same turnover-cost failure mode that killed the S&P 500 system, but structurally
  worse here.
- Survivorship/delisting: low-priced/small stocks are disproportionately subject to manipulation and
  non-compliance delistings — survivorship bias is a bigger risk than the S&P 500 case we already found
  problematic.
- Point-in-time fundamentals: financial restatements and reporting lag make "as-known-at-the-time" vs.
  "as-reported-today" data a real look-ahead-bias vector, distinct from and additional to the price-data
  point-in-time problem we already solved for the S&P 500.

**Honest base rates (the sobering part):** Bulkowski's ~14,000-pattern dataset found breakout failure
rates (to reach even a 20% target) ranging 22-64% depending on year, trending *worse* over time. A 240K-
trade opening-range-breakout study found ~66% hit their stop-loss. An independent academic study of AAII
stock screens (including CANSLIM-style ones) found **fewer than 38% were profitable after realistic
transaction costs at sub-$20K account sizes** — only becoming reliably profitable at ≥$100K, which is a
direct, material red flag given this project's account-size context. Minervini's audited real-money
results (220%/yr 1994-2000) are genuine but represent one elite discretionary trader, not a mechanical,
published strategy — "which trader gets famous" survivorship bias applies.

### Proposed architecture

```
PointInTimeSmallCapUniverse (NEW — small-cap analogue of utils/sp500_membership.py)
  + liquidity filter (minimum trailing avg-dollar-volume floor — non-negotiable per the cost research)
  → BreakoutDetectionAgent (NEW)
      technical: N-day-high proximity, ATR-contraction ratio, volume-multiple-of-average,
                 consolidation tightness (VCP-style)
      fundamental/event: PEAD indicator (days-since-earnings, surprise direction), insider cluster-buy
                 flag, short-interest level/change — from SEC EDGAR + FINRA short-interest data
      regime: market breadth, VIX level — as a GATE (trade smaller/not at all in adverse regimes),
                 not necessarily a per-stock feature
  → Event-based labeling (forward return CONDITIONAL on a breakout trigger firing — this is
      architecturally different from the S&P 500 system's daily cross-sectional ranking; it's an
      event-driven strategy, evaluated only when a trigger fires)
  → PortfolioAgent (reused, but needs a small-cap-realistic cost/slippage model — the flat 0.1% bps
      assumption is known-wrong at this liquidity tier per the research above)
```

### New data infrastructure required

- Small-cap point-in-time universe (Russell 2000 historical membership, or a market-cap-band filter
  applied to SEC EDGAR's point-in-time company list)
- Price/volume history for small caps **including delisted names** — the current Alpaca cache is
  large-cap only; this needs `utils/tiingo_loader.py` (already built) pointed at a small-cap universe,
  or a dedicated survivorship-bias-free provider (Sharadar, Norgate — paid)
- SEC EDGAR XBRL `companyfacts`/`frames` API (free, genuinely point-in-time) for fundamentals
- SEC Form 4 filings (EDGAR, free) for insider transactions
- FINRA short interest data (published bi-monthly, free)

### Phased POC plan

| Phase | Deliverable | Note |
|---|---|---|
| 0 | Write go/no-go bar before touching data | e.g. portfolio Sharpe > 0.5 *after* a small-cap-realistic cost model |
| 1 | Point-in-time small-cap universe + liquidity floor | Foundational; biggest new infra lift |
| 2 | Technical breakout detector only, walk-forward IC | Fast — reuses existing feature/walk-forward patterns |
| 3 | Add fundamental/event overlays (PEAD, insider, short interest), measure incremental contribution | Each overlay tested as its own atomic addition, autoresearch-style |
| 4 | Add regime gate, measure incremental contribution | |
| 5 | **Portfolio-level backtest with a realistic small-cap cost model** | This is where the S&P 500 system actually died — expect it to matter more here, not less |
| 6 | If it clears the Phase 0 bar: paper trade | |

---

## Idea 2: QQQ / Leveraged ETF Entry-Exit Timing

### Research summary

**Established methodologies, with real published numbers:**
- **Faber's 10-month SMA** (SSRN #962461): S&P 500 1901-2012, CAGR 10.2% vs 9.3% buy-and-hold, but
  stdev 12.0% vs 17.9% and max DD -50.3% vs -83.5% — **the "edge" is overwhelmingly volatility/drawdown
  reduction, not CAGR alpha.** Applied to the Nasdaq Composite since 1972, beat buy-and-hold by ~4%/yr
  with ~25% less volatility.
- **200-day SMA regime filter**: independently corroborated (1928-2015, including slippage) to beat
  buy-and-hold by only ~60bps/yr, ~half the drawdown, win rate just ~25% but asymmetric payoff (avg
  winner +26.5%, avg loser -6.0%).
- **Dual Momentum (Antonacci)**: the strongest numbers found (CAGR 17.4% vs 8.9%, max DD 22.7% vs
  60.2%, independently replicated) — but this is a **multi-asset** rotation strategy (US vs. ex-US
  equities vs. bonds), not single-instrument QQQ timing. Relevant as methodology inspiration, not
  directly transferable.
- **MACD-based timing**: weakest-evidenced of the classics — no credible source shows it beating
  buy-and-hold risk-adjusted.
- **Volatility-regime filters**: real academic backing (Moreira & Muir, *Journal of Finance* 2017) —
  scaling exposure inversely to realized vol improves Sharpe.

**Academic consensus is genuinely skeptical on close reading, more so than practitioner content
suggests.** Zakamulin directly rebuts the strongest pro-timing academic claim (Glabadanidis 2015),
showing it relied on look-ahead bias — corrected, MA timing is only "marginally better," statistically
indistinguishable from buy-and-hold. Sullivan, Timmermann & White (*Journal of Finance* 1999) bootstrap-
tested 26 famous technical rules and found profitability collapsed to near zero out-of-sample — a direct
data-snooping warning, structurally the same risk we're managing with our own multi-config search.
Dichev (*American Economic Review* 2007) found real investors' dollar-weighted returns trail buy-and-hold
by 1.3-1.5%/yr — **actual timing behavior destroys value on average**, independent of any specific
strategy's backtest. **Honest framing for this POC: expect timing to mainly buy volatility/drawdown
reduction, not CAGR improvement — and calibrate the go/no-go bar accordingly, not against a "beats
buy-and-hold return" standard that the literature says rarely survives.**

**Leveraged ETF decay — precise, well-sourced mechanism, must be modeled explicitly:**
TQQQ/SQQQ/QLD reset to constant leverage **daily**, not over the holding period. Decay ≈
`L(L-1)/2 × σ²` per period (Cheng & Madhavan 2009; Avellaneda & Zhang 2010) — real examples: 2022's
choppy bear market, QQQ -33% but TQQQ -82% (vs. a naive -98% or naive "still tracks 3x" intuition, both
wrong); COVID crash, SPX -34% drove a 3x fund down -76%. FINRA/SEC's joint alert states these products
are "typically unsuitable for retail investors who plan to hold longer than one trading session." Decay
scales with σ², so it's genuinely small over short, smooth, strongly-trending periods — logically sound
but only blog-level evidenced, not a single authoritative quantitative study.

**QQQ-specific signals:** Faber's Nasdaq Composite result above is the most rigorous evidence directly
adjacent to QQQ. A sector-rotation system built on Faber's own relative-strength methodology reports
13.94% CAGR, Sharpe 0.54. Breadth indicators (% of Nasdaq-100 above 200-day MA) are tracked by
StockCharts/SentimenTrader with some dated positive studies, though details are largely paywalled.
**One live red flag surfaced during research**: a widely-shared claim of "30% CAGR vs. SPY's 7%" for a
QQQ-rotation strategy was found, on inspection, to contain look-ahead and survivorship bias (selecting
today's top constituents and backtesting them historically) — a direct, concrete example of exactly the
failure mode this whole project has been trying to avoid.

**Practical account constraints:** QQQ has ~0.86bp average spread and enormous volume — liquidity is
simply not a concern here, unlike Idea 1. TQQQ trades ~$5.5B/day; QLD is thinner (~3.6M shares/day) but
still far more liquid than any small cap. **The classic PDT rule was reportedly eliminated as of June
2026** (FINRA Notice 26-10, replaced by an "Intraday Margin Deficit" framework) — verify this directly
with the broker before relying on it, but if accurate it removes a constraint that shaped earlier
decisions in this project. Cash-account T+1 settlement limits still apply regardless.

**Realistic expectations:** buy-and-hold QQQ/S&P 500 long-run Sharpe is roughly 0.4-0.5. Diversified
multi-asset trend-following systems report higher Sharpes (0.8-1.25), but those are not single-instrument
analogues — no rigorously-derived Sharpe specifically for single-ETF systematic timing was found; treat
0.4-0.8 as an informed inference, not a cited hard number.

### Proposed architecture

```
QQQ / TQQQ / SQQQ (or QLD) — single-instrument time series, no point-in-time universe problem,
    no survivorship bias concern (fixed, currently-listed instruments)
  → TimingSignalAgent (NEW)
      trend/regime: 200-day SMA position, 10-month SMA (Faber), MA slope
      volatility regime: realized vol, VIX level — explicitly to size the leveraged-ETF decay risk
      breadth: % of Nasdaq-100 above 200-day MA (NDTH-style), as a confirming/diverging signal
      momentum: absolute momentum (trailing N-month return sign), dual-momentum-style
  → Time-series backtest (NOT cross-sectional — this needs a different evaluation harness than the
      S&P 500 system's IC-based one; single-instrument regime-conditional performance breakdown instead)
  → Leverage-aware position sizing: explicitly estimate expected decay (L(L-1)/2 × σ²) given the
      current vol regime BEFORE choosing between QQQ / QLD / TQQQ exposure — this is the dominant risk
      lever for this idea, more so than signal quality
```

### New data infrastructure required

Much lighter than Idea 1:
- QQQ, TQQQ, SQQQ (or QLD) price history — straightforward via Tiingo/Alpaca, a handful of tickers, no
  universe-construction problem at all
- VIX historical data — not currently in the data layer; check Tiingo/FRED/CBOE for a free source
- Nasdaq-100 breadth — either construct from Nasdaq-100 constituent data (would need a point-in-time
  NDX membership source, a smaller version of the same problem solved for the S&P 500) or approximate
  with current constituents (NDX turnover is much lower than small caps, so this approximation is far
  less risky here than it would be for Idea 1)

### Phased POC plan

| Phase | Deliverable | Note |
|---|---|---|
| 0 | Write go/no-go bar — calibrated against the literature's honest finding that timing mostly reduces drawdown, not CAGR | Don't set a bar the research says rarely survives |
| 1 | Baseline data: QQQ/TQQQ/SQQQ price history + VIX | Cheapest, fastest infra step of either idea |
| 2 | **Replicate Faber's 10-month SMA / 200-day SMA on QQQ as a sanity check** | If we can't reasonably reproduce a published result, that's a methodology red flag, not a strategy finding |
| 3 | Build the richer signal (breadth + vol-regime + momentum), walk-forward vs. Phase 2 baseline and buy-and-hold | |
| 4 | Explicitly model the leveraged-ETF layer (signal → QQQ/QLD/TQQQ position, vol-sized) | Decay must be handled deliberately, not assumed away |
| 5 | Portfolio-level backtest with realistic costs | Much less turnover-sensitive than Idea 1 given QQQ's near-zero spread, but still real |
| 6 | If it clears the Phase 0 bar: paper trade | |

---

## Comparative read and recommendation

**Idea 2 (QQQ timing) is architecturally simpler and cheaper to POC**: single instrument, no
point-in-time universe problem, no new fundamentals-data sourcing, and a liquidity/cost profile that
makes the turnover failure mode from the S&P 500 system far less likely to recur. Its risk is
almost entirely in the literature's skepticism about timing producing real CAGR alpha (vs. just
volatility reduction) and in leveraged-ETF decay if that layer is used.

**Idea 1 (small-cap breakout) has more plausible avenues to genuine edge** — PEAD, insider clusters, and
short interest are real, evidenced anomalies, not folklore — but requires substantially more new data
infrastructure (point-in-time small-cap fundamentals, insider filings, short interest) and faces a
harsher cost/liquidity reality where the research explicitly warns that trading costs alone can erase
the premium. The AAII finding that CANSLIM-style screens only turn profitable after costs at ≥$100K
account size is a direct, material concern given this project's account-size context.

**My honest lean**: start with Idea 2. It validates the methodology (walk-forward discipline, pre-
registered go/no-go, honest reporting) again on a problem with a much better liquidity/cost profile and
far less new infrastructure — a faster, cheaper way to find out whether *this general approach* (not
just the specific S&P 500 setup) has anything real in it. Idea 1 is the bigger, more interesting bet if
there's appetite for the extra data-infrastructure work, but its cost/liquidity risk profile is a closer
echo of exactly what killed the last system.
