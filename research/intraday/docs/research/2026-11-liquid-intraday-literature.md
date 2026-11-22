# Liquid intraday literature: SPY/QQQ momentum and ORB, verified rules (2026-11-22)

Purpose: verify citations and extract each paper's own rules for pre-registering at most 3 long-only cells on 1-minute SPY/QQQ bars. Bibliographic data comes from the Crossref API. Quotes come from PDFs read with `pdftotext`, and each section names the version read. "Not verified" means no primary source was reached; nothing was filled from memory.

## Summary

| Paper | Verified citation | Instrument | Entry | Exit | Costs | Net Sharpe vs buy-and-hold | One bracket? |
|---|---|---|---|---|---|---|---|
| Gao, Han, Li, Zhou | JFE 129(2):394–414 (2018), doi:10.1016/j.jfineco.2018.05.009; WP SSRN 2440866 | SPY, 1993–2013 | 15:30, long if r1>0, else short | 16:00 close | WP: bid/ask, no commission | 1.08 **gross** vs 0.29 (WP); net Sharpe not reported | No (close exit) |
| Baltussen, Da, Lammers, Martens | JFE 142(1):377–403 (2021), doi:10.1016/j.jfineco.2021.04.029 | 60+ futures, 1974–2020 | Sign of r_ROD, 30 min before close | Close | None | 1.73 **gross** (equity 1/N) vs 0.51 | No (close exit) |
| Zarattini & Aziz | SSRN 4416622 (2023) | QQQ, 2016-01-01..2023-02-17 | 9:35 open, first-bar direction | Stop at first-bar low; TP 10R; end of day | $0.0005/sh, no slippage | 1.12; buy-and-hold Sharpe not given (~15%/yr) | Yes, plus end-of-day exit |
| Zarattini, **Aziz**, Barbon | SSRN 4824172; SFI RP 24-97 (2024) | SPY, May 2007–Apr 2024 | HH:00/HH:30 outside the Noise Area | Trailing stop max(UB, VWAP); 16:00 | $0.0035 + $0.001/sh | 1.33 vs 0.45 | No (moving stop) |
| Zarattini, Barbon, Aziz | SSRN 4729284; SFI RP 24-98 (2024) | ~7,000 stocks, 2016–2023 | Stop order at 5-min OR high | Stop at 10% of ATR(14); end of day | $0.0035/sh | 2.81 vs 0.78 | Stop entry, no TP, plus end-of-day exit |

Citation corrections:
- Paper 4's author order is Zarattini, Aziz, Barbon (title page, Crossref, IDEAS).
- Paper 5 is titled "A Profitable Day Trading Strategy For The U.S. Equity Market".

## 1. Gao, Han, Li & Zhou (2018), "Market intraday momentum"

- **Version read.** Working paper "Intraday Momentum: The First Half-Hour Return Predicts the Last Half-Hour Return", December 2015 (mql5.com mirror).
- **Journal version.** Abstract only (EconPapers), which adds "Bogousslavsky's (2016) model". **The journal's numbers are not verified.** Every number below is from the WP.
- **Data.** SPY from TAQ, "February 1, 1993 through December 31, 2013", 13 half-hour returns a day (WP p.3).
- **Signal.** "we calculate the first half-hour return using previous day's close price and the price at 10:00am" (p.3). So r1 runs from the prior close to 10:00 and **includes the overnight gap**. The journal abstract confirms "as measured from the previous day's market close".
- **Rule.** "take a long position… at the beginning of the last half-hour if the timing signal is positive, and take a short position otherwise… closed at the market close" (§3.1, p.8). There is no stop, target or sizing rule.
- **Results (p.9).** "6.67% per annum with a standard deviation of 6.19%. The Sharpe ratio is thus 1.08". Buy-and-hold: 6.04%/20.57%, Sharpe 0.29. Annualised ×252. Drawdown is not reported.
- **Costs (§5.2, pp.15–16).** Ask/bid at 15:30, with commissions ignored (footnote 12). After 2005: "6.52% after transaction costs compared with 7.96% before". Net Sharpe is not reported.
- **Long side.** "intraday momentum is a phenomenon specific to days when the first half-hour returns are positive" (§5.1, p.15). A long-only P&L is not reported.
- **QQQ.** Its out-of-sample R² of 0.70% is the lowest of the ten ETFs tested (§5.5, p.18).
- **Bracket.** No. Entry is at 15:30 with exit at the close; it needs a market-on-close or end-of-day exit.

## 2. Baltussen, Da, Lammers & Martens (2021), "Hedging demand and market intraday momentum"

- **Version read.** Published JFE PDF (nd.edu).
- **Data.** Tick data, December 1974–May 2020. The E-mini S&P 500 is used "between 09:30 and 16:00".
- **Signal.** r_ROD = P(c−30, t) / P(c, t−1) − 1: "buying at previous day's close and selling 30 minutes before today's close" (Eq. 2, p.381). Long if positive, else short, over the last half-hour (Eq. 12, p.385).
- **Results, gross (Table 6A, p.386).** Equity 1/N: 6.86%/3.96%, Sharpe 1.73. The Gao-style r_ONFH timing gets 1.07; buy-and-hold 0.51.
- **Costs.** "Note that we do not consider transaction costs" (p.386). They report a positive net Sharpe for S&P futures at one tick, without giving the number. The two subsamples, 1974–1999 and 2000–2020, are similar (§3.4).

## 3. Zarattini & Aziz (2023), "Can Day Trading Really Be Profitable?" (5-minute ORB, QQQ)

- **Version read.** 18-page PDF, "First published April 10, 2023", hosted on a wealth-lab.com forum. Its summary matches the SSRN 4416622 abstract in Crossref. A second SSRN record (ssrn.4898296, 2024) and any later revisions are **not verified**.
- **Data.** QQQ (TQQQ as an extension), "January 1, 2016 to February 17, 2023" (p.4). Interactive Brokers data (p.6).
- **Entry (p.5).** "if during the first 5 minutes the market moved up, we took a bullish position starting from the second candle's opening price… No positions were opened when the first 5-minute candle was a doji (open ~ close)." This is a market entry at 9:35, not a breakout stop.
- **Stop (p.5).** "the low of the first 5-minute candle" for longs; R = entry − stop.
- **Exit (p.5).** "We set the profit target at 10x the $R. Should the target not have been reached by the end of the day (EoD), we liquidated the position at market closure."
- **Sizing (p.6).** Shares = int(min(0.01·A/R, 4·A/P)), starting from $25,000. The 4× cap bound: "60% of the trades were conducted with an exposure 40% below the optimal" (p.10).
- **Costs (Table 1, p.6).** "$0.0005 per share", and "We assumed no slippage in fills".
- **Results, net.**
  - 675% total, "annualized Sharpe Ratio was 1.12… annualized rate of return was 31%" (p.8).
  - 1,795 trades, 51% long; win rate 24% (p.9).
  - QQQ buy-and-hold earned 15% a year (169% total, p.12).
  - Maximum drawdown and the long-only split are not reported.
- **Post-hoc variant.** A 5%-of-ATR(14) stop with no target on TQQQ was picked from a heat map (pp.14–15). It is an in-sample optimum.
- **Bracket.** Yes: marketable entry limit, TP limit and SL stop, plus an end-of-day flatten.

## 4. Zarattini, Aziz & Barbon (2024), "Beat the Market" (SPY)

- **Version read.** "This Version: February 3, 2025" (St. Gallen repository).
- **Data.** SPY 1-minute bars from IQFeed, "May 2007 to April 2024" (p.6).
- **Noise area (pp.6–7).**
  - move = |Close(t−i, HH:MM) / Open(t−i, 9:30) − 1|, "the absolute move from Open".
  - σ(t, HH:MM) = the mean of that move over the past 14 days.
  - UB = max(Open(t, 9:30), Close(t−1, 16:00)) × (1 + σ); LB = min(...) × (1 − σ).
  - The volatility multiplier is 1; a value of 1.5 was best but not used (p.22).
- **Checks.** "trading is restricted to semi-hourly intervals, specifically at HH:00 and HH:30" (p.9). The first position is at 10:00 (p.21).
- **Exits.** "Long TrailingStop = max(UB, VWAP)", closed "as soon as the price crosses" it (p.13). But p.9 says "Stop losses can also be triggered only on semi-hourly intervals". Everything is closed at 16:00.
- **Sizing (p.15).** Shares = AUM(t−1) × min(4, 0.02/σ_SPY) / Open(t), with σ_SPY the 14-day standard deviation of returns.
- **Costs.** "$0.0035 per share" plus "slippage of $0.001 per share", from their own test of SPY market orders (p.10, §4.6).
- **Results, net (Table 3, p.15).** Sharpe 1.33, 19.6% a year, max drawdown 25%, 1.8 trades a day. SPY buy-and-hold: Sharpe 0.45, 7.2% a year, max drawdown 56%. Long-only is not reported.
- **Bracket.** No: the stop moves, there is no TP, and there is an end-of-day exit.

## 5. Zarattini, Barbon & Aziz (2024), ORB on Stocks in Play

- **Version read.** "February 16, 2024" (St. Gallen copy, SSRN 4729284).
- **Universe.** US stocks 2016–2023, with filters:
  - price above $5;
  - 14-day average volume of at least 1M;
  - ATR(14) above $0.50;
  - Relative Volume (first 5 minutes vs the 14-day mean) of at least 100%;
  - top 20 by Relative Volume (p.14).
- **Entry and exit (p.8).** "a stop order… at the highest value of the 5-minute opening range"; the stop is "at a 10% ATR distance from the executed entry price"; otherwise exit at 16:00.
- **Results.** Sharpe 2.81 net vs 0.78 for the S&P 500 (Table 2, p.16).
- **Use here.** The rule is cross-sectional, so on SPY/QQQ alone it is **not the paper's rule**. It is excluded below.

## Additions (marked; not in the brief)

- **A1. Rosa (2022).** "Understanding intraday momentum strategies", J. Futures Markets 42(12):2218–2234, doi:10.1002/fut.22375. Abstract only: "The predictability disappears in the out‐of‐sample period." Its details are **not verified**.
- **A2. Li, Sakkas & Urquhart (2022).** J. Financial Markets 57:100619. Abstract only: the effect holds in 16 developed markets, in and out of sample.
- **A3. Paz (2026), SSRN 7290621.** Unrefereed; abstract only. Out-of-sample test of Beat the Market, May 2024–March 2026: "9.4% versus 29.5% for a passive buy-and-hold, with a Sharpe Ratio of 0.39".
- **A4. Delgado (2026), SSRN 7323419.** Unrefereed; abstract only.
  - The result reproduces only with "the authors' published Python reference code—the use of typical-price… VWAP and VWAP confirmation at entry". The paper text does not state either.
  - Sharpe was 1.06 for May 2024–Aug 2025 and −0.46 for Sep 2025–Aug 2026.
- **A5. Fetna (2026), SSRN 7428398.** Unrefereed; abstract only. A pre-registered grid of 225 ORB cells on futures: "Zero of 225 cells" pass.

## Candidate cells for pre-registration (papers' own parameters)

Every source paper is long/short and none reports a long-only result, so each cell below is a **derivative** of its paper, not a replication. The 2023-01-01..2025-08-06 confirmation window overlaps the post-publication evidence (A3, A4).

### C1: Last-half-hour momentum on SPY, long-only (Gao et al.)

**Rule.** r1 = P(10:00) / Close(t−1) − 1. If r1 > 0, buy at 15:30 and exit at the 16:00 close; otherwise stay flat. Size at 100% of equity (the paper gives no sizing).

**Needs.** The prior close, the 10:00 price, and a closing-price exit.

**Fix in advance:**
- (a) The 10:00 price: the close of the 09:59 bar or the open of the 10:00 bar.
- (b) r1 or Baltussen's r_ROD (prior close to 15:30). These are different cells.
- (c) The entry fill (Gao used the 15:30 ask).
- (d) The closing auction vs the last bar.
- (e) Half-days.
- (f) Whether to include QQQ (its out-of-sample R² was the weakest).

**Bracket.** It does not fit: there is no TP or SL, and it needs a market-on-close exit. Rosa (A1) reports the effect disappearing out of sample.

### C2: 5-minute ORB on QQQ, long-only (Zarattini & Aziz 2023)

**Rule.**
1. Build the 09:30–09:35 bar from 1-minute bars.
2. If close > open, buy at the open of the 09:35 bar.
3. Stop at that bar's low, with R = entry − stop; TP at entry + 10R.
4. Otherwise exit at 16:00.
5. Size shares = floor(min(0.01·A/R, 4·A/P)).

**Fix in advance:**
- (a) The doji threshold ("open ~ close").
- (b) Market fill vs a limit that may not fill.
- (c) A bar that touches both stop and TP.
- (d) Whether to add SPY (the paper is QQQ-only).
- (e) The leverage cap the live account allows.
- (f) Our own costs: the paper assumed no slippage.
- (g) Exclude the post-hoc ATR variant.

**Bracket.** It fits, plus an end-of-day flatten.

### C3: Noise-area momentum on SPY, long-only (Zarattini, Aziz & Barbon)

**Rule.**
1. σ(HH:MM) = the 14-day mean of |Close(t−i, HH:MM) / Open(t−i, 9:30) − 1|.
2. UB = max(Open(t), Close(t−1)) × (1 + σ).
3. At each HH:00/HH:30 from 10:00 to 15:30, if flat and price > UB, buy.
4. Exit when price < max(UB, VWAP), or at 16:00.
5. Size shares = AUM(t−1) × min(4, 0.02/σ_SPY) / Open(t).

**Needs.** A 14-day minute-of-day move table, session VWAP, and a 14-day daily standard deviation.

**Fix in advance:**
- (a) Check the stop at half-hours (p.9) or every minute (p.13).
- (b) Which bar's close counts as "price", and the fill bar.
- (c) Typical-price VWAP, and whether entry needs price > VWAP (A4: per the code, not the paper).
- (d) The return definition for σ_SPY.
- (e) Same-day re-entry (1.8 trades a day implies yes).
- (f) Half-days and missing bars.

**Bracket.** It does not fit a single static bracket: the stop must be replaced at each check and there is no TP. Either the live system supports stop replacement, or this cell is dropped. Post-publication evidence is weak (A3, A4).
