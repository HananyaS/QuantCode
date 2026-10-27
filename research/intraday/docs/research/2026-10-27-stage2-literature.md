# Stage 2 literature review: minute-level entry and exit on gapped low-priced stocks

Read-only review, 2026-10-27/28. Numbers come from the paper's own text or abstract unless tagged. UNVERIFIED = only a secondary source, or an abstract of a paper I could not open. Our costs: 0.4% (at $5) to 2% (at $1) round-trip commission, plus spread.

## Summary

1. **Labels are argued, not demonstrated.** The primary source for triple-barrier and meta-labeling (Lopez de Prado 2018b) has no out-of-sample, after-cost result in the article I read (the book was not accessed). The one head-to-head found (Li 2022) uses classification metrics on an index. Treat label design as a tested factor (E5).
2. **Short-horizon predictability is real but short-lived.** LASSO improved out-of-sample fit of 1-minute returns 23% over OLS, with signals tied to news (Chinco 2019). Order-book forecasts have an effective horizon of about two price changes (Kolm 2023). High accuracy does not necessarily give actionable signals (Briola 2024).
3. **Costs erase most published edges.** Zero-shot Chronos: gross Sharpe above 3.17 on daily residuals, negative after 3 bp slippage (Valeyre 2025). Daily LSTM and ensemble edges faded after about 2010 (Fischer 2018; Krauss 2017; secondary reports, UNVERIFIED). Our costs are 10 to 60 times 3 bp.
4. **No credible evidence that deep sequence models beat boosting on engineered features at minute frequency, after costs.** Trees lead on medium tabular data (Grinsztajn 2022). On Qlib's daily benchmark LightGBM leads on engineered features and GRU/LSTM on raw windows. LSTM and boosting both failed to beat the base rate on 5-minute futures (Mesfin 2026).
5. **Foundation models are low priority.** Off-the-shelf pretrained models do poorly on financial returns; only finance-pretrained ones help (Rahimikia 2025). LLM backbones add nothing in ablations (Tan 2024).
6. **Gap direction depends on context.** Overnight gains in high-attention stocks reverse intraday (Berkman 2012; Lou 2019; Akbas 2022), matching our -1.3%/day. Opening-range breakouts on high relative-volume (news) stocks continue (Zarattini 2024), but that study excludes penny stocks and charges commission only. Relative volume is the discriminating feature.
7. **Market-level intraday momentum is index-level.** The first half-hour predicts the last (Gao 2018; Baltussen 2021). Our entries stop at 15:30, and single low-priced names are untested.
8. **Time of day drives cost.** Spreads are widest at the open and close (McInish 1992); returns show half-hour periodicity (Heston 2010).
9. **Round numbers have support.** Over 70% of published FX support levels end in 0 (Osler 2000); buy-sell imbalance clusters at round prices (Bhattacharya 2012). The best simple technical rule of 1897-1986 showed no edge in 1987-96 (Sullivan 1999; secondary, UNVERIFIED).
10. **Search hygiene.** About 20 repeated backtests on the same data produce one false discovery at 5% (Lopez de Prado 2018b). Log every trial; use the Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO) and t above 3.0 (Harvey 2016).

## Q1. Labeling

| Paper | Year | Data | Method | Out-of-sample / cost result | Relevance |
|---|---|---|---|---|---|
| Lopez de Prado, 10 reasons | 2018 | conceptual | fixed-horizon vs triple-barrier (vol-scaled profit/stop, time barrier); meta-labeling | None in the article (book not read). Argues a fixed threshold gives mostly 0 labels when volatility varies and ignores stops; meta-labels filter primary false positives | Source of E5; argument only |
| Joubert | 2022 | framework | primary sets side, secondary sets size | Abstract only; OOS results UNVERIFIED | Fits a rule-based primary |
| Li et al. | 2022 | Nasdaq-100 | MLSTM-FCN, three label types | Triple-barrier best buy F1; fixed-horizon best accuracy; no costs | Weak, classification only |
| Carr, Lopez de Prado | 2014 | simulated OU | optimal profit/stop rules | Optimal rules exist numerically (conjecture) | Exits matter only with mean reversion |

Synthesis: no study found compares label types out of sample after costs at minute frequency. Volatility-scaled barriers are sound in logic (labels match how a stop-bounded trader exits), but the horizon and barrier ratios are hyperparameters that multiply the trial count. Our inference, not a citation: under a martingale price any profit/stop pair has zero gross edge, so exit gains come from the entry signal. Cost awareness is supported at the decision level only (Liu-Stentoft trade only when the forecast exceeds the spread). Meta-labeling fits best on a rule with a known side, such as an opening-range break.

## Q2. Deep sequence models

| Paper | Year | Data | Method | OOS result incl. costs | Relevance |
|---|---|---|---|---|---|
| Zhang (DeepLOB); Kolm | 2019; 2023 | LSE; 115 Nasdaq, order book | CNN+LSTM; LSTM | Stable OOS accuracy, transfers to unseen stocks; stationary order-flow inputs beat raw book; no cost P&L | Needs order book |
| Prata; Briola | 2024 | FI-2010; Nasdaq LOB | 15 DL models | All drop significantly on new data; accuracy not actionable | Replication, tradability negatives |
| Fischer; Krauss | 2018; 2017 | S&P 500 daily 1992-2015 | LSTM; DNN/GBT/RF | LSTM 0.46%/day gross; ensemble about 0.25% net; fades near 2010 (all secondary, UNVERIFIED) | Edge decays |
| Huddleston et al. | 2023 | 5-min S&P 500 | linear, trees, ensembles | Ensembles best; Sharpe 0.98 after costs | Index-level; trees win |
| Mesfin (preprint) | 2026 | MNQ 5-min OHLCV, 944 days | LSTM vs boosting | 50.0-50.9% and 50.6% vs 51.8% base rate; p = 0.135, 0.515 | Closest input; null |
| Qlib | repo | CSI300 daily | 15+ models | Alpha158: LightGBM IR 1.02, ann. 9.0%; LSTM 0.56, 3.8% (on 20 LightGBM-picked features); Transformer 0.40, 2.7%. Alpha360 raw: GRU IC 0.049 vs LightGBM 0.040; Transformer IC 0.011. Cost model not retrieved | Boosting wins on features |
| Grinsztajn; Zeng; Tan | 2022-24 | tabular; LTSF | trees vs DL; DLinear; LLM ablation | Trees stay SOTA near 10K rows; one linear layer beats Transformers; LLM adds nothing | Generic priors |
| Rahimikia; Valeyre; Noguer; Shi (Kronos) | 2025-26 | daily returns; K-lines | Chronos/TimesFM zero-shot, tuned, scratch | Generic models poor, finance-pretrained gain; Chronos gross Sharpe 3.17+, negative at 3 bp; gains over random walk "small and sparse"; Kronos RankIC +93%, no cost test | Not shown at 1-min |
| Sirignano, Cont | 2019 | US order book | pooled vs per-stock | Pooled beats stock-specific, transfers | Supports E2 pooling |

Synthesis: positive after-cost results cluster at order-book scale, which our OHLCV cannot reach. On OHLCV the record is null (Mesfin) or index-level with trees on top (Huddleston). Daily benchmarks suggest sequence models help on raw windows and boosting wins on engineered features, the comparison E4 sets up. Generalization failure (Prata) and post-2010 decay argue for date-blocked evaluation. PatchTST, iTransformer, TimesNet and TFT report no after-cost finance results.

## Q3. Intraday patterns for gappers and small caps

| Paper | Year | Data | Method | Result incl. costs | Relevance |
|---|---|---|---|---|---|
| Gao et al.; Baltussen et al. | 2018; 2021 | SPY 1993-2013; 60+ futures | first half-hour predicts last | R2 1.6% in-sample, 1.2% OOS, Sharpe 1.08 (2015 draft; published values UNVERIFIED); tied to short-gamma hedging | Market-level, last 30 min |
| Heston et al.; McInish, Wood | 2010; 1992 | NYSE TAQ | half-hour cross-section; spreads | Short-lag reversal, same-time-of-day continuation up to 40 days; spreads wider at open and close | Time-of-day, cost |
| Zarattini, Barbon, Aziz | 2024 | 7,000+ US stocks 2016-23 | 5-min opening-range breakout, stop 10% ATR | Filters: price over $5, volume over 1M ("exclude penny stocks"). All eligible: Sharpe 0.48 vs 0.78 S&P. Relative volume over 100%: +0.08R vs -0.02R; top-20 stocks in play: Sharpe 2.81. Commission $0.0035/share; no spread or slippage found in text; long and short | Best ORB evidence; not our universe |
| Zarattini et al. (SPY; VWAP) | 2024; 2023 | SPY; QQQ | noise area; VWAP trend | Reported Sharpe 1.33 (SPY, net) and about 2.10 (QQQ), both second-hand; third-party replications lower (UNVERIFIED) | Working papers |
| Berkman; Lou; Akbas | 2012; 2019; 2022 | US stocks | overnight vs intraday | Overnight gains reverse in attention stocks; buying near the open costs more than the half-spread | Explains -1.3%/day |
| Osler; Bhattacharya | 2000; 2012 | FX 1-min; 100M US trades | levels vs 10,000 random sets | 70%+ of levels end in 0, 96% in 0 or 5; bounces exceed random levels; imbalance peaks at integers | Round-number features |
| Brock; Sullivan | 1992; 1999 | DJIA | technical rules | Buys beat sells, no costs; best rule no edge 1987-96 (secondary) | Snooping warning |
| Barber; Chague | 2014; 2019 | Taiwan; Brazil | day-trader panels | Predictable skill rare (Barber draft); "virtually impossible" to day trade for a living (Chague abstract, second-hand) | Negative prior |

Synthesis: two literatures pull opposite ways on a gap-up. The attention literature predicts reversal, strongest in hard-to-arbitrage stocks, and our pool's drift fits. The opening-range literature predicts continuation on high relative-volume days, but its universe is liquid and its costs omit spread. Our arithmetic: the minimum tick is $0.01 at or above $1 (SEC Rule 612), so a 1-cent spread is 0.5% at $2 before commission. Branch-Ma (2018) report the overnight-intraday reversal persisting in small caps (secondary, UNVERIFIED). I found no peer-reviewed study of minute-level entry on $1-$20 gappers.

## Q4. Market and sector context

| Paper | Year | Data | Method | Result incl. costs | Relevance |
|---|---|---|---|---|---|
| Huddleston et al. | 2023 | 5-min S&P 500 | constituent lagged returns | Adds information beyond market returns, trends, liquidity; Sharpe 0.98 net; stronger mid-day, volatile days | Index-level |
| Liu, Stentoft | 2023 | SPY, 9 sector ETFs, S&P 500 stocks | 17 ML methods; stocks predicted from SPY and constituents; trade only above spread | Results paywalled; Sharpe near 4 net quoted second-hand (UNVERIFIED) | Closest design |
| Chinco et al. | 2019 | 1-min US stocks | LASSO on all lagged returns | +23% OOS fit vs OLS; signals rare, short-lived, news-linked | Context decays in minutes |
| Ferreira, Medeiros | 2021 | 1-min index | LSTM vs forest | VIX is the strongest predictor, LSTM only; preprint, no costs | Market, not stock |
| Gao; Baltussen | 2018; 2021 | 10 other ETFs | intraday momentum | Present in other ETFs; stronger on negative-gamma days | Regime variable |

Synthesis: context information exists at index level (constituents, VIX, ETF momentum) but is small and fast-decaying. The only single-stock test (Liu-Stentoft) was not retrievable. I found no peer-reviewed intraday evidence for breadth. Expect market and sector state to work as a regime gate rather than stock-level alpha.

## Q5. Evaluation hygiene

| Paper | Year | Method | Key claim | Relevance |
|---|---|---|---|---|
| Lopez de Prado, 10 reasons | 2018 | multiple testing | About 20 iterations find a false strategy at 5%, walk-forward included | Log all trials |
| Bailey, Lopez de Prado | 2014 | Deflated Sharpe Ratio | Corrects for trial count and non-normality | Primary gate |
| Bailey et al. | 2017 | PBO via combinatorially symmetric CV | P(in-sample best falls below OOS median) | Over E1-E6 candidates |
| Harvey, Liu, Zhu | 2016 | multiple testing | New factors need t above 3.0 | Our arithmetic: 1,361 days (5.4 yr) needs annual Sharpe about 1.3 |
| White; Hansen | 2000; 2005 | Reality Check; SPA | Best-of-many vs benchmark; SPA more powerful (secondary) | Benchmark: random entries, same exits |
| Arian et al. | 2024 | CPCV vs walk-forward, synthetic | Findings not retrieved; UNVERIFIED | Existence only |

Synthesis: any search over labels, horizons, features and architectures must count trials and report DSR and PBO, not the best run's Sharpe. Combinatorial purged CV (the AFML book, not read here; chapter mapping UNVERIFIED) yields a distribution of paths, but the 10 same-day names are dependent, so folds must group by date.

## Design implications

- **Labels (E5).** Keep vol-scaled triple-barrier with a profit barrier at least a cost multiple (Lopez de Prado 2018b). Run fixed-horizon net return as the control. Count every H and barrier-ratio point as a trial.
- **Models, in order.** (1) Boosting on engineered features (Grinsztajn 2022; Huddleston 2023; Qlib). (2) A small TCN or GRU on normalized raw bars plus the same features (Qlib raw-window result). (3) PatchTST-style only if (2) beats (1) (Zeng 2023). Skip zero-shot foundation models (Rahimikia 2025; Tan 2024; Valeyre 2025).
- **Population (E2).** Pool across stocks (Sirignano-Cont 2019); compare with pool-only.
- **Features (E3), by evidence.** Relative volume and opening-range direction (Zarattini 2024); gap size and attention proxies (Berkman 2012; Akbas 2022); minute of day and spread proxy (McInish 1992; Heston 2010); market return since the open, VIX, sector ETF (Gao 2018; Ferreira 2021; Huddleston 2023); distance to whole and half dollars (Osler 2000; Bhattacharya 2012). Drop breadth.
- **Policy (E6).** Price floor and early-minute exclusion follow from tick and spread arithmetic (SEC Rule 612; McInish 1992). Add a rule-based opening-range breakout on high relative volume as a baseline (Zarattini 2024).
- **Evaluation (E7-E8).** Date-grouped purged CV; White/Hansen against a random-entry benchmark; DSR with the true trial count, PBO over candidates, t above 3 (Bailey 2014, 2017; Harvey 2016; White 2000; Hansen 2005). Calibrate rank-IC to a martingale null.
- **Prior.** Expect a null: I found no peer-reviewed minute-level after-cost success on low-priced gappers.

## References

- Akbas, Boehmer, Jiang, Koch (2022). Overnight returns, daytime reversals, and future stock returns. JFE 145(3). doi:10.1016/j.jfineco.2021.09.019
- Arian, Norouzi Mobarekeh, Seco (2024). Backtest overfitting in the machine learning era. Knowledge-Based Systems 305. doi:10.1016/j.knosys.2024.112477
- Bailey, Lopez de Prado (2014). The Deflated Sharpe Ratio. JPM 40(5). doi:10.3905/jpm.2014.40.5.094
- Bailey, Borwein, Lopez de Prado, Zhu (2017). The probability of backtest overfitting. J. Computational Finance. doi:10.21314/JCF.2016.322
- Baltussen, Da, Lammers, Martens (2021). Hedging demand and market intraday momentum. JFE 142(1). https://www3.nd.edu/~zda/intramom.pdf
- Barber, Lee, Liu, Odean (2014). The cross-section of speculator skill. J. Financial Markets 18. doi:10.1016/j.finmar.2013.05.006
- Berkman, Koch, Tuttle, Zhang (2012). Paying attention. JFQA 47(4). doi:10.1017/S0022109012000270
- Bhattacharya, Holden, Jacobsen (2012). Penny wise, dollar foolish. Management Science 58(2). doi:10.1287/mnsc.1110.1364
- Branch, Ma (2018). Overnight-intraday returns, a retrospective. Summary only: https://www.cxoadvisory.com/?p=31761
- Briola, Bartolucci, Aste (2024). Deep limit order book forecasting. arXiv:2403.09267
- Brock, Lakonishok, LeBaron (1992). Simple technical trading rules. JF 47(5). https://ideas.repec.org/a/bla/jfinan/v47y1992i5p1731-64.html
- Carr, Lopez de Prado (2014). Determining optimal trading rules without backtesting. arXiv:1408.1159
- Chague, De-Losso, Giovannetti (2019). Day trading for a living? SSRN 3423101
- Chinco, Clark-Joseph, Ye (2019). Sparse signals in the cross-section of returns. JF 74(1). doi:10.1111/jofi.12733
- Ferreira, Medeiros (2021). Modeling and forecasting intraday market returns. arXiv:2112.15108
- Fischer, Krauss (2018). Deep learning with LSTM networks for financial market predictions. EJOR 270. https://hdl.handle.net/10419/157808
- Gao, Han, Li, Zhou (2018). Market intraday momentum. JFE 129(2). doi:10.1016/j.jfineco.2018.05.009
- Grinsztajn, Oyallon, Varoquaux (2022). Why do tree-based models still outperform deep learning on tabular data? NeurIPS. arXiv:2207.08815
- Hansen (2005). A test for superior predictive ability. JBES 23(4). doi:10.1198/073500105000000063
- Harvey, Liu, Zhu (2016). ... and the cross-section of expected returns. RFS 29(1). https://www.nber.org/papers/w20592
- Heston, Korajczyk, Sadka (2010). Intraday patterns in the cross-section of stock returns. JF 65(4). doi:10.1111/j.1540-6261.2010.01573.x
- Huddleston, Liu, Stentoft (2023). Intraday market predictability. J. Financial Econometrics 21(2). doi:10.1093/jjfinec/nbab007
- Joubert (2022). Meta-labeling: theory and framework. JFDS 4(3). doi:10.3905/jfds.2022.1.098
- Kolm, Turiel, Westray (2023). Deep order flow imbalance. Math. Finance 33(4). doi:10.1111/mafi.12413
- Krauss, Do, Huck (2017). Deep neural networks, gradient-boosted trees, random forests. EJOR 259(2). doi:10.1016/j.ejor.2016.10.031
- Li, Cao, Yang, Wang (2022). Model optimization for stock market prediction using multiple labelling techniques. IEEE ICSESS. https://research.mpu.edu.mo/en/publications/model-optimization-for-stock-market-prediction-using-multiple-lab/
- Liu, Stentoft (2023). Intraday stock predictability everywhere (working paper; summary only). https://www.cxoadvisory.com/?p=44469
- Lopez de Prado (2018a). Advances in Financial Machine Learning. Wiley. ISBN 978-1-119-48208-6
- Lopez de Prado (2018b). The 10 reasons most machine learning funds fail. JPM 44(6). https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf
- Lou, Polk, Skouras (2019). A tug of war. JFE 134(1). doi:10.1016/j.jfineco.2019.03.011
- McInish, Wood (1992). Intraday patterns in bid/ask spreads for NYSE stocks. JF 47(2). doi:10.1111/j.1540-6261.1992.tb04408.x
- Mesfin (2026). Sequential structure in intraday futures data: LSTM vs gradient boosting on MNQ. arXiv:2605.17724
- Microsoft Qlib benchmarks. https://github.com/microsoft/qlib/tree/main/examples/benchmarks
- Noguer i Alonso, Franklin (2026). Pretrained time-series foundation models for financial return forecasting. arXiv:2606.27100
- Osler (2000). Support for resistance. FRBNY Economic Policy Review 6(2). https://resources.newyorkfed.org/medialibrary/media/research/epr/00v06n2/0007osle.pdf
- Prata et al. (2024). LOB-based deep learning models for stock price trend prediction. Artif. Intell. Rev. 57. arXiv:2308.01915
- Rahimikia, Ni, Wang (2025). Re(visiting) time series foundation models in finance. arXiv:2511.18578
- SEC. 17 CFR 242.612. https://www.law.cornell.edu/cfr/text/17/242.612
- Shi et al. (2025). Kronos. arXiv:2508.02739
- Sirignano, Cont (2019). Universal features of price formation. Quantitative Finance. arXiv:1803.06917
- Sullivan, Timmermann, White (1999). Data-snooping, technical trading rule performance, and the bootstrap. JF 54. doi:10.1111/0022-1082.00163
- Tan et al. (2024). Are language models actually useful for time series forecasting? NeurIPS. arXiv:2406.16964
- Valeyre, Aboura (2025). LLMs for time series: single stocks and statistical arbitrage. arXiv:2412.09394
- White (2000). A reality check for data snooping. Econometrica 68(5). doi:10.1111/1468-0262.00152
- Zarattini, Barbon, Aziz (2024). A profitable day trading strategy for the U.S. equity market. SSRN 4729284. https://ssrn.com/abstract=4729284
- Zarattini, Aziz, Barbon (2024). Beat the market: intraday momentum for SPY. SFI RP 24-97. https://www.sfi.ch/de/publications/n-24-97-beat-the-market-an-effective-intraday-momentum-strategy-for-s-p500-etf-spy
- Zarattini, Aziz (2023). VWAP: the holy grail for day trading systems. SSRN 4631351
- Zeng, Chen, Zhang, Xu (2023). Are Transformers effective for time series forecasting? AAAI. arXiv:2205.13504
- Zhang, Zohren, Roberts (2019). DeepLOB. IEEE TSP. doi:10.1109/TSP.2019.2907260
- Architectures without finance claims: PatchTST arXiv:2211.14730; iTransformer arXiv:2310.06625; TimesNet arXiv:2210.02186; TFT doi:10.1016/j.ijforecast.2021.03.012; Chronos arXiv:2403.07815; TimesFM arXiv:2310.10688; Lag-Llama arXiv:2310.08278
