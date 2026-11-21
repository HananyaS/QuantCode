# Alpaca order behaviour the live runner relies on (fact sheet, 2026-11)

Accessed 2026-11-22. Docs quotes are verbatim (raw page Markdown). Forum quotes came via a summarising fetch tool. "Not found in docs" means searched and absent.

Short names: **Orders** = https://docs.alpaca.markets/us/docs/orders-at-alpaca · **CreateOrder** = https://docs.alpaca.markets/us/reference/postorder

## Summary

| # | Fact | Confidence | Source |
|---|---|---|---|
| 1 | GTC auto-cancel 90 days after creation (4:15 pm ET on `expires_at`) | documented | Orders |
| 2 | Bracket/OTO `day`/`gtc` only (no `opg`); OCO exit-only `limit`; SL stop or stop-limit ≥ $0.01 below base; legs inherit TIF | documented; inheritance forum-only | Orders |
| 3 | Fractional/notional DAY only; with bracket/OCO not found | documented | Orders, Fractional |
| 4 | OPG rejected 9:28 am–7 pm ET, CLS 3:50–7 pm; paper fills at quote | documented; paper forum-only | Orders; forum |
| 5 | Open sells hold shares; second exit rejected; cancel then new | documented; error text forum-only | Orders; Positions |
| 6 | GTC limits adjusted; bracket/OCO legs DNR/DNC; FAQ: reverse split cancels GTC | documented, conflicting | Orders; CA FAQ |
| 7 | 2 decimals ≥ $1.00, 4 below; finer rejected | documented | Orders |
| 8 | Paper: NBBO, marketable only, size unchecked, 10% partials, no queue/dividends/fees/splits | documented; splits forum-only | Paper |
| 9 | PDT and $25k removed (2026-06-04) | documented | Intraday Margin Rule |
| 10 | $0 commission; SEC $20.60/M; TAF $0 to 2026-12-31; CAT $0.000003/sh | documented | Fee schedule; SEC; FINRA |
| 11 | 200 req/min; `client_order_id` ≤ 128 chars; duplicates forum-only | documented; forum-only | Support; Websocket |
| 12 | Free: IEX, 30 symbols, 1 stream, SIP ≥ 15 min old; Plus $99/mo | documented | Market Data |
| 13 | `extended_hours` default false; pre-market stop election not found | documented; unverified | CreateOrder |

## 1. GTC expiry (documented)
Orders §Aged order policy: "automatically cancel GTC orders 90 days after creation … On the `expires_at` date there will be a job that will submit a cancel request … at 4:15 pm ET. The orders will remain in pending_cancel until canceled by the execution venue".

**Not found in docs:** whether these are calendar or trading days, whether the final event is `canceled` or `expired`, and whether paper applies the policy.

## 2. Order classes (documented unless marked)
Orders §Bracket, §OCO, §OTO, §Threshold.
- **Bracket:** "`time_in_force` must be `day` or `gtc`" (so no `opg`). "`extended_hours` must be "false" or omitted". The exits "won't be active until the first order is completely filled". The TP "must be higher than `stop_loss.stop_price` for a buy bracket". "both orders may fill before the cancellation occurs".
- **OTO:** "the rest of requirements are the same as the bracket orders".
- **OCO:** "currently only exit order is supported", and "Alpaca's system only accepts exit orders for existing positions". The type "must always be "limit"". The SL "is a stop order if only `stop_price` is specified, and is a stop-limit order if both". The allowed OCO TIF values are **not found in docs** (the examples use `gtc`).
- **Stop threshold:** "at least $0.01 below … the "base price"". The base price is the TP limit (OCO) or the entry limit (bracket), and "also the current market price".
- **Leg TIF (forum-only):** "the time-in-force of the take profit and stop loss order is the same as the order that triggered them" (https://forum.alpaca.markets/t/bracket-orders-should-be-able-to-specify-time-in-force/15060). alpaca-py does not check TIF by order class ([requests.py L264-289](https://github.com/alpacahq/alpaca-py/blob/60a5c3a045d70e48cbda7dd1a625cb3263671bfb/alpaca/trading/requests.py#L264-L289)).

## 3. Fractional and notional (documented)
Orders §Order Types vs Supported Time in Force: the fractional table marks GTC, IOC, FOK, OPG and CLS "No" and DAY "Yes". https://docs.alpaca.markets/us/docs/fractional-trading: "by setting `time_in_force` to `day`". **Not found in docs:** fractional quantities on bracket/OCO/OTO. A forum user says "fractional shares are only executing through simple orders" (https://forum.alpaca.markets/t/bracket-order-with-fractional-shares/12027).

## 4. Auction orders (documented / forum-only)
Orders §Time in Force: "OPG orders submitted after 9:28am but before 7:00pm ET will be rejected. OPG orders submitted after 7:00pm will be queued and routed to the following day's opening auction." "Any unfilled orders after the open will be cancelled." "CLS orders submitted after 3:50pm but before 7:00pm ET will be rejected." Such orders "are routed to the primary exchange".

The whole-share table marks OPG/CLS "Yes*" with the note "Please contact the sales team for any TIF marked with a*". The asterisk is not explained.

**Not found in docs:** cancel deadlines, and paper auction simulation. Forum (staff handle, 2023): "In paper trading all market orders are 'filled' at the current quote" and "paper trading doesn't differentiate Market On Open (MOO) and Market On Close (MOC) orders" (https://forum.alpaca.markets/t/opening-and-closing-market-prices-in-paper-trading/11664).

## 5. Shares held by an open OCO
- **Documented:** Orders §Bracket says a second exit would be seen "as exceeding the available position quantity". Position `qty_available` is "Total number of shares available minus open orders" (https://docs.alpaca.markets/us/reference/getallopenpositions).
- **Forum-only:** the error is code `40310000`, "insufficient qty available for order", with `held_for_orders` set (https://forum.alpaca.markets/t/liquidation-of-open-positions-dont-work-due-to-insufficient-qty/12267).
- **Documented (PATCH):** an OCO's `limit_price`/`stop_price` can be PATCHed, but "A success return code from a replaced order does NOT guarantee the existing open order has been replaced" (https://docs.alpaca.markets/us/reference/patchorderbyorderid-1).
- **Not found in docs:** PATCHing an OCO leg to `cls`. Use cancel, then confirm, then a new order.

## 6. Corporate actions (documented, conflicting)
- **Orders:** "Non-marketable GTC limit orders are subject to price adjustments to offset corporate actions". Bracket legs (and so OCO legs) carry "DNR/DNC … the order price will not be adjusted and the order will not be canceled in the event of a dividend or other corporate action".
- **Broker API FAQ** (https://docs.alpaca.markets/us/docs/mandatory-corporate-actions): "For reverse splits all GTC orders will be canceled … For forward splits, GTC buy limits and sell stops are adjusted". CA cancels are "enabled per correspondent". This conflicts with DNR/DNC, and the FAQ's scope for Trading API accounts is not stated.
- **Activity types** (https://docs.alpaca.markets/us/docs/account-activities): `SSP` (stock split), `REORG`, `SC` (symbol change), `NC`, `MA`, `SSO`. There is no `SPLIT` type.
- **Endpoint:** `GET https://data.alpaca.markets/v1/corporate-actions` (https://docs.alpaca.markets/us/reference/corporateactions-1). `/v2/corporate_actions/announcements` is "deprecated".
- **Not found in docs:** what happens to orders on a delisting.

## 7. Price increments (documented)
Orders §Sub-penny: "Limit price >=$1.00: Max Decimals= 2" and "Limit price <$1.00: Max Decimals = 4". The same rule applies to stop prices. "Orders received in excess of the minimum price variance will be rejected" with error `42210000`.

The minimum stop distance is the $0.01 base-price rule in §2. https://docs.alpaca.markets/us/docs/user-protection lists a "Limit order price away sanity check" but gives no threshold (**not found in docs**).

## 8. Paper fill model (documented / forum-only)
https://docs.alpaca.markets/us/docs/paper-trading: a buy limit is "not … filled until its limit price is equal to or greater than the best ask". "Your order quantity is not checked against the NBBO quantities." Orders get "partial fills for a random size 10% of the time". Paper "does not account for … Order queue position … Price improvement … Regulatory fees … Dividends".

Forum (staff, 2026-07-15): "paper accounts do not implement any corporate actions such as stock splits or dividends" (https://forum.alpaca.markets/t/corporate-actions-handeling/19325).

## 9. PDT (documented)
https://docs.alpaca.markets/us/docs/the-intraday-margin-rule: "The definition of a "pattern day trader" has been eliminated." "The specific $25,000 equity minimum required to day trade has been removed."

Effective June 4, 2026 (https://alpaca.markets/blog/finra-retires-the-pdt-rule-introducing-alpacas-new-intraday-margin-framework/). `daytrade_count` was removed 2026-07-06 (https://docs.alpaca.markets/us/changelog/2026-06-03-pdt-651df23). A same-day TP fill is no longer counted. **Not found in docs:** an explicit statement for paper.

## 10. Fees (documented)
- **Fee schedule, revised November 17, 2026** (https://files.alpaca.markets/disclosures/library/BrokFeeSched.pdf):
  - "does not charge commissions, except as described below" (index options, Elite Smart Router, non-retail flow).
  - On sells: "SEC Transaction Fee $0.0000206 * Trade Value" and "FINRA Trading Activity Fees (TAF) … $0 per share".
  - On both sides: CAT "$0.000003 per executed equivalent share".
- **SEC:** "$20.60 per million dollars" from April 4, 2026 (https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2).
- **FINRA:** the TAF is "$0.00" for "October 1, 2026 through December 31, 2026", "resuming on January 1, 2027" at $0.000195/share with a $9.79 cap (https://www.finra.org/sites/default/files/2026-09/SR-FINRA-2026-021.pdf).

## 11. Limits, events, ids
- **Rate limit:** "200 requests per minute", "per account", HTTP 429 (https://alpaca.markets/support/usage-limit-api-calls, a 2022 support article, not the docs).
- **`trade_updates` events** (https://docs.alpaca.markets/us/docs/websocket-streaming), documented:
  - Common: `new`, `fill`, `partial_fill`, `canceled`, `expired`, `done_for_day`, `replaced`.
  - Rare: `accepted`, `rejected`, `pending_new`, `stopped`, `pending_cancel`, `pending_replace`, `calculated`, `suspended`, `order_replace_rejected`, `order_cancel_rejected`.
  - Grouped orders carry `group_id`/`group_role` because "`legs` is not populated" on the stream (CreateOrder).
- **`client_order_id`** (CreateOrder): "A unique identifier for the order. Automatically generated if not sent. (<= 128 characters)". Look-up is `GET /v2/orders:by_client_order_id`. Duplicate rejection is **not found in docs**. Forum: "Each order must have a unique `client_order_id`" (https://forum.alpaca.markets/t/lifo-during-intraday-trading/2816).

## 12. Data plans (documented)
- https://docs.alpaca.markets/us/docs/about-market-data-api: Basic is "Free", "IEX", "30 symbols", "200 / min". Algo Trader Plus is "$99 / month" with "All US Stock Exchanges".
- https://docs.alpaca.markets/us/docs/market-data-faq: "the `end` parameter must be at least 15 minutes old to query SIP data without a subscription".
- https://docs.alpaca.markets/us/docs/streaming-market-data: "most users can have only 1 active stream connection".

## 13. Extended hours (documented / unverified)
CreateOrder: `extended_hours` is "(default) false … Only works with type `limit` and time_in_force set to either `day` or `gtc`". Orders §Order Handling: "Stop orders and trailing stops are elected on the consolidated print".

**Not found in docs:** whether a pre-market print can elect a GTC stop.

## Implications for the R15 runner
1. **Re-arm before expiry (1).** The 60-session hold is about 87 calendar days. Read `expires_at` and re-arm (cancel and resubmit) before it.
2. **OPG entries (2, 3, 4).** Use whole-share `limit`+`opg`, not a bracket parent. Send by 9:28 ET and treat a rejection as a miss. Smoke-test OPG/CLS acceptance because of the "Yes*" note.
3. **OCO after a full fill (2, 5).** Set qty to the filled shares. If the price is already at the stop, the OCO is rejected, so sell instead.
4. **Day 60 (4, 5).** Cancel the OCO, then wait for `canceled` on both legs and for `qty_available` to equal the position. Then send the `cls` sell before 15:50 ET.
5. **Ticks (7).** Round each price for its own level and keep TP − SL ≥ $0.01. Smoke-test the +100% TP against the price-away check.
6. **Corporate actions (6, 8).** Reconcile orders against positions and `SSP`/`REORG`/`SC` activities daily, and re-arm on any mismatch. Paper does not apply splits.
7. **Paper is optimistic (4, 8, 10).** Fills are at the quote, not the auction price, and carry no fees. Book SEC and CAT fees yourself, plus the TAF from 2027.
8. **Day brackets (2, 9).** Exits stay inactive until the entry fully fills, and both legs can fill. PDT no longer applies.
9. **Ids (11).** Give each order and each re-arm a unique, deterministic `client_order_id` (≤ 128 chars). Track by `group_id` and stay under 200 req/min.
