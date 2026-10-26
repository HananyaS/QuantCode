# Vendor facts: Alpaca market data and always-on hosts

Boards #33 and #39. Primary sources only (vendor docs and pricing pages). All pages accessed 2026-10-26.
Pages were read through a fetch tool that summarises the page, so quotes are as returned by that tool.
Where a page did not state a fact it is marked "not confirmed"; nothing is filled from memory or blogs.

## Q1. Alpaca market data, current terms (board #33)

| Fact | Status | Finding and source |
|---|---|---|
| Free/Basic real-time stock data is IEX only | confirmed | Basic covers only the IEX exchange; Algo Trader Plus covers "All US Stock Exchanges". https://docs.alpaca.markets/docs/about-market-data-api (accessed 2026-10-26). Also: "Our free market data offering includes live data only from the IEX exchange." https://docs.alpaca.markets/docs/market-data-faq (accessed 2026-10-26) |
| SIP is delayed 15 minutes on free | partly confirmed | A 15-minute delayed SIP feed exists as `v2/delayed_sip` (https://docs.alpaca.markets/docs/real-time-stock-pricing-data, accessed 2026-10-26). The pages read do not say outright that Basic gets it for real-time streaming; they say only that Basic real-time is IEX. Not confirmed: whether Basic can use `delayed_sip` on the stream. Tried: the three pages above plus the websocket page. |
| Historical SIP is free with a lag of 15 minutes | confirmed | "the `end` parameter must be at least 15 minutes old to query SIP data without a subscription"; newer queries error with "subscription does not permit querying recent SIP data". https://docs.alpaca.markets/docs/market-data-faq (accessed 2026-10-26). The plan table says Basic historical is limited to "latest 15 minutes" (https://docs.alpaca.markets/docs/about-market-data-api, accessed 2026-10-26). |
| Algo Trader Plus price and what it adds | confirmed | $99 per month. Adds: all US stock exchanges real-time (vs IEX only), OPRA options feed (vs indicative pricing), no historical restriction (vs 15 minutes), unlimited equity websocket symbols (vs 30), 1,000 option quotes (vs 200), 10,000 historical calls per minute (vs 200). https://docs.alpaca.markets/docs/about-market-data-api (accessed 2026-10-26) |
| Websocket limit: symbols per plan | confirmed | Basic 30 equity symbols and 200 option quotes; Algo Trader Plus unlimited equity symbols and 1,000 option quotes. https://docs.alpaca.markets/docs/about-market-data-api (accessed 2026-10-26) |
| Websocket limit: connections per plan | not confirmed | Neither the plan page nor https://docs.alpaca.markets/docs/websocket-streaming (accessed 2026-10-26) states a connection count. |
| REST rate limits per plan | confirmed | Basic 200 historical API calls per minute; Algo Trader Plus 10,000 per minute. https://docs.alpaca.markets/docs/about-market-data-api (accessed 2026-10-26). Trading API rate limits were not asked for and not checked. |
| Paper accounts differ in data | confirmed | "As an Alpaca Paper Only Account holder, you are only entitled to receive and make use of IEX market data." The Market Data API "works identically" in paper and live. https://docs.alpaca.markets/docs/paper-trading (accessed 2026-10-26). The plan page says Basic is the default for both paper and live. https://docs.alpaca.markets/docs/about-market-data-api (accessed 2026-10-26). Not confirmed: whether a paper-only account can buy Algo Trader Plus. |

## Q2. Always-on hosts for a 09:00-16:00 ET single-process engine (board #39)

### GitHub Actions

| Fact | Status | Finding and source |
|---|---|---|
| Job time limit, hosted runners | confirmed | "Each job in a workflow can run for up to 6 hours of execution time." The job is terminated and fails. A 09:00-16:00 session is 7 hours, so it does not fit in one job. Self-hosted runners: up to 5 days. Workflow run total: 35 days. https://docs.github.com/en/actions/reference/limits (accessed 2026-10-26) |
| Scheduled workflow delays | confirmed | "The `schedule` event can be delayed during periods of high loads of GitHub Actions workflow runs." "High load times include the start of every hour." Advice: run at a different minute. https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows (accessed 2026-10-26) |
| Dropped runs | confirmed | "If the load is sufficiently high enough, some queued jobs may be dropped." Same page, same date. |
| Auto-disable in inactive repos | confirmed | "In a public repository, scheduled workflows are automatically disabled when no repository activity has occurred in 60 days." Same page, same date. Not confirmed from this page: behaviour for private repos (the text names public repositories only). |
| Other schedule rules | confirmed | Shortest interval 5 minutes; UTC by default, optional IANA timezone; runs only on the default branch, latest commit. Same page, same date. |

### Cheap always-on options (order of magnitude)

| Option | Status | Smallest tier with >= 2 GB RAM, price |
|---|---|---|
| DigitalOcean Droplet (Basic) | confirmed | 2 GiB RAM, 1 vCPU, $12/month; the 2 GiB 2 vCPU tier is $18/month. https://www.digitalocean.com/pricing/droplets (accessed 2026-10-26) |
| AWS EC2 t3.small | confirmed (price) | 2 vCPU, 2.0 GiB, $0.0209/hour Linux on-demand, US East (N. Virginia). That is about $15/month if run 24x7 (0.0209 x ~730 h, my arithmetic); excludes storage and traffic. https://aws.amazon.com/ec2/instance-types/t3/ (accessed 2026-10-26). t3.medium: 2 vCPU, 4 GiB, $0.0418/hour (about $30/month). |
| Hetzner Cloud | not confirmed | https://www.hetzner.com/cloud/ (accessed 2026-10-26) shows shared plans "starting from" EUR 5.99 and 6.49 per month excl. VAT, but the fetched page did not state RAM or vCPU per plan, and the cheapest plan was labelled "currently not available". Plan pages were not opened. |

Order of magnitude: roughly $10-20 per month for a 2 GB, 1-2 vCPU machine.

## What this means for the plan

- Free data is IEX only, live. That is a single exchange, so volumes and prints will look smaller than the full market. Paper-only accounts get IEX only too. Whether that is good enough is for the replay on both feeds to decide, not this note.
- Full SIP live costs $99 a month. Free historical SIP works only if it is at least 15 minutes old, which is fine for replay and backtests but not for live trading.
- The free stream is capped at 30 symbols, and 200 REST history calls a minute. The screener should be built to stay inside those. The number of allowed connections is not documented in the pages read; do not assume more than one.
- A GitHub Actions job stops after 6 hours, and the session is 7. Scheduled runs can also start late or be dropped, and can be switched off in a quiet public repo. It is a poor fit for a live session. It is fine for short offline jobs.
- An always-on machine of about 2 GB RAM costs roughly $12-20 a month. The user picks the host; nothing here forces a choice.
