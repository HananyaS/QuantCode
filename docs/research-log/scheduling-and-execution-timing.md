# Research Log: Scheduling and Execution Timing

Why the live bot's trigger moved off GitHub cron, and what that does *not*
fix. Prompted by a complaint about duplicate push notifications, which
turned out to sit on top of a much larger measurement problem: nobody had
ever checked *when* the bot actually trades.

## 1. GitHub cron is not late sometimes — it is late always

Measured over every scheduled firing of `live-trading.yml` from the first
one (2026-07-28) through 2026-08-19, comparing each run's `created_at`
against its cron time. The schedule changed on 2026-07-28 17:20 UTC
(`cf40449`, 15:00 → 14:45 + 15:45), so firings are matched against
whichever schedule was live at the time.

| | delay past cron time |
|---|---|
| firings that were late | **33 / 33 (100%)** |
| median | **54 min** |
| p90 | 102 min |
| max | **135 min** (2026-08-03 primary) |
| ≥ 30 min late | 27 / 33 (82%) |
| ≥ 60 min late | 15 / 33 (45%) |

Primary firing alone (the one that trades, n=17): median 55 min, p90 107
min. So the nominal 14:45 UTC = 10:45 ET entry actually landed anywhere
between 11:03 and 13:00 ET, varying day to day.

Delays did shrink over the sample (median 100 min across the earliest ten
firings, 18 min across the last six) but never to zero, and GitHub
documents `schedule` as best-effort with no punctuality guarantee. This is
not a setting we can tune.

### Why it costs money

`KellyBacktestAgent` and `LeveragedBacktestAgent` earn `close.pct_change()`
— close(t-1) → close(t) — for the position held during session t, with no
execution-lag shift (`agents/timing/kelly_backtest_agent.py:71-74`, and see
that module's docstring for why the shift is deliberately absent). The
backtest therefore assumes the position is on before session t moves.

Live, on a rotation day the entire book is repriced at an unpredictable
time mid-morning, on a 3x ETF. That is tracking error nobody pays us for,
and — worse for diagnosis — it is *random*, so it cannot be modelled or
backed out afterwards.

## 2. The fix: punctual trigger outside GitHub, cron as watchdog

Only the *trigger* moved. Ledgers, secrets, the Pages dashboard, and every
line of Python stayed where they were.

* **Primary** — an external scheduler (cron-job.org, 09:45 Mon–Fri,
  timezone `America/New_York`) POSTs to the workflow's `/dispatches`
  endpoint. API dispatches do not go through the cron batch queue, and a
  timezone-aware scheduler also removes the hour of DST drift that
  GitHub's UTC-only cron cannot express.
* **Watchdog** — both crons stay, untouched. When the external trigger
  worked they find the session already complete and stand down silently.
  When it did not, they still trade, and the push is tagged
  `via cron fallback` so a dead trigger announces itself.

Rollback is disabling the external job; the crons alone are exactly the old
behaviour.

### Alternatives rejected

* **Sleep-to-target inside the job** — fire cron early, then wait until a
  target computed from Alpaca's calendar. Needs no new account and is
  DST-proof, but parks a runner idle ~2h/day. Free on a public repo, yet it
  sits badly with GitHub's Actions usage policy.
* **Move execution off GitHub** (VPS / Lambda + EventBridge / Cloud Run +
  Cloud Scheduler) — precise to the second, but forces migrating the
  git-committed SQLite ledgers, the secrets, and the Pages report, and
  hands us uptime and patching to own. Disproportionate for a scheduling
  problem.

### Why `workflow_dispatch` and not `repository_dispatch`

Both would work. `workflow_dispatch` was chosen for **credential blast
radius**: its API needs a fine-grained PAT with `Actions: write`, whereas
`repository_dispatch` needs `Contents: write`. This repo is public and its
Actions secrets hold live Alpaca credentials, so a `Contents: write` token
parked in a third-party scheduler is a path to pushing code that then runs
with those credentials. `Actions: write` cannot modify code.

That choice costs one footgun, closed in the workflow: `dry_run` defaults
to `true`, so a scheduler payload that omitted it would leave the bot
previewing forever — silently never trading while every run still reported
success. `dry_run` is therefore honoured **only** for `source: human`.
`tests/test_live_trading_workflow.py` holds both that line and the
`--manual` one, because they live in YAML expressions where a plausible
"simplification" would disable the once-per-session guard with every Python
test still green.

## 3. Deferred: punctuality has a ceiling, and it is low

Measured on 521 sessions of real Alpaca daily bars (2024-07-22 → 2026-08-19):

| | TQQQ | QQQ |
|---|---|---|
| session close→close, sd | 4.19% | 1.39% |
| overnight gap close(t-1)→open(t), sd | **2.72%** | 0.91% |
| intraday open→close, sd | 3.39% | 1.13% |

The overnight gap is **65% of TQQQ's session risk**, and no amount of
punctuality recovers any of it. The backtest earns close(t-1)→close(t), so
it assumes the position is on at the prior close, while the bot enters the
next morning. Fixing the trigger addresses the intraday slice — real, worth
having, and now consistent — but it is the smaller half.

Closing that gap means a different execution model: decide from the
near-final bar and trade the closing auction of day t-1 (MOC orders around
15:50 ET), which would match the backtest almost exactly. That is a change
to `live_run.py`'s bar/session logic with its own validation burden — in
particular the decision would then be made from an incomplete bar, which
`utils/live_decision.py::drop_incomplete_last_bar` currently exists
specifically to prevent — so it is left for its own evidence and its own
commit rather than bundled with a scheduling fix.
