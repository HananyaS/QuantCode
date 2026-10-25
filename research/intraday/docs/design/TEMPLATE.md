# <Service> design

Owner: `<intraday-agent>`. Package: `research/intraday/<pkg>/`. Read with the contracts
(`research/intraday/contracts/`) and ADR-001; those and this file are all a builder works from.

Rules for every design doc (enforced by `tests/test_intraday_design_docs.py`):
- Exactly the eleven H2 sections below, in this order, none empty.
- Under Consumes and Emits, every backticked capitalised name is a key of `EVENT_TYPES`
  (`research/intraday/contracts/events.py`). Name adapters and classes in prose without backticks there.
- Only `risk.md` lists the order-intent event under Emits.
- Each open question is a bullet tagged with exactly one of `[blocks-build]` or `[research]`.
- The Tracker section links exactly one issue: the service's epic.

Rules for every builder working from a design doc:
- Never emit while iterating a set or frozenset; sort first (set order varies with
  `PYTHONHASHSEED`, and the log hash with it).
- Event fields take the exact types the contracts require: `0.0`, not `0`, for a float field; a
  `tuple`, not a list or deque, for a tuple field; `datetime` with `tzinfo=timezone.utc`, not a
  `pd.Timestamp`.

## Responsibility

One sentence: the single thing this service does.

## Consumes

Each event type it subscribes to (exact class name, backticked) and what it does with it.

## Emits

Each event type it returns from `on_event` (exact class name), when, and with which field values.
Every emitted event carries the `ts` of its cause (bus rule, ADR-001).

## State

What it holds between events, keyed how, and the bound on its size. State is per session:
one run is one trading session, and agents are constructed fresh for each.

## Configuration

| Parameter | Default | Source |
|---|---|---|
| name | value | `spec` (cite the section), `R16` (set by the pre-registration), or `builder's call` |

Reuse pointers (existing functions, with paths) go here or under Test plan.

## Invariants

Properties that hold after every event. Each is a test.

## Failure modes

| Failure | Detection | Response |
|---|---|---|
| what goes wrong | how it is noticed | one of: raise, emit `Kill`, emit `Gap`, skip |

## Replay vs live

What differs between replay and live. Differences live in adapters only; the agent's
`on_event` code is the same in both.

## Test plan

Test file and test names, written before the code.

## Open questions

- **[blocks-build]** or **[research]** The question, and the default the builder uses meanwhile.

## Tracker

Epic: https://github.com/HananyaS/QuantCode-intraday-board/issues/<n>
