# research/intraday

Deterministic, event-driven intraday trading research engine. Lives on the
`research/intraday` branch and is never merged to main.

## Layout

| Path | Contents | Owner |
|---|---|---|
| `contracts/` | frozen events, codec, agent protocols, fixtures | intraday-architect |
| `engine/` | `EventBus`, replay driver | intraday-architect |
| `data/` | ScreenerAgent, FeedAgent, ETL | intraday-data |
| `state/`, `model/`, `research/` | StateAgent, PredictorAgent, research scripts | intraday-model |
| `risk/` | RiskAgent | intraday-risk |
| `exec/` | ExecutionAgent, brokers | intraday-exec |
| `ops/` | recorder, monitor | intraday-ops |
| `docs/adr/`, `docs/critiques/` | ADRs; critic verdicts | architect; critic |
| `ownership.py` | the `OWNERS` map | intraday-architect |

## Tests

```
PY="C:/Users/Shachar/AppData/Local/Programs/Python/Python313/python.exe"
"$PY" -m pytest tests/test_intraday_*.py -q
```

## Ownership check

Reject a diff that strays outside an agent's owned paths (prints offending files, exit 1):

```
"$PY" scripts/check_ownership.py --agent intraday-risk --base research/intraday
```

## Links

- Spec: [docs/superpowers/specs/2026-10-26-intraday-system-design.md](../../docs/superpowers/specs/2026-10-26-intraday-system-design.md)
- Plan: [docs/superpowers/plans/2026-10-26-intraday-phase1-contracts-engine.md](../../docs/superpowers/plans/2026-10-26-intraday-phase1-contracts-engine.md)
- ADR-001: [docs/adr/0001-event-driven-single-process.md](docs/adr/0001-event-driven-single-process.md)
- Board: https://github.com/users/HananyaS/projects/1
