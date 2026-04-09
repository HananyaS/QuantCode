# Project Memory: Quant AI Lab

## Goal
Build a modular multi-agent system for stock prediction and backtesting.

## Core Principles
- No data leakage
- Deterministic results
- Full validation before any commit
- Modular architecture (agent-based)
- All outputs must be testable

## Pipeline
data → features → labels → model → backtest → evaluation

## Constraints
- NEVER use future data
- ALWAYS validate data integrity
- ALWAYS compare vs SPY benchmark
- Tests must pass before committing
- NEVER modify tests automatically

## Coding Standards
- Type hints required
- Functions must be small and modular
- Use vectorized pandas ops
- Add assertions for sanity checks

## Agent Philosophy
Agents are deterministic modules, not autonomous decision-makers.
Each agent:
- receives context
- updates context
- logs outputs