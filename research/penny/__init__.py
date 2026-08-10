"""Penny/microcap veto research.

Question under test
-------------------
Does there exist a veto set V, computable from free data at 09:35 ET with no
lookahead, such that the retained population has E[net return] > 0 with a
confidence interval excluding zero after multiple-testing correction?

The honest default answer is "no".  This package is built so that a "no" is
cheap and fast, and a "yes" is trustworthy rather than a data-mining artifact.

Isolation contract
------------------
Nothing here is imported by the live trading path.  All persistent output goes
to ``data/research_penny/`` (gitignored).  No module in this package may import
from ``scripts.live_run``, ``agents.execution_agent``, ``utils.state_store``,
``utils.live_decision``, ``agents.live_orchestrator`` or ``run_live``.
"""
