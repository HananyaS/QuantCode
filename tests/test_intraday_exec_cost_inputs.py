"""Per symbol-day cost inputs from prior sessions only (plan Task 4).

Deviation from the plan (controller ruling P2-5, design docs updated in e09aed7 on
research/intraday): share volume is put on D's traded basis with the split-only
factor ``s = volume / volume_unadj`` of the last prior row, not with
``f = close_unadj / close``. The fixture gives ``f`` an extra dividend factor so the
two rules disagree, and a frame without ``volume_unadj`` raises.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research.intraday.exec.cost_inputs import build_cost_inputs

DATES = pd.bdate_range(end="2024-03-04", periods=26)
D = date(2024, 3, 4)
SPLIT = DATES[20]


def _panel() -> pd.DataFrame:
    i = np.arange(26)
    close = 1.0 + 0.01 * (i % 7)
    s = np.where(i < 20, 2.0, 1.0)              # 2:1 split effective at SPLIT
    div = np.where(i < 25, 1.02, 1.0)           # a dividend ex on D: prices only
    volume = np.where(i < 25, 1_000.0, 1e9)     # adjusted share volume; D's own row huge
    return pd.DataFrame({
        "symbol": "AAA", "date": DATES, "close": close, "volume": volume,
        "close_unadj": close * s * div, "volume_unadj": volume / s})


panel = _panel()


def test_only_prior_sessions_enter() -> None:
    ci = build_cost_inputs(panel, None, D).get("AAA", D)
    assert ci.adv_shares == pytest.approx(1_000.0)          # s_{D-1} = 1; D's 1e9 ignored
    # mixing per-day factors would give (15*500 + 5*1000)/20 = 625: the test tells them apart
    # dividing by f_{D-1} = 1.02 would give 980.4: so does this one


def test_adv_shares_uses_f_prev_for_every_session() -> None:
    d = SPLIT.date()                                         # last prior row (19) has s = 2
    ci = build_cost_inputs(panel, None, d).get("AAA", d)
    assert ci.adv_shares == pytest.approx(1_000.0 / 2.0)


def test_missing_volume_unadj_raises() -> None:
    with pytest.raises(ValueError, match="volume_unadj"):
        build_cost_inputs(panel.drop(columns="volume_unadj"), None, D)


def test_sigma_daily_matches_vol_20d_definition() -> None:
    closes = panel.loc[panel["date"] < pd.Timestamp(D), "close"].tail(21)
    assert build_cost_inputs(panel, None, D).get("AAA", D).sigma_daily == pytest.approx(
        closes.pct_change().dropna().std())


def test_summary_fields_attached_for_session_date_only() -> None:
    summ = pd.DataFrame({"symbol": ["AAA", "AAA"], "date": [DATES[-1], DATES[-2]],
                         "half_spread_bps": [40.0, 99.0], "ask_notional": [5e3, 1.0], "bid_notional": [4e3, 1.0]})
    ci = build_cost_inputs(panel, summ, D).get("AAA", D)
    assert (ci.half_spread_bps, ci.ask_notional, ci.bid_notional) == (40.0, 5e3, 4e3)


def test_short_history_symbol_omitted() -> None:
    assert build_cost_inputs(panel.tail(9), None, D).get("AAA", D) is None


def test_holdout_date_refused_without_unseal() -> None:
    with pytest.raises(ValueError):
        build_cost_inputs(panel, None, date(2025, 8, 7))


# ---- fix round 1 (controller review I1, M1): risk.adv semantics ---------------
def _with(rows: dict[int, dict[str, float]]) -> pd.DataFrame:
    p = panel.copy()
    for i, vals in rows.items():
        for col, v in vals.items():
            p.loc[i, col] = v
    return p


def test_zero_volume_prev_session_anchors_on_last_traded_row() -> None:
    p = _with({24: dict(volume=0.0, volume_unadj=0.0)})      # D-1 did not trade
    assert build_cost_inputs(p, None, D).get("AAA", D).adv_shares == pytest.approx(950.0)


def test_all_zero_volume_window_raises() -> None:
    p = _with({i: dict(volume=0.0, volume_unadj=0.0) for i in range(5, 25)})
    with pytest.raises(ValueError, match="AAA"):
        build_cost_inputs(p, None, D)


def test_non_finite_split_factor_on_traded_row_raises() -> None:
    p = _with({22: dict(volume=0.0)})                         # raw > 0, adjusted 0: s = 0
    with pytest.raises(ValueError, match="AAA"):
        build_cost_inputs(p, None, D)


def test_fewer_than_window_prior_sessions_gets_no_entry() -> None:
    assert build_cost_inputs(panel.tail(20), None, D).get("AAA", D) is None   # 19 prior sessions
