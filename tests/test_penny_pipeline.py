"""Integration test for dataset assembly.

The unit tests cover labelling and feature construction in isolation. This
covers the seam between them -- the join, the ordering, and the guarantee that
observation features and the label are built from disjoint slices of the same
minute path.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.labels.triple_barrier import BarrierSpec, label_day, observation_features


def _session(symbols, day="2024-03-05", winner=None):
    """Synthetic minute panel: a full session for each symbol."""
    rows = []
    for sym in symbols:
        price = 2.0
        # Observation window 09:30-09:34.
        for m in range(30, 35):
            rows.append((sym, f"09:{m:02d}", price, price * 1.01, price * 0.99, price, 5_000, 20))
        # Entry bar.
        rows.append((sym, "09:35", price, price * 1.01, price * 0.99, price, 5_000, 20))
        # Scan window: the designated winner runs, everyone else drifts.
        for i in range(1, 60):
            minute = 9 * 60 + 35 + i
            hhmm = f"{minute // 60:02d}:{minute % 60:02d}"
            if sym == winner and i > 5:
                p = price * 1.45
                rows.append((sym, hhmm, p, p * 1.02, p * 0.99, p, 9_000, 40))
            else:
                rows.append((sym, hhmm, price, price * 1.005, price * 0.995, price, 3_000, 15))
    df = pd.DataFrame(
        rows,
        columns=["symbol", "minute_et", "open", "high", "low", "close", "volume", "trade_count"],
    )
    df["date"] = pd.Timestamp(day)
    return df


SPEC = BarrierSpec(half_spread_floor_bps=0.0, impact_coefficient=0.0, stop_slippage_bps=0.0)
DAY = pd.Timestamp("2024-03-05")


def test_observation_and_label_use_disjoint_slices():
    """The leakage boundary, checked at the seam rather than in isolation."""
    panel = _session(["AAA", "BBB"], winner="AAA")

    obs = observation_features(panel)
    labels = label_day(panel, DAY, SPEC)

    # Observation window must not see the run-up that starts at 09:41.
    assert obs["high_obs"].max() < 2.05, "post-decision price leaked into features"
    assert (obs["minutes_obs"] == 5).all()

    # The label must see it.
    assert labels.set_index("symbol").loc["AAA", "outcome"] == "target"
    assert labels.set_index("symbol").loc["BBB", "outcome"] == "vertical"


def test_merge_is_one_to_one_per_symbol_day():
    panel = _session(["AAA", "BBB", "CCC"], winner="BBB")
    obs = observation_features(panel)
    labels = label_day(panel, DAY, SPEC)

    merged = labels.merge(obs, on="symbol", how="left", validate="one_to_one")
    assert len(merged) == 3
    assert merged["symbol"].nunique() == 3
    assert merged["close_obs"].notna().all()


def test_every_candidate_symbol_is_labelled():
    symbols = [f"S{i}" for i in range(8)]
    panel = _session(symbols, winner="S3")
    labels = label_day(panel, DAY, SPEC)
    assert set(labels["symbol"]) == set(symbols)


def test_net_return_is_finite_for_every_filled_trade():
    panel = _session(["AAA", "BBB"], winner="AAA")
    labels = label_day(panel, DAY, SPEC)
    filled = labels[labels["outcome"] != "no_fill"]
    assert filled["net_return"].notna().all()
    assert np.isfinite(filled["net_return"]).all()


def test_a_symbol_without_a_decision_bar_is_marked_unfillable():
    panel = _session(["AAA"], winner=None)
    panel = panel[panel["minute_et"] != "09:35"]
    labels = label_day(panel, DAY, SPEC)
    assert labels.iloc[0]["outcome"] == "no_fill"


def test_costs_reduce_net_below_gross_across_the_panel():
    panel = _session(["AAA", "BBB", "CCC"], winner="AAA")
    costly = BarrierSpec(half_spread_floor_bps=30.0, impact_coefficient=0.10)
    labels = label_day(panel, DAY, costly)
    assert (labels["net_return"] <= labels["gross_return"] + 1e-12).all()
    assert (labels["cost_bps"] > 0).all()


def test_outcome_mix_is_exhaustive():
    """Every labelled trade resolves to exactly one of the known outcomes."""
    panel = _session([f"S{i}" for i in range(6)], winner="S1")
    labels = label_day(panel, DAY, SPEC)
    valid = {"target", "stop", "vertical", "no_fill", "insufficient_data"}
    assert set(labels["outcome"]).issubset(valid)
