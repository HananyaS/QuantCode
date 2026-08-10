"""Regression tests for findings from the adversarial leakage audit.

Every test here corresponds to a bug that was live in the pipeline. They exist
because each one produced a *better* number, which is the failure mode no
amount of eyeballing catches.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.bars import add_cross_day_features
from research.penny.eval import conditional as cx
from research.penny.eval.folds import Fold, assert_folds_are_clean
from research.penny.features import veto as veto_mod
from research.penny.labels.triple_barrier import BarrierSpec, label_one


def _panel(symbols=("AAA", "BBB"), n=60, seed=0):
    """Two symbols with distinct price levels, so bleed between them is visible."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n)
    rows = []
    for i, sym in enumerate(symbols):
        base = 10.0 * (i + 1)
        px = base * np.cumprod(1 + rng.normal(0.001, 0.02, n))
        for d, p in zip(dates, px):
            rows.append(
                {
                    "symbol": sym, "date": d,
                    "open": p * 0.99, "high": p * 1.02, "low": p * 0.98,
                    "close": p, "volume": 1e6, "close_unadj": p,
                }
            )
    return pd.DataFrame(rows)


# ---- M1: shift applied outside the groupby ----


def test_ret_1d_prev_does_not_bleed_across_symbols():
    """A trailing .shift(1) on the concatenated result crosses the symbol
    boundary, handing each symbol's first row the previous symbol's LAST value
    -- data from years in the future."""
    out = add_cross_day_features(_panel())
    first_rows = out.sort_values(["symbol", "date"]).groupby("symbol").head(1)
    assert first_rows["ret_1d_prev"].isna().all(), (
        "the first row of each symbol must have no prior return"
    )


def test_ret_1d_prev_matches_the_symbols_own_history():
    out = add_cross_day_features(_panel()).sort_values(["symbol", "date"])
    for sym, g in out.groupby("symbol"):
        expected = g["close"].pct_change().shift(1)
        pd.testing.assert_series_equal(
            g["ret_1d_prev"], expected, check_names=False, check_index=False
        )


# ---- M2: consec_up counted NaN rows ----


def test_consec_up_counts_only_genuine_up_days():
    """`cumcount` counts rows, not True values, and a NaN yesterday neither
    qualifies nor resets -- inflating every streak after a data gap."""
    dates = pd.bdate_range("2024-01-02", periods=6)
    closes = [10.0, 11.0, 12.0, 11.0, 12.0, 13.0]   # up up down up up
    panel = pd.DataFrame(
        {
            "symbol": "AAA", "date": dates,
            "open": closes, "high": closes, "low": closes,
            "close": closes, "volume": 1e6, "close_unadj": closes,
        }
    )
    out = add_cross_day_features(panel)
    # daily_ret = [NaN, +, +, -, +, +]; shifted by one, streak at each row is:
    #   row0 unknown -> 0, row1 unknown -> 0, row2 one up -> 1, row3 two up -> 2,
    #   row4 yesterday down -> 0, row5 one up -> 1
    assert out["consec_up"].tolist() == [0, 0, 1, 2, 0, 1]


def test_consec_up_resets_on_an_unknown_day():
    out = add_cross_day_features(_panel(symbols=("AAA",)))
    assert out["consec_up"].iloc[0] == 0
    assert (out["consec_up"] >= 0).all()


# ---- C1: the adjusted/unadjusted ratio leaked FUTURE splits ----


def test_split_factor_uses_only_trailing_corporate_actions():
    """With adjustment="all" the adjusted series is anchored to the FETCH date,
    so close/close_unadj on day t encodes actions occurring AFTER t. A 1-for-20
    reverse split at date T makes that ratio ~20 on every day BEFORE T, which
    is a pure look-ahead announcement of future collapse."""
    dates = pd.bdate_range("2024-01-02", periods=10)
    raw = np.full(10, 1.0)
    # A 10:1 reverse split takes effect at index 5: bars before it are scaled
    # by 10 in the adjusted series, bars from index 5 onward are not.
    adj = np.array([10.0] * 5 + [1.0] * 5)
    panel = pd.DataFrame(
        {
            "symbol": "AAA", "date": dates,
            "open": adj, "high": adj, "low": adj, "close": adj,
            "volume": 1e6, "close_unadj": raw,
        }
    )
    out = add_cross_day_features(panel)

    # Before the split, nothing has happened yet -- the factor must be flat.
    assert out["split_factor_180d"].iloc[:5].max() == pytest.approx(1.0), (
        "pre-split rows must not know about the upcoming split"
    )
    # From the day after the event onward, the trailing product records it.
    assert out["split_factor_180d"].iloc[6:].max() > 5.0


def test_split_veto_does_not_fire_before_the_event():
    dates = pd.bdate_range("2024-01-02", periods=10)
    adj = np.array([10.0] * 5 + [1.0] * 5)
    panel = pd.DataFrame(
        {
            "symbol": "AAA", "date": dates,
            "open": adj, "high": adj, "low": adj, "close": adj,
            "volume": 1e6, "close_unadj": np.full(10, 1.0),
        }
    )
    out = add_cross_day_features(panel)
    flags = veto_mod._veto_recent_split_activity(out, {"max_adj_ratio": 2.0})
    assert not flags.iloc[:5].any(), "veto fired before the split occurred"


def test_old_leaky_veto_name_is_gone():
    """The leaky implementation must not survive under its old name."""
    assert not hasattr(veto_mod, "_veto_extreme_adjustment")
    assert "veto_extreme_adjustment" not in veto_mod.veto_names(include_diagnostics=True)


# ---- M3: entry spread estimated from the whole day ----


def test_entry_spread_ignores_the_afternoon():
    """A live trader entering at 09:35 cannot pay a spread inferred from the
    afternoon; charging one couples the label to future volatility."""
    calm_open = [
        ("09:30", 1.0, 1.001, 0.999, 1.0, 100),
        ("09:31", 1.0, 1.001, 0.999, 1.0, 100),
        ("09:34", 1.0, 1.001, 0.999, 1.0, 100),
        ("09:35", 1.0, 1.001, 0.999, 1.0, 100),
    ]
    quiet_rest = [("09:36", 1.0, 1.001, 0.999, 1.0, 100), ("09:37", 1.0, 1.001, 0.999, 1.0, 100)]
    wild_rest = [("09:36", 1.0, 1.50, 0.60, 1.0, 100), ("09:37", 1.0, 1.50, 0.60, 1.0, 100)]

    def frame(rows):
        df = pd.DataFrame(rows, columns=["minute_et", "open", "high", "low", "close", "volume"])
        df["symbol"] = "T"
        df["trade_count"] = 5
        return df

    spec = BarrierSpec(half_spread_floor_bps=1.0, impact_coefficient=0.0)
    quiet = label_one("T", pd.Timestamp("2026-01-05"), frame(calm_open + quiet_rest), spec)
    wild = label_one("T", pd.Timestamp("2026-01-05"), frame(calm_open + wild_rest), spec)

    assert quiet.cost_bps == pytest.approx(wild.cost_bps), (
        "entry cost must not depend on what happened after the entry"
    )


# ---- M5: the embargo assertion was a tautology ----


def test_embargo_assertion_can_actually_fail():
    """`gap >= min(embargo, gap)` holds for every value -- a dead tripwire in
    the function whose stated job is to fail loudly."""
    dates = pd.bdate_range("2024-01-02", periods=100)
    too_small = [Fold(0, dates[:50], dates[52:80], dates[50:52])]  # 2-day embargo
    with pytest.raises(AssertionError, match="embargo is 2 days, expected 10"):
        assert_folds_are_clean(too_small, embargo_days=10)


def test_embargo_assertion_passes_when_wide_enough():
    dates = pd.bdate_range("2024-01-02", periods=100)
    ok = [Fold(0, dates[:50], dates[60:80], dates[50:60])]
    assert_folds_are_clean(ok, embargo_days=10)


# ---- Structural guard against look-ahead diagnostics ----


def test_search_refuses_lookahead_diagnostic_vetoes():
    """Convention is not a safeguard: the diagnostic columns are written into
    trades.parquet and could be passed in by accident."""
    trades = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=50).repeat(4),
            "symbol": ["A", "B", "C", "D"] * 50,
            "net_return": 0.01,
            "outcome": "target",
            "veto_delisted_soon": False,
        }
    )
    with pytest.raises(ValueError, match="look-ahead diagnostic"):
        cx.search_veto_sets(trades, ["veto_delisted_soon"], max_size=1)


def test_search_accepts_ordinary_vetoes():
    trades = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=60).repeat(6),
            "symbol": list("ABCDEF") * 60,
            "net_return": np.random.default_rng(0).normal(0, 0.05, 360),
            "outcome": "stop",
            "veto_high_max": np.random.default_rng(1).random(360) < 0.3,
        }
    )
    table = cx.search_veto_sets(trades, ["veto_high_max"], max_size=1,
                                min_trades=50, min_retained=0.05, n_boot=100)
    assert not table.empty


def test_diagnostic_registry_is_not_part_of_the_tradable_set():
    tradable = set(veto_mod.veto_names(include_diagnostics=False))
    diagnostics = {d.name for d in veto_mod.LOOKAHEAD_DIAGNOSTICS}
    assert not (tradable & diagnostics)
