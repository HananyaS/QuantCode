"""Precision KPIs, and the multiple-testing correction that actually corrects.

Round 3 reported ``n_trials = len(cells) + 27`` and then ran Benjamini-Hochberg
on the cells array alone, so the 27 prior trials never harshened anything. The
padding test below pins the fix: adding trials that cannot themselves pass must
still make the threshold stricter, because BH's cutoff depends on the rank of
each p-value among *all* tests performed.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.eval.precision import kpi_table, max_consecutive, precision_kpis
from research.penny.eval.stats import benjamini_hochberg


def _trades(returns, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(returns))
    return pd.DataFrame({"date": dates, "net_return": returns})


# ---- Runs ----


def test_max_consecutive_counts_the_longest_run():
    assert max_consecutive(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == 3


def test_max_consecutive_handles_all_true_and_all_false():
    assert max_consecutive(np.ones(5, dtype=bool)) == 5
    assert max_consecutive(np.zeros(5, dtype=bool)) == 0


def test_max_consecutive_of_empty_is_zero():
    assert max_consecutive(np.array([], dtype=bool)) == 0


# ---- The KPIs a low-recall rule is judged on ----


def test_hit_rate_and_expectancy():
    k = precision_kpis(_trades([0.10, -0.05, 0.20, -0.05]))
    assert k["hit_rate"] == pytest.approx(0.5)
    assert k["expectancy"] == pytest.approx(0.05)


def test_profit_factor_is_gross_win_over_gross_loss():
    k = precision_kpis(_trades([0.30, -0.10, -0.05]))
    assert k["profit_factor"] == pytest.approx(0.30 / 0.15)


def test_profit_factor_is_nan_when_there_are_no_losers():
    """Infinity here is a sample-size artefact, not a perfect strategy."""
    k = precision_kpis(_trades([0.10, 0.20]))
    assert np.isnan(k["profit_factor"])


def test_payoff_ratio_compares_average_win_to_average_loss():
    k = precision_kpis(_trades([0.20, 0.20, -0.05, -0.05]))
    assert k["payoff_ratio"] == pytest.approx(4.0)


def test_max_consecutive_losers_is_reported():
    """A positive mean with a long losing run is not deployable."""
    k = precision_kpis(_trades([0.5, -0.01, -0.01, -0.01, -0.01, 0.5]))
    assert k["max_consecutive_losers"] == 4


def test_zero_return_counts_as_a_loss_not_a_win():
    k = precision_kpis(_trades([0.0, 0.10]))
    assert k["hit_rate"] == pytest.approx(0.5)


def test_worst_day_aggregates_within_a_session():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "net_return": [-0.10, -0.20, 0.05],
    })
    k = precision_kpis(df)
    assert k["worst_day"] == pytest.approx(-0.15), "worst day must be the day mean"


def test_trades_per_month_uses_the_observed_span():
    dates = pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-05", "2024-02-20"])
    k = precision_kpis(pd.DataFrame({"date": dates, "net_return": [0.01] * 4}))
    assert k["trades_per_month"] == pytest.approx(2.0)


def test_unpriced_trades_are_dropped_and_counted_not_treated_as_zero():
    """Silently zeroing them would flatter a rule that abstains on hard names."""
    df = _trades([0.10, np.nan, -0.05])
    k = precision_kpis(df)
    assert k["n"] == 2 and k["n_unpriced"] == 1
    assert k["expectancy"] == pytest.approx(0.025)


def test_empty_selection_reports_n_zero_rather_than_raising():
    assert precision_kpis(pd.DataFrame())["n"] == 0


def test_all_unpriced_reports_zero_with_the_count():
    k = precision_kpis(_trades([np.nan, np.nan]))
    assert k["n"] == 0 and k["n_unpriced"] == 2


def test_kpi_table_stacks_configurations():
    rows = {
        "a": precision_kpis(_trades([0.1, -0.05])),
        "b": precision_kpis(_trades([0.2, -0.01])),
    }
    out = kpi_table(rows)
    assert list(out["configuration"]) == ["a", "b"]
    assert "hit_rate" in out.columns


# ---- BH padding: the correction that round 3 reported but did not apply ----


def _bh_with_prior(p, prior_trials, q=0.05):
    padded = np.concatenate([np.asarray(p, dtype=float), np.ones(prior_trials)])
    return benjamini_hochberg(padded, q=q)[: len(p)]


def test_padding_with_prior_trials_makes_the_correction_stricter():
    p = np.array([0.004, 0.02, 0.30])
    without = benjamini_hochberg(p, q=0.05)
    with_prior = _bh_with_prior(p, prior_trials=72)
    assert without.sum() >= with_prior.sum()
    assert with_prior.sum() < without.sum(), (
        "prior trials did not harshen the threshold -- the round-3 bug"
    )


def test_padded_trials_can_never_pass_themselves():
    padded = np.concatenate([np.array([0.001]), np.ones(20)])
    out = benjamini_hochberg(padded, q=0.05)
    assert not out[1:].any()


def test_a_strong_enough_result_still_survives_the_prior_trials():
    """The correction must be harsher, not impossible."""
    assert _bh_with_prior(np.array([1e-9]), prior_trials=72).all()


def test_prior_trial_count_scales_the_penalty_monotonically():
    p = np.array([0.0005, 0.001])
    counts = [_bh_with_prior(p, n).sum() for n in (0, 20, 200, 5_000)]
    assert counts == sorted(counts, reverse=True), "more trials must never pass more"
