"""Insider and institutional flow from the EDGAR index.

These forms sat unused for four rounds: 568,896 Form 4, 73,880 SC 13G, 22,580
SC 13D. ``etl.edgar`` filters them out deliberately -- correctly, for a *supply*
question -- and nobody asked the demand question.

The failure guarded here is the usual one for filing features and it flatters
rather than crashes: EDGAR's ``filingDate`` is a date, but acceptance is an
instant, and an 18:00 acceptance carries the same filing date as the morning
that preceded it. A feature keyed on the date leaks several hours of hindsight
and looks like an excellent predictor.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.etl.filings_flow import (
    FLOW_FEATURES,
    FlowConfig,
    assert_no_future_flow,
    build_flow_features,
)


def _filings(rows, symbol="PENY"):
    """(form, 'YYYY-MM-DD HH:MM') -> an index frame with tz-aware acceptance."""
    df = pd.DataFrame(rows, columns=["form", "accepted"])
    df["symbol"] = symbol
    df["accepted_et"] = pd.to_datetime(df["accepted"]).dt.tz_localize("America/New_York")
    return df.drop(columns=["accepted"])


def _cands(dates, symbol="PENY"):
    return pd.DataFrame({"symbol": symbol, "date": pd.to_datetime(dates)})


# ---- Point-in-time discipline ----


def test_a_filing_accepted_after_the_decision_is_not_counted():
    """The whole guarantee, in one test."""
    f = _filings([("4", "2024-03-05 18:00")])
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert out["insider_txn_30d"] == 0.0
    assert np.isnan(out["days_since_insider_txn"])


def test_the_same_filing_counts_on_the_following_session():
    f = _filings([("4", "2024-03-05 18:00")])
    out = build_flow_features(_cands(["2024-03-06"]), f).iloc[0]
    assert out["insider_txn_30d"] == 1.0
    assert 0.5 < out["days_since_insider_txn"] < 1.0


def test_a_filing_accepted_minutes_before_the_decision_does_count():
    f = _filings([("4", "2024-03-05 09:30")])
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert out["insider_txn_30d"] == 1.0


def test_the_boundary_instant_itself_is_excluded():
    f = _filings([("4", "2024-03-05 09:35")])
    assert build_flow_features(_cands(["2024-03-05"]), f).iloc[0]["insider_txn_30d"] == 0.0


# ---- Counting windows ----


def test_windows_are_nested_correctly():
    f = _filings([("4", "2024-03-01 10:00"),   # 4 days back
                  ("4", "2024-01-20 10:00"),   # ~45 days back
                  ("4", "2023-06-01 10:00")])  # ~9 months back
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert out["insider_txn_30d"] == 1.0
    assert out["insider_txn_90d"] == 2.0


def test_days_since_uses_the_most_recent_prior_filing():
    f = _filings([("4", "2024-01-05 10:00"), ("4", "2024-03-04 10:00")])
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert 0.9 < out["days_since_insider_txn"] < 1.1


def test_amended_forms_count_as_events():
    """A 13G/A raising a stake is news, not noise."""
    f = _filings([("SC 13G/A", "2024-02-01 10:00")])
    assert build_flow_features(_cands(["2024-03-05"]), f).iloc[0]["passive_365d"] == 1.0


def test_form_families_do_not_bleed_into_each_other():
    f = _filings([("4", "2024-03-01 10:00"), ("SC 13D", "2024-03-01 10:00"),
                  ("SC 13G", "2024-03-01 10:00"), ("3", "2024-03-01 10:00")])
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert out["insider_txn_30d"] == 1.0
    assert out["activist_365d"] == 1.0
    assert out["passive_365d"] == 1.0
    assert out["insider_new_180d"] == 1.0


# ---- Normalisation against the symbol's own habit ----


def test_abnormal_flags_a_burst_against_a_quiet_history():
    """Four filings in a week, from a symbol that files monthly."""
    quiet = [("4", f"2023-{m:02d}-05 10:00") for m in range(4, 13)]
    burst = [("4", f"2024-03-0{d} 10:00") for d in range(1, 5)]
    out = build_flow_features(_cands(["2024-03-05"]), _filings(quiet + burst)).iloc[0]
    assert out["insider_txn_30d"] == 4.0
    assert out["insider_txn_abnormal_30d"] > 2.0, "a 4x burst was not flagged"


def test_abnormal_does_not_flag_a_habitually_busy_filer():
    """A firm with a dozen insiders files a dozen Form 4s per board decision.

    Both symbols below show three filings in the trailing week. Only the one
    with a quiet history should register as abnormal.
    """
    steady = [("4", f"2023-{m:02d}-{d:02d} 10:00")
              for m in range(3, 13) for d in (5, 15, 25)]
    steady += [("4", f"2024-{m:02d}-{d:02d} 10:00")
               for m in (1, 2) for d in (5, 15, 25)]
    recent = [("4", "2024-03-01 10:00"), ("4", "2024-03-02 10:00"),
              ("4", "2024-03-03 10:00")]
    busy = build_flow_features(_cands(["2024-03-05"]),
                               _filings(steady + recent)).iloc[0]
    quiet = build_flow_features(
        _cands(["2024-03-05"]),
        _filings([("4", "2023-06-05 10:00")] + recent),
    ).iloc[0]

    assert busy["insider_txn_30d"] > quiet["insider_txn_30d"], "fixture is wrong"
    assert busy["insider_txn_abnormal_30d"] < quiet["insider_txn_abnormal_30d"], (
        "the habitual filer scored as more abnormal than the quiet one"
    )


def test_abnormal_is_finite_on_a_first_ever_filing():
    out = build_flow_features(
        _cands(["2024-03-05"]), _filings([("4", "2024-03-01 10:00")])
    ).iloc[0]
    assert np.isfinite(out["insider_txn_abnormal_30d"])


def test_abnormal_is_zero_when_activity_matches_the_baseline():
    steady = [("4", f"2023-{m:02d}-{d:02d} 10:00")
              for m in range(3, 13) for d in (10,)]
    steady += [("4", f"2024-{m:02d}-10 10:00") for m in (1, 2)]
    out = build_flow_features(_cands(["2024-03-05"]), _filings(steady)).iloc[0]
    assert abs(out["insider_txn_abnormal_30d"]) < 1.0


def test_filing_burst_measures_overall_abnormal_activity():
    quiet = [("4", f"2023-{m:02d}-05 10:00") for m in range(4, 13)]
    burst = [("4", "2024-03-0%d 10:00" % d) for d in range(1, 6)]
    out = build_flow_features(_cands(["2024-03-06"]), _filings(quiet + burst)).iloc[0]
    assert out["filing_burst_30d"] > 2.0


# ---- Coverage and shape ----


def test_symbols_do_not_bleed_into_each_other():
    a = _filings([("4", "2024-03-01 10:00")], symbol="AAA")
    b = _filings([("SC 13D", "2024-03-01 10:00")], symbol="BBB")
    cands = pd.DataFrame({"symbol": ["AAA", "BBB"],
                          "date": [pd.Timestamp("2024-03-05")] * 2})
    out = build_flow_features(cands, pd.concat([a, b])).set_index("symbol")
    assert out.loc["AAA", "insider_txn_30d"] == 1.0
    assert out.loc["AAA", "activist_365d"] == 0.0
    assert out.loc["BBB", "activist_365d"] == 1.0


def test_uncovered_symbols_are_flagged_rather_than_silently_null():
    f = _filings([("4", "2024-03-01 10:00")], symbol="PENY")
    cands = pd.DataFrame({"symbol": ["PENY", "GONE"],
                          "date": [pd.Timestamp("2024-03-05")] * 2})
    out = build_flow_features(cands, f).set_index("symbol")
    assert bool(out.loc["PENY", "flow_has_coverage"])
    assert not bool(out.loc["GONE", "flow_has_coverage"])


def test_row_count_is_preserved():
    cands = pd.DataFrame({"symbol": ["PENY"] * 3,
                          "date": pd.to_datetime(["2024-03-05"] * 3)})
    assert len(build_flow_features(cands, _filings([("4", "2024-03-01 10:00")]))) == 3


def test_no_filings_at_all_yields_nulls_with_coverage_false():
    out = build_flow_features(_cands(["2024-03-05"]), pd.DataFrame()).iloc[0]
    assert not bool(out["flow_has_coverage"])
    assert all(np.isnan(out[c]) for c in FLOW_FEATURES)


def test_a_covered_symbol_with_no_prior_filings_counts_zero_not_null():
    """Zero is an observation; NaN is an absence of one.

    For a symbol EDGAR covers, "no insider filed in the last 30 days" is a fact
    worth recording. Collapsing it to NaN would throw that away and let the
    model treat a quiet company like an unknown one.
    """
    f = _filings([("4", "2024-03-05 18:00")])  # accepted after the decision
    out = build_flow_features(_cands(["2024-03-05"]), f).iloc[0]
    assert bool(out["flow_has_coverage"])
    assert out["insider_txn_30d"] == 0.0
    assert np.isnan(out["days_since_insider_txn"]), "recency must stay unknown"


def test_an_uncovered_symbol_gets_null_counts_not_zero():
    f = _filings([("4", "2024-03-01 10:00")], symbol="PENY")
    cands = pd.DataFrame({"symbol": ["GONE"], "date": [pd.Timestamp("2024-03-05")]})
    out = build_flow_features(cands, f).iloc[0]
    assert not bool(out["flow_has_coverage"])
    assert np.isnan(out["insider_txn_30d"])


def test_every_declared_feature_is_produced():
    out = build_flow_features(_cands(["2024-03-05"]), _filings([("4", "2024-03-01 10:00")]))
    assert set(FLOW_FEATURES) <= set(out.columns)


def test_filings_with_no_acceptance_timestamp_are_dropped_not_guessed():
    f = _filings([("4", "2024-03-01 10:00"), ("4", "2024-03-02 10:00")])
    f.loc[1, "accepted_et"] = pd.NaT
    assert build_flow_features(_cands(["2024-03-05"]), f).iloc[0]["insider_txn_30d"] == 1.0


# ---- The tripwire ----


def test_tripwire_passes_on_correctly_built_features():
    f = _filings([("4", "2024-03-01 10:00")])
    out = build_flow_features(_cands(["2024-03-05"]), f)
    assert_no_future_flow(out, f, "PENY", pd.Timestamp("2024-03-05"))


def test_tripwire_catches_a_feature_pointing_at_a_future_filing():
    f = _filings([("4", "2024-03-01 10:00")])
    out = build_flow_features(_cands(["2024-03-05"]), f)
    out.loc[0, "days_since_insider_txn"] = -1.0  # after the decision
    with pytest.raises(AssertionError, match="points at"):
        assert_no_future_flow(out, f, "PENY", pd.Timestamp("2024-03-05"))


def test_tripwire_catches_a_feature_invented_from_nothing():
    f = _filings([("SC 13G", "2024-03-01 10:00")])  # no Form 4 at all
    out = build_flow_features(_cands(["2024-03-05"]), f)
    out.loc[0, "days_since_insider_txn"] = 3.0
    with pytest.raises(AssertionError, match="none"):
        assert_no_future_flow(out, f, "PENY", pd.Timestamp("2024-03-05"))


def test_config_windows_are_honoured():
    f = _filings([("4", "2024-02-20 10:00")])
    tight = build_flow_features(_cands(["2024-03-05"]), f,
                                FlowConfig(short_days=7)).iloc[0]
    wide = build_flow_features(_cands(["2024-03-05"]), f,
                               FlowConfig(short_days=60)).iloc[0]
    assert tight["insider_txn_30d"] == 0.0
    assert wide["insider_txn_30d"] == 1.0
