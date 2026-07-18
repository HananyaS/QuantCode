"""Tests for utils/live_decision.py — converting a single-row timing-agent
decision into the full ticker->weight mapping ExecutionAgent needs.
"""
import pytest

from utils.live_decision import position_row_to_weights


def test_selected_ticker_gets_its_fraction():
    weights = position_row_to_weights("TQQQ", 0.83, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert weights["TQQQ"] == pytest.approx(0.83)


def test_untraded_tickers_are_explicitly_zeroed():
    # Critical: ExecutionAgent only sells a position if its ticker appears
    # in the weights row with a lower/zero target -- an untraded ticker
    # missing from the dict entirely would never get sold if a prior day's
    # position needs to be unwound.
    weights = position_row_to_weights("TQQQ", 1.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert weights["QQQ"] == 0.0
    assert weights["QLD"] == 0.0


def test_flat_position_zeros_every_tracked_ticker():
    weights = position_row_to_weights("QQQ", 0.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))
    assert all(v == 0.0 for v in weights.values())


def test_rejects_ticker_outside_tracked_set():
    with pytest.raises(AssertionError):
        position_row_to_weights("SPY", 1.0, tracked_tickers=("QQQ", "QLD", "TQQQ"))
