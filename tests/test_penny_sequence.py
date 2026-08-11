"""Tests for the pre-open sequence encoders.

Two properties matter most: the sequence must never contain a bar at or after
the decision, and the bar nearest the decision must sit at a fixed position so
the model can rely on it.
"""
import numpy as np
import pandas as pd
import pytest

from research.penny.model.sequence import (
    CHANNELS,
    SequenceConfig,
    blend_scores,
    build_sequences,
    train_sequence_model,
)


def _panel(rows, symbol="A", date="2026-04-07"):
    df = pd.DataFrame(rows, columns=["minute_et", "open", "high", "low", "close", "volume"])
    df["symbol"] = symbol
    df["date"] = pd.Timestamp(date)
    return df


def _bars(start="09:00", n=10, price=2.0, vol=1000):
    h, m = map(int, start.split(":"))
    base = h * 60 + m
    return [
        (f"{(base + i) // 60:02d}:{(base + i) % 60:02d}", price, price * 1.01,
         price * 0.99, price, vol)
        for i in range(n)
    ]


def _keys(symbol="A", date="2026-04-07"):
    return pd.DataFrame({"symbol": [symbol], "date": [pd.Timestamp(date)]})


# ---- Causality ----


def test_a_bar_at_the_decision_minute_is_rejected():
    panel = _panel(_bars(n=5) + [("09:35", 2, 9, 2, 9, 100)])
    with pytest.raises(AssertionError, match="at or after"):
        build_sequences(panel, _keys())


def test_clean_premarket_panel_is_accepted():
    x, keys = build_sequences(_panel(_bars("04:00", 50)), _keys())
    assert x.shape == (1, len(CHANNELS), SequenceConfig().seq_len)


# ---- Shape and alignment ----


def test_sequences_are_left_padded_so_the_last_bar_is_nearest_the_decision():
    """A right-padded sequence would move the most informative bar around."""
    cfg = SequenceConfig(seq_len=20)
    x, _ = build_sequences(_panel(_bars("09:20", 5)), _keys(), cfg)
    # Only five real bars: the first 15 timesteps are padding.
    assert np.allclose(x[0, :, :15], 0.0)
    assert not np.allclose(x[0, :, -1], 0.0)


def test_longer_sequences_are_truncated_from_the_left():
    """When bars exceed the window, the OLDEST are dropped, not the newest."""
    cfg = SequenceConfig(seq_len=10)
    rows = _bars("04:00", 30, price=1.0) + _bars("09:20", 10, price=5.0)
    x, _ = build_sequences(_panel(rows), _keys(), cfg)
    # The retained window should reflect the late, high-priced bars via the
    # minute_frac channel being near its maximum.
    assert x[0, CHANNELS.index("minute_frac"), -1] > 0.9


def test_missing_symbol_day_yields_zeros_not_a_crash():
    x, keys = build_sequences(_panel(_bars()), _keys(symbol="MISSING"))
    assert x.shape[0] == 1
    assert np.allclose(x[0], 0.0)
    assert len(keys) == 1


def test_empty_panel_returns_empty_tensor():
    x, keys = build_sequences(pd.DataFrame(), _keys())
    assert x.shape[0] == 0
    assert keys.empty


# ---- Channels must be scale-free ----


def test_channels_do_not_encode_absolute_price():
    """A model trained across $0.50 and $9 names must not key on price level."""
    cheap, _ = build_sequences(_panel(_bars(price=0.50)), _keys())
    dear, _ = build_sequences(_panel(_bars(price=9.00)), _keys())
    price_free = [c for c in CHANNELS if c != "logvol"]
    for name in price_free:
        i = CHANNELS.index(name)
        assert np.allclose(cheap[0, i], dear[0, i], atol=1e-5), (
            f"channel {name} changed with price level"
        )


def test_minute_frac_increases_through_the_session():
    x, _ = build_sequences(_panel(_bars("04:00", 90)), _keys())
    row = x[0, CHANNELS.index("minute_frac")]
    nonzero = row[row != 0]
    assert np.all(np.diff(nonzero) >= -1e-6), "time channel must be monotone"


def test_dv_share_sums_to_one_over_real_bars():
    x, _ = build_sequences(_panel(_bars(n=20)), _keys())
    assert x[0, CHANNELS.index("dv_share")].sum() == pytest.approx(1.0, abs=1e-4)


def test_no_nans_or_infinities_reach_the_tensor():
    rows = [("09:00", 0.0, 0.0, 0.0, 0.0, 0), ("09:01", 2.0, 2.0, 2.0, 2.0, 100)]
    x, _ = build_sequences(_panel(rows), _keys())
    assert np.isfinite(x).all()


# ---- Training ----


def _dataset(n=600, seq_len=32, signal=True, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(CHANNELS), seq_len)).astype(np.float32)
    if signal:
        # A learnable shape: late-window mean of channel 0 drives the label.
        y = (x[:, 0, -8:].mean(axis=1) > 0).astype(int)
    else:
        y = rng.integers(0, 2, size=n)
    return x, y


@pytest.mark.parametrize("kind", ["cnn", "gru", "transformer"])
def test_each_architecture_returns_valid_probabilities(kind):
    x, y = _dataset(n=400, seed=1)
    cfg = SequenceConfig(seq_len=32, epochs=3, patience=2)
    p = train_sequence_model(x[:300], y[:300], x[300:], kind=kind, config=cfg)
    assert p.shape == (100,)
    assert np.all((p >= 0) & (p <= 1))
    assert np.isfinite(p).all()


def test_a_learnable_shape_is_recovered():
    x, y = _dataset(n=1200, signal=True, seed=2)
    cfg = SequenceConfig(seq_len=32, epochs=25, patience=5)
    p = train_sequence_model(x[:900], y[:900], x[900:], kind="cnn", config=cfg)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(p, y[900:])
    assert rho > 0.25, f"planted sequence shape not recovered (rho={rho:.3f})"


def test_a_tiny_fold_returns_the_base_rate_rather_than_crashing():
    x, y = _dataset(n=100, seed=3)
    p = train_sequence_model(x[:50], y[:50], x[50:], config=SequenceConfig(seq_len=32))
    assert np.allclose(p, y[:50].mean())


def test_single_class_training_data_returns_the_base_rate():
    x, _ = _dataset(n=600, seed=4)
    y = np.zeros(600, dtype=int)
    p = train_sequence_model(x[:400], y[:400], x[400:], config=SequenceConfig(seq_len=32))
    assert np.allclose(p, 0.0)


def test_validation_split_is_chronological_not_random():
    """A random split would early-stop on days the model also trained on."""
    import inspect
    from research.penny.model import sequence

    src = inspect.getsource(sequence.train_sequence_model)
    assert "x_train[:cut]" in src and "x_train[cut:]" in src, (
        "validation split must be a contiguous tail, not a shuffle"
    )


# ---- Blending ----


def test_blend_uses_ranks_so_neither_model_dominates_by_confidence():
    gbdt = np.array([0.50, 0.51, 0.52])          # confident-looking, narrow
    seq = np.array([0.01, 0.50, 0.99])           # wide spread
    out = blend_scores(gbdt, seq, weight=0.5)
    assert np.all(np.diff(out) > 0), "ordering must be preserved"
    assert out.max() <= 1.0 and out.min() >= 0.0


def test_blend_weight_zero_is_the_gbdt_ranking():
    gbdt = np.array([0.9, 0.1, 0.5])
    seq = np.array([0.1, 0.9, 0.5])
    out = blend_scores(gbdt, seq, weight=0.0)
    assert list(np.argsort(out)) == list(np.argsort(gbdt))


def test_blend_weight_one_is_the_sequence_ranking():
    gbdt = np.array([0.9, 0.1, 0.5])
    seq = np.array([0.1, 0.9, 0.5])
    out = blend_scores(gbdt, seq, weight=1.0)
    assert list(np.argsort(out)) == list(np.argsort(seq))
