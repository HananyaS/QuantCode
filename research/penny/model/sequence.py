"""sequence.py — encoders over the pre-open bar sequence.

When a sequence model is and is not justified
---------------------------------------------
On the ~40 tabular features, it is not. Grinsztajn, Oyallon & Varoquaux (2022)
find tree ensembles still beat deep learning on tabular data at ~10k samples,
and this study has 17k; the inductive biases that make trees win (piecewise-
constant fits, robustness to uninformative features) are exactly what this
feature set needs.

On the pre-open **path**, it plausibly is. The GBDT sees ~16 hand-made
aggregates of ~96 bars. Whether a gap was accumulated steadily overnight or
spiked once on a headline and drifted is a shape, and a shape is precisely what
an aggregate destroys. That is information the tree genuinely cannot reach.

So the sequence model earns its place on the sequence or not at all, and it is
**ensembled with** the GBDT rather than replacing it.

Architecture choices, and why they are small
--------------------------------------------
17k samples is very little for a deep model. Everything here is deliberately
tiny and heavily regularised: a 2-layer 1D-CNN, a 1-layer GRU, and a 2-layer
transformer with 2 heads. Anything larger would memorise the training fold.
Fitting is early-stopped on a temporal split *inside* the training fold, never
on the test fold.

Causality
---------
The sequence is built from bars strictly before 09:35 and is padded/truncated
at the left, so the final timestep is always the bar nearest the decision.
:func:`build_sequences` asserts the boundary itself rather than trusting the
caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DECISION_MINUTE = "09:35"

# Per-bar channels. All are price-scale-free -- a model trained across $0.50
# and $9 stocks must not key on price level. `logvol` is share volume and is
# NOT size-scale-free; it is kept because absolute participation carries
# real information, and the dollar-scaled view is supplied by `dv_share`.
CHANNELS = ["ret", "logvol", "range", "close_pos", "dv_share", "minute_frac"]

SEQ_LEN = 96  # median pre-open bars per symbol-day


@dataclass
class SequenceConfig:
    seq_len: int = SEQ_LEN
    hidden: int = 32
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.3
    patience: int = 6
    val_fraction: float = 0.2
    seed: int = 42


def build_sequences(
    panel: pd.DataFrame,
    keys: pd.DataFrame,
    config: Optional[SequenceConfig] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Turn pre-open bars into a fixed-length (N, C, T) tensor.

    Left-padded and left-truncated, so index -1 is always the bar closest to
    the decision. That matters: a right-padded sequence would place the most
    informative bar at a position that varies with how much the name traded.
    """
    config = config or SequenceConfig()

    if panel.empty:
        return np.zeros((0, len(CHANNELS), config.seq_len), dtype=np.float32), keys.iloc[0:0]

    late = panel[panel["minute_et"] >= DECISION_MINUTE]
    assert late.empty, (
        f"{len(late)} bars at or after {DECISION_MINUTE} in the sequence input; "
        "the encoder would see the entry bar"
    )

    df = panel.sort_values(["symbol", "date", "minute_et"]).copy()
    parts = df["minute_et"].str.split(":", expand=True).astype(int)
    df["minute_idx"] = parts[0] * 60 + parts[1]
    df["dollar_volume"] = df["close"] * df["volume"]

    lookup: Dict[Tuple[str, pd.Timestamp], np.ndarray] = {}
    for (symbol, date), g in df.groupby(["symbol", "date"], sort=False):
        close = g["close"].to_numpy(dtype=float)
        high = g["high"].to_numpy(dtype=float)
        low = g["low"].to_numpy(dtype=float)
        vol = g["volume"].to_numpy(dtype=float)
        dv = g["dollar_volume"].to_numpy(dtype=float)
        idx = g["minute_idx"].to_numpy(dtype=float)

        with np.errstate(invalid="ignore", divide="ignore"):
            ret = np.diff(close, prepend=close[0]) / np.where(close != 0, close, np.nan)
            rng = (high - low) / np.where(close != 0, close, np.nan)
            span = np.where(high - low > 0, high - low, np.nan)
            close_pos = (close - low) / span
            total_dv = np.nansum(dv)
            dv_share = dv / total_dv if total_dv > 0 else np.zeros_like(dv)

        channels = np.vstack([
            np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0),
            np.log1p(np.nan_to_num(vol, nan=0.0)),
            np.nan_to_num(rng, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(close_pos, nan=0.5),
            np.nan_to_num(dv_share, nan=0.0),
            # Where in the pre-open session the bar sits: 04:00 -> 0, 09:34 -> 1.
            (idx - 4 * 60) / (5 * 60 + 34),
        ]).astype(np.float32)

        lookup[(symbol, pd.Timestamp(date))] = channels

    tensors: List[np.ndarray] = []
    kept: List[int] = []
    for pos, (symbol, date) in enumerate(zip(keys["symbol"], keys["date"])):
        ch = lookup.get((symbol, pd.Timestamp(date)))
        out = np.zeros((len(CHANNELS), config.seq_len), dtype=np.float32)
        if ch is not None and ch.shape[1] > 0:
            take = min(ch.shape[1], config.seq_len)
            # Left-pad: the last `take` bars land at the END of the window.
            out[:, -take:] = ch[:, -take:]
        tensors.append(out)
        kept.append(pos)

    return np.stack(tensors), keys.iloc[kept].reset_index(drop=True)


def _make_model(kind: str, config: SequenceConfig):
    """Small by design; 17k samples cannot support anything larger."""
    import torch
    import torch.nn as nn

    c, t, h = len(CHANNELS), config.seq_len, config.hidden

    if kind == "cnn":
        return nn.Sequential(
            nn.Conv1d(c, h, kernel_size=5, padding=2), nn.ReLU(),
            nn.BatchNorm1d(h), nn.Dropout(config.dropout),
            nn.Conv1d(h, h, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(config.dropout), nn.Linear(h, 1),
        )

    if kind == "gru":
        class GRUHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(c, h, batch_first=True)
                self.drop = nn.Dropout(config.dropout)
                self.fc = nn.Linear(h, 1)

            def forward(self, x):
                # (N, C, T) -> (N, T, C); the final state sits at the bar
                # nearest the decision because sequences are left-padded.
                _, hidden = self.gru(x.transpose(1, 2))
                return self.fc(self.drop(hidden[-1]))

        return GRUHead()

    if kind == "transformer":
        class TransformerHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(c, h)
                self.pos = nn.Parameter(torch.zeros(1, t, h))
                layer = nn.TransformerEncoderLayer(
                    d_model=h, nhead=2, dim_feedforward=2 * h,
                    dropout=config.dropout, batch_first=True, norm_first=True,
                )
                self.enc = nn.TransformerEncoder(layer, num_layers=2)
                self.fc = nn.Linear(h, 1)

            def forward(self, x):
                z = self.proj(x.transpose(1, 2)) + self.pos
                z = self.enc(z)
                return self.fc(z.mean(dim=1))

        return TransformerHead()

    raise ValueError(f"unknown architecture {kind!r}")


def train_sequence_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    kind: str = "cnn",
    config: Optional[SequenceConfig] = None,
) -> np.ndarray:
    """Fit on train, return P(clears cost) for test rows.

    Early stopping uses the **last** slice of the training fold as validation,
    chronologically rather than randomly. A random split would let the model
    stop on days it also trained on, which is the same leak the outer
    walk-forward exists to prevent, one level down.
    """
    import torch
    import torch.nn as nn

    config = config or SequenceConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if len(x_train) < 200 or len(np.unique(y_train)) < 2:
        logger.warning("sequence: fold too small or single-class; returning base rate")
        return np.full(len(x_test), float(y_train.mean()) if len(y_train) else 0.0)

    cut = max(1, int(len(x_train) * (1 - config.val_fraction)))
    xt = torch.tensor(x_train[:cut]); yt = torch.tensor(y_train[:cut], dtype=torch.float32)
    xv = torch.tensor(x_train[cut:]); yv = torch.tensor(y_train[cut:], dtype=torch.float32)

    model = _make_model(kind, config)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    # Class imbalance: the positive class is ~20-30% of rows.
    pos_weight = torch.tensor([(len(yt) - yt.sum()) / max(yt.sum(), 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss, best_state, waited = float("inf"), None, 0
    n = len(xt)

    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            opt.zero_grad()
            loss = loss_fn(model(xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(xv).squeeze(-1), yv)) if len(xv) else float(loss)

        if val_loss < best_loss - 1e-5:
            best_loss, waited = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x_test)).squeeze(-1)
        return torch.sigmoid(logits).numpy()


def blend_scores(
    gbdt: np.ndarray,
    sequence: np.ndarray,
    weight: float = 0.5,
    groups: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rank-average the two scores, ranking **within** ``groups``.

    Averaging ranks rather than probabilities: the two models are calibrated
    differently, and a raw average would let whichever is more confident
    dominate for reasons unrelated to skill.

    ``groups`` is not optional in spirit. Ranking across the pooled
    out-of-sample panel makes each trade's blended score a function of the
    score distribution of *other folds*, including later years -- an ordering
    that could not be produced at decision time. A pct-rank is monotone per
    model, but the **sum** of two rank transforms is not invariant to them, so
    the within-fold ordering genuinely shifts. Pass the fold (or the date) and
    the leak disappears.
    """
    a = pd.Series(gbdt)
    b = pd.Series(sequence)
    if groups is None:
        logger.warning(
            "blend_scores called without groups: ranking over the pooled panel "
            "uses information from other folds"
        )
        ra, rb = a.rank(pct=True), b.rank(pct=True)
    else:
        g = pd.Series(np.asarray(groups))
        ra = a.groupby(g).rank(pct=True)
        rb = b.groupby(g).rank(pct=True)
    return ((1 - weight) * ra + weight * rb).to_numpy()
