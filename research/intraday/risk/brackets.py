"""Exit-grid brackets (R16): one function per exit mode, chosen by a dispatch table."""
from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from research.intraday.contracts.events import Bar
from research.intraday.risk.config import EXIT_MODES, RiskConfig, SkipReason


@dataclass(frozen=True, slots=True)
class Bracket:
    take_profit: float | None
    stop_loss: float | None


BracketFn = Callable[[float, float | None, RiskConfig], "Bracket | SkipReason"]


def true_range(bar: Bar, prev_close: float | None) -> float:
    """High - low on a symbol's first bar of the session; else the gap-inclusive range."""
    if prev_close is None:
        return bar.high - bar.low
    return max(bar.high, prev_close) - min(bar.low, prev_close)


def mean_true_range(trs: Sequence[float], period: int) -> float | None:
    """Mean of the last `period` true ranges, or None during warm-up."""
    assert period >= 1
    if len(trs) < period:
        return None
    last = list(trs)[-period:]
    return sum(last) / period


def _time_stop(ref: float, atr: float | None, cfg: RiskConfig) -> Bracket | SkipReason:
    return Bracket(None, None)


def _fixed_pct(ref: float, atr: float | None, cfg: RiskConfig) -> Bracket | SkipReason:
    assert cfg.tp_pct is not None and cfg.sl_pct is not None
    return Bracket(ref * (1 + cfg.tp_pct), ref * (1 - cfg.sl_pct))


def _atr(ref: float, atr: float | None, cfg: RiskConfig) -> Bracket | SkipReason:
    assert cfg.tp_atr is not None and cfg.sl_atr is not None
    if atr is None:
        return "atr_warmup"
    if cfg.atr_floor is not None:
        atr = max(atr, cfg.atr_floor)   # R16 §5.5: one tick, so a flat window still has a bracket
    return Bracket(ref + cfg.tp_atr * atr, ref - cfg.sl_atr * atr)


BRACKETS: Mapping[str, BracketFn] = MappingProxyType({
    "time_stop": _time_stop,
    "fixed_pct": _fixed_pct,
    "atr": _atr,
})
assert set(BRACKETS) == set(EXIT_MODES)


def _valid(b: Bracket, ref_price: float) -> bool:
    if b.take_profit is None and b.stop_loss is None:
        return True
    if b.take_profit is None or b.stop_loss is None:
        return False
    if not (math.isfinite(b.take_profit) and math.isfinite(b.stop_loss)):
        return False
    return 0 < b.stop_loss < ref_price < b.take_profit


def make_bracket(ref_price: float, atr: float | None, cfg: RiskConfig) -> Bracket | SkipReason:
    """The bracket for `cfg.exit_mode`, or the skip reason (warm-up, degenerate levels)."""
    out = BRACKETS[cfg.exit_mode](ref_price, atr, cfg)
    if isinstance(out, str):
        return out
    return out if _valid(out, ref_price) else "bad_bracket"
