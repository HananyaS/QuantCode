"""book.py — one slot simulator, with the cost charged exactly once.

Why this replaces three loops
------------------------------
`scripts/slots.py:run_slots`, `scripts/risk_shaped.py:run` and
`scripts/long_reversion_portfolio.py:simulate` are three independent
implementations of the same object. They disagree about position sizing, about
whether entry is staggered, about whether exits are scheduled on an index or a
date -- and about how much the trade costs. A strategy compared across them is
not being compared at all.

The cost defect, stated exactly
--------------------------------
`cross_section.forward_return` returns a value **already net of one full round
trip**::

    r = cl[rows, ...] / entry - 1.0 - u["cost"].to_numpy()[pos]   # cross_section.py:88

while `risk_shaped.attach_target_returns` returns a **gross** value::

    out[f"R_t{int(100*t)}"] = px / entry - 1.0                    # risk_shaped.py:171

Every simulator then subtracts the same half a round trip regardless of which
column it was handed::

    net = float(row[col] - 0.5 * row["cost"])

So the time-stop arm is charged **1.5 round trips** and the profit-target arm
**0.5** -- a threefold difference between two arms of the same sweep, in the
direction that favours the target. Neither is one round trip, which is what a
long-only strategy that buys and later sells actually pays.

This module works in **gross** return space and applies exactly one cost, from
`strategy/cost_model.round_trip_frac`, at the position size actually traded.

Three further corrections
--------------------------
1. **Exits are scheduled on the market calendar, not on candidate days.** The
   older loops index `sess = np.sort(df["date"].unique())`, which contains only
   days with at least `MIN_CELL` candidates. A 60-session hold measured on that
   index is not 60 sessions, and the resulting equity path -- hence Sharpe and
   drawdown -- is measured on a clock that speeds up when candidates are scarce.
   Realised returns were never wrong; their *timing* was.
2. **The no-selection control actually selects nothing.** `slots.py`'s "noveto"
   arm skips the veto trim but still fills with `nlargest(..., "COMPOSITE")`, so
   it changes two things at once and is not a control. Here `fill` and
   `veto_frac` are independent knobs.
3. **The daily equity curve is returned.** All three older loops build one and
   throw it away, which is why no distributional metric has ever been computed
   on this strategy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from research.penny.strategy import cost_model
from research.penny.strategy.cross_section import MIN_CELL

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class BookConfig:
    """A complete, executable portfolio definition."""

    n_slots: int
    hold: int = 60
    # --- filters, both ranked WITHIN the day ------------------------------
    vol_cap: float = 0.50          # keep names <= this quantile of realised vol; 1.0 = off
    veto_frac: float = 0.25        # drop this fraction by composite; 0.0 = off
    # --- exit -------------------------------------------------------------
    target: float = 1.00           # profit target as a fraction; 0 = time stop only
    # --- construction -----------------------------------------------------
    weight: str = "equal"          # "equal" | "invvol"
    fill: str = "random"           # "random" | "top"   (independent of veto_frac)
    fill_col: str = "COMPOSITE"    # the column `fill="top"` ranks on
    stagger: bool = True           # cap entries at ceil(n_slots / hold) per session
    sector_cap: int = 0            # max names held per sector at once; 0 = off
    # --- money ------------------------------------------------------------
    account_usd: float = 50_000.0
    cost_mode: str = "size_aware"  # "size_aware" | "flat" | "none"
    cost_scale: float = 1.0        # 0.5 reproduces the legacy half-charge
    # --- accounting ---------------------------------------------------------
    # "realized": R15's original loop -- a position is carried at cost until
    #   signal + hold and its P&L lands on that day (kept to reproduce R15).
    # "mtm": PREREGISTRATION_R17 §2.5 -- positions marked to market every
    #   session from causal.holding_paths; needs `paths=` and a `_path` column.
    mark: str = "realized"

    def validate(self) -> None:
        assert self.n_slots >= 1, "n_slots must be >= 1"
        assert 1 <= self.hold <= 250, "hold out of range"
        assert 0.0 < self.vol_cap <= 1.0, "vol_cap is a quantile in (0, 1]"
        assert 0.0 <= self.veto_frac < 1.0, "veto_frac must be in [0, 1)"
        assert self.weight in ("equal", "invvol"), self.weight
        assert self.fill in ("random", "top"), self.fill
        assert self.cost_mode in ("size_aware", "flat", "none"), self.cost_mode
        assert self.account_usd > 0, "account_usd must be positive"
        assert self.cost_scale >= 0, "cost_scale must be non-negative"
        assert self.sector_cap >= 0, "sector_cap must be non-negative"
        assert self.mark in ("realized", "mtm"), self.mark

    @property
    def max_entries_per_session(self) -> int:
        if not self.stagger:
            return self.n_slots
        return max(1, int(np.ceil(self.n_slots / self.hold)))


@dataclass
class BookResult:
    """Everything the evaluation battery needs, not just the scalars."""

    daily: pd.Series               # date-indexed daily portfolio return
    equity: pd.Series              # date-indexed equity, starts at 1.0
    trades: pd.DataFrame           # one row per fill
    n_sessions: int
    n_entered: int
    n_skipped_no_slot: int
    orders_per_session: float
    weight_pct: float
    ann: float
    sharpe: float
    max_drawdown: float
    median_vol: float
    mean_cost_bps: float
    usd_per_name: float

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()
                if not isinstance(v, (pd.Series, pd.DataFrame))}


# --------------------------------------------------------------------------
# gross returns
# --------------------------------------------------------------------------
def gross_return(df: pd.DataFrame, target: float) -> np.ndarray:
    """The trade's return BEFORE any cost, whichever exit applies.

    `R` arrives net of one round trip (`cross_section.forward_return`) and the
    `R_t*` target columns arrive gross (`risk_shaped.attach_target_returns`).
    Undoing the first is the only way to put both on the same footing before a
    single honest charge is applied. Getting this backwards is the 1.5x/0.5x
    split described in the module docstring.
    """
    if target:
        col = f"R_t{int(round(100 * target))}"
        if col not in df.columns:
            raise KeyError(
                f"{col} missing -- call risk_shaped.attach_target_returns first")
        return df[col].to_numpy(dtype=float)          # already gross
    return (df["R"] + df["cost"]).to_numpy(dtype=float)   # undo the embedded cost


def _adv_from_features(df: pd.DataFrame) -> np.ndarray:
    """ADV in dollars. `B3_log_adv` is log10(adv20) -- cross_section.py:227."""
    if "adv20" in df.columns:
        return df["adv20"].to_numpy(dtype=float)
    return np.power(10.0, df["B3_log_adv"].to_numpy(dtype=float))


def _cost_frac(cfg: BookConfig, adv: float, position_usd: float,
               flat_cost: float) -> float:
    """One round trip, scaled.

    `cost_scale` exists only so the legacy half-charge can be reproduced
    side by side with an honest one. It is not a modelling knob.
    """
    if cfg.cost_mode == "none":
        return 0.0
    if cfg.cost_mode == "flat":
        return cfg.cost_scale * float(flat_cost)
    return cfg.cost_scale * float(
        cost_model.round_trip_frac(np.array([adv]), np.array([position_usd]))[0])


# --------------------------------------------------------------------------
# the simulation
# --------------------------------------------------------------------------
def run_book(df: pd.DataFrame, calendar: Sequence[pd.Timestamp],
             cfg: BookConfig, seed: int = 0,
             min_entries: int = 40, paths=None) -> Optional[BookResult]:
    """Event-driven, capital-constrained, one cost charge per trade.

    `calendar` is the **market** session list, not the candidate-day list. Days
    with no candidates still advance the clock and still count towards the
    return series, which is what makes the annualisation honest.

    With `cfg.mark == "mtm"`, `paths` is a `causal.Paths` and every row of `df`
    carries `_path`, its row in `paths`. The selection steps are unchanged; what
    changes is the trade itself (the limit may not fill, exits come from the
    causal path) and the accounting (marked every session, cost and fees at
    entry, slot released at the entry session + hold). See
    PREREGISTRATION_R17 §2.5.
    """
    cfg.validate()
    rng = np.random.default_rng(seed)
    mtm = cfg.mark == "mtm"
    if mtm:
        from research.penny.strategy.causal import fees_frac
        if paths is None or "_path" not in df.columns:
            raise ValueError("mark='mtm' needs paths= and a `_path` column")
        if paths.hold != cfg.hold:
            raise ValueError(f"paths built for hold={paths.hold}, book has {cfg.hold}")
        last_off = paths.marks.shape[1] - 1

    def _value(p, i: int) -> float:
        """A slot's current worth. Realized: its capital at cost. MTM: marked."""
        if not mtm or i < p[6]:
            return p[3]
        mk = paths.marks[p[5], min(i - p[6], last_off)]
        if not np.isfinite(mk):
            raise ValueError(f"no mark for {p[1]} at session {i}: data ends before the calendar")
        return p[3] * (mk - p[2])

    sessions = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Series(calendar)).unique()))
    if len(sessions) < 2:
        return None
    day_index = {d: i for i, d in enumerate(sessions)}

    work = df.copy()
    # In MTM mode the trade's outcome is not known at selection, so a missing
    # outcome cannot remove a candidate (that was R15's look-ahead).
    work["_gross"] = np.nan if mtm else gross_return(work, cfg.target)
    work["_adv"] = _adv_from_features(work)
    keep = np.isfinite(work["_adv"]) if mtm else (np.isfinite(work["_gross"]) & np.isfinite(work["_adv"]))
    work = work[keep]
    work = work[work["B7_realised_vol"] > 0]
    if work.empty:
        return None
    by_day: Dict[pd.Timestamp, pd.DataFrame] = dict(tuple(work.groupby("date")))

    if cfg.sector_cap and "sector" not in work.columns:
        raise KeyError("sector_cap set but the frame carries no `sector` column")

    # slot -> (exit_index, symbol, net_return, capital, sector)
    open_slots: List[Optional[Tuple[int, str, float, float, str]]] = [None] * cfg.n_slots
    cash = 1.0
    equity = np.ones(len(sessions))
    fills: List[Dict] = []
    n_skipped_no_slot = 0
    vols: List[float] = []

    for i, d in enumerate(sessions):
        # 1. release finished slots, returning capital plus P&L to cash
        for s in range(cfg.n_slots):
            p = open_slots[s]
            if p is not None and p[0] <= i:
                cash += _value(p, i) if mtm else p[3] * (1.0 + p[2])
                open_slots[s] = None

        free = [s for s in range(cfg.n_slots) if open_slots[s] is None]
        free = free[:cfg.max_entries_per_session]

        g = by_day.get(d)
        if free and g is not None and len(g) >= MIN_CELL:
            pool = g
            # --- risk filter, then selection filter, both within the day ---
            if cfg.vol_cap < 1.0:
                thr = pool["B7_realised_vol"].quantile(cfg.vol_cap)
                pool = pool[pool["B7_realised_vol"] <= thr]
            if cfg.fill == "top" and cfg.fill_col not in pool.columns:
                raise KeyError(f"fill_col {cfg.fill_col!r} not in the frame")
            if cfg.veto_frac > 0.0 and len(pool) >= MIN_CELL:
                keep = max(1, int(np.ceil((1.0 - cfg.veto_frac) * len(pool))))
                pool = pool.nlargest(keep, "COMPOSITE")

            held = {p[1] for p in open_slots if p is not None}
            pool = pool[~pool["symbol"].isin(held)]

            # --- concentration control: sigma_common, not sigma_idio -------
            # Ten slots filled from one theme is one position held ten times.
            # This caps CONCURRENT holdings per sector; it does not rank, so it
            # cannot smuggle in a selection effect.
            by_sec: Dict[str, int] = {}
            if cfg.sector_cap and len(pool):
                for q in open_slots:
                    if q is not None:
                        by_sec[q[4]] = by_sec.get(q[4], 0) + 1
                full = {k for k, v in by_sec.items() if v >= cfg.sector_cap}
                if full:
                    pool = pool[~pool["sector"].isin(full)]

            if len(pool):
                # Keep the pre-sample pool: it is the day's ELIGIBLE set, and
                # the inverse-vol normaliser has to come from it. Normalising
                # over the sampled subset is what made the weighting a no-op --
                # with `ceil(N/H) = 1` entry a session the subset holds one
                # name, so `iv / iv.mean()` is identically 1 and "inverse vol"
                # silently reduces to equal weight. `risk_shaped.py:120` has
                # the same construction under the same stagger, so every
                # inverse-vol arm reported at N <= 60 was an equal-weight arm.
                eligible_iv = 1.0 / pool["B7_realised_vol"].to_numpy()
                iv_ref = float(np.mean(eligible_iv))

                # --- fill rule, INDEPENDENT of whether the veto ran --------
                take = min(len(pool), len(free))
                if cfg.fill == "random":
                    pool = pool.sample(take, random_state=int(rng.integers(1 << 31)))
                else:
                    pool = pool.nlargest(take, cfg.fill_col)

                deployed = sum(_value(p, i) for p in open_slots if p is not None)
                eq_now = cash + deployed
                if cfg.weight == "invvol":
                    iv = 1.0 / pool["B7_realised_vol"].to_numpy()
                    # normalise against the DAY'S ELIGIBLE SET, not the sampled
                    # subset, so the tilt survives a one-name fill. Gross
                    # exposure is unchanged in expectation; only its
                    # distribution across names moves.
                    w = iv / max(iv_ref, 1e-12) / cfg.n_slots
                else:
                    w = np.full(len(pool), 1.0 / cfg.n_slots)

                taken = 0
                for s, (_, row), wi in zip(free, pool.iterrows(), w):
                    # The pool filter above saw only the slots already open. A
                    # session that fills several slots must also count what it
                    # takes as it goes, or a cap of one still admits N names
                    # from one sector on the same day.
                    if cfg.sector_cap:
                        sec = str(row["sector"])
                        if by_sec.get(sec, 0) >= cfg.sector_cap:
                            continue
                    if mtm:
                        pid = int(row["_path"])
                        if not paths.filled[pid]:
                            # the opening-auction limit did not fill: the slot
                            # stays free and this session's entry is spent
                            taken += 1
                            fills.append({"symbol": row["symbol"], "date": d, "filled": False,
                                          "kind": "unfilled", "sector": str(row.get("sector", ""))})
                            continue
                    size = min(eq_now * float(wi), cash)
                    if size <= 1e-12:
                        break
                    position_usd = cfg.account_usd * float(wi)
                    c = _cost_frac(cfg, float(row["_adv"]), position_usd,
                                   float(row["cost"]))
                    if mtm:
                        c += fees_frac(position_usd, float(paths.entry_unadj[pid]))
                        xo = int(paths.exit_off[pid])
                        net = float(paths.exit_value[pid]) - 1.0 - c   # NaN while open
                        exit_i = i + 1 + max(cfg.hold, xo)                 # may pass the calendar end
                    else:
                        net = float(row["_gross"]) - c
                        exit_i = min(i + cfg.hold, len(sessions) - 1)
                    cash -= size
                    sec = str(row["sector"]) if cfg.sector_cap else ""
                    if cfg.sector_cap:
                        by_sec[sec] = by_sec.get(sec, 0) + 1
                    taken += 1
                    if mtm:
                        open_slots[s] = (exit_i, row["symbol"], c, size, sec, pid, i + 1)
                        fills.append({
                            "symbol": row["symbol"], "date": d, "filled": True,
                            "kind": str(paths.kind[pid]), "exit_off": xo,
                            "release_date": sessions[exit_i] if exit_i < len(sessions) else pd.NaT,
                            "entry_unadj": float(paths.entry_unadj[pid]),
                            "gross": float(paths.exit_value[pid]) - 1.0, "cost_frac": c,
                            "net": net, "weight": float(wi), "capital": size,
                            "position_usd": position_usd, "adv20": float(row["_adv"]),
                            "realised_vol": float(row["B7_realised_vol"]),
                            "n_corr": int(paths.n_corr[pid]), "sector": sec or str(row.get("sector", "")),
                        })
                        vols.append(float(row["B7_realised_vol"]))
                        continue
                    open_slots[s] = (exit_i, row["symbol"], net, size, sec)
                    vols.append(float(row["B7_realised_vol"]))
                    fills.append({
                        "symbol": row["symbol"], "date": d,
                        "exit_date": sessions[exit_i],
                        "gross": float(row["_gross"]), "cost_frac": c,
                        "net": net, "weight": float(wi), "capital": size,
                        "position_usd": position_usd, "adv20": float(row["_adv"]),
                        "realised_vol": float(row["B7_realised_vol"]),
                        "sector": (str(row["sector"]) if "sector" in pool.columns
                                   else ""),
                    })
                n_skipped_no_slot += max(0, len(g) - taken)
            else:
                n_skipped_no_slot += len(g)
        elif g is not None:
            n_skipped_no_slot += len(g)

        equity[i] = cash + sum(_value(p, i) for p in open_slots if p is not None)

    if len(fills) < min_entries:
        return None

    eq = pd.Series(equity, index=sessions, name="equity")
    r = eq.pct_change().fillna(0.0).rename("ret")
    trades = pd.DataFrame(fills)
    n_orders = len(trades)
    if mtm:
        trades = trades[trades["filled"]].reset_index(drop=True)
        if trades.empty:
            return None

    years = len(sessions) / TRADING_DAYS
    ann = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    sharpe = (r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan
    dd = float(((eq - eq.cummax()) / eq.cummax()).min())

    return BookResult(
        daily=r, equity=eq, trades=trades,
        n_sessions=len(sessions), n_entered=len(trades),
        n_skipped_no_slot=int(n_skipped_no_slot),
        orders_per_session=n_orders / len(sessions),
        weight_pct=100.0 / cfg.n_slots,
        ann=100.0 * float(ann), sharpe=float(sharpe), max_drawdown=100.0 * dd,
        median_vol=float(np.median(vols)),
        mean_cost_bps=float(1e4 * trades["cost_frac"].mean()),
        usd_per_name=cfg.account_usd / cfg.n_slots,
    )
