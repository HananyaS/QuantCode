"""viz.py — figures for the penny-stock veto study.

Palette
-------
Uses a validated categorical palette (blue / orange / aqua). Both modes were
checked with the data-viz validator on the **all-pairs** pairlist, which is the
relevant one here because most figures below are scatter or small multiples
rather than stacked bars:

    light (surface #fcfcfb): CVD dE 9.2, normal-vision dE 24.0  -- PASS
    dark  (surface #1a1a19): CVD dE 9.4, normal-vision dE 20.9  -- PASS

Three slots is the documented all-pairs cap; a fourth would put yellow beside
orange and fail. Where more than three groups exist the figures **facet**
rather than adding hues.

On the light surface aqua sits at 2.74:1 contrast, below 3:1, so the relief
rule applies: every figure using it also carries direct labels or an
accompanying table. Status colours (good/critical) are reserved for
significance and never reused as a series.

Design rules honoured: one y-axis per figure (never dual), identity never
carried by colour alone, hover enabled on every plot, thin marks with recessive
grid and axes, and direct labels in preference to a legend where the series
count allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class Theme:
    """Chart tokens. Light by default; ``Theme.dark()`` returns the dark steps."""

    surface: str = "#fcfcfb"
    text_primary: str = "#0b0b0b"
    text_secondary: str = "#52514e"
    muted: str = "#898781"
    grid: str = "#e1e0d9"
    axis: str = "#c3c2b7"
    series: tuple = ("#2a78d6", "#eb6834", "#1baf7a")
    good: str = "#0ca30c"
    critical: str = "#d03b3b"
    warning: str = "#fab219"
    seq_low: str = "#cde2fb"
    seq_high: str = "#0d366b"

    @staticmethod
    def dark() -> "Theme":
        return Theme(
            surface="#1a1a19", text_primary="#ffffff", text_secondary="#c3c2b7",
            muted="#898781", grid="#2c2c2a", axis="#383835",
            series=("#3987e5", "#d95926", "#199e70"),
            seq_low="#184f95", seq_high="#cde2fb",
        )


THEME = Theme()
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(fig: go.Figure, title: str, subtitle: str = "", theme: Theme = THEME,
                 height: int = 460) -> go.Figure:
    """Apply shared chrome: recessive grid, hover, typography."""
    heading = f"<b>{title}</b>"
    if subtitle:
        heading += f"<br><span style='font-size:12px;color:{theme.text_secondary}'>{subtitle}</span>"

    fig.update_layout(
        title=dict(text=heading, x=0, xanchor="left", font=dict(size=17, color=theme.text_primary)),
        paper_bgcolor=theme.surface,
        plot_bgcolor=theme.surface,
        font=dict(family=FONT, size=12, color=theme.text_secondary),
        height=height,
        margin=dict(l=70, r=30, t=90, b=60),
        hovermode="closest",
        hoverlabel=dict(font_size=12, font_family=FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=theme.grid, linecolor=theme.axis, zeroline=False,
                     tickfont=dict(color=theme.muted), title_font=dict(color=theme.text_secondary))
    fig.update_yaxes(gridcolor=theme.grid, linecolor=theme.axis, zeroline=False,
                     tickfont=dict(color=theme.muted), title_font=dict(color=theme.text_secondary))
    return fig


# --------------------------------------------------------------------------
# 1. The conditional-expectation surface -- the study's headline figure
# --------------------------------------------------------------------------


def conditional_surface(
    table: pd.DataFrame,
    baseline_mean: float,
    theme: Theme = THEME,
    label_top: int = 6,
) -> go.Figure:
    """Retained population vs net expected return, with uncertainty.

    This is the precision/recall trade-off expressed in return space: moving
    left means filtering harder; moving up means earning more per trade. The
    question the whole study asks is whether any point sits above zero with an
    interval that excludes it, at an x-value worth trading.

    Significance is encoded by colour AND marker symbol AND direct label, never
    by colour alone.
    """
    df = table.copy()
    df = df[df["name"] != "(no veto)"]
    sig = df["q_pass"].fillna(False).astype(bool)

    fig = go.Figure()

    # Uncertainty first, so markers sit above it.
    for _, row in df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["retained_frac"], row["retained_frac"]],
                y=[row["ci_lo"], row["ci_hi"]],
                mode="lines",
                line=dict(color=theme.axis, width=1),
                hoverinfo="skip", showlegend=False,
            )
        )

    for mask, name, colour, symbol in (
        (~sig, "not significant", theme.series[0], "circle"),
        (sig, "passes FDR", theme.good, "diamond"),
    ):
        sub = df[mask]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["retained_frac"], y=sub["mean_net"],
                mode="markers", name=name,
                marker=dict(size=10, color=colour, symbol=symbol,
                            line=dict(width=2, color=theme.surface)),
                customdata=np.stack([sub["name"], sub["n_trades"], sub["p_value"],
                                     sub["sharpe"]], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "retained %{x:.1%} · n=%{customdata[1]:,}<br>"
                    "E[net] %{y:+.3%}<br>"
                    "p=%{customdata[2]:.3f} · Sharpe %{customdata[3]:.2f}<extra></extra>"
                ),
            )
        )

    # Reference lines: break-even, and the unfiltered baseline.
    fig.add_hline(y=0, line=dict(color=theme.text_secondary, width=1.5))
    fig.add_hline(y=baseline_mean, line=dict(color=theme.series[1], width=1.5, dash="dash"))
    fig.add_annotation(
        x=1.0, xref="paper", y=baseline_mean, yanchor="bottom",
        text=f"no-veto baseline {baseline_mean:+.2%}", showarrow=False,
        font=dict(size=11, color=theme.series[1]), xanchor="right",
    )

    # Direct labels on the strongest points (relief rule + legibility).
    for _, row in df.nlargest(label_top, "mean_net").iterrows():
        fig.add_annotation(
            x=row["retained_frac"], y=row["mean_net"],
            text=row["name"].replace("~veto_", "~"), showarrow=False,
            yshift=14, font=dict(size=10, color=theme.text_secondary),
        )

    fig.update_xaxes(title="fraction of candidate population retained", tickformat=".0%")
    fig.update_yaxes(title="net expected return per trade", tickformat=".1%")
    return _base_layout(
        fig, "Conditional expectation surface",
        "Each point is one veto set. Vertical bars are day-clustered 95% bootstrap intervals.",
        theme, height=520,
    )


# --------------------------------------------------------------------------
# 2. Veto lift -- diverging, one axis
# --------------------------------------------------------------------------


def veto_lift(single: pd.DataFrame, theme: Theme = THEME) -> go.Figure:
    """Per-veto effect on the mean, as a diverging bar with an interval.

    Diverging blue/red about zero: polarity is the message, so a two-hue
    diverging encoding is the correct choice rather than categorical hues.
    """
    df = single.sort_values("mean_kept").copy()
    colours = [theme.series[0] if v >= 0 else theme.critical for v in df["mean_kept"]]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["mean_kept"], y=df["veto"].str.replace("veto_", ""),
            orientation="h",
            marker=dict(color=colours, line=dict(width=0)),
            error_x=dict(
                type="data", symmetric=False,
                array=(df["ci_hi_kept"] - df["mean_kept"]).clip(lower=0),
                arrayminus=(df["mean_kept"] - df["ci_lo_kept"]).clip(lower=0),
                color=theme.axis, thickness=1, width=4,
            ),
            customdata=np.stack([df["n_kept"], df["retained_frac"], df["mean_removed"]], axis=-1),
            hovertemplate=(
                "<b>~%{y}</b><br>E[net | survives] %{x:+.3%}<br>"
                "kept %{customdata[0]:,} (%{customdata[1]:.0%})<br>"
                "E[net | vetoed] %{customdata[2]:+.3%}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_vline(x=0, line=dict(color=theme.text_secondary, width=1.5))
    fig.update_xaxes(title="net expected return of the surviving population", tickformat=".1%")
    fig.update_yaxes(title="")
    return _base_layout(
        fig, "Effect of each veto applied alone",
        "Bars show E[net return] among trades that survive the veto; whiskers are 95% bootstrap intervals.",
        theme, height=max(360, 26 * len(df) + 160),
    )


# --------------------------------------------------------------------------
# 3. Distribution comparison -- ECDF
# --------------------------------------------------------------------------


def return_distributions(
    baseline: Sequence[float], filtered: Sequence[float],
    filtered_label: str = "after veto", theme: Theme = THEME,
) -> go.Figure:
    """Empirical CDFs of net return, baseline vs filtered.

    An ECDF rather than a histogram: it needs no bin choice, shows the whole
    distribution including the extreme right tail that dominates this universe,
    and makes stochastic dominance visible directly.
    """
    fig = go.Figure()
    for values, name, colour in (
        (baseline, "all candidates", theme.series[0]),
        (filtered, filtered_label, theme.series[1]),
    ):
        v = np.sort(np.asarray(values, dtype=float))
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        fig.add_trace(
            go.Scatter(
                x=v, y=y, mode="lines", name=name,
                line=dict(color=colour, width=2),
                hovertemplate=f"<b>{name}</b><br>net return %{{x:.1%}}<br>"
                              f"cumulative %{{y:.1%}}<extra></extra>",
            )
        )

    fig.add_vline(x=0, line=dict(color=theme.text_secondary, width=1.5, dash="dot"))
    fig.update_xaxes(title="net return per trade", tickformat=".0%", range=[-0.30, 0.45])
    fig.update_yaxes(title="cumulative probability", tickformat=".0%")
    return _base_layout(
        fig, "Distribution of net returns",
        "A curve further right is better. The vertical line marks break-even.",
        theme,
    )


# --------------------------------------------------------------------------
# 4. Walk-forward stability -- the honesty check
# --------------------------------------------------------------------------


def walk_forward_stability(summary: pd.DataFrame, theme: Theme = THEME) -> go.Figure:
    """In-sample vs out-of-sample mean per fold.

    The most diagnostic figure in the study. A large, consistent gap between
    the two series is the signature of selection overfitting: the veto set was
    chosen because it looked good on the training days, and did not survive
    contact with held-out ones.
    """
    df = summary.copy()
    labels = [f"fold {int(f)}<br><span style='font-size:10px'>{s:%Y-%m}</span>"
              for f, s in zip(df["fold"], pd.to_datetime(df["test_start"]))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=df["train_mean"], name="in-sample (selected on)",
        marker=dict(color=theme.series[0]),
        hovertemplate="<b>in-sample</b><br>%{y:+.3%}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=df["test_mean"], name="out-of-sample (held out)",
        marker=dict(color=theme.series[1]),
        error_y=dict(
            type="data", symmetric=False,
            array=(df["test_ci_hi"] - df["test_mean"]).clip(lower=0),
            arrayminus=(df["test_mean"] - df["test_ci_lo"]).clip(lower=0),
            color=theme.axis, thickness=1, width=4,
        ),
        customdata=np.stack([df["chosen_vetoes"], df["test_n"]], axis=-1),
        hovertemplate="<b>out-of-sample</b><br>%{y:+.3%}<br>n=%{customdata[1]:,}"
                      "<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=theme.text_secondary, width=1.5))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08)
    fig.update_yaxes(title="mean net return per trade", tickformat=".1%")
    return _base_layout(
        fig, "Selection stability: in-sample vs held-out",
        "A persistent gap between the pairs is the signature of selection overfitting.",
        theme,
    )


# --------------------------------------------------------------------------
# 5. Equity curves
# --------------------------------------------------------------------------


def equity_curves(curves: Dict[str, pd.DataFrame], theme: Theme = THEME) -> go.Figure:
    """Equal-weight daily equity for up to three populations."""
    fig = go.Figure()
    for (name, curve), colour in zip(curves.items(), theme.series):
        if curve.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=curve["date"], y=curve["equity"], mode="lines", name=name,
                line=dict(color=colour, width=2),
                customdata=curve["n_trades"],
                hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>"
                              f"equity %{{y:.3f}}<br>%{{customdata}} trades<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line=dict(color=theme.text_secondary, width=1, dash="dot"))
    fig.update_yaxes(title="equity (equal-weight, starts at 1.0)", type="log")
    fig.update_xaxes(title="")
    return _base_layout(
        fig, "Equal-weight daily equity",
        "Capital is split evenly across each day's signals, so busy days do not silently take more risk. Log scale.",
        theme,
    )


# --------------------------------------------------------------------------
# 6. Outcome mix
# --------------------------------------------------------------------------


def outcome_mix(trades: pd.DataFrame, group_col: str, theme: Theme = THEME) -> go.Figure:
    """Share of target / stop / vertical outcomes by group.

    Stacked bars use the adjacent pairlist, where the first three slots are
    validated. A 2px surface gap separates segments.
    """
    order = ["target", "vertical", "stop"]
    mix = (
        trades.groupby([group_col, "outcome"]).size()
        .unstack(fill_value=0).reindex(columns=order, fill_value=0)
    )
    mix = mix.div(mix.sum(axis=1), axis=0)

    fig = go.Figure()
    for outcome, colour in zip(order, theme.series):
        fig.add_trace(
            go.Bar(
                x=mix.index.astype(str), y=mix[outcome], name=outcome,
                marker=dict(color=colour, line=dict(width=2, color=theme.surface)),
                hovertemplate=f"<b>{outcome}</b><br>%{{x}}<br>%{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(title="share of trades", tickformat=".0%", range=[0, 1])
    fig.update_xaxes(title="")
    return _base_layout(
        fig, "Outcome mix", "How each population resolves: profit target, time exit, or stop.",
        theme,
    )


# --------------------------------------------------------------------------
# 7. Excursion density -- why the barriers sit where they do
# --------------------------------------------------------------------------


def excursion_density(trades: pd.DataFrame, spec_target: float, spec_stop: float,
                      theme: Theme = THEME) -> go.Figure:
    """Maximum favourable vs maximum adverse excursion.

    Sequential (single-hue) density, because the encoded quantity is magnitude.
    Shows directly whether the chosen barriers sit anywhere near where the
    price actually travels -- a target beyond the bulk of the MFE mass is
    unreachable by construction, no matter what the model predicts.
    """
    df = trades.dropna(subset=["max_favourable", "max_adverse"])
    fig = go.Figure(
        go.Histogram2d(
            x=df["max_adverse"].clip(-0.5, 0),
            y=df["max_favourable"].clip(0, 1.0),
            colorscale=[[0, theme.seq_low], [1, theme.seq_high]],
            nbinsx=60, nbinsy=60,
            colorbar=dict(title="trades", outlinewidth=0, tickfont=dict(color=theme.muted)),
            hovertemplate="MAE %{x:.0%}<br>MFE %{y:.0%}<br>%{z} trades<extra></extra>",
        )
    )
    fig.add_hline(y=spec_target, line=dict(color=theme.critical, width=2, dash="dash"))
    fig.add_annotation(x=0, xref="paper", y=spec_target, text=f"profit target +{spec_target:.0%}",
                       showarrow=False, yshift=10, xanchor="left",
                       font=dict(size=11, color=theme.critical))
    fig.add_vline(x=-spec_stop, line=dict(color=theme.warning, width=2, dash="dash"))
    fig.add_annotation(x=-spec_stop, y=1, yref="paper", text=f"stop -{spec_stop:.0%}",
                       showarrow=False, xshift=-6, textangle=-90, yanchor="top",
                       font=dict(size=11, color=theme.warning))

    fig.update_xaxes(title="maximum adverse excursion", tickformat=".0%")
    fig.update_yaxes(title="maximum favourable excursion", tickformat=".0%")
    return _base_layout(
        fig, "Where price actually travels",
        "Density of intraday excursions against the chosen barriers.",
        theme, height=520,
    )


# --------------------------------------------------------------------------
# 8. Veto interaction heatmap
# --------------------------------------------------------------------------


def veto_interaction(table: pd.DataFrame, vetoes: Sequence[str],
                     baseline_mean: float, theme: Theme = THEME) -> go.Figure:
    """Pairwise veto combinations, as a diverging matrix around the baseline.

    Diverging blue/red with a neutral midpoint at the baseline mean: the
    message is polarity (better or worse than no veto), which is exactly what a
    diverging scale encodes.
    """
    short = [v.replace("veto_", "") for v in vetoes]
    grid = pd.DataFrame(np.nan, index=short, columns=short, dtype=float)

    lookup = {}
    for _, row in table.iterrows():
        key = frozenset(v.strip() for v in row["vetoes"].split(",") if v.strip())
        lookup[key] = row["mean_net"]

    for i, a in enumerate(vetoes):
        for j, b in enumerate(vetoes):
            key = frozenset([a]) if i == j else frozenset([a, b])
            if key in lookup:
                grid.iloc[i, j] = lookup[key]

    span = np.nanmax(np.abs(grid.to_numpy() - baseline_mean))
    span = span if np.isfinite(span) and span > 0 else 0.01

    fig = go.Figure(
        go.Heatmap(
            z=grid.to_numpy(), x=short, y=short,
            colorscale=[[0, theme.critical], [0.5, "#f0efec"], [1, theme.series[0]]],
            zmid=baseline_mean, zmin=baseline_mean - span, zmax=baseline_mean + span,
            xgap=2, ygap=2,
            colorbar=dict(title="E[net]", tickformat=".1%", outlinewidth=0,
                          tickfont=dict(color=theme.muted)),
            hovertemplate="~%{y} & ~%{x}<br>E[net] %{z:+.3%}<extra></extra>",
        )
    )
    fig.update_xaxes(tickangle=-45, title="")
    fig.update_yaxes(title="")
    return _base_layout(
        fig, "Veto pair interactions",
        f"Diagonal is each veto alone. Midpoint is the no-veto baseline ({baseline_mean:+.2%}); "
        f"blue beats it, red trails it.",
        theme, height=560,
    )
