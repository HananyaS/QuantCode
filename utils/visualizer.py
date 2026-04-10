"""visualizer.py — interactive Plotly dashboard for multi-asset pipeline results."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Helpers ───────────────────────────────────────────────────────────────────

def _drawdown_series(equity: pd.Series) -> pd.Series:
    rolling_max = equity.cummax()
    return (equity - rolling_max) / rolling_max


def _benchmark_equity(
    context: dict, test_index: pd.Index, initial_capital: float
) -> pd.Series | None:
    bm_data = context.get("benchmark_data")
    if bm_data is None:
        return None
    bm_close = bm_data["Close"].reindex(test_index).dropna()
    bm_returns = bm_close.pct_change().dropna()
    return (1 + bm_returns).cumprod() * initial_capital


# ── Multi-asset dashboard ─────────────────────────────────────────────────────

def plot_multi_results(context: dict) -> None:
    """Render a 4-panel interactive dashboard for the multi-asset pipeline.

    Panels:
        1. Portfolio equity curve vs benchmark
        2. Drawdown
        3. Daily holdings heatmap (asset presence per day)
        4. Net daily returns distribution
    """
    bt = context["multi_backtest"]
    metrics = context["multi_metrics"]

    equity: pd.Series = bt["equity_curve"]
    returns: pd.Series = bt["returns"]
    initial_capital: float = bt["initial_capital"]
    weights: pd.DataFrame = context["portfolio_weights"]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Portfolio Equity vs Benchmark",
            "Drawdown",
            "Daily Holdings Heatmap (top-K assets)",
            "Net Daily Returns Distribution",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # ── 1. Equity curve ───────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            name="Portfolio",
            line=dict(color="#2196F3", width=2),
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Portfolio</extra>",
        ),
        row=1, col=1,
    )

    bm_equity = _benchmark_equity(context, returns.index, initial_capital)
    if bm_equity is not None:
        fig.add_trace(
            go.Scatter(
                x=bm_equity.index,
                y=bm_equity.values,
                name="Benchmark",
                line=dict(color="#FF9800", width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Benchmark</extra>",
            ),
            row=1, col=1,
        )

    fig.add_hline(y=initial_capital, line_dash="dot", line_color="gray",
                  line_width=1, row=1, col=1)

    # ── 2. Drawdown ───────────────────────────────────────────────────────────
    drawdown = _drawdown_series(equity)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=(drawdown * 100).values,
            fill="tozeroy",
            fillcolor="rgba(244,67,54,0.15)",
            line=dict(color="#F44336", width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f}%<extra>Drawdown</extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # ── 3. Holdings heatmap ───────────────────────────────────────────────────
    held = (weights > 0).astype(int)
    held_weekly = held.resample("W").last().dropna(how="all")
    fig.add_trace(
        go.Heatmap(
            z=held_weekly.T.values,
            x=held_weekly.index,
            y=list(held_weekly.columns),
            colorscale=[[0, "#1a1a2e"], [1, "#2196F3"]],
            showscale=False,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Ticker: %{y}<br>Held: %{z}<extra></extra>",
        ),
        row=2, col=1,
    )

    # ── 4. Returns distribution ───────────────────────────────────────────────
    fig.add_trace(
        go.Histogram(
            x=(returns * 100).values,
            nbinsx=50,
            marker_color="#9C27B0",
            opacity=0.75,
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=2,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1, row=2, col=2)

    # ── Layout ────────────────────────────────────────────────────────────────
    ann_ret = metrics["annualized_return"] * 100
    bm_ann_ret = metrics.get("benchmark", {}).get("annualized_return", 0) * 100
    sharpe = metrics["sharpe"]
    max_dd = metrics["max_drawdown"] * 100
    ic = metrics.get("ic", float("nan"))

    fig.update_layout(
        title=dict(
            text=(
                f"Multi-Asset Results  |  "
                f"Ann.Return: {ann_ret:.1f}% vs Benchmark {bm_ann_ret:.1f}%  |  "
                f"Sharpe: {sharpe:.2f}  |  "
                f"MaxDD: {max_dd:.1f}%  |  "
                f"IC: {ic:.4f}"
            ),
            font=dict(size=14),
        ),
        template="plotly_dark",
        height=750,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1, tickformat="$,.0f")
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)
    fig.update_xaxes(title_text="Daily Return (%)", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    fig.show()
