"""make_report.py — build the round-9 findings page with inline SVG charts.

Charts are generated here rather than hand-written so they cannot drift from the
numbers. Everything is inline: the artifact CSP blocks external hosts, so no CDN
scripts, no webfonts, no remote images.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("chartdata.json")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("round9.html")
d: Dict = json.loads(DATA.read_text(encoding="utf-8"))

W, H = 860, 300
PAD_L, PAD_R, PAD_T, PAD_B = 56, 18, 18, 34


def esc(s: str) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def equity_chart() -> str:
    a, b = d["curve_all"], d["curve_nocovid"]
    n = max(len(a["equity"]), len(b["equity"]))
    ymax = max(max(a["equity"]), max(b["equity"])) * 1.06
    ymin = 0.8

    def sx(i: int) -> float:
        return PAD_L + (W - PAD_L - PAD_R) * i / max(n - 1, 1)

    def sy(v: float) -> float:
        return H - PAD_B - (H - PAD_T - PAD_B) * (v - ymin) / (ymax - ymin)

    def path(vals: List[float]) -> str:
        return " ".join(("M" if i == 0 else "L") + f"{sx(i):.1f},{sy(v):.1f}"
                        for i, v in enumerate(vals))

    grid, labels = [], []
    for gv in (1.0, 1.5, 2.0, 2.5):
        if gv > ymax:
            continue
        y = sy(gv)
        grid.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{W-PAD_R}" y2="{y:.1f}" class="grid"/>')
        labels.append(f'<text x="{PAD_L-9}" y="{y+4:.1f}" class="ax" text-anchor="end">{gv:.1f}&#215;</text>')
    ticks = []
    for i in range(0, n, max(n // 6, 1)):
        if i < len(a["dates"]):
            ticks.append(f'<text x="{sx(i):.1f}" y="{H-12}" class="ax" text-anchor="middle">'
                         f'{a["dates"][i][:4]}</text>')
    # shade the COVID window
    cov = [i for i, dt in enumerate(a["dates"]) if "2020-02" <= dt <= "2020-07"]
    band = ""
    if cov:
        x0, x1 = sx(cov[0]), sx(cov[-1])
        band = (f'<rect x="{x0:.1f}" y="{PAD_T}" width="{max(x1-x0,3):.1f}" '
                f'height="{H-PAD_T-PAD_B}" class="band"/>'
                f'<text x="{x1+6:.1f}" y="{PAD_T+14}" class="bandlab">COVID crash</text>')
    return f"""<svg viewBox="0 0 {W} {H}" class="chart" role="img"
     aria-label="Equity curves with and without the COVID crash window">
  {band}{''.join(grid)}{''.join(labels)}{''.join(ticks)}
  <path d="{path(b['equity'])}" class="line-b"/>
  <path d="{path(a['equity'])}" class="line-a"/>
  <g class="key">
    <line x1="{PAD_L+8}" y1="{PAD_T+10}" x2="{PAD_L+34}" y2="{PAD_T+10}" class="line-a"/>
    <text x="{PAD_L+40}" y="{PAD_T+14}" class="ax">all trades &#8212; 2.09&#215;</text>
    <line x1="{PAD_L+8}" y1="{PAD_T+28}" x2="{PAD_L+34}" y2="{PAD_T+28}" class="line-b"/>
    <text x="{PAD_L+40}" y="{PAD_T+32}" class="ax">excluding Feb&#8211;Jun 2020 &#8212; 1.04&#215;</text>
  </g>
</svg>"""


def loo_chart() -> str:
    rows = d["loo"]
    h = 34 * len(rows) + 40
    lo = min(r["lo"] for r in rows) - 3
    hi = max(r["hi"] for r in rows) + 3
    x0, x1 = 118, W - 24

    def sx(v: float) -> float:
        return x0 + (x1 - x0) * (v - lo) / (hi - lo)

    parts = [f'<line x1="{sx(0):.1f}" y1="18" x2="{sx(0):.1f}" y2="{h-22}" class="zero"/>',
             f'<text x="{sx(0):.1f}" y="{h-8}" class="ax" text-anchor="middle">0%</text>']
    for i, r in enumerate(rows):
        y = 34 * i + 30
        kill = r["lo"] <= 0
        cls = "bar-dead" if kill else "bar-ok"
        parts.append(f'<text x="108" y="{y+4}" class="ax" text-anchor="end">'
                     f'{"without " + r["removed"] if r["removed"] != "none" else "all years"}</text>')
        parts.append(f'<line x1="{sx(r["lo"]):.1f}" y1="{y}" x2="{sx(r["hi"]):.1f}" y2="{y}" class="{cls}"/>')
        parts.append(f'<circle cx="{sx(r["mean"]):.1f}" cy="{y}" r="4" class="{cls}-dot"/>')
        if kill:
            parts.append(f'<text x="{sx(r["hi"]):.1f}" y="{y+4}" dx="8" class="killlab">'
                         f'interval crosses zero</text>')
    return f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="Leave one year out confidence intervals">{"".join(parts)}</svg>'


def dist_chart() -> str:
    c, e = d["dist"]["counts"], d["dist"]["edges"]
    hh = 250
    mx = max(c)
    bw = (W - PAD_L - PAD_R) / len(c)
    bars = []
    for i, v in enumerate(c):
        x = PAD_L + i * bw
        bh = (hh - PAD_T - PAD_B) * v / mx
        neg = e[i] < 0
        bars.append(f'<rect x="{x:.1f}" y="{hh-PAD_B-bh:.1f}" width="{bw*0.86:.1f}" '
                    f'height="{bh:.1f}" class="{"hn" if neg else "hp"}"/>')

    def sx(v: float) -> float:
        return PAD_L + (W - PAD_L - PAD_R) * (v - e[0]) / (e[-1] - e[0])

    med, mean = d["dist"]["median"], d["dist"]["mean"]
    marks = (f'<line x1="{sx(med):.1f}" y1="{PAD_T}" x2="{sx(med):.1f}" y2="{hh-PAD_B}" class="mmed"/>'
             f'<text x="{sx(med):.1f}" y="{PAD_T-2}" class="ax" text-anchor="middle">median {med:+.1f}%</text>'
             f'<line x1="{sx(mean):.1f}" y1="{PAD_T}" x2="{sx(mean):.1f}" y2="{hh-PAD_B}" class="mmean"/>'
             f'<text x="{sx(mean):.1f}" y="{PAD_T-2}" class="ax" text-anchor="middle">mean {mean:+.1f}%</text>')
    ticks = "".join(f'<text x="{sx(v):.1f}" y="{hh-10}" class="ax" text-anchor="middle">{v:+.0f}%</text>'
                    for v in (-100, -50, 0, 50, 100, 150, 200, 250))
    return f'<svg viewBox="0 0 {W} {hh}" class="chart" role="img" aria-label="Distribution of trade returns">{"".join(bars)}{marks}{ticks}</svg>'


def year_chart() -> str:
    rows = [r for r in d["per_year"] if r["period"] == "dev"]
    hh = 260
    lo = min(min(r["mean"], r["median"]) for r in rows) - 6
    hi = max(max(r["mean"], r["median"]) for r in rows) + 8
    bw = (W - PAD_L - PAD_R) / len(rows)

    def sy(v: float) -> float:
        return hh - PAD_B - (hh - PAD_T - PAD_B) * (v - lo) / (hi - lo)

    parts = [f'<line x1="{PAD_L}" y1="{sy(0):.1f}" x2="{W-PAD_R}" y2="{sy(0):.1f}" class="zero"/>']
    for i, r in enumerate(rows):
        cx = PAD_L + bw * (i + 0.5)
        for val, cls, w in ((r["mean"], "ymean", bw * 0.30), (r["median"], "ymed", bw * 0.30)):
            x = cx - bw * 0.32 + (0 if cls == "ymean" else bw * 0.34)
            y0, y1 = sy(0), sy(val)
            parts.append(f'<rect x="{x:.1f}" y="{min(y0,y1):.1f}" width="{w:.1f}" '
                         f'height="{abs(y1-y0):.1f}" class="{cls}"/>')
        parts.append(f'<text x="{cx:.1f}" y="{hh-10}" class="ax" text-anchor="middle">{r["year"]}</text>')
        if r["year"] == 2020:
            parts.append(f'<text x="{cx:.1f}" y="{sy(r["mean"])-6:.1f}" class="ax" '
                         f'text-anchor="middle">+{r["mean"]:.0f}%</text>')
    key = (f'<rect x="{PAD_L+8}" y="{PAD_T}" width="11" height="11" class="ymean"/>'
           f'<text x="{PAD_L+24}" y="{PAD_T+10}" class="ax">mean</text>'
           f'<rect x="{PAD_L+74}" y="{PAD_T}" width="11" height="11" class="ymed"/>'
           f'<text x="{PAD_L+90}" y="{PAD_T+10}" class="ax">median</text>')
    return f'<svg viewBox="0 0 {W} {hh}" class="chart" role="img" aria-label="Per-year mean and median trade return">{"".join(parts)}{key}</svg>'


b = d["bug"]
html = f"""<title>Four Months in Eleven Years</title>
<style>
:root {{
  --ink:#15171C; --body:#3A3F4A; --mute:#767E8C; --paper:#F5F6F8; --card:#FFFFFF;
  --rule:#DFE3E9; --accent:#2F4A9C; --bad:#B3261E; --good:#1D6B45; --band:#F0E4D8;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --ink:#EEF1F6; --body:#BCC3CF; --mute:#8B94A3; --paper:#101318; --card:#171B22;
  --rule:#282E38; --accent:#8FA6E8; --bad:#F0857C; --good:#63C08F; --band:#2A2318;
}} }}
:root[data-theme="dark"] {{
  --ink:#EEF1F6; --body:#BCC3CF; --mute:#8B94A3; --paper:#101318; --card:#171B22;
  --rule:#282E38; --accent:#8FA6E8; --bad:#F0857C; --good:#63C08F; --band:#2A2318;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--body); margin:0;
  font:16px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
.wrap {{ max-width:940px; margin:0 auto; padding:0 24px 80px; }}
header {{ border-bottom:1px solid var(--rule); margin-bottom:40px;
  padding:64px 0 32px; }}
.kick {{ font-size:11.5px; letter-spacing:.18em; text-transform:uppercase;
  color:var(--accent); font-weight:650; }}
h1 {{ font:650 40px/1.12 ui-serif,Georgia,"Times New Roman",serif; color:var(--ink);
  margin:14px 0 12px; letter-spacing:-.015em; text-wrap:balance; }}
h2 {{ font:600 25px/1.2 ui-serif,Georgia,serif; color:var(--ink); margin:52px 0 8px;
  text-wrap:balance; }}
h3 {{ font:600 17px/1.35 inherit; color:var(--ink); margin:26px 0 6px; }}
.stand {{ font-size:18.5px; color:var(--body); max-width:64ch; }}
.meta {{ display:flex; flex-wrap:wrap; gap:20px; margin-top:22px; font-size:13px; color:var(--mute); }}
.meta b {{ color:var(--ink); font-weight:600; }}
p {{ max-width:68ch; }}
.verdict {{ background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--bad);
  border-radius:3px; padding:22px 26px; margin:30px 0; }}
.verdict .lab {{ font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--bad); font-weight:700; }}
.note {{ background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:3px; padding:20px 24px; margin:26px 0; }}
.note .lab {{ font-size:11px; letter-spacing:.16em; text-transform:uppercase;
  color:var(--accent); font-weight:700; }}
figure {{ margin:26px 0; background:var(--card); border:1px solid var(--rule);
  border-radius:3px; padding:18px 16px 10px; }}
figcaption {{ font-size:13.5px; color:var(--mute); margin-top:10px; padding:0 8px;
  max-width:74ch; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--rule); stroke-width:1; }}
.zero {{ stroke:var(--mute); stroke-width:1; stroke-dasharray:3 3; }}
.ax {{ fill:var(--mute); font-size:11.5px; font-family:inherit; }}
.line-a {{ fill:none; stroke:var(--accent); stroke-width:2.2; }}
.line-b {{ fill:none; stroke:var(--mute); stroke-width:2.2; stroke-dasharray:5 4; }}
.band {{ fill:var(--band); }}
.bandlab {{ fill:var(--mute); font-size:11px; font-family:inherit; }}
.bar-ok {{ stroke:var(--accent); stroke-width:3; }} .bar-ok-dot {{ fill:var(--accent); }}
.bar-dead {{ stroke:var(--bad); stroke-width:3; }} .bar-dead-dot {{ fill:var(--bad); }}
.killlab {{ fill:var(--bad); font-size:11.5px; font-weight:600; font-family:inherit; }}
.hn {{ fill:var(--bad); opacity:.62; }} .hp {{ fill:var(--accent); opacity:.62; }}
.mmed {{ stroke:var(--good); stroke-width:2; }} .mmean {{ stroke:var(--bad); stroke-width:2; }}
.ymean {{ fill:var(--accent); }} .ymed {{ fill:var(--mute); opacity:.75; }}
.scroll {{ overflow-x:auto; margin:22px 0; }}
table {{ border-collapse:collapse; width:100%; font-size:14.5px;
  font-variant-numeric:tabular-nums; }}
th {{ text-align:left; font-size:11px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--mute); border-bottom:1px solid var(--rule); padding:0 14px 8px 0; font-weight:650; }}
td {{ padding:8px 14px 8px 0; border-bottom:1px solid var(--rule); color:var(--body); }}
td strong {{ color:var(--ink); }}
.neg {{ color:var(--bad); font-weight:600; }} .pos {{ color:var(--good); font-weight:600; }}
ul,ol {{ max-width:68ch; }} li {{ margin:7px 0; }}
code {{ font:13px ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--card);
  border:1px solid var(--rule); border-radius:2px; padding:1px 5px; color:var(--ink); }}
hr {{ border:0; border-top:1px solid var(--rule); margin:56px 0 0; }}
</style>

<header><div class="wrap">
  <div class="kick">Penny research &middot; Round 9</div>
  <h1>Four Months in Eleven Years</h1>
  <p class="stand">An automated search over 195 strategy specifications found one that passed a sealed
    holdout with a Sharpe of 2.40. It was wrong twice &mdash; once because of a hole in the data, and
    once because of the COVID crash.</p>
  <div class="meta">
    <span><b>Panel</b> 6,710,204 sessions</span>
    <span><b>Span</b> 2016&ndash;2026</span>
    <span><b>Specs tested</b> 195</span>
    <span><b>Strategies found</b> 0</span>
  </div>
</div></header>

<div class="wrap">

<div class="verdict">
  <div class="lab">Verdict</div>
  <h3 style="margin-top:6px">The candidate was a single market event</h3>
  <p style="margin-bottom:0">Removing <strong>2020-02-15 to 2020-06-30</strong> &mdash; 59 trades of
    274, four and a half months out of eleven years &mdash; leaves
    <strong class="neg">+1.17% per trade, CI [&minus;6.36%, +8.79%]</strong>, median &minus;8.08%,
    Sharpe 0.09, compounded +3.7%. Seven of the ten largest trades fall between 10 and 13 March 2020
    and every one is an oil and gas name. Buying beaten-down energy at the bottom of the COVID crash
    worked. Nothing else did.</p>
</div>

<figure>
  {equity_chart()}
  <figcaption>Compounded equity on development data, 30 concurrent positions. The two curves differ
    only in whether the shaded window is traded. A rule that needs a once-in-a-decade dislocation to
    be profitable is a description of that dislocation.</figcaption>
</figure>

<h2>The bug that produced a false positive first</h2>
<p>Before the COVID finding, the search produced a candidate that passed everything: development
  Sharpe 1.55, and a <strong>sealed-holdout</strong> result of <strong>+36.25% per trade, CI
  [+24.54%, +49.23%], Sharpe 2.40</strong>, three of three years positive. Then the individual trades
  were read.</p>
<p><strong>MI</strong> showed 2026-03-17 at <strong>$0.48</strong> and the next panel row at
  <strong>$10.01</strong> on 2026-07-02. Those are not consecutive sessions.</p>

<div class="note">
  <div class="lab">Root cause</div>
  <p style="margin-bottom:0">The universe file is <em>pre-filtered</em> &mdash; it keeps only rows whose
    previous close sits in $0.50&ndash;$10.00. Every <code>shift()</code> and every forward path
    computed on it walks over <strong>surviving rows, not sessions</strong>. A stock that drops below
    the $0.50 floor vanishes and reappears weeks later at a different price, and the time series steps
    silently across the hole. No error, no missing values &mdash; the frame is perfectly well-formed.</p>
</div>

<div class="scroll"><table>
  <thead><tr><th>Measure</th><th>First panel</th><th>Corrected</th></tr></thead>
  <tbody>
    <tr><td>rows</td><td>3,431,765</td><td><strong>6,710,204</strong></td></tr>
    <tr><td>consecutive-row gaps &gt; 10 days</td><td class="neg">{b['gaps10_before']:,}</td><td class="pos">{b['gaps10_after']}</td></tr>
    <tr><td>gaps &gt; 90 days</td><td class="neg">{b['gaps90_before']:,}</td><td class="pos">{b['gaps90_after']}</td></tr>
    <tr><td>signals whose &ldquo;5-day&rdquo; window spanned &gt; 10 calendar days</td>
        <td class="neg">{b['bad_signals']} of {b['total_signals']:,} (19%)</td><td class="pos">0</td></tr>
  </tbody>
</table></div>

<p>MI alone had <strong>73 of 91 sessions invisible</strong> to its forward path, so a 90-bar hold
  skipped the months the stock spent at its lows and booked the recovery. The fix is a rule rather
  than a patch: <strong>filter at signal time; never filter the series a shift walks over.</strong>
  Eligibility is now a boolean column on a continuous panel.</p>

<h2>What the correction left</h2>
<figure>
  {loo_chart()}
  <figcaption>Leave-one-year-out on corrected development data, per-trade return with 95%
    day-clustered intervals. Removing any year but 2020 leaves the result intact. Removing 2020
    collapses it to nothing.</figcaption>
</figure>

<figure>
  {year_chart()}
  <figcaption>Mean and median trade return by year. 2020 is not a good year among good years &mdash;
    it is the only one. In five of eight years the median trade loses money.</figcaption>
</figure>

<figure>
  {dist_chart()}
  <figcaption>Distribution of trade returns, clipped at &plusmn;250% for display. Mean
    {d['dist']['mean']:+.1f}% against a median of {d['dist']['median']:+.1f}% &mdash; the average is a
    property of a handful of trades, not of the strategy.</figcaption>
</figure>

<h2>Three findings that survive</h2>
<ol>
  <li><strong>Every stop hurts.</strong> The proposed structural stop at the bounce low fired on
    <strong>74%</strong> of trades and returned &minus;2.27%; fixed stops were no better. These names
    routinely break the bounce low and then recover &mdash; the drawdown <em>is</em> the entry. This
    reproduces an earlier finding on a completely different entry rule.</li>
  <li><strong>The metric was wrong, and changing it changed the answer.</strong> Maximising the lower
    confidence bound of per-trade return selected a lottery: +41.4% per trade with negative medians in
    six of eight years, Sharpe 0.90, and a 2020 that earned +65% per trade while compounding to
    +14.3%, because &sigma;&sup2;/2 exceeded &mu;. Switching to portfolio Sharpe changed which
    strategies won.</li>
  <li><strong>Two thirds of the proposed design was right.</strong> On a portfolio metric the
    confirmation and the profit target both earn their place. Only the stop never does.</li>
</ol>

<h2>What was built</h2>
<p>The infrastructure outlives the negative result, and each piece exists because of a specific past
  failure:</p>
<ul>
  <li><code>etl/screens.py</code> &mdash; raises if a <em>filter</em> touches an adjusted price level.
    Adjusted prices are computed backwards from future corporate actions, so screening on one is a
    survivorship screen. That bug invalidated an earlier strategy; it is now an exception rather than
    something to remember.</li>
  <li><code>strategy/engine.py</code> &mdash; a strategy is data, and there is exactly one
    implementation of selection, barriers and exits. Ties go to the stop, gaps through the stop fill
    at the open, and a delisting exits at the last print rather than being dropped.</li>
  <li><strong>A sealed holdout.</strong> Signals from 2024 are refused without an explicit flag,
    because a search that can see the holdout optimises it.</li>
</ul>

<div class="note">
  <div class="lab">The lesson</div>
  <p><strong>Reading ten rows of raw tape has now overturned four conclusions that survived every
    aggregate test put to them.</strong> The contaminated version here had a <em>tighter</em>
    confidence interval, a higher Sharpe, a cleaner drawdown and a passing sealed holdout. Four
    separate bias checks &mdash; leave-one-year-out, a date-matched control, a delisting worst case
    and a split-exclusion audit &mdash; all passed on corrupted data.</p>
  <p style="margin-bottom:0">Reading the trades belongs in the pipeline, not the post-mortem.</p>
</div>

<hr>
<p style="font-size:13px;color:var(--mute);margin-top:26px">Free data only, unadjusted prices in every
  filter, paper research &mdash; nothing traded. Development 2016&ndash;2023, holdout 2024&ndash;2026
  sealed during the search. The +36.25% holdout figure is void: it was computed on the buggy panel and
  measures the artefact, not the rule.</p>
</div>"""

OUT.write_text(html, encoding="utf-8")
print(f"wrote {OUT} ({len(html):,} bytes)")
