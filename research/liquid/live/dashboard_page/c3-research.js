// Research tab: the dev window 2016-2022 only (fixed files built by dashboard.py --dev).
(function () {
  const C3 = window.C3;
  const num = C3.num;
  const rowsOf = (id) => { const d = dash.data(id); return d.status === 'ok' && Array.isArray(d.data) ? d.data : []; };
  const SERIES = [['c3', 'C3', 0], ['spy', 'SPY', 1], ['qqq', 'QQQ', 2]];
  const dayFmt = d3.utcFormat('%a %-d %b %Y');

  const growth = (rows) => {
    const xs = rows.map((r) => C3.parseDay(r.date));
    return SERIES.map(([k, label, ci]) => {
      let g = 1, peak = 1;
      const cum = [], dd = [];
      for (const r of rows) {
        g *= 1 + (num(r[k]) || 0);
        peak = Math.max(peak, g);
        cum.push(g);
        dd.push(g / peak - 1);
      }
      return { key: k, label, color: C3.color(ci), cum: cum.map((v, i) => [xs[i], v]), dd: dd.map((v, i) => [xs[i], v]) };
    });
  };

  dash.draw('dev-equity', (el) => {
    if (C3.notReady(el, 'dev_daily')) return;
    const s = growth(rowsOf('dev_daily'));
    C3.lineChart(el, {
      label: 'Growth of one dollar in C3 after costs, SPY and QQQ, 2016 to 2022',
      x: 'time', height: 250, series: s.map((v) => ({ key: v.key, label: v.label, color: v.color, values: v.cum })),
      yFmt: (v) => `$${d3.format('.2f')(v)}`, xFmt: d3.utcFormat('%Y'), tipTitle: (d) => dayFmt(d),
    });
  });

  dash.draw('dev-dd', (el) => {
    if (C3.notReady(el, 'dev_daily')) return;
    const s = growth(rowsOf('dev_daily'));
    C3.lineChart(el, {
      label: 'Drawdown from the running peak for C3, SPY and QQQ, 2016 to 2022',
      x: 'time', height: 250, zero: true, series: s.map((v) => ({ key: v.key, label: v.label, color: v.color, values: v.dd, width: v.key === 'c3' ? 2 : 1.5 })),
      yFmt: d3.format('.0%'), tipFmt: d3.format('.1%'), xFmt: d3.utcFormat('%Y'), tipTitle: (d) => dayFmt(d),
    });
  });

  dash.draw('dev-summary', (tb) => {
    const d = dash.data('dev_summary');
    if (d.status !== 'ok') { tb.replaceChildren(); return; }
    C3.fillBody(tb, d.data, [
      { field: 'series' }, { field: 'sharpe', fmt: 'f2' }, { field: 'cagr', fmt: 'pct1' }, { field: 'vol', fmt: 'pct1' },
      { field: 'max_dd', fmt: 'pct1' }, { field: 'total_return', fmt: 'pct1' },
    ], 'series');
  });

  dash.draw('dev-years', (el) => {
    if (C3.notReady(el, 'dev_years')) return;
    const rows = rowsOf('dev_years');
    const by = new Map(rows.map((r) => [String(r.year), r]));
    const groups = SERIES.map(([k, label, ci]) => ({ key: k, label, color: C3.color(ci) }));
    C3.barChart(el, {
      label: 'Sharpe ratio by calendar year for C3, SPY and QQQ',
      cats: rows.map((r) => String(r.year)), groups, value: (c, g) => num(by.get(c)[`${g.key}_sharpe`]),
      yFmt: d3.format('.1f'), tipTitle: (c) => c,
      tipRows: (c) => groups.map((g) => ({ color: g.color, label: g.label,
        value: `Sharpe ${C3.fmt('f2', by.get(c)[`${g.key}_sharpe`])} · ${C3.fmt('spct1', by.get(c)[`${g.key}_ret`])}` })),
    });
  });

  dash.draw('dev-cost', (el) => {
    if (C3.notReady(el, 'dev_costcurve', 'dev_c3')) return;
    const curve = rowsOf('dev_costcurve'), c3 = rowsOf('dev_c3')[0];
    const ls = dash.data('live_status');
    const live = ls.status === 'ok' && ls.data[0] && num(ls.data[0].cost_mean_bp) !== null ? ls.data[0] : null;
    const be = num(c3.breakeven_bp), model = num(c3.cost_bp);
    const vrefs = [{ x: model, label: `research model ${d3.format('.2f')(model)}` }, { x: be, label: `break-even ${d3.format('.2f')(be)}`, solid: true, row: 1 }];
    if (live) vrefs.push({ x: num(live.cost_mean_bp), label: `paper ${d3.format('.2f')(num(live.cost_mean_bp))}`, color: C3.color(0), solid: true, row: 2,
      band: num(live.cost_lo_bp) !== null ? [num(live.cost_lo_bp), num(live.cost_hi_bp)] : null });
    C3.lineChart(el, {
      label: 'C3 dev Sharpe as the cost per round trip rises from 0 to 3.5 basis points, with SPY\'s Sharpe and the break-even',
      height: 250, endLabels: false, xDomain: [0, 3.5],
      series: [{ key: 'c3', label: 'C3 Sharpe', color: C3.color(0), values: curve.map((r) => [num(r.cost_bp), num(r.sharpe)]) }],
      refs: [{ y: num(c3.spy_sharpe), label: `SPY ${d3.format('.2f')(num(c3.spy_sharpe))}` }], vrefs,
      yFmt: d3.format('.1f'), tipFmt: d3.format('.2f'), xFmt: (v) => `${v} bp`,
      tipTitle: (v) => `${d3.format('.1f')(v)} bp per round trip`,
      points: [{ x: be, y: num(c3.spy_sharpe), color: C3.color(0) }],
    });
  });

  dash.draw('dev-hist', (el) => {
    if (C3.notReady(el, 'dev_trades')) return;
    const trades = rowsOf('dev_trades');
    const LO = -40, HI = 100, W = 5;
    const cats = d3.range(LO - W, HI + W, W);
    const binOf = (v) => (v < LO ? LO - W : v >= HI ? HI : Math.floor(v / W) * W);
    const count = new Map();
    for (const t of trades) {
      const k = `${binOf(num(t.net_bp))}|${t.exit}`;
      count.set(k, (count.get(k) || 0) + 1);
    }
    const groups = [{ key: 'stop', label: 'Stopped out', color: C3.color(0) }, { key: 'close', label: 'Held to the close', color: C3.color(5) }];
    const name = (c) => (c === LO - W ? `below ${LO} bp` : c === HI ? `${HI} bp or more` : `${c} to ${c + W} bp`);
    C3.barChart(el, {
      label: 'Number of research round trips by net return in basis points, split by stop exits and close exits',
      cats, groups, stacked: true, value: (c, g) => count.get(`${c}|${g.key}`) || 0, yFmt: d3.format(','),
      xFmt: (c) => (c === LO - W ? `<${LO}` : c === HI ? `${HI}+` : String(c)), minLabel: 34,
      tipTitle: (c) => `Net ${name(c)}`,
      tipRows: (c) => groups.map((g) => ({ color: g.color, label: g.label, value: d3.format(',')(count.get(`${c}|${g.key}`) || 0) })),
    });
  });

  document.querySelectorAll('.c3 #check-view button[data-view]').forEach((b, i, all) => {
    b.addEventListener('click', () => {
      C3.state.checkView = b.dataset.view;
      all.forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      dash.draw('dev-bycheck');
    });
  });
  dash.draw('dev-bycheck', (el) => {
    if (C3.notReady(el, 'dev_by_check')) return;
    const rows = rowsOf('dev_by_check');
    const by = new Map(rows.map((r) => [r.check, r]));
    const view = C3.state.checkView;
    const pick = { contrib: ['contrib', d3.format('+.1%'), 'Return contributed, summed'], trades: ['trades', d3.format(','), 'Round trips'],
      mean: ['mean_net_bp', (v) => `${d3.format('+.1f')(v)}`, 'Mean net bp'] }[view];
    C3.barChart(el, {
      label: `${pick[2]} by the check time of entry, research 2016 to 2022`,
      cats: rows.map((r) => r.check), groups: [{ key: pick[0], label: pick[2], color: C3.color(0) }],
      value: (c) => num(by.get(c)[pick[0]]), yFmt: pick[1], minLabel: 40, tipTitle: (c) => `Entered at the ${c} check`,
      tipRows: (c) => {
        const r = by.get(c);
        return [
          { label: 'Round trips', value: d3.format(',')(r.trades) },
          { label: 'Won', value: d3.format('.0%')(r.win_rate) },
          { label: 'Mean net', value: `${d3.format('+.2f')(r.mean_net_bp)} bp` },
          { color: C3.color(0), label: 'Return contributed', value: d3.format('+.2%')(r.contrib) },
        ];
      },
    });
  });

  dash.draw('dev-exit', (tb) => {
    const d = dash.data('dev_by_exit');
    if (d.status !== 'ok' || !Array.isArray(d.data)) { tb.replaceChildren(); return; }
    C3.fillBody(tb, d.data, [
      { field: 'exit', cls: 'is-text' }, { field: 'trades', fmt: 'int' }, { field: 'share', fmt: 'pct1' }, { field: 'win_rate', fmt: 'pct1' },
      { field: 'mean_gross_bp', fmt: 'f2', tone: true }, { field: 'mean_net_bp', fmt: 'f2', tone: true }, { field: 'mean_hold_min', fmt: 'int' },
    ], 'exit');
  });

  const yearSel = document.querySelector('.c3 #dev-year');
  if (yearSel) yearSel.addEventListener('change', () => { C3.state.devYear = yearSel.value; dash.draw('dev-trades'); dash.draw('dev-trades-count'); });
  const devRows = () => {
    const y = C3.state.devYear;
    return rowsOf('dev_trades').filter((t) => !y || String(t.year) === y);
  };
  dash.draw('dev-trades-count', (el) => {
    if (dash.data('dev_trades').status !== 'ok') { el.textContent = '—'; return; }
    const r = devRows();
    const y = C3.state.devYear;
    el.textContent = `${d3.format(',')(r.length)} round trips${y ? ` in ${y}` : ', newest first'}. Mean net ${d3.format('+.2f')(d3.mean(r, (t) => num(t.net_bp)) || 0)} bp; ${d3.format('.0%')(d3.mean(r, (t) => (num(t.net_bp) > 0 ? 1 : 0)) || 0)} won.`;
  });
  dash.draw('dev-trades', (tb) => {
    if (dash.data('dev_trades').status !== 'ok') { tb.replaceChildren(); return; }
    C3.fillBody(tb, [...devRows()].reverse(), [
      { field: 'id', fmt: 'int' }, { field: 'date' }, { field: 'check' }, { field: 'close_at_check', fmt: 'f2' }, { field: 'ub', fmt: 'f2' },
      { field: 'vwap', fmt: 'f2' }, { field: 'weight', fmt: 'x2' }, { field: 'exit', cls: 'is-text' }, { field: 'exit_time' },
      { field: 'gross_bp', fmt: 'f2', tone: true }, { field: 'net_bp', fmt: 'f2', tone: true }, { field: 'contrib_bp', fmt: 'f2', tone: true },
    ], 'id');
  });
})();
