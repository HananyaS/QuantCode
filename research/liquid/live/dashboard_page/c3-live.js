// Live paper tab: cumulative return, P&L per session, the gates, the trade blotter, sessions
// and the day replay. Everything here reads the server's nightly file through the live_* sources.
(function () {
  const C3 = window.C3;
  const num = C3.num;

  // Fill a <tbody> drawing. cols: [{field, fmt, text: (row) => string, cls}]; key: the row-key column.
  C3.fillBody = (tb, rows, cols, key, onClick) => {
    tb.replaceChildren();
    for (const r of rows) {
      const tr = document.createElement('tr');
      tr.dataset.key = String(r[key]);
      for (const c of cols) {
        const td = document.createElement('td');
        td.textContent = c.text ? c.text(r) : C3.fmt(c.fmt || 'text', r[c.field]);
        if (c.cls) td.className = c.cls;
        if (c.tone && num(r[c.field]) !== null) td.classList.add(num(r[c.field]) > 0 ? 'c3-good' : num(r[c.field]) < 0 ? 'c3-bad' : 'x');
        tr.appendChild(td);
      }
      if (onClick) {
        tr.classList.add('is-click');
        tr.tabIndex = 0;
        tr.addEventListener('click', () => onClick(r, tr));
        tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(r, tr); } });
      }
      tb.appendChild(tr);
    }
  };

  const rowsOf = (id) => { const d = dash.data(id); return d.status === 'ok' && Array.isArray(d.data) ? d.data : []; };
  const status = () => rowsOf('live_status')[0] || null;
  const selectedDay = () => {
    const days = rowsOf('live_days').map((r) => r.day);
    if (C3.state.liveDay && days.includes(C3.state.liveDay)) return C3.state.liveDay;
    return days.length ? days[days.length - 1] : null;
  };
  C3.showDay = (day, scroll) => {
    C3.state.liveDay = day;
    ['replay-day', 'replay-chart', 'replay-steps', 'live-trades'].forEach((n) => dash.draw(n));
    if (scroll) { const r = document.getElementById('replay'); if (r) r.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  };

  // ---- cumulative return on the paper days ----
  dash.draw('live-cum', (el) => {
    if (C3.notReady(el, 'live_days')) return;
    const days = rowsOf('live_days');
    if (!days.length) { C3.empty(el, 'No paper session yet.'); return; }
    const first = C3.parseDay(days[0].day);
    const xs = [d3.utcDay.offset(first, -1), ...days.map((r) => C3.parseDay(r.day))];
    const cum = (f) => { let g = 1; return [0, ...days.map((r) => { g *= 1 + (num(r[f]) || 0); return g - 1; })]; };
    const mk = (key, label, ci, f) => ({ key, label, color: C3.color(ci), values: cum(f).map((v, i) => [xs[i], v]) });
    C3.lineChart(el, {
      label: 'Cumulative return of the paper account, the research rule, SPY and QQQ over the paper sessions',
      x: 'time', zero: true, height: 240, xTickValues: xs,
      series: [mk('c3', 'C3 paper', 0, 'ret'), mk('res', 'Research', 3, 'research_ret'), mk('spy', 'SPY', 1, 'spy_ret'), mk('qqq', 'QQQ', 2, 'qqq_ret')],
      yFmt: d3.format('+.1%'), tipFmt: d3.format('+.2%'),
      xFmt: (d) => d3.utcFormat('%-d %b')(d),
      tipTitle: (d, i) => (i === 0 ? 'Before the first session' : C3.dayLabel(days[i - 1].day)),
    });
  });

  // ---- P&L per session ----
  dash.draw('live-pnl', (el) => {
    if (C3.notReady(el, 'live_days')) return;
    const days = rowsOf('live_days');
    if (!days.length) { C3.empty(el, 'No paper session yet.'); return; }
    const by = new Map(days.map((r) => [r.day, r]));
    C3.barChart(el, {
      label: 'Paper P&L in dollars for each session, after fees',
      cats: days.map((r) => r.day), groups: [{ key: 'pnl', label: 'P&L', color: C3.color(0) }],
      value: (c) => num(by.get(c).pnl) || 0, yFmt: d3.format('$,.0f'), xFmt: C3.dayShort, minLabel: 52,
      tipTitle: (c) => C3.dayLabel(c) + ' · click to replay',
      tipRows: (c) => {
        const r = by.get(c);
        return [
          { color: C3.color(0), label: 'P&L after fees', value: C3.fmt('susd', r.pnl) },
          { label: 'Round trips', value: C3.fmt('int', r.trades) },
          { color: C3.color(0), label: 'C3 return', value: C3.fmt('spct2', r.ret) },
          { color: C3.color(3), label: 'Research, same day', value: C3.fmt('spct2', r.research_ret) },
          { color: C3.color(1), label: 'SPY', value: C3.fmt('spct2', r.spy_ret) },
        ];
      },
      onClick: (c) => C3.showDay(c, true),
    });
  });

  // ---- G-cost: trips counted toward the looks, and the cost against the break-even ----
  dash.draw('gate-cost', (el) => {
    if (C3.notReady(el, 'live_status')) return;
    const s = status();
    if (!s) { C3.empty(el, 'No status yet.'); return; }
    const { svg, w, h } = C3.frame(el, 150, 'Usable round trips counted toward the G-cost looks, and the measured cost per round trip against the break-even');
    const m = { l: 8, r: 12 };
    const n = num(s.usable_trips) || 0, looks = [[num(s.interim_n), 'kill-only look'], [num(s.final_n), 'decision'], [num(s.extend_n), 'extension']];
    const xmax = Math.max(n, looks[2][0] || 80);
    const xp = d3.scaleLinear().domain([0, xmax]).range([m.l, w - m.r]);
    const g = svg.append('g');
    g.append('text').attr('class', 'c3-muted-text').attr('x', m.l).attr('y', 12).text(`Usable round trips: ${n}`);
    g.append('rect').attr('x', xp(0)).attr('y', 20).attr('width', xp(xmax) - xp(0)).attr('height', 8).attr('rx', 4)
      .style('fill', 'var(--color-bg)').style('stroke', 'var(--color-border-line)');
    if (n > 0) g.append('rect').attr('x', xp(0)).attr('y', 20).attr('width', Math.max(4, xp(n) - xp(0))).attr('height', 8).attr('rx', 4).style('fill', C3.color(0));
    looks.forEach(([v, lab], i) => {
      if (!v) return;
      g.append('line').attr('x1', xp(v)).attr('x2', xp(v)).attr('y1', 16).attr('y2', 32).style('stroke', 'var(--color-fg-muted)');
      g.append('text').attr('class', 'c3-muted-text').attr('x', xp(v)).attr('y', 44).attr('text-anchor', i === 2 ? 'end' : 'middle').text(`${v} ${w < 360 && i === 0 ? '' : lab}`);
    });
    // cost scale
    const be = num(s.breakeven_bp), mean = num(s.cost_mean_bp), lo = num(s.cost_lo_bp), hi = num(s.cost_hi_bp);
    const dev = rowsOf('dev_c3')[0];
    const model = dev ? num(dev.cost_bp) : null;
    const cmax = Math.max(3, (hi || 0) + 0.5, (mean || 0) + 0.5);
    const xc = d3.scaleLinear().domain([Math.min(0, lo || 0), cmax]).nice().range([m.l + 4, w - m.r - 4]);
    const y0 = 104;
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(0,${y0 + 14})`)
      .call(d3.axisBottom(xc).ticks(Math.max(3, Math.floor(w / 80))).tickFormat((v) => `${v} bp`).tickSizeOuter(0));
    g.append('rect').attr('x', xc(be)).attr('y', 62).attr('width', Math.max(0, xc.range()[1] - xc(be))).attr('height', y0 + 14 - 62)
      .style('fill', 'var(--color-bad)').style('opacity', 0.08);
    g.append('line').attr('class', 'c3-ref').attr('x1', xc(be)).attr('x2', xc(be)).attr('y1', 62).attr('y2', y0 + 14).style('stroke-dasharray', 'none');
    g.append('text').attr('class', 'c3-reflabel').attr('x', xc(be) + 4).attr('y', 72).text(`break-even ${d3.format('.2f')(be)}`);
    if (model !== null) {
      g.append('line').attr('class', 'c3-ref').attr('x1', xc(model)).attr('x2', xc(model)).attr('y1', 80).attr('y2', y0 + 14);
      g.append('text').attr('class', 'c3-reflabel').attr('x', xc(model) - 4).attr('y', 88).attr('text-anchor', 'end').text(`research model ${d3.format('.2f')(model)}`);
    }
    if (mean !== null) {
      if (lo !== null && hi !== null) {
        g.append('line').attr('x1', xc(lo)).attr('x2', xc(hi)).attr('y1', y0).attr('y2', y0).style('stroke', C3.color(0)).style('stroke-width', 2);
        [lo, hi].forEach((v) => g.append('line').attr('x1', xc(v)).attr('x2', xc(v)).attr('y1', y0 - 5).attr('y2', y0 + 5).style('stroke', C3.color(0)).style('stroke-width', 2));
      }
      g.append('circle').attr('cx', xc(mean)).attr('cy', y0).attr('r', 5).style('fill', C3.color(0)).style('stroke', 'var(--color-bg)').style('stroke-width', 2);
    } else {
      g.append('text').attr('class', 'c3-muted-text').attr('x', m.l + 4).attr('y', y0 + 4).text('No measured round trip yet');
    }
    const tip = C3.tip(el);
    svg.append('rect').attr('class', 'c3-hit').attr('x', 0).attr('y', 56).attr('width', w).attr('height', h - 56)
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev, svg.node());
        const sc = el.clientWidth / w;
        tip.show('Cost per round trip', [
          { color: C3.color(0), label: 'Paper, mean', value: mean === null ? '—' : `${d3.format('.2f')(mean)} bp` },
          { label: '95% interval', value: lo === null ? '—' : `${d3.format('.2f')(lo)} to ${d3.format('.2f')(hi)} bp` },
          { label: 'Break-even', value: `${d3.format('.2f')(be)} bp` },
          { label: 'Research model', value: model === null ? '—' : `${d3.format('.2f')(model)} bp` },
        ], px * sc, py * sc);
      })
      .on('pointerleave', () => tip.hide());
  });

  // ---- G-fidelity: the pooled Jaccard against its bar ----
  dash.draw('gate-fid', (el) => {
    if (C3.notReady(el, 'live_status')) return;
    const s = status();
    if (!s) { C3.empty(el, 'No status yet.'); return; }
    const { svg, w } = C3.frame(el, 70, 'Share of entries shared by live and research, against the 0.85 bar');
    const j = num(s.jaccard), bar = num(s.fidelity_bar);
    const x = d3.scaleLinear().domain([0, 1]).range([8, w - 12]);
    const g = svg.append('g');
    g.append('rect').attr('x', x(0)).attr('y', 14).attr('width', x(1) - x(0)).attr('height', 10).attr('rx', 5)
      .style('fill', 'var(--color-bg)').style('stroke', 'var(--color-border-line)');
    if (j !== null) g.append('rect').attr('x', x(0)).attr('y', 14).attr('width', Math.max(5, x(j) - x(0))).attr('height', 10).attr('rx', 5).style('fill', C3.color(0));
    g.append('line').attr('x1', x(bar)).attr('x2', x(bar)).attr('y1', 8).attr('y2', 30).style('stroke', 'var(--color-fg)').style('stroke-width', 1.5);
    g.append('text').attr('class', 'c3-reflabel').attr('x', x(bar)).attr('y', 44).attr('text-anchor', 'middle').text(`bar ${d3.format('.2f')(bar)}`);
    g.append('text').attr('class', 'c3-muted-text').attr('x', x(0)).attr('y', 44)
      .text(j === null ? 'No entries yet' : `${d3.format('.2f')(j)} · ${s.matched_entries} shared, ${s.live_entries} live, ${s.research_entries} research`);
    g.append('g').attr('class', 'c3-axis').attr('transform', 'translate(0,56)')
      .call(d3.axisBottom(x).tickValues([0, 0.5, 1]).tickFormat(d3.format('.1f')).tickSize(0)).call((a) => a.select('.domain').remove());
  });

  // ---- blotter ----
  dash.draw('live-trades', (tb) => {
    const d = dash.data('live_trades');
    if (d.status !== 'ok') { tb.replaceChildren(); return; }
    const day = selectedDay();
    const rows = [...d.data].reverse();
    C3.fillBody(tb, rows, [
      { field: 'trade' }, { field: 'check' }, { field: 'entry_time' }, { field: 'entry_price', fmt: 'f2' },
      { field: 'qty', fmt: 'int' }, { field: 'weight', fmt: 'x2' }, { field: 'exit_time' }, { field: 'exit_price', fmt: 'f2' },
      { field: 'exit', cls: 'is-text', text: (r) => ({ sl: 'stop', time: 'close', tp: 'target', kill: 'kill switch' }[r.exit] || r.exit || 'open') },
      { field: 'pnl', fmt: 'f2', tone: true }, { field: 'gross_bp', fmt: 'f2', tone: true }, { field: 'cost_bp', fmt: 'f2' },
      { field: 'research_match', cls: 'is-text', text: (r) => (C3.bool(r.research_match) ? 'same check' : 'live only') },
    ], 'trade', (r) => C3.showDay(r.day, true));
    tb.querySelectorAll('tr').forEach((tr, i) => tr.classList.toggle('is-on', rows[i].day === day));
  });

  dash.draw('live-days', (tb) => {
    const d = dash.data('live_days');
    if (d.status !== 'ok') { tb.replaceChildren(); return; }
    C3.fillBody(tb, [...d.data].reverse(), [
      { field: 'day', text: (r) => r.day }, { field: 'trades', fmt: 'int' }, { field: 'pnl', fmt: 'f2', tone: true },
      { field: 'ret', fmt: 'spct2', tone: true }, { field: 'research_ret', fmt: 'spct2' }, { field: 'spy_ret', fmt: 'spct2' },
      { field: 'qqq_ret', fmt: 'spct2' }, { field: 'weight', fmt: 'x2' }, { field: 'jaccard', fmt: 'f2' },
      { field: 'safety_ok', cls: 'is-text', text: (r) => (C3.bool(r.safety_ok) ? 'ok' : 'FAILED') },
    ], 'day', (r) => C3.showDay(r.day, true));
  });

  // ---- day replay ----
  dash.draw('replay-day', (sel) => {
    const days = rowsOf('live_days');
    sel.replaceChildren(...[...days].reverse().map((r) => new Option(`${C3.dayLabel(r.day)} · ${r.trades} trade${+r.trades === 1 ? '' : 's'}`, r.day)));
    sel.value = selectedDay() || '';
    sel.onchange = () => C3.showDay(sel.value, false);
  });

  const dayParts = (day) => {
    const intr = rowsOf('live_intraday').find((r) => r.day === day);
    const checks = rowsOf('live_checks').filter((r) => r.day === day).sort((a, b) => a.d - b.d);
    const trades = rowsOf('live_trades').filter((r) => r.day === day).sort((a, b) => a.n - b.n);
    return { intr, checks, trades };
  };

  dash.draw('replay-chart', (el) => {
    if (C3.notReady(el, 'live_intraday', 'live_checks', 'live_trades')) return;
    const day = selectedDay();
    if (!day) { C3.empty(el, 'No paper session yet.'); return; }
    const { intr, checks, trades } = dayParts(day);
    if (!intr) { C3.empty(el, `No minute bars for ${day}: the replay keeps the last 40 sessions.`); return; }
    const n = num(intr.n_bars);
    const tr = trades.map((t) => {
      const x0 = C3.clockToMin(t.entry_time), x1 = C3.clockToMin(t.exit_time);
      const stops = [[x0, num(t.stop_at_entry)]];
      for (const c of checks) {
        const cx = num(c.d) + 1;
        if (cx > x0 && (x1 === null || cx < x1) && num(c.stop) !== null && C3.bool(c.in_position)) stops.push([cx, num(c.stop)]);
      }
      return { x0, p0: num(t.entry_price), x1, p1: num(t.exit_price), stops };
    });
    C3.dayChart(el, {
      label: `SPY minute closes on ${day} with the noise area, VWAP, the checks and the paper trades`,
      n, close: C3.arr(intr.close), ub: C3.arr(intr.ub), lb: C3.arr(intr.lb), vwap: C3.arr(intr.vwap),
      checks: checks.map((c) => ({ x: num(c.d) + 1, action: c.action })),
      trades: tr,
      ghosts: checks.filter((c) => C3.bool(c.research_enters)).map((c) => ({ x: num(c.d) + 1, y: num(c.price) })),
    });
  });

  dash.draw('replay-steps', (ol) => {
    ol.replaceChildren();
    const day = selectedDay();
    if (!day || dash.data('live_checks').status !== 'ok') return;
    const { checks, trades } = dayParts(day);
    const f2 = (v) => C3.fmt('f2', v);
    const items = [];
    for (const c of checks) {
      const x = num(c.d) + 1;
      let say;
      if (c.action === 'enter') say = `Close ${f2(c.price)} is above the upper bound ${f2(c.ub)} and VWAP ${f2(c.vwap)}: buy, stop at ${f2(c.stop)}.`;
      else if (c.action === 'hold') say = `Still long at ${f2(c.price)}; stop re-set to ${f2(c.stop)}.`;
      else if (c.action === 'exit') say = `Close ${f2(c.price)} is under the stop level: sold at market.`;
      else if (c.action === 'enter_unfilled') say = `Signal at ${f2(c.price)}, but the limit order did not fill.`;
      else if (c.action === 'inside') say = `Close ${f2(c.price)} is inside the noise area (upper bound ${f2(c.ub)}).`;
      else say = `Close ${f2(c.price)} is above the upper bound ${f2(c.ub)} but below VWAP ${f2(c.vwap)}.`;
      const res = C3.bool(c.research_enters);
      if (res && c.action !== 'enter') say += ' The research rule entered here; live did not.';
      if (!res && c.action === 'enter') say += ' The research rule did not enter here.';
      items.push({ x, time: c.time, tag: C3.actionWord(c.action), say });
    }
    for (const t of trades) {
      const x0 = C3.clockToMin(t.entry_time), x1 = C3.clockToMin(t.exit_time);
      if (x0 !== null) items.push({ x: x0 + 0.01, time: String(t.entry_time).slice(0, 5), tag: 'filled', say: `Bought ${t.qty} shares at ${f2(t.entry_price)} (limit ${f2(t.limit_price)}), ${C3.fmt('x2', t.weight)} equity.` });
      if (x1 !== null) {
        const how = { sl: 'Stopped out', time: 'Sold at the close', tp: 'Target filled', kill: 'Flattened by the kill switch' }[t.exit] || 'Sold';
        items.push({ x: x1, time: String(t.exit_time).slice(0, 5), tag: 'sold', say: `${how} at ${f2(t.exit_price)}: ${C3.fmt('susd', t.pnl)} after fees, cost ${f2(t.cost_bp)} bp.` });
      }
    }
    items.sort((a, b) => a.x - b.x);
    for (const it of items) {
      const li = document.createElement('li');
      const tm = document.createElement('time');
      tm.textContent = it.time;
      const sp = document.createElement('span');
      const tag = document.createElement('span');
      tag.className = 'c3-tag';
      tag.textContent = it.tag;
      sp.append(tag, document.createTextNode(it.say));
      li.append(tm, sp);
      ol.appendChild(li);
    }
  });
})();
