// Chart helpers shared by the Live, Strategy and Research tabs: a frame, a tooltip kept inside
// its chart, text measurement, a line chart with a crosshair, a bar chart, and the intraday
// day chart (price, noise area, VWAP, checks, entries, stop path, exits).
(function () {
  const C3 = window.C3;

  C3.frame = (el, height, label) => {
    el.replaceChildren();
    const w = Math.max(260, Math.floor(el.clientWidth || 600));
    const svg = d3.select(el).append('svg')
      .attr('viewBox', `0 0 ${w} ${height}`).attr('width', '100%').attr('height', height)
      .attr('role', 'img').attr('aria-label', label);
    return { svg, w, h: height };
  };

  C3.empty = (el, text) => {
    el.replaceChildren();
    const p = document.createElement('p');
    p.className = 'c3-note';
    p.textContent = text;
    el.appendChild(p);
  };

  // Shows a drawing's own state when its data is not there: loading, failed, or nothing yet.
  C3.notReady = (el, ...ids) => {
    for (const id of ids) {
      const d = dash.data(id);
      if (d.status === 'ok') continue;
      if (d.status === 'loading') { el.replaceChildren(); el.classList.add('dash-skeleton'); return true; }
      el.classList.remove('dash-skeleton');
      if (d.status === 'error') C3.empty(el, 'This chart could not load its data; Sources shows the error.');
      else el.replaceChildren(document.createTextNode('—'));
      return true;
    }
    el.classList.remove('dash-skeleton');
    return false;
  };

  C3.tip = (el) => {
    const tip = document.createElement('div');
    tip.className = 'c3-tip';
    tip.hidden = true;
    el.appendChild(tip);
    return {
      show(title, rows, x, y) {
        tip.replaceChildren();
        const h = document.createElement('div');
        h.className = 'c3-tip-h';
        h.textContent = title;
        tip.appendChild(h);
        for (const r of rows) {
          const row = document.createElement('div');
          row.className = 'c3-tip-r';
          const sw = document.createElement('i');
          sw.className = 'c3-sw';
          if (r.color) sw.style.background = r.color;
          const l = document.createElement('span');
          l.textContent = r.label;
          const v = document.createElement('b');
          v.textContent = r.value;
          row.append(sw, l, v);
          tip.appendChild(row);
        }
        tip.hidden = false;
        const W = el.clientWidth, H = el.clientHeight, tw = tip.offsetWidth, th = tip.offsetHeight;
        let left = x + 14;
        if (left + tw > W) left = x - tw - 14;
        left = Math.max(0, Math.min(left, W - tw));
        const top = Math.max(0, Math.min(y - th / 2, H - th));
        tip.style.left = `${left}px`;
        tip.style.top = `${top}px`;
      },
      hide() { tip.hidden = true; },
    };
  };

  C3.measure = (svg, texts) => {
    const g = svg.append('g').style('visibility', 'hidden');
    let w = 0;
    for (const t of texts) w = Math.max(w, g.append('text').text(t).node().getComputedTextLength() || t.length * 6.2);
    g.remove();
    return w;
  };

  // A bar whose data end is rounded and whose baseline end is square.
  C3.bar = (x, base, end, w, r = 3) => {
    const h = Math.abs(end - base);
    if (h < 0.5 || w < 0.5) return `M${x},${base}h${w}v0h${-w}Z`;
    const rr = Math.min(r, w / 2, h);
    if (end < base) {
      return `M${x},${base}V${end + rr}Q${x},${end} ${x + rr},${end}H${x + w - rr}Q${x + w},${end} ${x + w},${end + rr}V${base}Z`;
    }
    return `M${x},${base}V${end - rr}Q${x},${end} ${x + rr},${end}H${x + w - rr}Q${x + w},${end} ${x + w},${end - rr}V${base}Z`;
  };

  const gridY = (g, y, ticks, x0, x1) => {
    g.append('g').attr('class', 'c3-grid').selectAll('line').data(ticks).join('line')
      .attr('x1', x0).attr('x2', x1).attr('y1', (t) => y(t)).attr('y2', (t) => y(t));
  };
  C3.gridY = gridY;

  // ---- line chart ---------------------------------------------------------------------------
  // o.series: [{key, label, color, values: [[x, y], ...], width}] sharing one x grid.
  C3.lineChart = (el, o) => {
    const { svg, w, h } = C3.frame(el, o.height || 240, o.label);
    const all = o.series.flatMap((s) => s.values.filter((v) => v[1] !== null && Number.isFinite(v[1])));
    if (!all.length) { C3.empty(el, o.emptyText || 'No data yet.'); return; }
    const yExt = d3.extent(all, (v) => v[1]);
    for (const r of o.refs || []) { yExt[0] = Math.min(yExt[0], r.y); yExt[1] = Math.max(yExt[1], r.y); }
    if (o.zero) { yExt[0] = Math.min(0, yExt[0]); yExt[1] = Math.max(0, yExt[1]); }
    const y = d3.scaleLinear().domain(yExt).nice(5);
    const yt = y.ticks(5);
    const yFmt = o.yFmt || d3.format('.2f');
    const ml = C3.measure(svg, yt.map(yFmt)) + 10;
    const ends = o.endLabels === false ? 0 : C3.measure(svg, o.series.map((s) => s.label)) + 16;
    const m = { t: o.top || 12, r: Math.max(8, ends), b: 26, l: ml };
    const xs = o.series[0].values.map((v) => v[0]);
    const x = (o.x === 'time' ? d3.scaleUtc() : d3.scaleLinear()).domain(o.xDomain || d3.extent(xs)).range([m.l, w - m.r]);
    y.range([h - m.b, m.t]);
    const g = svg.append('g');
    gridY(g, y, yt, m.l, w - m.r);
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(0,${h - m.b})`)
      .call((o.xTickValues ? d3.axisBottom(x).tickValues(o.xTickValues.filter((v, i, a) => i % Math.max(1, Math.ceil(a.length / Math.max(2, Math.floor((w - m.l - m.r) / 70)))) === 0))
        : d3.axisBottom(x).ticks(Math.max(3, Math.floor((w - m.l - m.r) / 90)))).tickSizeOuter(0).tickFormat(o.xFmt || null));
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues(yt).tickFormat(yFmt).tickSize(0).tickPadding(6))
      .call((a) => a.select('.domain').remove());
    for (const vr of o.vrefs || []) {
      if (vr.band) {
        g.append('rect').attr('x', x(vr.band[0])).attr('width', Math.max(1, x(vr.band[1]) - x(vr.band[0])))
          .attr('y', m.t).attr('height', h - m.b - m.t).style('fill', vr.color || 'var(--cds-chart-muted)').style('opacity', 0.18);
      }
      g.append('line').attr('class', 'c3-ref').attr('x1', x(vr.x)).attr('x2', x(vr.x)).attr('y1', m.t).attr('y2', h - m.b)
        .style('stroke', vr.color || null).style('stroke-dasharray', vr.solid ? 'none' : null);
      g.append('text').attr('class', 'c3-reflabel').attr('x', x(vr.x) + 4).attr('y', m.t + 10 + (vr.row || 0) * 12).text(vr.label);
    }
    for (const r of o.refs || []) {
      g.append('line').attr('class', 'c3-ref').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(r.y)).attr('y2', y(r.y));
      g.append('text').attr('class', 'c3-reflabel').attr('x', w - m.r - 4).attr('y', y(r.y) - 4).attr('text-anchor', 'end').text(r.label);
    }
    const line = d3.line().defined((v) => v[1] !== null && Number.isFinite(v[1])).x((v) => x(v[0])).y((v) => y(v[1]));
    for (const s of [...o.series].reverse()) {
      g.append('path').attr('d', line(s.values)).style('fill', 'none').style('stroke', s.color)
        .style('stroke-width', s.width || 2).style('stroke-linejoin', 'round').style('stroke-linecap', 'round');
    }
    for (const p of o.points || []) {
      g.append('circle').attr('cx', x(p.x)).attr('cy', y(p.y)).attr('r', 4.5).style('fill', p.color)
        .style('stroke', 'var(--color-bg)').style('stroke-width', 2);
    }
    if (ends) {
      const lab = o.series.map((s) => {
        const last = [...s.values].reverse().find((v) => v[1] !== null && Number.isFinite(v[1]));
        return last ? { s, x: x(last[0]), y: y(last[1]), want: y(last[1]) } : null;
      }).filter(Boolean).sort((a, b) => a.want - b.want);
      for (let i = 1; i < lab.length; i++) lab[i].y = Math.max(lab[i].y, lab[i - 1].y + 13);
      for (const l of lab) {
        g.append('circle').attr('cx', l.x).attr('cy', l.want).attr('r', 3.5).style('fill', l.s.color)
          .style('stroke', 'var(--color-bg)').style('stroke-width', 2);
        g.append('text').attr('class', 'c3-endlabel c3-label').attr('x', l.x + 8).attr('y', l.y + 4).text(l.s.label);
      }
    }
    // crosshair
    const tip = C3.tip(el);
    const cross = g.append('line').attr('class', 'c3-cross').attr('y1', m.t).attr('y2', h - m.b).style('opacity', 0);
    const dots = o.series.map((s) => g.append('circle').attr('r', 4).style('fill', s.color)
      .style('stroke', 'var(--color-bg)').style('stroke-width', 2).style('opacity', 0));
    const bis = d3.bisector((v) => v).center;
    svg.append('rect').attr('class', 'c3-hit').attr('x', m.l).attr('y', m.t).attr('width', Math.max(1, w - m.l - m.r))
      .attr('height', h - m.t - m.b)
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev, svg.node());
        const sc = el.clientWidth / w;
        const i = bis(xs, x.invert(px));
        const xv = xs[i];
        cross.attr('x1', x(xv)).attr('x2', x(xv)).style('opacity', 1);
        const rows = [];
        o.series.forEach((s, k) => {
          const v = s.values[i];
          const ok = v && v[1] !== null && Number.isFinite(v[1]);
          dots[k].attr('cx', x(xv)).attr('cy', ok ? y(v[1]) : 0).style('opacity', ok ? 1 : 0);
          rows.push({ color: s.color, label: s.label, value: ok ? (o.tipFmt || yFmt)(v[1]) : '—' });
        });
        tip.show(o.tipTitle ? o.tipTitle(xv, i) : String(xv), (o.tipExtra ? rows.concat(o.tipExtra(i)) : rows), x(xv) * sc, py * sc);
      })
      .on('pointerleave', () => { cross.style('opacity', 0); dots.forEach((d) => d.style('opacity', 0)); tip.hide(); });
  };

  // ---- bar chart: grouped or stacked, one category axis ------------------------------------------
  // o.cats: [cat]; o.groups: [{key, label, color}]; o.value(cat, group) -> number
  C3.barChart = (el, o) => {
    const { svg, w, h } = C3.frame(el, o.height || 240, o.label);
    const vals = o.cats.flatMap((c) => o.groups.map((gr) => o.value(c, gr) || 0));
    let lo = Math.min(0, d3.min(vals)), hi = Math.max(0, d3.max(vals));
    if (o.stacked) {
      const sums = o.cats.map((c) => d3.sum(o.groups, (gr) => Math.max(0, o.value(c, gr) || 0)));
      hi = Math.max(hi, d3.max(sums));
    }
    for (const r of o.refs || []) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
    const y = d3.scaleLinear().domain([lo, hi]).nice(5);
    const yt = y.ticks(5);
    const yFmt = o.yFmt || d3.format(',');
    const m = { t: 12, r: 8, b: 26, l: C3.measure(svg, yt.map(yFmt)) + 10 };
    const x = d3.scaleBand().domain(o.cats).range([m.l, w - m.r]).paddingInner(o.stacked ? 0.12 : 0.22).paddingOuter(0.1);
    y.range([h - m.b, m.t]);
    const g = svg.append('g');
    gridY(g, y, yt, m.l, w - m.r);
    const every = Math.max(1, Math.ceil(o.cats.length / Math.max(2, Math.floor((w - m.l - m.r) / (o.minLabel || 44)))));
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(0,${h - m.b})`)
      .call(d3.axisBottom(x).tickSizeOuter(0).tickFormat((c, i) => (i % every === 0 ? (o.xFmt ? o.xFmt(c) : c) : '')));
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues(yt).tickFormat(yFmt).tickSize(0).tickPadding(6))
      .call((a) => a.select('.domain').remove());
    g.append('line').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(0)).attr('y2', y(0)).style('stroke', 'var(--cds-chart-axis)');
    for (const r of o.refs || []) {
      g.append('line').attr('class', 'c3-ref').attr('x1', m.l).attr('x2', w - m.r).attr('y1', y(r.y)).attr('y2', y(r.y));
      g.append('text').attr('class', 'c3-reflabel').attr('x', w - m.r - 4).attr('y', y(r.y) - 4).attr('text-anchor', 'end').text(r.label);
    }
    const inner = o.stacked ? null : d3.scaleBand().domain(o.groups.map((gr) => gr.key)).range([0, x.bandwidth()]).paddingInner(0.12);
    const gap = 2;
    for (const c of o.cats) {
      let acc = 0;
      for (const gr of o.groups) {
        const v = o.value(c, gr) || 0;
        if (o.stacked) {
          if (v <= 0) continue;
          let base = y(acc);
          const end = y(acc + v);
          if (acc > 0 && base - gap > end) base -= gap;            // a 2px surface gap between segments
          g.append('path').attr('d', C3.bar(x(c), base, end, x.bandwidth(), 2)).style('fill', gr.color);
          acc += v;
        } else {
          g.append('path').attr('d', C3.bar(x(c) + inner(gr.key), y(0), y(v), inner.bandwidth())).style('fill', gr.color);
        }
      }
    }
    const tip = C3.tip(el);
    const hl = g.append('rect').style('fill', 'var(--cds-chart-muted)').style('opacity', 0).attr('y', m.t).attr('height', h - m.t - m.b);
    g.lower(); hl.lower();
    svg.append('g').selectAll('rect').data(o.cats).join('rect').attr('class', 'c3-hit')
      .attr('x', (c) => x(c) - (x.step() - x.bandwidth()) / 2).attr('width', x.step()).attr('y', m.t).attr('height', h - m.t - m.b)
      .on('pointermove', (ev, c) => {
        const [px, py] = d3.pointer(ev, svg.node());
        const sc = el.clientWidth / w;
        hl.attr('x', x(c) - 2).attr('width', x.bandwidth() + 4).style('opacity', 0.35);
        const rows = o.tipRows ? o.tipRows(c) : o.groups.map((gr) => ({ color: gr.color, label: gr.label, value: yFmt(o.value(c, gr) || 0) }));
        tip.show(o.tipTitle ? o.tipTitle(c) : String(c), rows, px * sc, py * sc);
      })
      .on('pointerleave', () => { hl.style('opacity', 0); tip.hide(); })
      .on('click', (ev, c) => { if (o.onClick) o.onClick(c); });
  };

  // ---- the intraday day chart ---------------------------------------------------------------------
  const hhmm = (min) => { const t = 570 + Math.round(min); return `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`; };
  C3.hhmm = hhmm;
  C3.clockToMin = (s) => {
    if (!s) return null;
    const [hh, mm, ss] = String(s).split(':').map(Number);
    return hh * 60 + mm + (ss || 0) / 60 - 570;
  };
  // o: {n, close, ub, lb, vwap (arrays by bar), upto (minutes revealed), checks: [{x, action}], hiX,
  //     trades: [{x0, p0, x1, p1, stops: [[x, level]]}], ghosts: [{x, y}], label, height}
  C3.dayChart = (el, o) => {
    const { svg, w, h } = C3.frame(el, o.height || 330, o.label);
    const n = o.n, upto = o.upto === undefined ? n : o.upto;
    const pts = (a) => a.slice(0, n).map((v, b) => [b + 1, C3.num(v)]);
    const close = pts(o.close), ub = pts(o.ub), lb = pts(o.lb), vw = pts(o.vwap);
    const vals = [close, ub, lb, vw].flatMap((a) => a.map((p) => p[1])).filter((v) => v !== null);
    const stopVals = (o.trades || []).flatMap((t) => (t.stops || []).map((s) => s[1]));
    const y = d3.scaleLinear().domain(d3.extent(vals.concat(stopVals))).nice(6);
    const yt = y.ticks(6);
    const yFmt = d3.format('.2f');
    const m = { t: 14, r: 10, b: 26, l: C3.measure(svg, yt.map(yFmt)) + 10 };
    const x = d3.scaleLinear().domain([0, n]).range([m.l, w - m.r]);
    y.range([h - m.b, m.t]);
    const g = svg.append('g');
    gridY(g, y, yt, m.l, w - m.r);
    const hours = d3.range(30, n + 1, 60);
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(0,${h - m.b})`)
      .call(d3.axisBottom(x).tickValues(w < 420 ? hours.filter((v, i) => i % 2 === 0) : hours).tickFormat(hhmm).tickSizeOuter(0));
    g.append('g').attr('class', 'c3-axis').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(y).tickValues(yt).tickFormat(yFmt).tickSize(0).tickPadding(6))
      .call((a) => a.select('.domain').remove());
    // checks: faint verticals, the current one stronger
    g.append('g').selectAll('line').data(o.checks || []).join('line')
      .attr('x1', (c) => x(c.x)).attr('x2', (c) => x(c.x)).attr('y1', m.t).attr('y2', h - m.b)
      .style('stroke', 'var(--cds-chart-grid)').style('stroke-dasharray', '2 3');
    // noise area: known at the open for the whole day
    const band = d3.area().defined((p, i) => p[1] !== null && lb[i][1] !== null)
      .x((p) => x(p[0])).y0((p, i) => y(lb[i][1])).y1((p) => y(p[1]));
    g.append('path').attr('d', band(ub)).style('fill', 'var(--cds-chart-muted)').style('opacity', 0.45);
    const line = d3.line().defined((p) => p[1] !== null).x((p) => x(p[0])).y((p) => y(p[1]));
    for (const a of [ub, lb]) g.append('path').attr('d', line(a)).style('fill', 'none').style('stroke', 'var(--cds-chart-reference)').style('stroke-width', 1);
    const shown = (a) => a.filter((p) => p[0] <= upto);
    g.append('path').attr('d', line(shown(vw))).style('fill', 'none').style('stroke', C3.color(4)).style('stroke-width', 1.5).style('stroke-dasharray', '5 3');
    g.append('path').attr('d', line(shown(close))).style('fill', 'none').style('stroke', C3.color(1)).style('stroke-width', 2).style('stroke-linejoin', 'round');
    // trades: stop path, entry triangle, exit ring
    const tri = d3.symbol(d3.symbolTriangle, 70);
    for (const t of o.trades || []) {
      if (t.x0 > upto) continue;
      const end = Math.min(t.x1 ?? n, upto);
      const st = (t.stops || []).filter((s) => s[0] <= end && s[1] !== null);
      if (st.length) {
        const path = st.concat([[end, st[st.length - 1][1]]]);
        g.append('path').attr('d', d3.line().curve(d3.curveStepAfter).x((p) => x(p[0])).y((p) => y(p[1]))(path))
          .style('fill', 'none').style('stroke', C3.color(0)).style('stroke-width', 1.5).style('stroke-dasharray', '4 3');
      }
      g.append('path').attr('d', tri()).attr('transform', `translate(${x(t.x0)},${y(t.p0) + 9})`)
        .style('fill', C3.color(0)).style('stroke', 'var(--color-bg)').style('stroke-width', 2);
      if (t.x1 !== null && t.x1 !== undefined && t.x1 <= upto && t.p1 !== null) {
        g.append('circle').attr('cx', x(t.x1)).attr('cy', y(t.p1)).attr('r', 4.5)
          .style('fill', 'var(--color-bg)').style('stroke', C3.color(0)).style('stroke-width', 2);
      }
    }
    for (const gh of o.ghosts || []) {
      if (gh.x > upto || gh.y === null) continue;
      g.append('path').attr('d', tri()).attr('transform', `translate(${x(gh.x)},${y(gh.y) + 20})`)
        .style('fill', 'none').style('stroke', C3.color(3)).style('stroke-width', 1.5);
    }
    if (o.hiX !== undefined && o.hiX !== null) {
      g.append('line').attr('x1', x(o.hiX)).attr('x2', x(o.hiX)).attr('y1', m.t).attr('y2', h - m.b)
        .style('stroke', 'var(--color-fg)').style('stroke-width', 1).style('opacity', 0.55);
      const p = close[Math.max(0, Math.min(n - 1, Math.round(o.hiX) - 1))];
      if (p && p[1] !== null) g.append('circle').attr('cx', x(p[0])).attr('cy', y(p[1])).attr('r', 4.5)
        .style('fill', C3.color(1)).style('stroke', 'var(--color-bg)').style('stroke-width', 2);
    }
    // crosshair over the bars
    const tip = C3.tip(el);
    const cross = g.append('line').attr('class', 'c3-cross').attr('y1', m.t).attr('y2', h - m.b).style('opacity', 0);
    const byX = new Map((o.checks || []).map((c) => [c.x, c]));
    svg.append('rect').attr('class', 'c3-hit').attr('x', m.l).attr('y', m.t).attr('width', Math.max(1, w - m.l - m.r)).attr('height', h - m.t - m.b)
      .on('pointermove', (ev) => {
        const [px, py] = d3.pointer(ev, svg.node());
        const sc = el.clientWidth / w;
        const b = Math.max(1, Math.min(n, Math.round(x.invert(px))));
        cross.attr('x1', x(b)).attr('x2', x(b)).style('opacity', 1);
        const i = b - 1, rows = [];
        if (b <= upto) rows.push({ color: C3.color(1), label: 'SPY close', value: yFmt(close[i][1]) });
        rows.push({ color: 'var(--cds-chart-reference)', label: 'Upper bound', value: ub[i][1] === null ? '—' : yFmt(ub[i][1]) });
        rows.push({ color: 'var(--cds-chart-reference)', label: 'Lower bound', value: lb[i][1] === null ? '—' : yFmt(lb[i][1]) });
        if (b <= upto) rows.push({ color: C3.color(4), label: 'VWAP', value: vw[i][1] === null ? '—' : yFmt(vw[i][1]) });
        const c = byX.get(b);
        if (c && b <= upto) rows.push({ label: 'Check', value: C3.actionWord(c.action) });
        tip.show(`${hhmm(b - 1)}–${hhmm(b)} ET`, rows, x(b) * sc, py * sc);
      })
      .on('pointerleave', () => { cross.style('opacity', 0); tip.hide(); });
  };

  C3.actionWord = (a) => ({
    enter: 'buy', hold: 'hold, move stop', exit: 'exit', enter_unfilled: 'signal, not filled',
    inside: 'inside the band', below_vwap: 'above band, below VWAP',
  }[a] || String(a));
})();
