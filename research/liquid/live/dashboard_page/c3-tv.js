// TradingView view of one session, drawn with TradingView's open-source Lightweight Charts (v5.1.0):
// minute candles and volume, the noise area, VWAP, the half-hour checks, C3's entries, stop path and exits.
// The library loads from jsDelivr. If it cannot, the page shows the line chart and disables "Candles".
// Day objects come from C3.exDay() (How it trades) and C3.liveDayObj(day) (Live paper).
(function () {
  const C3 = window.C3;
  const num = C3.num;
  const LIB = 'https://cdn.jsdelivr.net/npm/lightweight-charts@5.1.0/dist/lightweight-charts.standalone.production.js';
  const VIEWS = [
    { group: 'example-view', tv: 'ex-tv', tvBox: 'example-tv', line: 'ex-chart', lineBox: 'example-chart' },
    { group: 'replay-view', tv: 'replay-tv', tvBox: 'replay-tv', line: 'replay-chart', lineBox: 'replay-chart' },
  ];
  C3.state.dayView = C3.state.dayView || 'tv';
  C3.state.tf = C3.state.tf || 5;
  let lib = window.LightweightCharts ? 'ready' : 'loading';

  // ---- which chart shows: hidden / aria only, outside the drawings ----
  const applyView = () => {
    const v = lib === 'failed' ? 'line' : C3.state.dayView;
    for (const w of VIEWS) {
      const tvEl = document.getElementById(w.tvBox), lineEl = document.getElementById(w.lineBox);
      if (tvEl) tvEl.hidden = v !== 'tv';
      if (lineEl) lineEl.hidden = v !== 'line';
      document.querySelectorAll(`.c3 [data-legend-for="${w.group}"]`).forEach((l) => { l.hidden = l.dataset.view !== v; });
      document.querySelectorAll(`.c3 #${w.group} button[data-dayview]`).forEach((b) => {
        b.setAttribute('aria-pressed', b.dataset.dayview === v ? 'true' : 'false');
        if (b.dataset.dayview === 'tv') b.disabled = lib === 'failed';
      });
      dash.draw(v === 'tv' ? w.tv : w.line);
    }
  };
  document.querySelectorAll('.c3 button[data-dayview]').forEach((b) => b.addEventListener('click', () => {
    C3.state.dayView = b.dataset.dayview;
    applyView();
  }));
  if (lib === 'loading') {
    const done = (ok) => {
      if (lib !== 'loading') return;
      lib = ok && window.LightweightCharts ? 'ready' : 'failed';
      applyView();
    };
    const s = document.createElement('script');
    s.src = LIB;
    s.async = true;
    s.onload = () => done(true);
    s.onerror = () => done(false);
    setTimeout(() => done(false), 15000);
    document.head.appendChild(s);
  }

  // ---- colours: theme tokens resolved to rgba, which the canvas library needs ----
  const cvs = document.createElement('canvas');
  cvs.width = 1; cvs.height = 1;
  const ctx = cvs.getContext('2d', { willReadFrequently: true });
  const toRgba = (css, a) => {
    ctx.clearRect(0, 0, 1, 1);
    ctx.fillStyle = '#000';
    ctx.fillStyle = css;
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b, al] = ctx.getImageData(0, 0, 1, 1).data;
    return `rgba(${r},${g},${b},${+((a ?? 1) * al / 255).toFixed(3)})`;
  };
  const palette = (el) => {
    let p = el.querySelector(':scope > .c3-tv-probe');
    if (!p) { p = document.createElement('i'); p.className = 'c3-tv-probe'; p.hidden = true; el.append(p); }
    const cs = getComputedStyle(el);
    const res = (v, fb) => {
      const m = /^var\((--[\w-]+)/.exec(String(v));
      if (m && !cs.getPropertyValue(m[1]).trim()) return fb;
      p.style.color = '';
      p.style.color = v;
      return getComputedStyle(p).color || fb;
    };
    const c = (v, a, fb) => toRgba(res(v, fb || '#888888'), a);
    return {
      text: c('var(--cds-text-secondary)', 1), fg: c('var(--color-fg)', 1), grid: c('var(--cds-chart-grid)', 1),
      axis: c('var(--cds-chart-axis)', 1), ref: c('var(--cds-chart-reference)', 1),
      up: c('var(--color-ok)', 1, '#26a69a'), down: c('var(--color-bad)', 1, '#ef5350'),
      upA: c('var(--color-ok)', 0.4, '#26a69a'), downA: c('var(--color-bad)', 0.4, '#ef5350'),
      s: (i, a) => c(C3.color(i), a),
      font: cs.fontFamily,
    };
  };

  // ---- one session in bars of tf minutes, up to o.upto (a replay reveals the day as it unfolds) ----
  const dayStart = (day) => Date.UTC(+day.slice(0, 4), +day.slice(5, 7) - 1, +day.slice(8, 10), 9, 30) / 1000;
  const hhmm = (b) => C3.hhmm(b);
  const shape = (o, tf) => {
    const n = o.n, shown = Math.max(0, Math.min(o.upto ?? n, n));
    const t0 = dayStart(o.day);
    const T = (b) => t0 + Math.floor(b / tf) * tf * 60;
    const bucket = (b) => Math.floor(b / tf) * tf;
    const hasOhlc = [o.open, o.high, o.low].every((a) => Array.isArray(a) && a.length >= n);
    const hasVol = Array.isArray(o.volume) && o.volume.length >= n;
    // minute stop level while a position is open (the stop in force during that minute)
    const stopAt = new Array(n).fill(null), inPos = new Array(n).fill(false);
    for (const t of o.trades || []) {
      if (t.x0 === null || t.x0 === undefined) continue;
      const a = Math.max(0, Math.floor(t.x0)), z = Math.min(n - 1, t.x1 === null || t.x1 === undefined ? n - 1 : Math.floor(t.x1));
      const st = (t.stops || []).filter((s) => s[1] !== null).sort((p, q) => p[0] - q[0]);
      for (let b = a; b <= z; b++) {
        inPos[b] = true;
        let v = null;
        for (const s of st) if (Math.floor(s[0]) <= b) v = s[1];
        stopAt[b] = v;
      }
    }
    const bars = [], line = { ub: [], lb: [], vwap: [], stop: [] }, rec = new Map();
    for (let s = 0; s < n; s += tf) {
      const time = T(s), e = Math.min(s + tf, n) - 1;
      line.ub.push(num(o.ub[e]) === null ? { time } : { time, value: num(o.ub[e]) });
      line.lb.push(num(o.lb[e]) === null ? { time } : { time, value: num(o.lb[e]) });
      const r = { s, e: Math.min(e, shown - 1), time, ub: num(o.ub[e]), lb: num(o.lb[e]) };
      rec.set(time, r);
      if (s >= shown) continue;
      let op = null, hi = -Infinity, lo = Infinity, cl = null, vol = 0, stop = null;
      for (let b = s; b <= r.e; b++) {
        const c = num(o.close[b]);
        if (c === null) continue;
        const oo = hasOhlc ? num(o.open[b]) ?? c : c;
        if (op === null) op = oo;
        hi = Math.max(hi, hasOhlc ? num(o.high[b]) ?? c : c);
        lo = Math.min(lo, hasOhlc ? num(o.low[b]) ?? c : c);
        cl = c;
        vol += hasVol ? num(o.volume[b]) || 0 : 0;
        if (inPos[b] && stopAt[b] !== null) stop = stopAt[b];
      }
      const vw = num(o.vwap[r.e]);
      line.vwap.push(vw === null ? { time } : { time, value: vw });
      line.stop.push(stop === null ? { time } : { time, value: stop });
      Object.assign(r, { vwap: vw, stop });
      if (cl === null) continue;
      Object.assign(r, { open: op, high: hi, low: lo, close: cl, volume: vol, shown: true });
      bars.push({ time, open: op, high: hi, low: lo, close: cl, volume: vol });
    }
    const lastShown = bars.length ? bars[bars.length - 1].time : null;
    const visible = (b) => lastShown !== null && T(b) <= lastShown && b < shown;
    return { n, shown, T, bucket, bars, line, rec, hasOhlc, hasVol, lastShown, visible };
  };

  // ---- the drawing ----
  const mount = (el) => {
    el.replaceChildren();
    const bar = document.createElement('div');
    bar.className = 'c3-tv-bar';
    const seg = document.createElement('div');
    seg.className = 'c3-seg';
    seg.setAttribute('role', 'group');
    seg.setAttribute('aria-label', 'Candle length');
    const tfBtns = [1, 5].map((tf) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = `${tf} min`;
      b.dataset.tf = String(tf);
      seg.append(b);
      return b;
    });
    const fit = document.createElement('button');
    fit.type = 'button';
    fit.className = 'c3-btn c3-tv-fit';
    fit.textContent = 'Whole day';
    const note = document.createElement('span');
    note.className = 'c3-note c3-tv-note';
    note.textContent = 'Scroll to zoom, drag to pan. Times are New York.';
    bar.append(seg, fit, note);
    const box = document.createElement('div');
    box.className = 'c3-tv-box';
    const legend = document.createElement('div');
    legend.className = 'c3-tv-legend';
    const msg = document.createElement('div');
    msg.className = 'c3-tv-msg';
    box.append(legend, msg);
    el.append(bar, box);
    const made = { bar, box, legend, msg, tfBtns, fit, chart: null, key: null, tf: null };
    tfBtns.forEach((b) => b.addEventListener('click', () => {
      C3.state.tf = +b.dataset.tf;
      VIEWS.forEach((w) => dash.draw(w.tv));
    }));
    fit.addEventListener('click', () => { if (made.chart) made.chart.timeScale().fitContent(); });
    made.repaint = () => { if (made.chart && made.update) made.update(); };      // keeps the zoom
    made.obs = new MutationObserver(made.repaint);
    made.obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme', 'style'] });
    made.mq = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;
    if (made.mq && made.mq.addEventListener) made.mq.addEventListener('change', made.repaint);
    return made;
  };

  const say = (made, text) => {
    made.msg.textContent = text;
    made.msg.hidden = !text;
    made.legend.hidden = !!text;
  };

  const ensureChart = (made, P) => {
    const LW = window.LightweightCharts;
    if (made.chart) return;
    const chart = LW.createChart(made.box, {
      autoSize: true,
      layout: { background: { type: LW.ColorType.Solid, color: 'rgba(0,0,0,0)' }, textColor: P.text, fontSize: 11, fontFamily: P.font,
        panes: { separatorColor: P.grid, separatorHoverColor: P.axis, enableResize: false } },
      grid: { vertLines: { color: P.grid }, horzLines: { color: P.grid } },
      crosshair: { mode: LW.CrosshairMode.Normal },
      rightPriceScale: { borderColor: P.axis, scaleMargins: { top: 0.12, bottom: 0.06 } },
      timeScale: { borderColor: P.axis, timeVisible: true, secondsVisible: false, rightOffset: 3, fixLeftEdge: true },
    });
    const pf = { type: 'price', precision: 2, minMove: 0.01 };
    const quiet = { priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false, priceFormat: pf };
    const S = {};
    S.checks = chart.addSeries(LW.HistogramSeries, { priceScaleId: 'checks', priceLineVisible: false, lastValueVisible: false, base: 0 });
    S.checks.priceScale().applyOptions({ scaleMargins: { top: 0, bottom: 0 }, visible: false });
    S.ub = chart.addSeries(LW.LineSeries, { ...quiet, lineWidth: 1 });
    S.lb = chart.addSeries(LW.LineSeries, { ...quiet, lineWidth: 1 });
    made.chart = chart;
    made.S = S;
    made.pf = pf;
    made.quiet = quiet;
    chart.subscribeCrosshairMove((param) => {
      const r = param && param.time !== undefined && made.view ? made.view.rec.get(param.time) : null;
      made.showLegend(r && r.shown ? r : made.lastRec);
    });
  };

  // price series: candles when the day has OHLC, else a close line; VWAP and the stop drawn on top
  const ensurePrice = (made, P, candles) => {
    const LW = window.LightweightCharts;
    const { chart, S } = made;
    if (made.priceKind !== (candles ? 'candles' : 'line')) {
      for (const k of ['price', 'vwap', 'stop']) if (S[k]) { chart.removeSeries(S[k]); S[k] = null; }
      S.price = candles
        ? chart.addSeries(LW.CandlestickSeries, { borderVisible: false, priceLineVisible: false, lastValueVisible: true, priceFormat: made.pf })
        : chart.addSeries(LW.LineSeries, { lineWidth: 2, priceLineVisible: false, lastValueVisible: true, priceFormat: made.pf });
      S.vwap = chart.addSeries(LW.LineSeries, { ...made.quiet, lineWidth: 2, lineStyle: LW.LineStyle.Dashed });
      S.stop = chart.addSeries(LW.LineSeries, { ...made.quiet, lineWidth: 2, lineType: LW.LineType.WithSteps, lineStyle: LW.LineStyle.SparseDotted });
      made.markers = LW.createSeriesMarkers(S.price, []);
      made.priceKind = candles ? 'candles' : 'line';
    }
    if (candles) S.price.applyOptions({ upColor: P.up, downColor: P.down, wickUpColor: P.up, wickDownColor: P.down });
    else S.price.applyOptions({ color: P.s(1) });
  };

  const ensureVolume = (made, P, on) => {
    const LW = window.LightweightCharts;
    if (on && !made.S.vol) {
      made.S.vol = made.chart.addSeries(LW.HistogramSeries, { priceFormat: { type: 'volume' }, priceLineVisible: false, lastValueVisible: false }, 1);
      const panes = made.chart.panes();
      if (panes[0]) panes[0].setStretchFactor(5);
      if (panes[1]) panes[1].setStretchFactor(1);
    }
    if (!on && made.S.vol) { made.chart.removeSeries(made.S.vol); made.S.vol = null; }
  };

  const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(2));
  const kfmt = (v) => (v >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : v >= 1e3 ? `${(v / 1e3).toFixed(0)}K` : String(Math.round(v)));

  const makeUpdate = (provider) => (el, made) => {
    made.update = () => update(el, made);
    update(el, made);
    function update() {
      made.tfBtns.forEach((b) => b.setAttribute('aria-pressed', +b.dataset.tf === C3.state.tf ? 'true' : 'false'));
      if (lib === 'loading') { say(made, 'Loading the TradingView chart library…'); return; }
      if (lib === 'failed' || !window.LightweightCharts) { say(made, 'The TradingView library could not load here; use “Line”.'); return; }
      const got = provider();
      if (got.wait) { say(made, got.wait); if (made.chart) { made.chart.remove(); made.chart = null; made.priceKind = null; made.key = null; } return; }
      const o = got.day;
      const P = palette(el);
      ensureChart(made, P);
      ensurePrice(made, P, [o.open, o.high, o.low].every((a) => Array.isArray(a) && a.length >= o.n));
      const tf = C3.state.tf;
      const v = shape(o, tf);
      made.view = v;
      ensureVolume(made, P, v.hasVol);
      const { chart, S } = made;
      chart.applyOptions({
        layout: { textColor: P.text, fontFamily: P.font, panes: { separatorColor: P.grid, separatorHoverColor: P.axis } },
        grid: { vertLines: { color: P.grid }, horzLines: { color: P.grid } },
        crosshair: { vertLine: { color: P.ref, labelBackgroundColor: P.fg }, horzLine: { color: P.ref, labelBackgroundColor: P.fg } },
        rightPriceScale: { borderColor: P.axis }, timeScale: { borderColor: P.axis },
      });
      S.ub.applyOptions({ color: P.ref });
      S.lb.applyOptions({ color: P.ref });
      S.vwap.applyOptions({ color: P.s(4) });
      S.stop.applyOptions({ color: P.s(0) });
      S.ub.setData(v.line.ub);
      S.lb.setData(v.line.lb);
      S.price.setData(made.priceKind === 'candles' ? v.bars.map(({ volume, ...b }) => b) : v.bars.map((b) => ({ time: b.time, value: b.close })));
      S.vwap.setData(v.line.vwap);
      S.stop.setData(v.line.stop);
      if (S.vol) S.vol.setData(v.bars.map((b) => ({ time: b.time, value: b.volume, color: b.close >= b.open ? P.upA : P.downA })));
      // the twelve checks: a faint column at each, the entering ones in C3's colour, the current one darker
      const seen = new Set();
      S.checks.setData((o.checks || []).map((c) => {
        const b = Math.floor(c.x);
        const time = v.T(b);
        if (b >= v.n || seen.has(time)) return null;
        seen.add(time);
        const color = o.hiX !== null && o.hiX !== undefined && Math.floor(o.hiX) === b ? toRgba(P.fg, 0.22)
          : c.action === 'enter' && b <= v.shown ? toRgba(P.s(0), 0.2) : toRgba(P.ref, 0.1);
        return { time, value: 1, color };
      }).filter(Boolean).sort((p, q) => p.time - q.time));
      // markers: entries, exits and (live) where the research rule entered
      const mk = [];
      for (const t of o.trades || []) {
        const b0 = Math.floor(t.x0);
        if (v.visible(b0)) mk.push({ time: v.T(b0), position: 'belowBar', shape: 'arrowUp', color: P.s(0), text: t.inText || 'Buy' });
        if (t.x1 !== null && t.x1 !== undefined) {
          const b1 = Math.min(v.n - 1, Math.floor(t.x1));
          if (v.visible(b1) && b1 < (o.upto ?? v.n)) mk.push({ time: v.T(b1), position: 'aboveBar', shape: 'arrowDown', color: P.fg, text: t.outText || 'Exit' });
        }
      }
      for (const g of o.ghosts || []) {
        const b = Math.floor(g.x);
        if (v.visible(b)) mk.push({ time: v.T(b), position: 'belowBar', shape: 'circle', color: P.s(3), text: 'research' });
      }
      made.markers.setMarkers(mk.sort((p, q) => p.time - q.time));
      // legend: OHLC, the band, VWAP and the stop at the hovered bar (the last bar shown otherwise)
      const checkAt = new Map((o.checks || []).map((c) => [Math.floor(c.x), c]));
      made.showLegend = (r) => {
        made.legend.replaceChildren();
        if (!r) return;
        const row = (parts, cls) => {
          const d = document.createElement('div');
          if (cls) d.className = cls;
          for (const [label, value, color] of parts) {
            const s = document.createElement('span');
            if (color) { const i = document.createElement('i'); i.className = 'c3-sw'; i.style.background = color; s.append(i); }
            s.append(document.createTextNode(`${label} `));
            const b = document.createElement('b');
            b.textContent = value;
            s.append(b);
            d.append(s);
          }
          made.legend.append(d);
        };
        const head = document.createElement('div');
        head.className = 'c3-tv-head';
        head.textContent = `SPY · ${tf} min · ${C3.dayLabel(o.day)} · ${hhmm(r.s)}–${hhmm(r.e + 1)} ET`;
        made.legend.append(head);
        if (r.shown) {
          const up = r.close >= r.open;
          row([['O', fmt(r.open)], ['H', fmt(r.high)], ['L', fmt(r.low)], ['C', fmt(r.close)],
            ...(v.hasVol ? [['Vol', kfmt(r.volume)]] : [])].map(([a, b], i) => (i === 3 ? [a, b, up ? P.up : P.down] : [a, b])));
        }
        row([['Upper', fmt(r.ub), P.ref], ['Lower', fmt(r.lb), P.ref], ...(r.shown ? [['VWAP', fmt(r.vwap), P.s(4)]] : []),
          ...(r.stop !== null && r.stop !== undefined && r.shown ? [['Stop', fmt(r.stop), P.s(0)]] : [])], 'is-extra');
        const checks = [];
        for (let b = r.s; b < r.s + tf; b++) if (checkAt.has(b) && b <= v.shown) checks.push(checkAt.get(b));
        if (checks.length) {
          const d = document.createElement('div');
          d.className = 'is-extra';
          d.textContent = checks.map((c) => `${hhmm(Math.floor(c.x))} check: ${C3.actionWord(c.action)}`).join(' · ');
          made.legend.append(d);
        }
      };
      made.lastRec = v.bars.length ? v.rec.get(v.bars[v.bars.length - 1].time) : null;
      say(made, v.bars.length ? '' : 'No bars yet at this step.');
      made.showLegend(made.lastRec);
      const key = `${o.key}|${tf}`;
      if (made.key !== key) {
        made.key = key;
        requestAnimationFrame(() => { if (made.chart) made.chart.timeScale().fitContent(); });
      }
      made.box.setAttribute('aria-label', got.label || `SPY on ${o.day}`);
      made.box.setAttribute('role', 'img');
    }
  };

  const unmount = (el, made) => {
    if (made.obs) made.obs.disconnect();
    if (made.mq && made.mq.removeEventListener) made.mq.removeEventListener('change', made.repaint);
    if (made.chart) made.chart.remove();
    made.chart = null;
  };

  dash.draw('ex-tv', {
    mount,
    update: makeUpdate(() => {
      const d = dash.data('dev_examples');
      if (d.status === 'loading') return { wait: 'Loading the example days…' };
      if (d.status !== 'ok') return { wait: 'The example days could not load; Sources shows the error.' };
      const o = C3.exDay();
      if (!o) return { wait: 'No example days.' };
      return { day: o, label: `SPY candles on ${o.day} with the noise area, VWAP, the checks and C3's trades, up to ${C3.hhmm(o.upto)} ET` };
    }),
    unmount,
  });

  dash.draw('replay-tv', {
    mount,
    update: makeUpdate(() => {
      for (const id of ['live_intraday', 'live_checks', 'live_trades', 'live_days']) {
        const d = dash.data(id);
        if (d.status === 'loading') return { wait: 'Loading the paper sessions…' };
        if (d.status !== 'ok') return { wait: 'The paper sessions are not available yet (Sources shows why).' };
      }
      const day = C3.liveSelectedDay();
      if (!day) return { wait: 'No paper session yet.' };
      const o = C3.liveDayObj(day);
      if (!o) return { wait: `No minute bars for ${day}: the replay keeps the last 40 sessions.` };
      return { day: o, label: `SPY candles on ${day} with the noise area, VWAP, the checks and the paper trades` };
    }),
    unmount,
  });

  applyView();
})();
