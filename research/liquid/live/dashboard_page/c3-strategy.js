// "How it trades": the example-day stepper over dev_examples (four 2021-2022 days picked by rule).
(function () {
  const C3 = window.C3;
  const num = C3.num;
  const f2 = (v) => C3.fmt('f2', v);
  const LAST = 12;                               // step 12 = the close; steps 0-11 are the twelve checks

  const example = () => {
    const d = dash.data('dev_examples');
    if (d.status !== 'ok') return null;
    return d.data.find((r) => r.key === C3.state.ex) || d.data[0] || null;
  };
  const parts = (ex) => {
    const checks = C3.arr(ex.checks), trades = C3.arr(ex.trades);
    const n = num(ex.n_bars);
    return { n, checks, trades };
  };
  // minutes after 09:30 that a step reveals, and the check it highlights
  const reveal = (ex) => {
    const { n, checks } = parts(ex);
    const s = Math.min(C3.state.step, LAST);
    if (s >= checks.length) return { upto: n, hiX: null, check: null };
    const c = checks[s];
    return { upto: num(c.d) + 1, hiX: num(c.d) + 1, check: c };
  };

  const redraw = () => ['ex-chart', 'ex-tv', 'ex-steps', 'ex-where', 'ex-head'].forEach((nm) => dash.draw(nm));
  const range = document.querySelector('.c3 #ex-range');
  const setStep = (s) => {
    C3.state.step = Math.max(0, Math.min(LAST, s));
    if (range) range.value = String(C3.state.step);
    const prev = document.querySelector('.c3 #ex-prev'), next = document.querySelector('.c3 #ex-next');
    if (prev) prev.disabled = C3.state.step === 0;
    if (next) next.disabled = C3.state.step === LAST;
    redraw();
  };
  if (range) range.addEventListener('input', () => setStep(+range.value));
  const prevB = document.querySelector('.c3 #ex-prev'), nextB = document.querySelector('.c3 #ex-next');
  if (prevB) prevB.addEventListener('click', () => setStep(C3.state.step - 1));
  if (nextB) nextB.addEventListener('click', () => setStep(C3.state.step + 1));
  if (nextB) nextB.disabled = true;
  document.querySelectorAll('.c3 #example-pick button[data-ex]').forEach((b, i, all) => {
    b.addEventListener('click', () => {
      C3.state.ex = b.dataset.ex;
      all.forEach((x) => x.setAttribute('aria-pressed', x === b ? 'true' : 'false'));
      setStep(0);
    });
  });

  dash.draw('ex-head', (el) => {
    el.replaceChildren();
    const ex = example();
    if (!ex) { el.textContent = '—'; return; }
    const p1 = document.createElement('p');
    const b = document.createElement('b');
    b.textContent = `${C3.dayLabel(ex.day)}. `;
    const nt = num(ex.n_trades);
    p1.append(b, document.createTextNode(
      `SPY closed ${C3.fmt('spct2', ex.day_ret)} on the day; C3 made ${C3.fmt('spct2', ex.c3_ret)} on equity with ` +
      `${nt === 0 ? 'no trade' : nt === 1 ? 'one round trip' : `${nt} round trips`}.`));
    const p2 = document.createElement('p');
    p2.className = 'c3-note';
    p2.textContent = `Picked as: ${ex.rule}.`;
    el.append(p1, p2);
  });

  dash.draw('ex-where', (el) => {
    const ex = example();
    if (!ex) { el.textContent = '—'; return; }
    const { check } = reveal(ex);
    el.textContent = check ? `Check ${C3.state.step + 1} of ${parts(ex).checks.length}, ${check.time} ET` : 'The close, 16:00 ET: the whole day';
  });

  // The example day as one object (minutes after 09:30 on x), for the line chart and the TradingView chart.
  C3.exDay = () => {
    const ex = example();
    if (!ex) return null;
    const { n, checks, trades } = parts(ex);
    const { upto, hiX } = reveal(ex);
    const tr = trades.map((t) => {
      const x0 = num(t.entry_bar), x1 = t.exit_kind === 'close' ? n : num(t.exit_bar) + 0.5;
      const stops = checks.filter((c) => (c.action === 'enter' || c.action === 'hold') && num(c.d) + 1 >= x0 && num(c.d) + 1 < x1 && num(c.stop) !== null)
        .map((c) => [num(c.d) + 1, num(c.stop)]);
      return { x0, p0: num(t.entry), x1, p1: num(t.exit), stops,
        inText: `Buy ${t.check}`, outText: t.exit_kind === 'close' ? 'Close' : 'Stop' };
    });
    return {
      key: `ex-${ex.key}`, day: ex.day, n, upto, hiX,
      open: C3.arr(ex.open), high: C3.arr(ex.high), low: C3.arr(ex.low), close: C3.arr(ex.close), volume: C3.arr(ex.volume),
      ub: C3.arr(ex.ub), lb: C3.arr(ex.lb), vwap: C3.arr(ex.vwap),
      checks: checks.map((c) => ({ x: num(c.d) + 1, action: c.action })), trades: tr, ghosts: [],
    };
  };

  dash.draw('ex-chart', (el) => {
    if (C3.notReady(el, 'dev_examples')) return;
    const o = C3.exDay();
    if (!o) { C3.empty(el, 'No example days.'); return; }
    C3.dayChart(el, { ...o, label: `SPY on ${o.day}: minute closes, the noise area, VWAP and C3's trades, revealed up to ${C3.hhmm(o.upto)} ET` });
  });

  dash.draw('ex-steps', (ol) => {
    ol.replaceChildren();
    const ex = example();
    if (!ex) return;
    const { n, checks, trades } = parts(ex);
    const { upto } = reveal(ex);
    const items = checks.map((c) => {
      let say;
      if (c.action === 'enter') {
        const t = trades.find((x) => x.check === c.time);
        say = `Close ${f2(c.close)} is above the upper bound ${f2(c.ub)} and above VWAP ${f2(c.vwap)}. Buy ${t ? C3.fmt('x2', t.weight) : ''} equity with a limit 0.1% above; the first stop is ${f2(c.stop)}.`;
      } else if (c.action === 'hold') say = `Still long, close ${f2(c.close)}. The stop moves to max(upper bound, VWAP) = ${f2(c.stop)}.`;
      else if (c.action === 'inside') say = `Close ${f2(c.close)} is inside the noise area (upper bound ${f2(c.ub)}). No signal.`;
      else if (c.action === 'below_vwap') say = `Close ${f2(c.close)} is above the upper bound ${f2(c.ub)} but below VWAP ${f2(c.vwap)}. No entry.`;
      else say = `A signal at ${f2(c.close)}, but the limit order did not fill.`;
      return { x: num(c.d) + 1, time: c.time, tag: C3.actionWord(c.action), say, check: true };
    });
    for (const t of trades) {
      const x1 = t.exit_kind === 'close' ? n : num(t.exit_bar) + 0.5;
      const say = t.exit_kind === 'close'
        ? `Sold the rest at ${f2(t.exit)} for the close: ${C3.fmt('sf2', t.net_bp)} bp after costs on the round trip.`
        : `The price fell through the stop: sold at ${f2(t.exit)}, ${C3.fmt('sf2', t.net_bp)} bp after costs.`;
      items.push({ x: x1, time: t.exit_kind === 'close' ? '15:59' : C3.hhmm(num(t.exit_bar)), tag: t.exit_kind === 'close' ? 'sold, close' : 'stopped', say });
    }
    if (!trades.length) {
      items.push({ x: n, time: '16:00', tag: 'no trade', say: `No trade today. Long only, C3 ignores breaks below the lower bound; SPY closed ${C3.fmt('spct2', ex.day_ret)}.` });
    }
    items.sort((a, b) => a.x - b.x);
    const shown = items.filter((it) => it.x <= upto + 1e-9);
    shown.forEach((it, i) => {
      const li = document.createElement('li');
      if (i === shown.length - 1) li.classList.add('is-on');
      const tm = document.createElement('time');
      tm.textContent = it.time;
      const sp = document.createElement('span');
      const tag = document.createElement('span');
      tag.className = 'c3-tag';
      tag.textContent = it.tag;
      sp.append(tag, document.createTextNode(it.say));
      li.append(tm, sp);
      ol.appendChild(li);
    });
  });
})();
