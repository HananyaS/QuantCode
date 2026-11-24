// C3 dashboard core: shared state, formatting, the generic value drawing, the phase switch,
// chip tones, legend swatches, tabs and the print layout. Charts live in c3-charts.js.
(function () {
  const C3 = (window.C3 = window.C3 || {});
  C3.state = C3.state || { ex: 'held', step: 12, liveDay: null, devYear: '', checkView: 'contrib' };
  const ET = 'America/New_York';

  const num = (v) => (v === null || v === undefined || v === '' || !Number.isFinite(+v) ? null : +v);
  const arr = (v) => {
    if (Array.isArray(v)) return v;
    if (typeof v === 'string' && v.trim().startsWith('[')) { try { return JSON.parse(v); } catch (e) { return []; } }
    return [];
  };
  const bool = (v) => v === true || v === 'true' || v === 1 || v === '1';
  C3.num = num; C3.arr = arr; C3.bool = bool;
  C3.color = (i) => (dash.colors && dash.colors.length ? dash.colors[Math.min(i, dash.colors.length - 1)] : 'currentColor');

  // ---- formatting -------------------------------------------------------------------------------
  const dfmt = (opts) => new Intl.DateTimeFormat('en-GB', Object.assign({ timeZone: ET }, opts));
  const etDay = dfmt({ weekday: 'short', day: 'numeric', month: 'short' });
  const etTime = dfmt({ hour: '2-digit', minute: '2-digit', hour12: false });
  const localTime = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
  const parseDay = d3.utcParse('%Y-%m-%d');
  C3.parseDay = parseDay;
  C3.dayLabel = (s) => { const d = parseDay(String(s)); return d ? d3.utcFormat('%a %-d %b %Y')(d) : String(s); };
  C3.dayShort = (s) => { const d = parseDay(String(s)); return d ? d3.utcFormat('%-d %b')(d) : String(s); };
  const ago = (ms) => {
    const m = Math.round(ms / 60000);
    if (m < 2) return 'just now';
    if (m < 90) return `${m} min ago`;
    const h = Math.round(m / 60);
    return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
  };
  C3.ago = ago;
  const F = {
    text: (v) => String(v),
    int: (v) => d3.format(',d')(Math.round(v)),
    f2: (v) => d3.format('.2f')(v),
    f3: (v) => d3.format('.3f')(v),
    sf2: (v) => d3.format('+.2f')(v),
    pct0: (v) => d3.format('.0%')(v),
    pct1: (v) => d3.format('.1%')(v),
    spct1: (v) => d3.format('+.1%')(v),
    spct2: (v) => d3.format('+.2%')(v),
    usd0: (v) => d3.format('$,.0f')(v),
    usd2: (v) => d3.format('$,.2f')(v),
    susd: (v) => (v < 0 ? '−' : '+') + d3.format('$,.2f')(Math.abs(v)),
    x2: (v) => d3.format('.2f')(v) + '×',
    date: (v) => C3.dayLabel(v),
    when: (v) => {
      const t = new Date(v);
      if (isNaN(t)) return String(v);
      const loc = localTime.format(t), et = etTime.format(t);
      return `${etDay.format(t)}, ${et} ET` + (loc !== et ? ` (${loc} your time)` : '');
    },
    ago: (v) => {
      const t = new Date(v);
      if (isNaN(t)) return String(v);
      return `${etDay.format(t)} ${etTime.format(t)} ET, ${ago(Date.now() - t)}`;
    },
    phase: (v) => ({ waiting: 'waiting for the first session', running: 'trading on schedule', stopped: 'stopped by the KILL file' }[v] || String(v)),
    gate: (v) => String(v).toLowerCase() === 'pending' ? 'pending' : String(v).toLowerCase(),
  };
  const NUMERIC = new Set(['int', 'f2', 'f3', 'sf2', 'pct0', 'pct1', 'spct1', 'spct2', 'usd0', 'usd2', 'susd', 'x2']);
  C3.fmt = (kind, v) => {
    if (v === null || v === undefined || v === '') return '—';
    if (NUMERIC.has(kind)) { const n = num(v); return n === null ? '—' : F[kind](n); }
    return (F[kind] || F.text)(v);
  };
  C3.tone = (s) => {
    const t = String(s || '').toLowerCase();
    if (t.includes('fail') || t === 'stopped') return 'is-bad';
    if (t.includes('pass') || t === 'running') return 'is-ok';
    if (t.includes('extend') || t.includes('inconclusive')) return 'is-warn';
    return '';
  };

  // ---- picking a row the way a mark names it ----------------------------------------------------------
  C3.pick = (rows, el) => {
    if (!Array.isArray(rows)) return null;
    if (el.dataset.row !== undefined) return rows[+el.dataset.row] || null;
    if (el.dataset.where) {
      const conds = el.dataset.where.split('&').map((kv) => kv.split('=').map(decodeURIComponent));
      return rows.find((r) => conds.every(([k, v]) => String(r[k]) === v)) || null;
    }
    return rows[0] || null;
  };

  // One drawing for every single value on the page: its marks say where it comes from,
  // data-fmt how it reads, data-tone="sign" colours it by whether it is good.
  dash.draw('val', (el) => {
    const d = dash.data(el.dataset.source);
    el.classList.toggle('dash-skeleton', d.status === 'loading');
    if (d.status !== 'ok') { el.textContent = '—'; return; }
    const v = el.dataset.field === undefined ? d.data : (C3.pick(d.data, el) || {})[el.dataset.field];
    el.textContent = C3.fmt(el.dataset.fmt || 'text', v);
    if (el.dataset.tone === 'sign') {
      const n = num(v);
      el.classList.toggle('c3-good', n !== null && n > 0);
      el.classList.toggle('c3-bad', n !== null && n < 0);
    }
    if (el.classList.contains('c3-pill')) {
      el.classList.remove('is-ok', 'is-bad', 'is-warn');
      const t = C3.tone(v); if (t) el.classList.add(t);
    }
  });

  // Which of the phase sentences and blocks show: waiting, running, stopped, loading or offline.
  C3.phase = () => {
    const d = dash.data('live_status');
    if (d.status === 'loading') return 'loading';
    if (d.status !== 'ok' || !d.data[0]) return 'offline';
    const s = d.data[0];
    if (s.phase === 'stopped') return 'stopped';
    return num(s.sessions) > 0 ? 'running' : 'waiting';
  };
  dash.draw('phase', () => {
    const p = C3.phase();
    document.querySelectorAll('#dash-root [data-phase], .c3 [data-phase]').forEach((n) => {
      n.hidden = !n.dataset.phase.split(/\s+/).includes(p);
    });
  });

  // Countdown to the next session, ticking inside its own drawing.
  dash.draw('countdown', {
    mount: (el) => ({ timer: null }),
    update: (el, made) => {
      const tick = () => {
        const d = dash.data('live_status');
        if (d.status !== 'ok' || !d.data[0] || !d.data[0].next_open) { el.textContent = '—'; return; }
        const open = new Date(d.data[0].next_open), now = Date.now();
        const ms = open - now;
        if (ms > 0) {
          const h = Math.floor(ms / 3600000), m = Math.floor((ms % 3600000) / 60000);
          el.textContent = `The session opens in ${h ? h + ' h ' : ''}${m} min. The server starts at 08:00 ET.`;
        } else if (-ms < 6.85 * 3600000) {
          el.textContent = 'The session is under way. Its results arrive here after the 16:20 ET report.';
        } else {
          el.textContent = 'That session has closed; this page updates when the server publishes its evening file.';
        }
      };
      tick();
      clearInterval(made.timer);
      made.timer = setInterval(tick, 30000);
    },
    unmount: (el, made) => clearInterval(made.timer),
  });

  // Chip tones and legend swatches: class and style only, outside any drawing.
  dash.onData(() => {
    document.querySelectorAll('.c3 .c3-sw[data-color]').forEach((sw) => {
      const c = C3.color(+sw.dataset.color);
      if (sw.classList.contains('is-dash') || sw.classList.contains('is-tri')) sw.style.color = c;
      else sw.style.background = c;
    });
    document.querySelectorAll('.c3 .c3-sw[data-muted]').forEach((sw) => {
      if (sw.classList.contains('is-dash')) sw.style.color = 'var(--cds-chart-reference)';
      else sw.style.background = 'var(--cds-chart-muted)';
    });
    const d = dash.data('live_status');
    const s = d.status === 'ok' && d.data[0] ? d.data[0] : null;
    document.querySelectorAll('.c3 .c3-chip[data-chip]').forEach((chip) => {
      chip.classList.remove('is-ok', 'is-bad', 'is-warn');
      if (!s) return;
      const k = chip.dataset.chip;
      let t = '';
      if (k === 'phase') t = s.phase === 'stopped' ? 'is-bad' : 'is-ok';
      else if (k === 'fresh') {
        const age = (Date.now() - new Date(s.updated)) / 3600000;
        t = age > 30 ? 'is-warn' : 'is-ok';                     // a weekday file is never older than a day and a bit
      } else if (k.startsWith('g_')) t = C3.tone(s[k]);
      if (t) chip.classList.add(t);
    });
  });

  // ---- tabs -----------------------------------------------------------------------------------------
  const tabs = () => Array.from(document.querySelectorAll('.c3 .c3-tab[aria-controls]'));
  C3.openTab = (id, focus) => {
    const all = tabs();
    if (!all.some((t) => t.getAttribute('aria-controls') === id)) return;
    all.forEach((t) => {
      const on = t.getAttribute('aria-controls') === id;
      t.setAttribute('aria-selected', on ? 'true' : 'false');
      t.tabIndex = on ? 0 : -1;
      const pane = document.getElementById(t.getAttribute('aria-controls'));
      if (pane) pane.hidden = !on;
      if (on && focus) t.focus();
    });
    try { dash.setLink(id); } catch (e) { /* links are a convenience */ }
  };
  tabs().forEach((t, i, all) => {
    t.addEventListener('click', () => C3.openTab(t.getAttribute('aria-controls')));
    t.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const j = (i + (e.key === 'ArrowRight' ? 1 : all.length - 1)) % all.length;
      C3.openTab(all[j].getAttribute('aria-controls'), true);
    });
  });
  const start = (location.hash || '').slice(1);
  if (start) C3.openTab(start);

  // Print every tab, each on its own page; put the tabs back afterwards.
  let open = null;
  window.addEventListener('beforeprint', () => {
    if (open === null) open = (tabs().find((t) => t.getAttribute('aria-selected') === 'true') || tabs()[0]).getAttribute('aria-controls');
    document.querySelectorAll('.c3 .c3-pane').forEach((p) => { p.hidden = false; });
  });
  window.addEventListener('afterprint', () => { if (open) C3.openTab(open); open = null; });
})();
