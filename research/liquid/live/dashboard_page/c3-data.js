// Loaders (each stands alone, as Sources shows its text) and calculations.
dash.loader('live_status', {
  description: 'The server writes c3_live.json as gzip+base64 JSON in a small envelope (Drive may escape its text as Markdown); this decodes it and takes the status table.',
  fn: async ({answer, items}) => {
    const pieces = items && items.length ? items : [answer];
    let text = '';
    for (const p of pieces) {
      const t = typeof p === 'string' ? p : p && (p.fileContent ?? p.content ?? p.text);
      if (typeof t === 'string' && t.includes('data')) { text = t; break; }
    }
    if (!text) throw new Error('No file text in the Drive answer: ' + JSON.stringify(answer).slice(0, 160));
    const m = text.split(String.fromCharCode(92)).join('').match(/"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"/);
    if (!m) throw new Error('c3_live.json holds no data field: ' + text.slice(0, 160));
    const bin = Uint8Array.from(atob(m[1].replace(/\s+/g, '')), (c) => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    const doc = JSON.parse(await new Response(stream).text());
    const t = doc['status'];
    if (!t || !Array.isArray(t.rows)) throw new Error('c3_live.json has no status table');
    const st = doc.status && doc.status.rows && doc.status.rows[0];
    const meta = st ? [{label: 'Written by the server', value: String(st.updated)}, {label: 'Code', value: String(st.code_sha) + (st.code_dirty ? ' (uncommitted changes)' : '')}] : [];
    return {columns: t.columns, data: t.rows, meta};
  },
});
dash.loader('live_days', {
  description: 'The server writes c3_live.json as gzip+base64 JSON in a small envelope (Drive may escape its text as Markdown); this decodes it and takes the days table.',
  fn: async ({answer, items}) => {
    const pieces = items && items.length ? items : [answer];
    let text = '';
    for (const p of pieces) {
      const t = typeof p === 'string' ? p : p && (p.fileContent ?? p.content ?? p.text);
      if (typeof t === 'string' && t.includes('data')) { text = t; break; }
    }
    if (!text) throw new Error('No file text in the Drive answer: ' + JSON.stringify(answer).slice(0, 160));
    const m = text.split(String.fromCharCode(92)).join('').match(/"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"/);
    if (!m) throw new Error('c3_live.json holds no data field: ' + text.slice(0, 160));
    const bin = Uint8Array.from(atob(m[1].replace(/\s+/g, '')), (c) => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    const doc = JSON.parse(await new Response(stream).text());
    const t = doc['days'];
    if (!t || !Array.isArray(t.rows)) throw new Error('c3_live.json has no days table');
    const st = doc.status && doc.status.rows && doc.status.rows[0];
    const meta = st ? [{label: 'Written by the server', value: String(st.updated)}, {label: 'Code', value: String(st.code_sha) + (st.code_dirty ? ' (uncommitted changes)' : '')}] : [];
    return {columns: t.columns, data: t.rows, meta};
  },
});
dash.loader('live_trades', {
  description: 'The server writes c3_live.json as gzip+base64 JSON in a small envelope (Drive may escape its text as Markdown); this decodes it and takes the trades table.',
  fn: async ({answer, items}) => {
    const pieces = items && items.length ? items : [answer];
    let text = '';
    for (const p of pieces) {
      const t = typeof p === 'string' ? p : p && (p.fileContent ?? p.content ?? p.text);
      if (typeof t === 'string' && t.includes('data')) { text = t; break; }
    }
    if (!text) throw new Error('No file text in the Drive answer: ' + JSON.stringify(answer).slice(0, 160));
    const m = text.split(String.fromCharCode(92)).join('').match(/"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"/);
    if (!m) throw new Error('c3_live.json holds no data field: ' + text.slice(0, 160));
    const bin = Uint8Array.from(atob(m[1].replace(/\s+/g, '')), (c) => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    const doc = JSON.parse(await new Response(stream).text());
    const t = doc['trades'];
    if (!t || !Array.isArray(t.rows)) throw new Error('c3_live.json has no trades table');
    const st = doc.status && doc.status.rows && doc.status.rows[0];
    const meta = st ? [{label: 'Written by the server', value: String(st.updated)}, {label: 'Code', value: String(st.code_sha) + (st.code_dirty ? ' (uncommitted changes)' : '')}] : [];
    return {columns: t.columns, data: t.rows, meta};
  },
});
dash.loader('live_checks', {
  description: 'The server writes c3_live.json as gzip+base64 JSON in a small envelope (Drive may escape its text as Markdown); this decodes it and takes the checks table.',
  fn: async ({answer, items}) => {
    const pieces = items && items.length ? items : [answer];
    let text = '';
    for (const p of pieces) {
      const t = typeof p === 'string' ? p : p && (p.fileContent ?? p.content ?? p.text);
      if (typeof t === 'string' && t.includes('data')) { text = t; break; }
    }
    if (!text) throw new Error('No file text in the Drive answer: ' + JSON.stringify(answer).slice(0, 160));
    const m = text.split(String.fromCharCode(92)).join('').match(/"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"/);
    if (!m) throw new Error('c3_live.json holds no data field: ' + text.slice(0, 160));
    const bin = Uint8Array.from(atob(m[1].replace(/\s+/g, '')), (c) => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    const doc = JSON.parse(await new Response(stream).text());
    const t = doc['checks'];
    if (!t || !Array.isArray(t.rows)) throw new Error('c3_live.json has no checks table');
    const st = doc.status && doc.status.rows && doc.status.rows[0];
    const meta = st ? [{label: 'Written by the server', value: String(st.updated)}, {label: 'Code', value: String(st.code_sha) + (st.code_dirty ? ' (uncommitted changes)' : '')}] : [];
    return {columns: t.columns, data: t.rows, meta};
  },
});
dash.loader('live_intraday', {
  description: 'The server writes c3_intraday.json as gzip+base64 JSON in a small envelope (Drive may escape its text as Markdown); this decodes it and takes the intraday table.',
  fn: async ({answer, items}) => {
    const pieces = items && items.length ? items : [answer];
    let text = '';
    for (const p of pieces) {
      const t = typeof p === 'string' ? p : p && (p.fileContent ?? p.content ?? p.text);
      if (typeof t === 'string' && t.includes('data')) { text = t; break; }
    }
    if (!text) throw new Error('No file text in the Drive answer: ' + JSON.stringify(answer).slice(0, 160));
    const m = text.split(String.fromCharCode(92)).join('').match(/"data"\s*:\s*"([A-Za-z0-9+/=\s]+)"/);
    if (!m) throw new Error('c3_intraday.json holds no data field: ' + text.slice(0, 160));
    const bin = Uint8Array.from(atob(m[1].replace(/\s+/g, '')), (c) => c.charCodeAt(0));
    const stream = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
    const doc = JSON.parse(await new Response(stream).text());
    const t = doc['intraday'];
    if (!t || !Array.isArray(t.rows)) throw new Error('c3_intraday.json has no intraday table');
    const st = doc.status && doc.status.rows && doc.status.rows[0];
    const meta = st ? [{label: 'Written by the server', value: String(st.updated)}, {label: 'Code', value: String(st.code_sha) + (st.code_dirty ? ' (uncommitted changes)' : '')}] : [];
    return {columns: t.columns, data: t.rows, meta};
  },
});
dash.loader('dev_daily', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_summary', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_c3', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_years', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_trades', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_costcurve', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});
dash.loader('dev_examples', {
  description: 'dashboard.py --dev writes {columns, rows, meta}; this keeps the column order it names.',
  fn: ({answer}) => ({columns: answer.columns, data: answer.rows}),
});

dash.calc('live_excess_spy', {
  inputs: ['live_status'],
  title: 'Paper return minus SPY',
  description: 'Compounded paper return less SPY compounded over the same sessions.',
  fn: (status) => {
    const s = status && status[0];
    if (!s || s.ret_total === null || s.ret_total === undefined || s.spy_total === null || s.spy_total === undefined) return null;
    return Number(s.ret_total) - Number(s.spy_total);
  },
});

dash.calc('dev_by_exit', {
  inputs: ['dev_trades'],
  title: 'Research round trips by exit',
  description: 'Count, share, win rate, mean gross and net bp and mean minutes held, per way out, 2016-2022.',
  fn: (trades) => {
    const n = trades.length;
    const names = {stop: 'Stop', close: 'Close (15:59)', target: 'Target'};
    return d3.rollups(trades, (g) => ({
      trades: g.length,
      share: g.length / n,
      win_rate: d3.mean(g, (t) => (Number(t.net_bp) > 0 ? 1 : 0)),
      mean_gross_bp: d3.mean(g, (t) => Number(t.gross_bp)),
      mean_net_bp: d3.mean(g, (t) => Number(t.net_bp)),
      mean_hold_min: d3.mean(g, (t) => Number(t.hold_min)),
    }), (t) => t.exit)
      .map(([exit, v]) => ({exit: names[exit] || exit, ...v}))
      .sort((a, b) => b.trades - a.trades);
  },
});

dash.calc('dev_by_check', {
  inputs: ['dev_trades'],
  title: 'Research round trips by check time',
  description: 'Per entry check: round trips, win rate, mean net bp, and return contributed (weight x net, summed over 2016-2022).',
  fn: (trades) => d3.rollups(trades, (g) => ({
    trades: g.length,
    win_rate: d3.mean(g, (t) => (Number(t.net_bp) > 0 ? 1 : 0)),
    mean_net_bp: d3.mean(g, (t) => Number(t.net_bp)),
    contrib: d3.sum(g, (t) => Number(t.contrib_bp)) / 1e4,
  }), (t) => t.check)
    .map(([check, v]) => ({check, ...v}))
    .sort((a, b) => d3.ascending(a.check, b.check)),
});
