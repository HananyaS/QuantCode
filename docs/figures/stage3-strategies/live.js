// Live layer of the Stage 3 trade recorder. Runs after the kit, the lathe (G) and the page data
// (CFG, DATA, ARMS, PLATES, ...). Every moving value comes from a real dev trade in DATA.
const stage = document.querySelector(".sr-stage");
const svg = stage.querySelector("svg");
const $ = (sel) => svg.querySelector(sel);
const part = (name) => $(`[data-part="${name}"]`);
const reduce = matchMedia("(prefers-reduced-motion: reduce)");
const readout = document.querySelector("[data-readout]");
const chips = [...document.querySelectorAll(".sr-chip")];
const field = (name) => document.querySelector(`[data-sr="${name}"]`);

const E = {
  gantry: part("gantry"), car: part("carriage"), tp: part("tp-collar"), sl: part("sl-collar"),
  fence: part("fence"), follower: part("follower"), trigger: part("trigger"),
  needle: part("dial-needle"), lever: part("mode-lever"), range: part("range-pointer"),
  blade: part("blade"), knob: part("knob-pointer"), colL: part("col-l"), colR: part("col-r"),
  panL: part("pan-l"), panR: part("pan-r"), bal: part("bal-needle"), paper: part("paper-ink"),
  pre: $('[data-trace="pre"]'), trade: $('[data-trace="trade"]'),
  lv: Object.fromEntries(["tp", "sl", "low", "high"].map((n) => [n, $(`[data-level="${n}"]`)])),
  dot: Object.fromEntries([...svg.querySelectorAll("[data-dot]")].map((c) => [c.dataset.dot, c])),
  pins: [...svg.querySelectorAll("[data-pin]")],
};

const { ZE, HALF, KX, PRE, XE, XS, PY, PZ0, PZ1, CAR_HALF, COLLAR_H, PARK_TP, PARK_SL } = CFG;
const PEN_SPEED = 14; // chart minutes per second
const SET = 1.1, TRIG = 0.8, HOLD = 2.0, FADE = 0.35;
const PER_ARM = 2; // trades per strategy on the idle tour
const IDLE_BACK = 45; // seconds without input before the tour resumes

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = (t) => { const u = clamp(t, 0, 1); return u * u * (3 - 2 * u); };
const lerp = (a, b, t) => a + (b - a) * t;
const pt = (p) => at(p);
const move = (el, by) => el.setAttribute("transform", translateAlong(P, by));
const fmt = (v, nd = 2, sign = true) => (v == null ? "—" : `${v < 0 ? "−" : sign && v > 0 ? "+" : ""}${Math.abs(v).toFixed(nd)}`);
const clock = (m) => { const total = 9 * 60 + 30 + m; return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`; };
const CLS = { TP: "take-profit", SL: "stop-loss", TIME: "time exit" };

// ------------------------------------------------------------ a trade, as points in chart time
// tau is minutes from the entry open (bar t+1 opens at tau 0); values are basis points of entry.
function barSeq(s, i) {
  const o = s.o[i], h = s.h[i], l = s.l[i], c = s.cl[i];
  return c > o ? [o, l, h, c] : [o, h, l, c];
}
function tradeOf(s) {
  const pre = [], trade = [];
  let entry = null;
  for (let i = 0; i < s.cl.length; i++) {
    const m = s.m0 + i;
    if (s.o[i] == null || s.cl[i] == null) continue;
    const tau0 = m - s.t - 1;
    if (m <= s.t) {
      if (m >= s.t - PRE) barSeq(s, i).forEach((v, j) => pre.push([tau0 + j / 3, v]));
      continue;
    }
    if (entry == null) { entry = [tau0, s.o[i]]; pre.push(entry); trade.push(entry); }
    if (m > s.exit) break;
    const seq = barSeq(s, i);
    if (m < s.exit) { seq.slice(1).forEach((v, j) => trade.push([tau0 + (j + 1) / 3, v])); continue; }
    // the exit bar: stop where the bracket fills, or at the open for a time exit
    if (s.cls === "TIME") { if (tau0 > entry[0]) trade.push([tau0, s.o[i]]); trade.push([tau0, s.xr]); break; }
    const hit = s.cls === "TP" ? (v) => v >= s.u : (v) => v <= -s.d;
    const level = s.cls === "TP" ? s.u : -s.d;
    if (hit(seq[0])) { trade.push([tau0, s.cls === "TP" ? s.u : s.xr]); break; }
    let done = false;
    for (let j = 1; j < 4 && !done; j++) {
      if (hit(seq[j])) {
        const f = (level - seq[j - 1]) / (seq[j] - seq[j - 1] || 1);
        trade.push([tau0 + (j - 1 + clamp(f, 0, 1)) / 3, level]);
        if (s.cls === "SL" && s.xr < level) trade.push([tau0 + (j - 1 + clamp(f, 0, 1)) / 3, s.xr]);
        done = true;
      } else trade.push([tau0 + j / 3, seq[j]]);
    }
    if (!done) trade.push([trade[trade.length - 1][0], s.cls === "TP" ? s.u : s.xr]); // a touch inside a tick
    break;
  }
  return { pre, trade, end: trade[trade.length - 1][0] };
}
function valueAt(points, tau) {
  if (tau <= points[0][0]) return points[0][1];
  for (let i = 1; i < points.length; i++) {
    const [t1, v1] = points[i];
    if (t1 >= tau) {
      const [t0, v0] = points[i - 1];
      return t1 === t0 ? v1 : lerp(v0, v1, (tau - t0) / (t1 - t0));
    }
  }
  return points[points.length - 1][1];
}
function upTo(points, tau) {
  const out = [];
  for (const p of points) {
    if (p[0] > tau) { out.push([tau, valueAt(points, tau)]); return out; }
    out.push(p);
  }
  return out;
}

// ------------------------------------------------------------ state
const st = {
  arm: 0, sample: 0, played: 0, touring: !reduce.matches, lastInput: -1e9, phase: "set", phaseT: 0,
  next: { arm: 0, sample: 0 }, cur: null, R: 10, x: XS, penZ: ZE,
  fenceX: XE + 30 * KX, follow: 0, rangeA: 0, knobA: 0, blade: 0, bal: 0, needleA: -2.48, pins: [0, 0, 0, 0, 0],
  tpZ: PARK_TP - CAR_HALF, slZ: PARK_SL + CAR_HALF, levels: 0, ink: 1,
};
const sampleIdx = ARMS.map(() => 0);

function zOf(bp) { return ZE + (bp / 100) * (HALF / st.R); }
const paperX = (tau) => XE + tau * KX;
const paperPt = (tau, bp) => pt([paperX(tau), PY, zOf(bp)]);
const dOf = (points) => (points.length > 1 ? pathOf(points.map(([tau, v]) => paperPt(tau, v))) : "");

function prepare(armI, sI) {
  const arm = ARMS[armI];
  const s = DATA.arms[arm.id].samples[sI];
  const tr = tradeOf(s);
  const plate = plateOf(arm, s);
  const fenceTau = s.cls === "TIME" ? s.exit - s.t - 1 : s.T - s.t;
  return { arm, s, tr, plate, plateI: PLATE_INDEX[plate] ?? 1, R: rangeOf(s), fenceX: XE + fenceTau * KX, n: DATA.arms[arm.id].samples.length, sI };
}

// ------------------------------------------------------------ drawing
function setDot(name, p) {
  const c = E.dot[name];
  if (!p) { c.style.display = "none"; return; }
  c.style.display = "";
  c.setAttribute("cx", p[0].toFixed(1));
  c.setAttribute("cy", p[1].toFixed(1));
}
function levelLine(name, z, x0, x1) {
  const el = E.lv[name];
  if (z == null) { el.setAttribute("d", ""); setDot(`${name}0`, null); setDot(`${name}1`, null); return; }
  const a = pt([x0, PY, z]), b = pt([x1, PY, z]);
  el.setAttribute("d", pathOf([a, b]));
  setDot(`${name}0`, a);
  setDot(`${name}1`, b);
}
function drawLevels() {
  const c = st.cur;
  if (!c) return;
  const { s, arm } = c;
  E.lv.tp.parentNode.style.opacity = st.levels.toFixed(3);
  levelLine("tp", zOf(s.u), XE, c.fenceX);
  levelLine("sl", s.d == null ? null : zOf(-s.d), XE, c.fenceX);
  const bounce = arm.trigger === "pins" && s.lowr != null;
  levelLine("low", bounce ? zOf(s.lowr) : null, XE + (s.jl - s.t - 1) * KX, XE);
  levelLine("high", bounce && zOf(s.highr) <= PZ1 ? zOf(s.highr) : null, XS, XE);
}
function drawTrace(tau) {
  const c = st.cur;
  if (!c) return;
  const { pre, trade } = c.tr;
  E.pre.setAttribute("d", dOf(upTo(pre, Math.min(tau, 0))));
  E.trade.setAttribute("d", tau > 0 ? dOf(upTo(trade, tau)) : "");
  setDot("start", paperPt(pre[0][0], pre[0][1]));
  setDot("entry", tau >= 0 ? paperPt(trade[0][0], trade[0][1]) : null);
  const done = st.phase === "hold";
  const last = trade[trade.length - 1];
  setDot("exit", done ? paperPt(last[0], last[1]) : null);
}
function drawGantry() {
  move(E.gantry, [st.x, 0, 0]);
  const lo = st.slZ, hi = st.tpZ;
  const z = clamp(st.penZ, Math.max(PZ0, lo), Math.min(PZ1, hi));
  move(E.car, [0, 0, z - ZE]);
  move(E.tp, [0, 0, hi + CAR_HALF - (ZE + CAR_HALF)]);
  move(E.sl, [0, 0, lo - CAR_HALF - (ZE - CAR_HALF)]);
  move(E.fence, [st.fenceX + CFG.TROLLEY_HALF, 0, 0]);
}
function drawNeedle() {
  const [x, y] = CFG.DIAL;
  const a = st.needleA;
  const r = CFG.DIAL_R - 3;
  const tip = pt([x + Math.sin(a) * r, y - Math.cos(a) * r, CFG.DIAL_TOP]);
  const tail = pt([x - Math.sin(a) * 2.6, y + Math.cos(a) * 2.6, CFG.DIAL_TOP]);
  E.needle.setAttribute("d", pathOf([tail, tip]));
}
function drawPins() {
  st.pins.forEach((lift, i) => {
    const g = E.pins[i];
    const key = lift.toFixed(2);
    if (g.dataset.lift === key) return;
    g.dataset.lift = key;
    g.innerHTML = pinSvg(CFG.PIN_X[i], lift * CFG.PIN_LIFT);
  });
}
function drawLever(mode) {
  const [x, y] = CFG.MODE;
  const [tx, ty] = CFG.PINS_DIAL_TARGETS[mode];
  const len = Math.hypot(tx - x, ty - y);
  const tip = [x + ((tx - x) / len) * 6.5, y + ((ty - y) / len) * 6.5];
  E.lever.setAttribute("d", pathOf([pt([x, y, CFG.MODE_Z]), pt([tip[0], tip[1], CFG.MODE_Z])]));
}
function drawRange() {
  const [rx, rz] = CFG.RANGE_KNOB;
  const a = st.rangeA;
  const y = CFG.YF + 5;
  E.range.setAttribute("d", pathOf([pt([rx, y, rz]), pt([rx + Math.sin(a) * 4.2, y, rz + Math.cos(a) * 4.2])]));
}
function drawKnob() {
  const [x, y] = CFG.KNOB;
  const a = st.knobA;
  E.knob.setAttribute("d", pathOf([pt([x, y, 8]), pt([x + Math.sin(a) * 4, y - Math.cos(a) * 4, 8])]));
}
function drawBlade() {
  const key = st.blade.toFixed(1);
  if (E.blade.dataset.angle === key) return;
  E.blade.dataset.angle = key;
  E.blade.innerHTML = bladeSvg(st.blade);
}
let pans = { arm: null };
function drawBalance() {
  const id = ARMS[st.arm].id;
  if (pans.arm !== id) {
    const sp = DATA.arms[id].split;
    E.panL.innerHTML = panSvg(CFG.COL[0], discsOf(sp.gross));
    E.panR.innerHTML = panSvg(CFG.COL[1], discsOf(sp.cost));
    pans = { arm: id };
  }
  const dz = st.bal * 4.5; // cost heavier: the right pan goes down, the left comes up
  const key = dz.toFixed(2);
  if (pans.key !== key) {
    pans.key = key;
    E.colL.innerHTML = columnSvg(CFG.COL[0], dz);
    E.colR.innerHTML = columnSvg(CFG.COL[1], -dz);
    move(E.panL, [0, 0, dz]);
    move(E.panR, [0, 0, -dz]);
  }
  const th = (st.bal * 30 * Math.PI) / 180;
  const [px, py, pz] = CFG.PIVOT;
  E.bal.setAttribute("d", pathOf([pt([px, py, pz]), pt([px + Math.sin(th) * 6.6, py, pz + Math.cos(th) * 6.6])]));
}
function drawFollower() {
  move(E.follower, [st.follow, 0, 0]);
}

// ------------------------------------------------------------ text
function setReadout(text) {
  if (readout.textContent !== text) readout.textContent = text;
}
function scoreAt(s, m) {
  const row = (s.sc ?? []).find(([t]) => t === m);
  return row ? row[1] : null;
}
function condAt(s, m) {
  const row = (s.cond ?? []).find(([t]) => t === m);
  return row ? row.slice(1) : [0, 0, 0, 0, 0];
}
function panel(arm) {
  const a = DATA.arms[arm.id];
  const log = a.log;
  field("title").textContent = `${arm.label} · ${arm.group}`;
  field("what").textContent = arm.what;
  field("entry").textContent = arm.entry;
  field("exit").textContent = arm.exit;
  field("reentry").textContent = REENTRY_WORDS[arm.reentry];
  field("gate").textContent = arm.gate === "on"
    ? "none: every day trades"
    : `trades on ${DATA.h5_on[arm.id]} of ${DATA.days} dev days, the days the 09:44 gate says “on”`;
  field("result").innerHTML = `<span class="sr-num">${fmt(log.mean, 3)}%</span> a day [${fmt(log.lo, 3)}, ${fmt(log.hi, 3)}] · <span class="sr-num">${fmt(log.pd)}%</span> per $ traded [${fmt(log.pd_lo)}, ${fmt(log.pd_hi)}] · ${log.tpd} trades a day on ${log.days} days`;
  field("split").innerHTML = `<span class="sr-num">${fmt(a.split.gross)}%</span> gross − <span class="sr-num">${fmt(a.split.cost, 2, false)}%</span> cost = <span class="sr-num">${fmt(a.split.net)}%</span> net, per $ traded`;
  const mix = `take-profit ${(a.mix.TP * 100).toFixed(1)}% · stop ${(a.mix.SL * 100).toFixed(1)}% · time ${(a.mix.TIME * 100).toFixed(1)}%`;
  const plans = Object.keys(a.plan_mix).length
    ? ` · plans ${Object.entries(a.plan_mix).sort((p, q) => q[1] - p[1]).map(([k, v]) => `${k} ${(v * 100).toFixed(1)}%`).join(", ")}`
    : "";
  field("mix").textContent = mix + plans;
  field("vsb0").innerHTML = log.vs_b0
    ? `<span class="sr-num">${fmt(log.vs_b0[0])} pp</span> per $ [${fmt(log.vs_b0[1])}, ${fmt(log.vs_b0[2])}] · Holm p ${log.p_holm} · ${log.halves.filter((h) => h > 0).length}/6 half-years positive`
    : "the baseline every arm is paired against";
  chips.forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.arm === arm.id)));
  E.trigger.dataset.mode = arm.trigger;
  drawLever(arm.trigger);
}

// ------------------------------------------------------------ the loop
const targets = (c) => {
  return {
    follow: CFG.PLATE_X[c.plateI],
    rangeA: ((-60 + 30 * RANGES.indexOf(c.R)) * Math.PI) / 180,
    knobA: (CFG.KNOB_ANGLES[REENTRY.indexOf(c.arm.reentry)] * Math.PI) / 180,
    bal: clamp(-DATA.arms[c.arm.id].split.net / 1.4, -1, 1),
  };
};
let from = null;
function begin(armI, sI) {
  const fresh = armI !== st.arm || !st.cur;
  st.arm = armI;
  st.sample = sI;
  st.phase = "set";
  st.phaseT = 0;
  from = { x: st.x, penZ: st.penZ, fenceX: st.fenceX, follow: st.follow, rangeA: st.rangeA, knobA: st.knobA, bal: st.bal, tpZ: st.tpZ, slZ: st.slZ, ink: st.ink, levels: st.levels };
  st.pending = prepare(armI, sI);
  if (fresh) panel(ARMS[armI]);
}
function step(dt) {
  st.phaseT += dt;
  const c = st.cur;
  if (st.phase === "set") {
    const u = st.phaseT / SET;
    st.ink = from.ink * (1 - ease(st.phaseT / FADE));
    st.levels = from.levels * (1 - ease(st.phaseT / FADE));
    if (st.phaseT >= FADE && st.pending) {
      st.cur = st.pending;
      st.pending = null;
      st.R = st.cur.R;
      E.pre.setAttribute("d", "");
      E.trade.setAttribute("d", "");
      ["start", "entry", "exit"].forEach((n) => setDot(n, null));
    }
    const k = ease(u);
    const nc = st.pending ?? st.cur;
    const tg = targets(nc);
    st.x = lerp(from.x, XS, k);
    st.fenceX = lerp(from.fenceX, nc.fenceX, k);
    st.tpZ = lerp(from.tpZ, PARK_TP - CAR_HALF, k);
    st.slZ = lerp(from.slZ, PARK_SL + CAR_HALF, k);
    st.follow = lerp(from.follow, tg.follow, k);
    st.rangeA = lerp(from.rangeA, tg.rangeA, k);
    st.knobA = lerp(from.knobA, tg.knobA, k);
    st.bal = lerp(from.bal, tg.bal, k);
    st.penZ = lerp(from.penZ, ZE + (nc.tr.pre[0][1] / 100) * (HALF / nc.R), k);
    const gated = nc.arm.gate !== "on";
    st.blade = gated ? CFG.OPEN_ANGLE * (u < 0.45 ? ease(u / 0.45) : 1 - ease((u - 0.45) / 0.5)) : lerp(st.blade, 0, k);
    st.needleA = lerp(st.needleA, -2.48, 1 - Math.exp(-dt / 0.15));
    st.pins = st.pins.map((p) => lerp(p, 0, 1 - Math.exp(-dt / 0.12)));
    const s = nc.s;
    setReadout(`${nc.arm.label} · trade ${nc.sI + 1}/${nc.n} · ${s.sym} ${s.date}${gated ? " · gate on" : ""}`);
    if (st.phaseT >= SET) { st.phase = "watch"; st.phaseT = 0; st.ink = 1; }
  } else if (st.phase === "watch") {
    const tau = -(PRE + 1) + st.phaseT * PEN_SPEED;
    const t = Math.min(tau, 0);
    st.x = paperX(t);
    st.penZ = zOf(valueAt(c.tr.pre, t));
    const m = c.s.t + Math.floor(t);
    watchInstruments(m, dt);
    const sc = scoreAt(c.s, m);
    const what = c.arm.trigger === "gauge" ? (sc == null ? "no score" : `score ${fmt(sc / 100)}%`) : `${condAt(c.s, m).filter(Boolean).length}/5 conditions`;
    setReadout(`${c.arm.label} · ${c.s.sym} · ${clock(m + 1)} · ${what}`);
    if (tau >= 0) { st.phase = "trigger"; st.phaseT = 0; }
  } else if (st.phase === "trigger") {
    const k = ease(st.phaseT / TRIG);
    const tpZ = zOf(c.s.u);
    const slZ = c.s.d == null ? PARK_SL + CAR_HALF : zOf(-c.s.d);
    st.tpZ = lerp(PARK_TP - CAR_HALF, tpZ, k);
    st.slZ = lerp(PARK_SL + CAR_HALF, slZ, k);
    st.levels = k;
    st.penZ = zOf(c.tr.trade[0][1]);
    watchInstruments(c.s.t, dt);
    const sc = scoreAt(c.s, c.s.t);
    setReadout(`${c.arm.label} · ${c.s.sym} · ${c.arm.trigger === "gauge" ? `score ${fmt(sc / 100)}% > 0` : "5/5 conditions"} · buy at ${clock(c.s.t + 1)}`);
    if (st.phaseT >= TRIG) { st.phase = "trade"; st.phaseT = 0; }
  } else if (st.phase === "trade") {
    const tau = Math.min(st.phaseT * PEN_SPEED, c.tr.end);
    st.x = paperX(tau);
    const v = valueAt(c.tr.trade, tau);
    st.penZ = zOf(v);
    st.tpZ = zOf(c.s.u);
    st.slZ = c.s.d == null ? PARK_SL + CAR_HALF : zOf(-c.s.d);
    setReadout(`${c.arm.label} · ${c.s.sym} · ${clock(c.s.t + 1 + Math.floor(tau))} · ${fmt(v / 100)}%`);
    if (tau >= c.tr.end) { st.phase = "hold"; st.phaseT = 0; }
  } else if (st.phase === "hold") {
    holdText();
    if (st.phaseT >= HOLD) nextTrade(false);
  }
  const tau = st.phase === "watch" ? Math.min(-(PRE + 1) + st.phaseT * PEN_SPEED, 0) : st.phase === "trigger" ? 0 : st.phase === "trade" ? Math.min(st.phaseT * PEN_SPEED, c.tr.end) : st.phase === "hold" ? c.tr.end : -(PRE + 1);
  render(tau);
}
function holdText() {
  const c = st.cur;
  setReadout(`${c.arm.label} · ${c.s.sym} · ${CLS[c.s.cls]} at ${clock(c.s.exit)} · net ${fmt(c.s.r / 100)}%`);
}
function watchInstruments(m, dt) {
  const c = st.cur;
  const f = 1 - Math.exp(-dt / 0.12);
  if (c.arm.trigger === "gauge") {
    const v = scoreAt(c.s, m);
    const a = v == null ? -2.48 : clamp(v / 200, -1, 1) * (135 * Math.PI) / 180;
    st.needleA = lerp(st.needleA, a, f);
    st.pins = st.pins.map((p) => lerp(p, 0, f));
  } else {
    const cond = condAt(c.s, m);
    st.pins = st.pins.map((p, i) => lerp(p, cond[i] ? 1 : 0, f));
    st.needleA = lerp(st.needleA, -2.48, f);
  }
}
function render(tau) {
  if (st.cur && !st.pending) {
    if (st.phase !== "set") drawTrace(tau);
    E.paper.style.opacity = st.ink.toFixed(3);
    drawLevels();
  } else E.paper.style.opacity = st.ink.toFixed(3);
  drawGantry();
  drawNeedle();
  drawPins();
  drawRange();
  drawKnob();
  drawBlade();
  drawBalance();
  drawFollower();
}
function nextTrade(user) {
  const armI = st.arm;
  sampleIdx[armI] = (st.sample + 1) % DATA.arms[ARMS[armI].id].samples.length;
  st.played += 1;
  if (st.touring && !user && st.played >= PER_ARM) {
    st.played = 0;
    const next = (armI + 1) % ARMS.length;
    begin(next, sampleIdx[next]);
  } else begin(armI, sampleIdx[armI]);
  if (reduce.matches) still();
}

// Reduced motion: show each trade finished, with no travel.
function still() {
  st.cur = st.pending ?? st.cur;
  st.pending = null;
  const c = st.cur;
  st.R = c.R;
  const tg = targets(c);
  Object.assign(st, { phase: "hold", phaseT: 0, ink: 1, levels: 1, x: paperX(c.tr.end), penZ: zOf(c.tr.trade[c.tr.trade.length - 1][1]), fenceX: c.fenceX,
    tpZ: zOf(c.s.u), slZ: c.s.d == null ? PARK_SL + CAR_HALF : zOf(-c.s.d), follow: tg.follow, rangeA: tg.rangeA, knobA: tg.knobA, bal: tg.bal, blade: 0 });
  if (c.arm.trigger === "gauge") st.needleA = clamp((scoreAt(c.s, c.s.t) ?? 0) / 200, -1, 1) * (135 * Math.PI) / 180;
  else st.pins = [1, 1, 1, 1, 1];
  render(c.tr.end);
  holdText();
}

// ------------------------------------------------------------ input
function choose(armI, user = true) {
  if (user) { st.touring = false; st.lastInput = performance.now(); }
  st.played = 0;
  begin(armI, sampleIdx[armI]);
  if (reduce.matches) still();
  wake();
}
chips.forEach((chip) => chip.addEventListener("click", () => choose(ARMS.findIndex((a) => a.id === chip.dataset.arm))));
stage.addEventListener("click", () => { st.touring = false; st.lastInput = performance.now(); nextTrade(true); wake(); });
stage.addEventListener("keydown", (e) => {
  if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
    e.preventDefault();
    choose((st.arm + (e.key === "ArrowRight" ? 1 : ARMS.length - 1)) % ARMS.length);
  } else if (e.key === " " || e.key === "Enter") {
    e.preventDefault();
    st.touring = false;
    st.lastInput = performance.now();
    nextTrade(true);
    wake();
  }
});

let raf = 0, last = 0, visible = true;
function frame(now) {
  raf = 0;
  if (!visible || reduce.matches) return;
  const dt = Math.min(0.05, (now - (last || now)) / 1000);
  last = now;
  if (!st.touring && now - st.lastInput > IDLE_BACK * 1000) st.touring = true;
  step(dt);
  raf = requestAnimationFrame(frame);
}
function wake() {
  if (!raf && visible && !reduce.matches) { last = 0; raf = requestAnimationFrame(frame); }
}
new IntersectionObserver((entries) => {
  visible = entries.some((e) => e.isIntersecting);
  if (visible) wake();
}).observe(stage);
reduce.addEventListener("change", () => { if (reduce.matches) still(); else wake(); });

begin(0, 0);
if (reduce.matches) still();
else { step(0); wake(); }
window.__sr = { st, choose, nextTrade, still, tradeOf };
