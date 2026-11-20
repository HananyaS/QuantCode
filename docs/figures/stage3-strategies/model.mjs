// The Stage 3 strategies as machine settings. Definitions follow the Stage 3 program
// (research/intraday/docs/design/2026-11-05-stage3-algo-program.md): B0 in §3 and §9,
// Phase C in §9.3-§9.6 as amended by §9a, H7 in §10 as amended by §10a.
// Numbers are not typed here: they come from data/figure.json (logged results + extraction).

// The plan library on the cam rack, left to right. H minutes, k_u, k_d (null: no stop).
export const PLATES = [
  { id: "x15", H: 15, ku: 1, kd: 1, name: "H15 (1,1)" },
  { id: "x30", H: 30, ku: 1, kd: 1, name: "H30 (1,1)" },
  { id: "x30d2", H: 30, ku: 1, kd: 2, name: "H30 (1,2)" },
  { id: "x60", H: 60, ku: 1, kd: 1, name: "H60 (1,1)" },
  { id: "x60u2", H: 60, ku: 2, kd: 1, name: "H60 (2,1)" },
  { id: "x60d2", H: 60, ku: 1, kd: 2, name: "H60 (1,2)" },
  { id: "x60n", H: 60, ku: 1, kd: null, name: "H60 (1,∞)" },
  { id: "chart", H: 60, ku: null, kd: null, name: "chart levels, H60" },
];
export const PLATE_INDEX = Object.fromEntries(PLATES.map((p, i) => [p.id, i]));

// Re-entry rule detents on the knob, in angle order.
export const REENTRY = ["cool5", "once", "nosl", "episode"];
export const REENTRY_WORDS = {
  cool5: "5-minute cooldown, then the same name may re-enter",
  once: "one entry per name-day",
  nosl: "no re-entry after a stop-out that day",
  episode: "one entry per bounce episode (name-day, j_L), 5-minute cooldown",
};

const EV = "S2 model's EV at H30 (1,1) > 0 (M1, isotonic)";
const B0_EXIT = "H30 (1,1): TP at +1σ√30 (at least 1.5× cost), stop at −1σ√30, else sell at the open 30 minutes on";

// trigger: "gauge" reads a score each minute and fires when it crosses 0; "pins" fires when all
// five true-bounce conditions hold. gate: "on" trades every day, "drift"/"pnl" only on gated days.
export const ARMS = [
  { id: "b0", label: "B0", group: "Baseline", trigger: "gauge", gauge: "EV", plate: "x30", reentry: "cool5", gate: "on",
    entry: `${EV}; no entry before 09:45`, exit: B0_EXIT,
    what: "The Stage 2m model as it stands: enter whenever the EV is positive, exit on the σ brackets." },
  { id: "h1a", label: "h1a", group: "H1 fewer round trips", trigger: "gauge", gauge: "EV", plate: "x30", reentry: "once", gate: "on",
    entry: "B0's entry", exit: B0_EXIT,
    what: "B0 with one entry per name-day: the same first entries, no re-entries." },
  { id: "h1b", label: "h1b", group: "H1 fewer round trips", trigger: "gauge", gauge: "EV", plate: "x30d2", reentry: "nosl", gate: "on",
    entry: "B0's entry", exit: "H30 (1,2): the stop moves out to −2σ√30",
    what: "A wider stop, and no re-entry after a stop-out." },
  { id: "h1c", label: "h1c", group: "H1 fewer round trips", trigger: "gauge", gauge: "EV", plate: "x60n", reentry: "once", gate: "on",
    entry: "B0's entry (X*'s EV, unchanged)", exit: "H60 (1,∞): TP at +1σ√60, no stop at all, sell at the open 60 minutes on",
    what: "One entry per name-day, held up to an hour with no stop." },
  { id: "h1d", label: "h1d", group: "H1 fewer round trips", trigger: "gauge", gauge: "EV", plate: "x60d2", reentry: "nosl", gate: "on",
    entry: "B0's entry", exit: "H60 (1,2): TP +1σ√60, stop −2σ√60",
    what: "An hour's hold with a wide stop, and no re-entry after a stop-out." },
  { id: "h5_drift", label: "h5 drift", group: "H5 day gate", trigger: "gauge", gauge: "EV", plate: "x30", reentry: "cool5", gate: "drift",
    entry: "B0's entry, only on 'on' days", exit: B0_EXIT,
    what: "At 09:44 a logistic gate on 17 day features predicts whether the pool drifts up from 09:45 to the close; on 'off' days nothing trades." },
  { id: "h5_pnl", label: "h5 pnl", group: "H5 day gate", trigger: "gauge", gauge: "EV", plate: "x30", reentry: "cool5", gate: "pnl",
    entry: "B0's entry, only on 'on' days", exit: B0_EXIT,
    what: "The same gate, trained to predict whether B0's own day will be positive." },
  { id: "h2q", label: "h2q", group: "H2 selection", trigger: "gauge", gauge: "q35", plate: "x30", reentry: "cool5", gate: "on",
    entry: "LightGBM 35% quantile of r_net above a per-fold cut τ (as many trades as B0)", exit: B0_EXIT,
    what: "Enter on a pessimistic quantile of the trade's return instead of its mean." },
  { id: "h3meta", label: "h3 meta", group: "H3 setups", trigger: "gauge", gauge: "meta", plate: "x30", reentry: "cool5", gate: "on",
    entry: "only at a setup minute (ORB15, ORB15+RVOL, VWAP reclaim, pre-market high break, high-of-day break) whose meta-model EV > 0", exit: B0_EXIT,
    what: "Classic day-trading setups, filtered by a meta-model." },
  { id: "h7_x", label: "h7 x30", group: "H7 true bounce", trigger: "pins", plate: "x30", reentry: "episode", gate: "on",
    entry: "every true bounce: all five conditions at minute t, read from bars ≤ t", exit: "x30, B0's exit",
    what: "The bounce as the entry trigger, with B0's exit." },
  { id: "h7_chart", label: "h7 chart", group: "H7 true bounce", trigger: "pins", plate: "chart", reentry: "episode", gate: "on",
    entry: "every true bounce", exit: "stop one tick under the low L, TP at the prior high H0 (at least 1.5× cost), H60",
    what: "A trader's levels: risk to the low that held, target the high before the drop." },
  { id: "h7_menu", label: "h7 menu", group: "H7 true bounce", trigger: "pins", plate: "menu", reentry: "episode", gate: "on",
    entry: "every true bounce", exit: "per trade, the plan with the highest calibrated EV of six (x15, x30, x60, x60u2, x60d2, chart)",
    what: "Choose the exit per trade from six plans' EVs." },
  { id: "h7_direct", label: "h7 direct", group: "H7 true bounce", trigger: "pins", plate: "menu", reentry: "episode", gate: "on",
    entry: "every true bounce", exit: "per trade, the plan a LightGBM classifier picks among the six",
    what: "Choose the exit per trade with a direct classifier." },
];
export const ARM_INDEX = Object.fromEntries(ARMS.map((a, i) => [a.id, i]));

export const BOUNCE_CONDITIONS = [
  "held low: no lower low in the last 3 bars",
  "real drop: 1 − L/H0 ≥ max(σ√30, 3× cost)",
  "reclaim: the close is above every high since the low",
  "retrace: the close is at least 38.2% of the way back to H0",
  "volume: at least the median of the last 30 bars",
];

// Chart scale: the smallest range that holds the trade's brackets and its path (and, for a
// bounce, the drop before it). The watch trace before entry is clipped at the paper's edge.
export const RANGES = [2.5, 5, 10, 25, 50];
export function rangeOf(sample, pre = 35) {
  const need = [sample.u, sample.d ?? 0];
  const from = sample.lowr != null ? sample.t - pre : sample.t + 1;
  const last = Math.max(sample.exit, sample.T + 1);
  for (let i = 0; i < sample.cl.length; i++) {
    const m = sample.m0 + i;
    if (m < from || m > last) continue;
    for (const k of ["h", "l"]) if (sample[k][i] != null) need.push(Math.abs(sample[k][i]));
  }
  if (sample.lowr != null) need.push(Math.abs(sample.lowr), Math.abs(sample.highr));
  const bpNeed = Math.max(...need) / 100; // basis points -> per cent
  return RANGES.find((r) => r >= bpNeed * 1.04) ?? RANGES[RANGES.length - 1];
}

// The plate a sample runs on: fixed per arm, or the trade's own plan for the menu arms.
export function plateOf(arm, sample) {
  if (arm.plate !== "menu") return arm.plate;
  return sample.plan;
}
