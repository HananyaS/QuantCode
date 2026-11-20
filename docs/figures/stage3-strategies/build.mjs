// Stage 3 strategies as a bench trade recorder (anatomy kit, framework-free).
//   node build.mjs [--light]      -> stage3-strategies[-light].html
// The kit is read from $ANATOMY_HOME (default ~/.claude/skills/anatomy).
import { readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import * as M from "./model.mjs";

const HERE = dirname(fileURLToPath(import.meta.url));
const ANATOMY = process.env.ANATOMY_HOME ?? join(homedir(), ".claude", "skills", "anatomy");
const k = await import(pathToFileURL(join(ANATOMY, "kit", "iso-kit.mjs")).href);
const G = await import(pathToFileURL(join(ANATOMY, "kit", "lathe.mjs")).href);
const { kitScript } = await import(pathToFileURL(join(ANATOMY, "scripts", "inline-kit.mjs")).href);
const theme = process.argv.includes("--light") ? "light" : "dark";
const DATA = JSON.parse(readFileSync(join(HERE, "data", "figure.json"), "utf8"));

// ---------------------------------------------------------------- world
const W = 640;
const H = 452;
const AZ = 58;
const BASE = { x: -216, y: -56, w: 440, d: 120, r: 8 };
const BASE_H = 12;
const FOOT = 5;

const BOARD = { x: -162, y: -38, w: 324, d: 8, r: 1.5 };
const BOARD_TOP = 178;
const YF = BOARD.y + BOARD.d; // board front face, -30
const PAPER = { x: -128, y: YF, w: 256, d: 0.6, r: 0.3 };
const PY = YF + PAPER.d; // paper face
const PZ0 = 42;
const PZ1 = 152;
const ZE = 97; // entry level on the paper
const HALF = 55; // the range's full value, 10 divisions
const DIV = HALF / 10;
const KX = 2.5; // units per minute
const PRE = 35; // minutes watched before the decision minute
const XE = PAPER.x + (PRE + 1) * KX; // entry x (open of bar t+1)
const XS = PAPER.x; // gantry start (bar t-35)
const STRIP_FACE = YF + 0.8;
const PSTRIP = { x: -142, y: YF, w: 10, d: 0.8, r: 0.4 };
const TSTRIP = { x: PAPER.x, y: YF, w: PAPER.w, d: 0.8, r: 0.4 };
const TSTRIP_Z = [33.5, 39.5];
const BAR = { x: -146, y: YF, w: 292, d: 3, r: 0.6 };
const BAR_B = [23, 29];
const BAR_T = [165, 171];
const ENDS = [-150, 144];
const END_B = [19, 33];
const END_T = [161, 177];

// gantry, built with the pen at x = 0
const TROLLEY = { x: -7, y: YF + 3, w: 14, d: 8, r: 1.2 };
const TROLLEY_B = [21, 31];
const TROLLEY_T = [163, 173];
const GBAR = { x: -3, y: YF + 5, w: 6, d: 3, r: 0.6 };
const CAR = { x: -8, y: YF + 8, w: 16, d: 8, r: 1.4 };
const CAR_HALF = 5;
const COLLAR = { x: -6, y: YF + 8, w: 12, d: 7, r: 1 };
const COLLAR_H = 4;
const PARK_TP = 158; // collar bottom when parked
const PARK_SL = 36; // collar top when parked
const FENCE = { x: 0, y: YF + 3, w: 5, d: 9, r: 1 };
const FENCE_Z = [20, 32];
const RANGE_KNOB = [-152, 136];

// trigger head
const HOUSING = { x: -212, y: -42, w: 42, d: 52, r: 3 };
const HOUSING_H = 20;
const DIAL = [-193, -26];
const DIAL_R = 11;
const DIAL_Z = HOUSING_H;
const DIAL_TOP = DIAL_Z + 6;
const PINS = { x: -209, y: -9, w: 36, d: 9, r: 1.5 };
const PINS_TOP = HOUSING_H + 4;
const PIN_X = [-203, -197, -191, -185, -179];
const PIN_Y = PINS.y + PINS.d / 2;
const MODE = [-178, -26];
const CONNECTOR = { x: -159, y: YF, w: 8, d: 3, r: 0.8 };

// day gate (knife switch)
const SW = { x: -208, y: 20, w: 56, d: 16, r: 2 };
const SW_TOP = 4;
const BLADE_Y = 28;
const HINGE = [-200, BLADE_Y, 12];
const BLADE_L = 40;
const OPEN_ANGLE = 64;

// plan rack
const RACK = { x: -114, y: 15, w: 150, d: 24, r: 2 };
const RACK_TOP = 4;
const PLATE_X = M.PLATES.map((_, i) => -108 + 17 * i);
const PLATE_W = 12;
const PLATE_Y = 19.5;
const PLATE_D = 2.2;
const plateHeight = (p) => (p.H === 15 ? 9 : p.H === 30 ? 15 : 23);
const ROD = { x: -112, y: 34.5, w: 146, d: 2.4, r: 0.8 };
const ROD_Z = [6, 8.4];
const FOLLOW = { x: -5, y: 32, w: 10, d: 7.6, r: 1.2 };
const FOLLOW_Z = [4.4, 11.4];

// re-entry knob, label plate, ruler
const KNOB = [56, 27];
const LABEL = { x: 84, y: 37, w: 56, d: 16, r: 1.6 };
const RULER = { x: -208, y: 50, w: 282, d: 6, r: 1 };

// balance
const BAL = { x: 160, y: -18, w: 56, d: 24, r: 9 };
const BAL_H = 15;
const COL = [[173, -6], [203, -6]];
const PAN_Z = 30;
const PAN_R = 12.5;
const DISC_R = 5.2;
const DISC_H = 1.25;
const BAL_FACE = BAL.y + BAL.d; // 6
const PIVOT = [188, BAL_FACE + 0.8, 4.6];
const TRAY = { x: 152, y: 20, w: 60, d: 30, r: 3 };
const TRAY_H = 4;
const POCKETS = [[162, 28, 3], [182, 28, 2], [202, 28, 1], [162, 42, 2], [182, 42, 0], [202, 42, 1]];

const P = k.fitProjection(
  [...k.boxCorners(BASE, -BASE_H - FOOT, 0), ...k.boxCorners(BOARD, 0, BOARD_TOP), [COL[0][0], COL[0][1], 58], [COL[1][0], COL[1][1], 58]],
  W, H, { pad: 18, azimuth: AZ },
);

const at = (p) => k.iso(p, P);
const S = (paths, style) => k.solidSvg(paths, style);
const L = (d, style) => k.lineSvg(d, style);
const D = (points, style) => k.dotsSvg(points, style);
const seg = (a, b) => k.segment(a, b, P);
const byDepth = (points) => [...points].sort((a, b) => k.depthOf([a[0], a[1], 0], P) - k.depthOf([b[0], b[1], 0], P));
const along = (o, a) => G.frameAlong(o, a);
const knurlY = (F, s0, s1, r, n) => k.fadedSvg(G.ribsOf(F, s0, s1, r, n, P, { fade: [0.1, 0.55] }), { tone: "lo" });

// ---------------------------------------------------------------- base
function base() {
  const out = [`<path class="iso-halo" d="${k.haloOf(BASE, -BASE_H - FOOT, BASE_H, P)}"/>`];
  for (const [x, y] of byDepth(k.corners(BASE, 24))) {
    out.push(S(k.cylinder(x, y, 8, -BASE_H - FOOT, 1.6, P, 28), { tone: "lo" }));
    out.push(S(k.cylinder(x, y, 6.2, -BASE_H - FOOT + 1.6, FOOT - 1.6, P, 28), { tone: "mid" }));
  }
  out.push(S(k.slabOf(BASE, -BASE_H, BASE_H, P, 8, 1.6), { tone: "mid", inner: L(k.sideSeam(BASE, -4, P, 8), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(BASE, 5), 0, P), { tone: "lo" }));
  const screws = k.corners(BASE, 11);
  out.push(L(screws.map(([x, y]) => k.ring(x, y, 2.4, 0, P, 16)).join(""), { tone: "lo" }));
  out.push(D(screws.map(([x, y]) => at([x, y, 0])), { size: 0.55 }));
  // ruler strip along the front: one tick per 4 units, a major every 20
  out.push(S(k.slabOf(RULER, 0, 0.6, P, 3), { tone: "lo", crease: "none" }));
  const t = k.topTicks(RULER.x + 4, RULER.x + RULER.w - 4, 4, 5, RULER.y + 0.6, 0.6, [1.8, 3.4], P);
  out.push(L(t.minor, { tone: "lo" }), L(t.major, { tone: "mid" }));
  out.push(D([at([RULER.x + 2, RULER.y + 3, 0.6]), at([RULER.x + RULER.w - 2, RULER.y + 3, 0.6])], { size: 0.45 }));
  // label plate: three rule lines stand for printed text
  out.push(S(k.slabOf(LABEL, 0, 0.8, P, 4, 0.4), { tone: "lo" }));
  out.push(L(k.planOutline(k.insetPlan(LABEL, 1.8), 0.8, P, 4), { tone: "faint" }));
  const lx = LABEL.x + 6;
  out.push(L(k.lineOnTop([lx, LABEL.y + 5], [lx + 34, LABEL.y + 5], 0.8, P), { tone: "mid", free: true }));
  out.push(L(k.lineOnTop([lx, LABEL.y + 8.5], [lx + 24, LABEL.y + 8.5], 0.8, P), { tone: "lo", free: true }));
  out.push(L(k.lineOnTop([lx, LABEL.y + 12], [lx + 29, LABEL.y + 12], 0.8, P), { tone: "lo", free: true }));
  out.push(D([at([LABEL.x + LABEL.w - 5, LABEL.y + 4, 0.8]), at([LABEL.x + LABEL.w - 5, LABEL.y + LABEL.d - 4, 0.8])], { size: 0.5 }));
  return out.join("");
}

// ---------------------------------------------------------------- trigger head
function triggerHead() {
  const out = [];
  out.push(`<path class="iso-halo" d="${k.haloOf(k.insetPlan(HOUSING, -1), 0, 0.1, P)}" style="filter:blur(3px)"/>`);
  out.push(S(k.slabOf(HOUSING, 0, HOUSING_H, P, 6, 0.8), { tone: "mid", inner: L(k.sideSeam(HOUSING, 15, P, 6), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(HOUSING, 2.6), HOUSING_H, P, 6), { tone: "faint" }));
  out.push(D(k.corners(HOUSING, 4.5).map(([x, y]) => at([x, y, HOUSING_H])), { size: 0.5 }));
  // vent slots on the +x face (closed outlines)
  for (let i = 0; i < 4; i++) {
    const y0 = HOUSING.y + 12 + i * 7;
    const face = HOUSING.x + HOUSING.w;
    out.push(L(k.pathOf([at([face, y0, 5]), at([face, y0 + 4.4, 5]), at([face, y0 + 4.4, 6.4]), at([face, y0, 6.4])], true), { tone: "lo" }));
  }
  // dial gauge (score)
  const [x, y] = DIAL;
  out.push(S(k.cylinder(x, y, DIAL_R + 1.6, DIAL_Z, 1.4, P, 56), { tone: "mid" }));
  out.push(S(k.cylinder(x, y, DIAL_R, DIAL_Z + 1.4, DIAL_TOP - DIAL_Z - 1.4, P, 64, 0.6), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, DIAL_R, DIAL_TOP - 2.6, DIAL_TOP, 72, P, 0.3), { tone: "lo" }));
  out.push(L(k.ring(x, y, DIAL_R - 2, DIAL_TOP, P, 64), { tone: "lo" }));
  const span = (270 * Math.PI) / 180;
  const ticks = k.radialTicks(x, y, DIAL_R - 2.6, 40, 10, DIAL_TOP, [1, 2.2], P, -span / 2, span); // 0.1% a tick, ±2% full scale
  out.push(L(ticks.minor, { tone: "lo" }), L(ticks.major, { tone: "mid" }));
  out.push(L(k.lineOnTop([x, y - (DIAL_R - 2.6)], [x, y - (DIAL_R - 6)], DIAL_TOP, P), { tone: "hi" }));
  // rest pin for "no score"
  const rest = (-150 * Math.PI) / 180;
  out.push(D([at([x + Math.sin(rest) * (DIAL_R - 4), y - Math.cos(rest) * (DIAL_R - 4), DIAL_TOP])], { size: 0.6, tone: "mid" }));
  out.push(L(k.ring(x, y, 1.5, DIAL_TOP, P, 16), { tone: "mid" }));
  // pin row (the five bounce conditions)
  out.push(S(k.slabOf(PINS, HOUSING_H, PINS_TOP - HOUSING_H, P, 3, 0.5), { tone: "mid" }));
  out.push(L(PIN_X.map((px) => k.ring(px, PIN_Y, 2.3, PINS_TOP, P, 20)).join(""), { tone: "lo" }));
  out.push(D([at([PINS.x + 2.2, PIN_Y, PINS_TOP]), at([PINS.x + PINS.w - 2.2, PIN_Y, PINS_TOP])], { size: 0.45 }));
  // mode lever boss
  out.push(S(k.cylinder(MODE[0], MODE[1], 2.6, HOUSING_H, 1.4, P, 24), { tone: "mid" }));
  return out.join("");
}

const PIN_REST = 1;
const PIN_LIFT = 4.5;
const pinSvg = (px, lift) => S(k.cylinder(px, PIN_Y, 1.4, PINS_TOP, PIN_REST + lift, P, 18), { tone: "hi", crease: "lo" });

// ---------------------------------------------------------------- board
function boardBack() {
  const out = [];
  out.push(`<path class="iso-halo" d="${k.haloOf(k.insetPlan(BOARD, -2), 0, 0.1, P)}" style="filter:blur(3px)"/>`);
  out.push(S(k.slabOf(BOARD, 0, BOARD_TOP, P, 3, 1), { tone: "mid", inner: L(k.sideSeam(BOARD, 8, P, 3), { tone: "faint" }) }));
  // price strip with ticks every division, a major every five, zero in hi
  out.push(S(k.slabOf(PSTRIP, PZ0, PZ1 - PZ0, P, 3), { tone: "lo", crease: "none" }));
  const edge = PSTRIP.x + PSTRIP.w;
  const minor = [], major = [];
  for (let i = -10; i <= 10; i++) {
    if (i === 0) continue;
    const z = ZE + i * DIV;
    const len = i % 5 === 0 ? 4.4 : 2.4;
    (i % 5 === 0 ? major : minor).push(seg([edge, STRIP_FACE, z], [edge - len, STRIP_FACE, z]));
  }
  out.push(L(minor.join(""), { tone: "lo" }), L(major.join(""), { tone: "mid" }));
  out.push(L(seg([edge, STRIP_FACE, ZE], [PSTRIP.x, STRIP_FACE, ZE]), { tone: "hi" }));
  // time strip: a tick per 5 minutes, a major per 15, from its top edge
  out.push(S(k.slabOf(TSTRIP, TSTRIP_Z[0], TSTRIP_Z[1] - TSTRIP_Z[0], P, 3), { tone: "lo", crease: "none" }));
  const tmin = [], tmaj = [];
  for (let j = -7; j <= 13; j++) {
    const x = XE + j * 5 * KX;
    if (x < PAPER.x + 0.5 || x > PAPER.x + PAPER.w - 0.5) continue;
    const big = j % 3 === 0;
    (big ? tmaj : tmin).push(seg([x, STRIP_FACE, TSTRIP_Z[1]], [x, STRIP_FACE, TSTRIP_Z[1] - (big ? 3.6 : 2)]));
  }
  out.push(L(tmin.join(""), { tone: "lo" }), L(tmaj.join(""), { tone: "mid" }));
  out.push(L(seg([XE, STRIP_FACE, TSTRIP_Z[1]], [XE, STRIP_FACE, TSTRIP_Z[0]]), { tone: "hi" }));
  // paper with a printed grid: divisions across, 5-minute lines down
  out.push(S(k.slabOf(PAPER, PZ0, PZ1 - PZ0, P, 2), { tone: "hi", crease: "faint" }));
  const gridFaint = [], gridLo = [];
  for (let i = -9; i <= 9; i++) {
    const z = ZE + i * DIV;
    (i % 5 === 0 ? gridLo : gridFaint).push(seg([PAPER.x, PY, z], [PAPER.x + PAPER.w, PY, z]));
  }
  for (let j = -7; j <= 13; j++) {
    const x = XE + j * 5 * KX;
    if (x < PAPER.x + 0.5 || x > PAPER.x + PAPER.w - 0.5) continue;
    (j % 3 === 0 ? gridLo : gridFaint).push(seg([x, PY, PZ0], [x, PY, PZ1]));
  }
  out.push(L(gridFaint.join(""), { tone: "faint" }), L(gridLo.join(""), { tone: "lo" }));
  // paper clips at the corners
  for (const [cx, cz] of [[PAPER.x + 5, PZ1 - 3], [PAPER.x + PAPER.w - 13, PZ1 - 3], [PAPER.x + 5, PZ0 + 0.5], [PAPER.x + PAPER.w - 13, PZ0 + 0.5]]) {
    out.push(S(k.slabOf({ x: cx, y: PY, w: 8, d: 1.2, r: 0.4 }, cz, 2.6, P, 2, 0.3), { tone: "mid" }));
    out.push(D([at([cx + 4, PY + 1.2, cz + 1.3])], { size: 0.4 }));
  }
  // range knob and its five detents (2.5, 5, 10, 25, 50 %)
  const [rx, rz] = RANGE_KNOB;
  out.push(D(M.RANGES.map((_, i) => {
    const a = ((-60 + 30 * i) * Math.PI) / 180;
    return at([rx + Math.sin(a) * 8, YF, rz + Math.cos(a) * 8]);
  }), { size: 0.55, tone: "mid" }));
  const RF = along([rx, YF, rz], [0, 1, 0]);
  out.push(S(G.disc(0, 1.2, 6.2, RF, P), { tone: "mid" }));
  out.push(S(G.disc(1.2, 5, 5, RF, P), { tone: "mid", inner: knurlY(RF, 1.6, 4.6, 5, 40) }));
  // the trigger cable's connector
  out.push(S(k.slabOf(CONNECTOR, 7, 6, P, 2, 0.4), { tone: "mid" }));
  out.push(D([at([CONNECTOR.x + 4, CONNECTOR.y + CONNECTOR.d, 10])], { size: 0.7, tone: "lo" }));
  // guide bars and their end blocks
  for (const [z0, z1] of [BAR_B, BAR_T]) {
    out.push(S(k.slabOf(BAR, z0, z1 - z0, P, 2, 0.4), { tone: "mid" }));
  }
  for (const x of ENDS) {
    for (const [z0, z1] of [END_B, END_T]) {
      const plan = { x, y: YF, w: 6, d: 12, r: 1 };
      out.push(S(k.slabOf(plan, z0, z1 - z0, P, 3, 0.5), { tone: "mid" }));
      out.push(D([at([x + 3, YF + 12, z0 + 3]), at([x + 3, YF + 12, z1 - 3])], { size: 0.45 }));
    }
  }
  // front feet clamping the board
  for (const x of [-124, 106]) {
    const foot = { x, y: YF, w: 18, d: 9, r: 1.5 };
    out.push(S(k.slabOf(foot, 0, 10, P, 3, 0.6), { tone: "mid" }));
    out.push(D([at([x + 4.5, YF + 4.5, 10]), at([x + 13.5, YF + 4.5, 10])], { size: 0.5 }));
  }
  return out.join("");
}

// ---------------------------------------------------------------- gantry (pen at x = 0)
function gantryFrame() {
  const out = [];
  for (const [z0, z1] of [TROLLEY_B, TROLLEY_T]) {
    out.push(S(k.slabOf(TROLLEY, z0, z1 - z0, P, 3, 0.5), { tone: "mid" }));
    const fy = TROLLEY.y + TROLLEY.d;
    out.push(D([at([-4, fy, z0 + 2.5]), at([4, fy, z0 + 2.5]), at([-4, fy, z1 - 2.5]), at([4, fy, z1 - 2.5])], { size: 0.45 }));
  }
  out.push(S(k.slabOf(GBAR, TROLLEY_B[1], TROLLEY_T[0] - TROLLEY_B[1], P, 2, 0.3), { tone: "mid", crease: "none" }));
  return out.join("");
}

function collarSvg(z0) {
  const out = [S(k.slabOf(COLLAR, z0, COLLAR_H, P, 3, 0.4), { tone: "hi", crease: "lo" })];
  const F = along([3, COLLAR.y + COLLAR.d, z0 + COLLAR_H / 2], [0, 1, 0]);
  out.push(S(G.disc(0, 2.4, 1.5, F, P), { tone: "mid", inner: knurlY(F, 0.4, 2.2, 1.5, 14) }));
  out.push(D([at([-3.6, COLLAR.y + COLLAR.d, z0 + COLLAR_H / 2])], { size: 0.45, tone: "lo" }));
  return out.join("");
}

function carriageSvg() {
  const z0 = ZE - CAR_HALF;
  const out = [S(k.slabOf(CAR, z0, CAR_HALF * 2, P, 3, 0.6), { tone: "hi", lit: true })];
  const F = along([0, CAR.y + CAR.d, ZE], [0, 1, 0]);
  out.push(S(G.disc(0, 2.6, 3.2, F, P), { tone: "hi", lit: true, inner: L(G.circleOf(F, 2.6, 1.8, P), { tone: "lo" }) }));
  out.push(D([at([0, CAR.y + CAR.d + 2.6, ZE])], { size: 0.5, tone: "hi" }));
  const fy = CAR.y + CAR.d;
  out.push(D([at([-6, fy, z0 + 2]), at([6, fy, z0 + 2]), at([-6, fy, z0 + 8]), at([6, fy, z0 + 8])], { size: 0.4, tone: "mid" }));
  return out.join("");
}

function fenceSvg() {
  const out = [S(k.slabOf(FENCE, FENCE_Z[0], FENCE_Z[1] - FENCE_Z[0], P, 3, 0.5), { tone: "hi", crease: "lo" })];
  const cx = FENCE.x + FENCE.w / 2, cy = FENCE.y + FENCE.d / 2;
  out.push(S(k.cylinder(cx, cy, 1.9, FENCE_Z[1], 3.2, P, 20), { tone: "mid" }));
  out.push(L(k.knurl(cx, cy, 1.9, FENCE_Z[1], FENCE_Z[1] + 3.2, 14, P, 0.3), { tone: "lo" }));
  out.push(D([at([FENCE.x + FENCE.w, FENCE.y + 2, FENCE_Z[0] + 3]), at([FENCE.x + FENCE.w, FENCE.y + 7, FENCE_Z[0] + 3])], { size: 0.4, tone: "lo" }));
  return out.join("");
}

// ---------------------------------------------------------------- day gate
function gateBase() {
  const out = [S(k.slabOf(SW, 0, SW_TOP, P, 4, 0.5), { tone: "mid" })];
  out.push(L(k.planOutline(k.insetPlan(SW, 1.6), SW_TOP, P, 4), { tone: "faint" }));
  out.push(D(k.corners(SW, 3).map(([x, y]) => at([x, y, SW_TOP])), { size: 0.45 }));
  return out.join("");
}
const JAW = (x, back) => ({ x, y: back ? BLADE_Y - 2.4 : BLADE_Y + 0.8, w: 6, d: 1.6, r: 0.3 });
function gateJaws(back) {
  const out = [];
  out.push(S(k.slabOf(JAW(HINGE[0] - 3, back), SW_TOP, 11, P, 2, 0.3), { tone: "mid" }));
  out.push(S(k.slabOf(JAW(HINGE[0] + BLADE_L - 6, back), SW_TOP, 9.6, P, 2, 0.3), { tone: "mid" }));
  if (!back) {
    const F = along([HINGE[0], BLADE_Y + 2.4, HINGE[2]], [0, 1, 0]);
    out.push(S(G.disc(0, 1, 1.6, F, P), { tone: "mid" }));
    out.push(D([at([HINGE[0], BLADE_Y + 3.4, HINGE[2]])], { size: 0.5, tone: "lo" }));
  }
  return out.join("");
}
function bladeSvg(angleDeg) {
  const a = (angleDeg * Math.PI) / 180;
  const dir = [Math.cos(a), 0, Math.sin(a)];
  const nrm = [-Math.sin(a), 0, Math.cos(a)];
  const F = G.frameOf(HINGE, dir, [0, 1, 0], nrm);
  const out = [S(G.prismOf(F, [[-0.8, -2], [0.8, -2], [0.8, 2], [-0.8, 2]], -2, BLADE_L, P), { tone: "hi", crease: "lo" })];
  const grip = G.add3(HINGE, G.mul3(dir, BLADE_L - 4));
  const GF = G.frameOf(G.add3(grip, G.mul3(nrm, 2)), nrm, [0, 1, 0], G.mul3(dir, -1));
  out.push(S(G.disc(0, 6.5, 1.3, GF, P), { tone: "mid" }));
  out.push(S(G.sphereOf(G.add3(grip, G.mul3(nrm, 10.2)), 2.4, P), { tone: "mid" }));
  return out.join("");
}

// ---------------------------------------------------------------- plan rack
function rack() {
  const out = [S(k.slabOf(RACK, 0, RACK_TOP, P, 4, 0.5), { tone: "mid" })];
  out.push(L(k.planOutline(k.insetPlan(RACK, 1.6), RACK_TOP, P, 4), { tone: "faint" }));
  M.PLATES.forEach((p, i) => {
    const x = PLATE_X[i];
    const h = plateHeight(p);
    const plan = { x, y: PLATE_Y, w: PLATE_W, d: PLATE_D, r: 0.4 };
    out.push(S(k.slabOf(plan, RACK_TOP, h, P, 2, 0.3), { tone: "hi", crease: "lo" }));
    if (p.id === "chart") out.push(S(k.slabOf({ x: x + 6, y: PLATE_Y, w: 6, d: PLATE_D, r: 0.3 }, RACK_TOP + h, 4, P, 2, 0.3), { tone: "hi", crease: "lo" }));
    const c = RACK_TOP + h / 2;
    const fy = PLATE_Y + PLATE_D;
    const tab = (z) => S(k.slabOf({ x: x + 2, y: fy, w: 8, d: 1.3, r: 0.3 }, z, 1.2, P, 2), { tone: "mid", crease: "none" });
    const ku = p.ku ?? 1.6, kd = p.kd;
    out.push(tab(c + 1.6 + 2.2 * (ku - 1) + (p.H === 15 ? 0 : 1)));
    if (kd != null) out.push(tab(c - 2.8 - 2.6 * (kd - 1) - (p.H === 15 ? 0 : 1)));
    if (p.id === "chart") out.push(tab(c - 6.2));
    out.push(D([at([x + 1.4, fy, RACK_TOP + h - 1.4]), at([x + PLATE_W - 1.4, fy, RACK_TOP + h - 1.4])], { size: 0.35, tone: "lo" }));
  });
  for (const x of [ROD.x - 2, ROD.x + ROD.w - 2]) {
    out.push(S(k.slabOf({ x, y: ROD.y - 2, w: 4, d: 6.4, r: 0.8 }, RACK_TOP, 7.4, P, 2, 0.3), { tone: "mid" }));
    out.push(D([at([x + 2, ROD.y + 1.2, RACK_TOP + 7.4])], { size: 0.45 }));
  }
  out.push(S(k.slabOf({ ...ROD, x: ROD.x + 2, w: ROD.w - 4 }, ROD_Z[0], ROD_Z[1] - ROD_Z[0], P, 2), { tone: "mid", crease: "none" }));
  return out.join("");
}
function followerSvg() {
  const out = [];
  out.push(S(k.slabOf({ x: -1.2, y: PLATE_Y + PLATE_D + 1.3, w: 2.4, d: FOLLOW.y - PLATE_Y - PLATE_D - 1.3, r: 0.4 }, 9.4, 1.4, P, 2), { tone: "mid" }));
  out.push(S(k.slabOf(FOLLOW, FOLLOW_Z[0], FOLLOW_Z[1] - FOLLOW_Z[0], P, 3, 0.5), { tone: "hi", crease: "lo" }));
  out.push(D([at([-2.6, FOLLOW.y + FOLLOW.d, 8]), at([2.6, FOLLOW.y + FOLLOW.d, 8])], { size: 0.45, tone: "lo" }));
  return out.join("");
}

// ---------------------------------------------------------------- re-entry knob
const KNOB_ANGLES = [-66, -22, 22, 66];
function knob() {
  const [x, y] = KNOB;
  const out = [S(k.cylinder(x, y, 8.4, 0, 3, P, 40, 0.4), { tone: "mid" })];
  out.push(D(KNOB_ANGLES.map((deg) => {
    const a = (deg * Math.PI) / 180;
    return at([x + Math.sin(a) * 7, y - Math.cos(a) * 7, 3]);
  }), { size: 0.55, tone: "mid" }));
  out.push(S(k.cylinder(x, y, 4.8, 3, 5, P, 36), { tone: "hi", crease: "lo" }));
  out.push(L(k.knurl(x, y, 4.8, 3.4, 7.6, 30, P, 0.3), { tone: "lo" }));
  return out.join("");
}

// ---------------------------------------------------------------- balance
function balanceBody() {
  const out = [`<path class="iso-halo" d="${k.haloOf(k.insetPlan(BAL, -1), 0, 0.1, P)}" style="filter:blur(3px)"/>`];
  out.push(S(k.slabOf(BAL, 0, BAL_H, P, 8, 0.8), { tone: "mid", inner: L(k.sideSeam(BAL, 3, P, 8), { tone: "faint" }) }));
  out.push(L(k.planOutline(k.insetPlan(BAL, 2.4), BAL_H, P, 8), { tone: "faint" }));
  for (const [cx, cy] of COL) {
    out.push(L(k.planOutline({ x: cx - 5, y: cy - 3.2, w: 10, d: 6.4, r: 3.2 }, BAL_H, P, 6), { tone: "lo" }));
    out.push(S(k.cylinder(cx, cy, 2.7, BAL_H, 1.4, P, 24), { tone: "mid" }));
  }
  out.push(D([at([BAL.x + 6, BAL.y + 6, BAL_H]), at([BAL.x + BAL.w - 6, BAL.y + 6, BAL_H])], { size: 0.45 }));
  // scale plate on the front face, ticks from its top edge
  const plate = { x: PIVOT[0] - 10, y: BAL_FACE, w: 20, d: 0.6, r: 0.3 };
  out.push(S(k.slabOf(plate, 2.4, 9.6, P, 2), { tone: "lo", crease: "none" }));
  const fy = BAL_FACE + 0.6;
  const mi = [], ma = [];
  for (let i = -4; i <= 4; i++) {
    const x = PIVOT[0] + i * 2;
    (i % 2 === 0 ? ma : mi).push(seg([x, fy, 12], [x, fy, 12 - (i % 2 === 0 ? 2.4 : 1.4)]));
  }
  out.push(L(mi.join(""), { tone: "lo" }), L(ma.join(""), { tone: "mid" }));
  out.push(L(seg([PIVOT[0], fy, 12], [PIVOT[0], fy, 8.8]), { tone: "hi" }));
  return out.join("");
}
const columnSvg = ([cx, cy], dz) => S(k.cylinder(cx, cy, 1.5, BAL_H + 1.4, PAN_Z - 2 - BAL_H - 1.4 + dz, P, 18), { tone: "mid", crease: "none" });
function panSvg([cx, cy], discs) {
  const out = [S(k.cylinder(cx, cy, 3, PAN_Z - 2, 2, P, 24), { tone: "mid" })];
  out.push(S(k.cylinder(cx, cy, PAN_R, PAN_Z, 1.4, P, 72, 0.5), { tone: "hi", crease: "lo" }));
  out.push(L(k.ring(cx, cy, PAN_R - 1.8, PAN_Z + 1.4, P, 72), { tone: "faint" }));
  let z = PAN_Z + 1.4;
  for (const h of discs) {
    out.push(S(k.cylinder(cx, cy, DISC_R, z, h, P, 36), { tone: "mid", crease: "lo" }));
    z += h;
  }
  return out.join("");
}
const discsOf = (value) => {
  const tenths = Math.max(0, value) * 10; // one disc per 0.1% per $
  const n = Math.floor(tenths + 1e-9);
  const out = Array(n).fill(DISC_H);
  const rest = tenths - n;
  if (rest > 0.12) out.push(Math.max(0.3, rest * DISC_H));
  return out;
};

// spare weights beside the balance: 0.1% discs in pockets
function weightBox() {
  const out = [S(k.slabOf(TRAY, 0, TRAY_H, P, 4, 0.6), { tone: "mid", inner: L(k.sideSeam(TRAY, 1.4, P, 4), { tone: "faint" }) })];
  out.push(L(k.planOutline(k.insetPlan(TRAY, 2), TRAY_H, P, 4), { tone: "faint" }));
  out.push(L(POCKETS.map(([x, y]) => k.ring(x, y, 5.6, TRAY_H, P, 40)).join(""), { tone: "lo" }));
  out.push(D(k.corners(TRAY, 3.6).map(([x, y]) => at([x, y, TRAY_H])), { size: 0.45 }));
  for (const [x, y, n] of byDepth(POCKETS)) {
    for (let i = 0; i < n; i++) out.push(S(k.cylinder(x, y, DISC_R, TRAY_H + i * DISC_H, DISC_H, P, 36), { tone: "mid", crease: "lo" }));
    if (!n) out.push(D([at([x, y, TRAY_H])], { size: 0.5, tone: "lo" }));
  }
  return out.join("");
}

// ---------------------------------------------------------------- page
const ARM0 = DATA.arms.b0;
const svg = k.figureSvg({
  width: W,
  height: H,
  label: "A bench trade recorder. A vertical chart board carries a gantry that sweeps across in time; a lit pen carriage on the gantry rides the real price of a dev trade between a take-profit collar above and a stop-loss collar below, and a horizon fence on the lower rail ends the trade on time. A trigger head on the left reads the strategy's score on a dial or five bounce pins, a knife switch in front is the day gate, a rack of eight cam plates holds the exit plans, a knob sets the re-entry rule, and a balance on the right weighs gross return against cost per dollar.",
  body: [
    base(),
    `<g data-part="trigger">${triggerHead()}<g data-part="pins">${PIN_X.map((px, i) => `<g data-pin="${i}">${pinSvg(px, 0)}</g>`).join("")}</g>`,
    `<path data-part="dial-needle" class="iso-line" data-tone="hi" data-free="" style="stroke-width:1"/>`,
    D([at([DIAL[0], DIAL[1], DIAL_TOP])], { size: 0.7, tone: "hi" }),
    `<path data-part="mode-lever" class="iso-line" data-tone="mid" data-free="end" style="stroke-width:1.2"/>`,
    D([at([MODE[0], MODE[1], HOUSING_H + 1.4])], { size: 0.7, tone: "mid" }),
    `</g>`,
    boardBack(),
    `<path data-part="range-pointer" class="iso-line" data-tone="hi" data-free="end" style="stroke-width:1"/>`,
    D([at([RANGE_KNOB[0], YF + 5, RANGE_KNOB[1]])], { size: 0.6, tone: "hi" }),
    k.wireSvg(at([HOUSING.x + HOUSING.w, -20, 9]), at([CONNECTOR.x + 4, CONNECTOR.y + CONNECTOR.d, 10]), { via: [at([HOUSING.x + HOUSING.w + 6, -20, 9])], tone: "lo", size: 0.9 }),
    `<defs><clipPath id="sr-paper"><path d="${k.pathOf([at([PAPER.x, PY, PZ0]), at([PAPER.x + PAPER.w, PY, PZ0]), at([PAPER.x + PAPER.w, PY, PZ1]), at([PAPER.x, PY, PZ1])], true)}"/></clipPath></defs>`,
    `<g data-part="paper-ink" clip-path="url(#sr-paper)">`,
    `<g data-part="levels" style="opacity:0"><path data-level="tp" class="iso-line" data-tone="mid"/><path data-level="sl" class="iso-line" data-tone="mid"/><path data-level="low" class="iso-line" data-tone="lo" data-dashed=""/><path data-level="high" class="iso-line" data-tone="lo" data-dashed=""/>`,
    `<g class="iso-dots" data-tone="mid">${["tp0", "tp1", "sl0", "sl1", "low0", "low1", "high0", "high1"].map((n) => `<circle data-dot="${n}" r="0.6" style="display:none"/>`).join("")}</g></g>`,
    `<g data-part="trace"><path data-trace="pre" class="iso-line" data-tone="mid" style="stroke-width:.8"/><path data-trace="trade" class="iso-line" data-tone="lit" style="stroke-width:1.1"/>`,
    `<g class="iso-dots" data-tone="hi">${["start", "entry", "exit"].map((n) => `<circle data-dot="${n}" r="0.75" style="display:none"/>`).join("")}</g></g></g>`,
    `<g data-part="gantry">${gantryFrame()}<g data-part="sl-collar">${collarSvg(ZE - CAR_HALF - COLLAR_H)}</g><g data-part="carriage">${carriageSvg()}</g><g data-part="tp-collar">${collarSvg(ZE + CAR_HALF)}</g></g>`,
    `<g data-part="fence">${fenceSvg()}</g>`,
    gateBase(),
    gateJaws(true),
    `<g data-part="blade">${bladeSvg(0)}</g>`,
    gateJaws(false),
    rack(),
    `<g data-part="follower">${followerSvg()}</g>`,
    knob(),
    `<path data-part="knob-pointer" class="iso-line" data-tone="hi" data-free="end" style="stroke-width:1.1"/>`,
    D([at([KNOB[0], KNOB[1], 8])], { size: 0.6, tone: "hi" }),
    balanceBody(),
    `<g data-part="col-l">${columnSvg(COL[0], 0)}</g><g data-part="pan-l">${panSvg(COL[0], discsOf(ARM0.split.gross))}</g>`,
    `<g data-part="col-r">${columnSvg(COL[1], 0)}</g><g data-part="pan-r">${panSvg(COL[1], discsOf(ARM0.split.cost))}</g>`,
    `<path data-part="bal-needle" class="iso-line" data-tone="hi" data-free="end" style="stroke-width:1"/>`,
    D([at(PIVOT)], { size: 0.8, tone: "hi" }),
    weightBox(),
  ],
});

// geometry the live code needs
const CFG = {
  P, ZE, HALF, DIV, KX, PRE, XE, XS, PY, PZ0, PZ1, CAR_HALF, COLLAR_H, PARK_TP, PARK_SL,
  FENCE_X0: FENCE.x, TROLLEY_HALF: TROLLEY.w / 2, DIAL, DIAL_R, DIAL_TOP, MODE, MODE_Z: HOUSING_H + 1.4,
  PIN_X, PIN_Y, PINS_TOP, PIN_REST, PIN_LIFT, HINGE, BLADE_L, BLADE_Y, OPEN_ANGLE,
  PLATE_X: PLATE_X.map((x) => x + PLATE_W / 2), KNOB, KNOB_ANGLES, RANGE_KNOB, YF,
  COL, PAN_Z, PAN_R, DISC_R, DISC_H, BAL_H, PIVOT, BAL_FACE,
  PINS_DIAL_TARGETS: { gauge: DIAL, pins: [(PIN_X[0] + PIN_X[4]) / 2, PIN_Y] },
};

const latheSrc = readFileSync(join(ANATOMY, "kit", "lathe.mjs"), "utf8").replace(/^import[^\n]*\n/m, "").replace(/^export /gm, "");
const latheScript = `const G = (() => {\n${latheSrc}\nreturn { prismOf, frameOf, frameAlong, disc, sphereOf, add3, mul3 };\n})();\n`;
const shared = Object.entries({ plateHeight, discsOf, pinSvg, columnSvg, panSvg, bladeSvg }).map(([name, f]) => `const ${name} = ${f.toString()};`).join("\n");
const live = readFileSync(join(HERE, "live.js"), "utf8");
const pageData = `
const CFG = ${JSON.stringify(CFG)};
const P = CFG.P;
const DATA = ${JSON.stringify(DATA)};
const ARMS = ${JSON.stringify(M.ARMS)};
const PLATES = ${JSON.stringify(M.PLATES)};
const PLATE_INDEX = ${JSON.stringify(M.PLATE_INDEX)};
const REENTRY = ${JSON.stringify(M.REENTRY)};
const REENTRY_WORDS = ${JSON.stringify(M.REENTRY_WORDS)};
const RANGES = ${JSON.stringify(M.RANGES)};
const BOUNCE_CONDITIONS = ${JSON.stringify(M.BOUNCE_CONDITIONS)};
const rangeOf = ${M.rangeOf.toString()};
const plateOf = ${M.plateOf.toString()};
const k = { cylinder, solidSvg, lineSvg, dotsSvg, ring, knurl, iso };
const S = (paths, style) => solidSvg(paths, style);
const at = (p) => iso(p, P);
const D = (points, style) => dotsSvg(points, style);
const L = (d, style) => lineSvg(d, style);
const PINS_TOP = CFG.PINS_TOP, PIN_Y = CFG.PIN_Y, PIN_REST = CFG.PIN_REST, BAL_H = CFG.BAL_H, PAN_Z = CFG.PAN_Z, PAN_R = CFG.PAN_R, DISC_R = CFG.DISC_R, DISC_H = CFG.DISC_H, HINGE = CFG.HINGE, BLADE_L = CFG.BLADE_L;
${shared}
`;

const chips = `<div class="sr-chips" role="group" aria-label="Strategy">${[...new Set(M.ARMS.map((a) => a.group))].map((g) => `<div class="sr-row"><span class="sr-group">${g}</span>${M.ARMS.filter((a) => a.group === g).map((a) => `<button type="button" class="sr-chip" data-arm="${a.id}" aria-pressed="false">${a.label}</button>`).join("")}</div>`).join("")}</div>`;
const panel = `<section class="sr-panel" aria-live="polite"><h2 class="sr-title" data-sr="title"></h2><p class="sr-what" data-sr="what"></p><dl class="sr-spec"><dt>Entry</dt><dd data-sr="entry"></dd><dt>Exit</dt><dd data-sr="exit"></dd><dt>Re-entry</dt><dd data-sr="reentry"></dd><dt>Day gate</dt><dd data-sr="gate"></dd><dt>Result</dt><dd data-sr="result"></dd><dt>Gross − cost</dt><dd data-sr="split"></dd><dt>Exits</dt><dd data-sr="mix"></dd><dt>vs B0</dt><dd data-sr="vsb0"></dd></dl></section>`;
const body = k.plateHtml({
  fig: "Fig 1",
  title: "Stage 3 trade recorder",
  hint: "Click for the next trade · ← → strategy",
  readout: "B0 · loading",
  keys: [
    { mark: "lit", label: "Pen carriage and its trace: a real dev trade of the chosen strategy, replayed bar by bar" },
    { mark: "raised", label: "Collars: take-profit above, stop-loss below; the fence on the lower rail is the time limit" },
    { mark: "raised", label: "Balance: gross per $ on the left pan, cost per $ on the right, one disc per 0.1%" },
  ],
  caption: `Dev window 2020-01 to 2022-12 (${DATA.days} test days), pool P1 (10 names a day), $25k in 10 slots of $2,500, long only, market entries. Each strategy's numbers are its logged dev result; the trades replayed are a seeded random sample of its own trades, relabelled through the labeller. No arm passes the pre-registered bar; N = ${DATA.n_trials} trial lines. ${Object.keys(DATA.not_run).join(", ")} did not run: their pre-checks failed.`,
  body: `<div class="iso-stage sr-stage" tabindex="0" role="group" aria-label="Trade recorder. Arrow keys step through the strategies; Space replays the next trade.">${svg}</div>`,
});
const CSS = `
.sr-stage:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:6px}
.sr-chips{display:flex;flex-direction:column;gap:6px;margin:18px 0 6px}
.sr-row{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.sr-group{font-size:12px;color:var(--anatomy-muted);min-width:132px}
.sr-chip{font:12px/1 var(--anatomy-mono);color:var(--anatomy-text);background:var(--anatomy-card);border:1px solid var(--anatomy-faint);border-radius:6px;padding:6px 9px;cursor:pointer}
.sr-chip:hover{border-color:var(--anatomy-lo)}
.sr-chip[aria-pressed="true"]{color:var(--anatomy-strong);border-color:var(--anatomy-hi)}
.sr-chip:focus-visible{outline:1px solid var(--anatomy-muted);outline-offset:2px}
.sr-panel{margin-top:14px;padding:16px 18px;border-radius:10px;background:var(--anatomy-card);box-shadow:var(--anatomy-lift)}
.sr-title{margin:0 0 4px;font-size:15px;font-weight:600;color:var(--anatomy-strong)}
.sr-what{margin:0 0 10px}
.sr-spec{display:grid;grid-template-columns:96px 1fr;gap:4px 12px;margin:0}
.sr-spec dt{color:var(--anatomy-muted);font-size:12px;padding-top:2px}
.sr-spec dd{margin:0;font-size:13px}
.sr-num{font-family:var(--anatomy-mono);font-size:12px;color:var(--anatomy-strong)}
[data-part="trigger"][data-mode="gauge"] [data-part="pins"]{opacity:.5}
[data-part="trigger"][data-mode="pins"] [data-part="dial-needle"]{opacity:.5}
@media (max-width:520px){.sr-group{min-width:100%}.sr-spec{grid-template-columns:1fr}.sr-spec dt{padding-top:6px}}
`;
const page = k.pageHtml({ title: "Stage 3 trade recorder", theme, body: body + chips + panel, script: kitScript() + latheScript + pageData + live, width: 760 })
  .replace("</style>", `${CSS}</style>`);
const out = join(HERE, theme === "light" ? "stage3-strategies-light.html" : "stage3-strategies.html");
writeFileSync(out, page);
console.log("wrote", out, `${(page.length / 1024).toFixed(0)} KB`);

// The same page as a theme-following artifact: content only (the host adds the document
// skeleton), the kit's tokens on :root (dark first), light values for a light viewer.
if (process.argv.includes("--artifact")) {
  const iso = k.ISO_CSS;
  const lightOpen = '.iso[data-theme="light"]{';
  const a = iso.indexOf(lightOpen);
  const b = iso.indexOf("}", a);
  const light = iso.slice(a + lightOpen.length, b);
  const tokens = iso.slice(0, a).replace(".iso{", ":root{color-scheme:dark;");
  const themed = `${tokens}@media (prefers-color-scheme: light){:root:not([data-theme="dark"]){${light};color-scheme:light}}:root[data-theme="light"]{${light};color-scheme:light}${iso.slice(b + 1)}`;
  const shell = `
body{margin:0;background:var(--anatomy-page);color:var(--anatomy-text);font:14px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased;padding-left:max(16px, env(safe-area-inset-left, 0px));padding-right:max(16px, env(safe-area-inset-right, 0px))}
*{box-sizing:border-box}
.iso-page{width:min(760px,100%);margin-inline:auto;padding-block:40px 80px}
.sr-head{margin:0 0 22px;max-width:62ch}
.sr-head h1{margin:0 0 6px;font-size:22px;line-height:1.25;font-weight:600;color:var(--anatomy-strong);text-wrap:balance}
.sr-head p{margin:0}
`;
  const head = `<header class="sr-head"><h1>Stage 3 trade recorder</h1><p>Thirteen intraday strategies from the pool-then-trade study, each set up on one machine: what fires the entry, where the take-profit and stop sit, when time runs out, and how gross return weighs against cost. Each replays its own real dev trades. None passed.</p></header>`;
  const doc = `<title>Stage 3 trade recorder</title><meta name="claude:viewport-fit-sides" content="cover"><style>${themed}${shell}${CSS}</style><main class="iso iso-page">${head}${body}${chips}${panel}</main><script type="module">${(kitScript() + latheScript + pageData + live).replace(/<\/(script)/gi, "<\\/$1")}</script>`;
  const art = join(HERE, "stage3-strategies-artifact.html");
  writeFileSync(art, doc);
  console.log("wrote", art, `${(doc.length / 1024).toFixed(0)} KB`);
}
