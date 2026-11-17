// gta6-twitch-ccu-forecast-01 — forecast Twitch viewers in the "Grand Theft Auto VI"
// category for hours 2..24 of release day, as five quantiles per hour.
//
// MODEL (local log-ratio regression on the case's comparables)
// -----------------------------------------------------------
// A launch-day curve is shaped by the rolling midnight unlock: at 00:00 UTC only
// New Zealand / Australia / Asia / Europe have the game, the US unlocks at 05:00
// (East) and 08:00 (West) UTC, so the count usually climbs steeply through the day
// and peaks in the late UTC hours. How steeply it climbs depends on how much of the
// audience was already unlocked in hour 1: a launch with a huge hour-1 print is
// already near its peak (ratio ~1), a launch with a tiny hour-1 print explodes
// later (ratios of 100x+ are real). Across the case's 38-42 comparable launches
// the log-ratio ln(v_h / v_1) is almost a linear function of ln(v_1) (correlation
// about -0.93 at the evening hours), but the relation saturates for the biggest
// launches, so a single global line under-predicts them badly. We therefore fit,
// separately for each hour h, a LOCAL linear regression of the log-ratio on
// ln(v_1) using only the comparables whose hour-1 print lies within +-1.5 natural
// log units of this case's hour-1 print (falling back to all of them when fewer
// than 8 are near). The spread is the local residual standard deviation, clipped
// to [0.30, 1.20] log units, so a case whose neighbours disagree gets an honestly
// wide band and a case whose neighbours agree gets a tight one. Quantiles are
// log-normal: q_tau = p1 * exp(median_lr_h + z_tau * sigma_h).
//
// Everything the program knows is in the case file: it reads every *.in in the
// input directory and writes one *.out per case. Standard library only, no
// network, deterministic (same input bytes -> same output bytes), well inside the
// 8 s / 1 CPU / 1024 MB cap.
//
// Input:  /input/<id>.in  (schema "makefun/twitch-ccu-case/v1")
// Output: /output/<id>.out {"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}
//   exactly 23 rows, hours 2..24 each once, each q a finite number > 0 and
//   <= 1e9, non-decreasing, at most 65,536 bytes.
// Env: MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output
//   when set (local runs); the grader leaves both unset.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? "/input";
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Hours to forecast. */
const HOURS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
/** Half-width of the local window in ln(hour-1 viewers). Tuned by leave-one-out
 *  on the 37 usable comparables: window 2.0 scores mean pinball 0.384, versus
 *  0.400 at 1.5, 0.409 at 1.0 and 0.385 at 2.5 — 2.0 is the sweet spot, wide
 *  enough to be stable yet local enough to respect the saturation. */
const WINDOW = 2.0;
/** Minimum comparables in the window before falling back to all of them. */
const MIN_LOCAL = 8;
/** Spread floor / cap in log units. */
const SIGMA_MIN = 0.3;
const SIGMA_MAX = 1.2;
/** Fallback spread when a regression has no residual (single point). */
const SIGMA_FALLBACK = 0.6;

interface HourlyPoint {
  from: string;
  viewers: number | null;
  channels?: number | null;
}

interface Comparable {
  name: string;
  day_start: string;
  hourly: HourlyPoint[];
}

interface CaseFile {
  levels: number[];
  known: Array<{ hour: number; viewers: number }>;
  forecast: Array<{ hour: number }>;
  comparables: Comparable[];
}

/** Parse an ISO-8601 UTC timestamp to milliseconds. */
function parseTime(s: string): number {
  return Date.parse(s);
}

/**
 * A comparable's release-day curve: viewers for hours 1..24 of its UTC release
 * day, or null when the curve is incomplete (the GTA VI pre-release comparable
 * has no release day yet, and a sparse category may miss hours).
 */
function releaseDayCurve(c: Comparable): number[] | null {
  const dayStart = parseTime(c.day_start);
  if (!Number.isFinite(dayStart)) return null;
  const byHour = new Map<number, number>();
  for (const pt of c.hourly ?? []) {
    if (pt.viewers === null || pt.viewers === undefined) continue;
    const t = parseTime(pt.from);
    if (!Number.isFinite(t)) continue;
    const h = Math.floor((t - dayStart) / 3_600_000) + 1;
    if (h >= 1 && h <= 24 && !byHour.has(h)) byHour.set(h, Math.max(1, pt.viewers));
  }
  if (byHour.size !== 24 || !byHour.has(1)) return null;
  const curve: number[] = [];
  for (let h = 1; h <= 24; h++) curve.push(byHour.get(h) as number);
  return curve;
}

/** Least-squares fit of y ~ a + b*x; returns [a, b]. */
function linfit(xs: number[], ys: number[]): [number, number] {
  const n = xs.length;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) {
    sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i];
  }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return [sy / n, 0];
  const b = (n * sxy - sx * sy) / denom;
  const a = (sy - b * sx) / n;
  return [a, b];
}

/** Population standard deviation. */
function stdev(xs: number[]): number {
  if (xs.length === 0) return 0;
  const m = xs.reduce((a, b) => a + b, 0) / xs.length;
  const v = xs.reduce((a, b) => a + (b - m) * (b - m), 0) / xs.length;
  return Math.sqrt(Math.max(0, v));
}

function clamp(x: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, x));
}

/** Round to 2 decimals, half up (matches the house baseline's rounding). */
function round2(x: number): number {
  return Math.round(x * 100) / 100;
}

/**
 * Forecast one case. Returns the 23 output rows.
 * p1 is hour 1's print; curves are the comparables' release-day curves.
 */
function forecastCase(p1: number, curves: number[][]): Array<{ hour: number; q: number[] }> {
  const logP1 = Math.log(Math.max(1, p1));
  // Log-ratios lr[h] = ln(v_h / v_1) for each comparable, h = 2..24.
  const logRatios: number[][] = curves.map((c) => {
    const row: number[] = [];
    for (let h = 2; h <= 24; h++) row.push(Math.log(c[h - 1] / c[0]));
    return row;
  });
  const logP1s: number[] = curves.map((c) => Math.log(c[0]));

  const rows: Array<{ hour: number; q: number[] }> = [];
  for (let hi = 0; hi < HOURS.length; hi++) {
    const h = HOURS[hi];
    // Local window: comparables whose hour-1 print is near this case's.
    let idx: number[] = [];
    for (let i = 0; i < curves.length; i++) {
      if (Math.abs(logP1s[i] - logP1) <= WINDOW) idx.push(i);
    }
    if (idx.length < MIN_LOCAL) idx = curves.map((_, i) => i);
    // Local linear fit of this hour's log-ratio on ln(v_1).
    const xs = idx.map((i) => logP1s[i]);
    const ys = idx.map((i) => logRatios[i][hi]);
    const [a, b] = linfit(xs, ys);
    const medLr = a + b * logP1;
    const resid = idx.map((i, k) => ys[k] - (a + b * xs[k]));
    let sigma = stdev(resid);
    if (!Number.isFinite(sigma) || sigma <= 0) sigma = SIGMA_FALLBACK;
    sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX);
    // Quantiles around the median, log-normal spread.
    const q = Z.map((z) => {
      const v = Math.max(1, p1) * Math.exp(medLr + z * sigma);
      return round2(clamp(v, 1, 1e9));
    });
    // Enforce non-decreasing (rounding can tie adjacent values; that is fine,
    // but a decrease from floating error must not happen).
    for (let k = 1; k < q.length; k++) q[k] = Math.max(q[k], q[k - 1]);
    rows.push({ hour: h, q });
  }
  return rows;
}

/** House-baseline fallback (persistence) when a case has no usable comparables. */
function fallbackRows(p1: number): Array<{ hour: number; q: number[] }> {
  const p = Math.max(1, p1);
  return HOURS.map((h) => ({
    hour: h,
    q: Z.map((z) => round2(clamp(p * Math.exp(z * 0.15 * Math.sqrt(h - 1)), 1, 1e9))),
  }));
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith(".in")).sort();
for (const file of inputs) {
  const id = file.slice(0, -".in".length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), "utf8")) as CaseFile;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  const first = c.known.find((k) => k.hour === 1);
  if (first === undefined) throw new Error(`${file} has no hour-1 print`);
  const p1 = Math.max(1, first.viewers);
  const curves: number[][] = [];
  for (const comp of c.comparables ?? []) {
    const curve = releaseDayCurve(comp);
    if (curve !== null) curves.push(curve);
  }
  const hours = curves.length >= 3 ? forecastCase(p1, curves) : fallbackRows(p1);
  // Exactly the forecast hours, in order.
  const wanted = new Set(c.forecast.map((f) => f.hour));
  const out = hours.filter((r) => wanted.has(r.hour));
  if (out.length !== 23) throw new Error(`${file}: expected 23 forecast rows, got ${out.length}`);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ hours: out })}\n`);
}
