// Trophy Rarity Forecaster v2 — GTA VI week-one trophy earn-rate quantiles.
//
// Upgrade over v1 (single global blend, fixed 1.0*sqrt(days) spread):
// per-band calibration + multi-horizon backtest.
//
// 1. Earn-rate bands. Trophies are grouped by last observed rate into
//    four bands (COMMON >= 5%, MID 1-5%, RARE 0.2-1%, ULTRA < 0.2%).
//    Blend weights are learned separately per band, so a band whose
//    dilution dynamics differ gets its own candidate mix instead of one
//    global mix (the post-mortem lesson from gta6-twitch-ccu-forecast-01:
//    never one global model — calibrate per tier).
// 2. Multi-horizon rolling-origin backtest. Origins k = 1..n-2 forecast
//    snapshots k+1, k+4, k+12 and k+24 ahead (weights 1/2/3/4 by horizon),
//    scored by pinball loss per band. Short-horizon-only backtests reward
//    persistence even when it fails 4.5 days out; the longer horizons
//    carry the most weight. Softmax w = exp(-loss/0.5) per band.
// 3. Horizon-calibrated spread. The blend's residual sd is measured per
//    band at each backtest horizon and extrapolated to the target horizon
//    with a power law (log sd vs log hours, slope clamped to [0, 1]);
//    fallback is sqrt-scaling from the longest measured horizon. Floor
//    0.75 logit (protects flat histories), cap 3.0 logit. This replaces
//    v1's fixed 1.0*sqrt(days) spread, which the synthetic harness showed
//    is far too wide once the median is roughly right.
//    A band-pooled logit-linear candidate and a long-horizon bias
//    correction were both tried in ablation and DROPPED: the pooled
//    candidate hurt in every world, and the bias correction
//    over-corrected (12h-horizon bias does not extrapolate to 108h).
//
// Candidates (per trophy, in logit space):
//   P — persistence: last observed rate.
//   L — logit-linear OLS on ln(1+hours) over all snapshots.
//   R — logit-linear OLS over the last 24h only (fallback P if <3 points).
//   C — counts: separate OLS on ln(E) and ln(N) (E = rate*N/100, only
//       snapshots with E >= 0.5, >= 3 points else fallback P).
//
// Special case: a trophy at exactly 100% in every snapshot gets
// [97, 99, 100, 100, 100] instead of a wide spread around 100.
//
// Input:  /input/<id>.in (or $MAKEFUN_INPUT_DIR), one case per file.
// Output: /output/<id>.out (or $MAKEFUN_OUTPUT_DIR):
//   {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}
// Rates are percents 0..100, quantiles non-decreasing, file <= 65536
// bytes. Node 22, stdlib only, no network, fully deterministic.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const T0 = Date.parse("2026-11-19T00:00:00Z");
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const TEMP = 0.5;
const SPREAD_FLOOR = 0.75;
const SPREAD_CAP = 3.0;
// Backtest horizons: [snapshots ahead, weight]. Longer horizons weigh more.
const HORIZONS: [number, number][] = [[1, 1], [4, 2], [12, 3], [24, 4]];

type Read = { slot: string; players_tracked: number; rates: (number | null)[] };
type Snap = { hours: number; n: number; rates: (number | null)[] };

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}
function logit(p: number): number {
  const c = clamp(p, 0.01, 99.99) / 100;
  return Math.log(c / (1 - c));
}
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0;
}
function sd(xs: number[]): number {
  if (xs.length < 2) return 0;
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1));
}
// Ordinary least squares y ~ a + b*x. Returns [a, b]; null when degenerate.
function ols(xs: number[], ys: number[]): [number, number] | null {
  if (xs.length < 2) return null;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < xs.length; i++) {
    sxx += (xs[i] - mx) * (xs[i] - mx);
    sxy += (xs[i] - mx) * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  const b = sxy / sxx;
  return [my - b * mx, b];
}
// Pinball loss on the logit scale, levels from the case.
function pinball(actualPct: number, qPct: number, tau: number): number {
  const d = logit(actualPct) - logit(qPct);
  return 2 * Math.max(tau * d, (tau - 1) * d);
}

// Earn-rate band of a trophy from its last observed rate.
function bandOf(lastRate: number): number {
  if (lastRate >= 5) return 0; // COMMON
  if (lastRate >= 1) return 1; // MID
  if (lastRate >= 0.2) return 2; // RARE
  return 3; // ULTRA
}

function dedup(reads: Read[]): Snap[] {
  const out: Snap[] = [];
  for (const r of reads) {
    const hours = (Date.parse(r.slot) - T0) / 3600000;
    const last = out[out.length - 1];
    if (
      last &&
      last.n === r.players_tracked &&
      last.rates.length === r.rates.length &&
      last.rates.every((v, i) => v === r.rates[i])
    ) {
      continue;
    }
    out.push({ hours, n: r.players_tracked, rates: r.rates.slice() });
  }
  return out;
}

// Per-trophy series: valid (hours, rate) pairs, skipping null rates.
function series(snaps: Snap[], upto: number, ti: number): { x: number; y: number; h: number }[] {
  const pts: { x: number; y: number; h: number }[] = [];
  for (let s = 0; s <= upto && s < snaps.length; s++) {
    const r = snaps[s].rates[ti];
    if (r === null || r === undefined) continue;
    pts.push({ x: Math.log1p(snaps[s].hours), y: logit(r), h: snaps[s].hours });
  }
  return pts;
}

// Fit the four candidates for trophy ti on snapshots 0..upto and read
// each at target log-time xt. Returns medians in percent.
function candidates(snaps: Snap[], upto: number, ti: number, xt: number): number[] {
  const pts = series(snaps, upto, ti);
  const lastPct = pts.length
    ? clamp(100 * sigmoid(pts[pts.length - 1].y), 0.01, 99.99)
    : 50;
  const fromLogit = (v: number) => clamp(100 * sigmoid(v), 0.01, 99.99);

  // P — persistence.
  const p = lastPct;

  // L — logit-linear, all history.
  let l = p;
  if (pts.length >= 2) {
    const fit = ols(pts.map((q) => q.x), pts.map((q) => q.y));
    if (fit) l = fromLogit(fit[0] + fit[1] * xt);
  }

  // R — logit-linear, last 24h only.
  let r = p;
  const hLast = pts.length ? pts[pts.length - 1].h : 0;
  const recent = pts.filter((q) => hLast - q.h <= 24);
  if (recent.length >= 3) {
    const fit = ols(recent.map((q) => q.x), recent.map((q) => q.y));
    if (fit) r = fromLogit(fit[0] + fit[1] * xt);
  }

  // C — counts E/N, only snapshots with E >= 0.5, >= 3 points.
  let c = p;
  const ePts: { x: number; e: number }[] = [];
  const nPts: { x: number; n: number }[] = [];
  for (let s = 0; s <= upto && s < snaps.length; s++) {
    const rate = snaps[s].rates[ti];
    if (rate === null || rate === undefined) continue;
    const e = (rate * snaps[s].n) / 100;
    if (e >= 0.5) {
      ePts.push({ x: Math.log1p(snaps[s].hours), e: Math.log(e) });
      nPts.push({ x: Math.log1p(snaps[s].hours), n: Math.log(snaps[s].n) });
    }
  }
  if (ePts.length >= 3) {
    const fe = ols(ePts.map((q) => q.x), ePts.map((q) => q.e));
    const fn = ols(nPts.map((q) => q.x), nPts.map((q) => q.n));
    if (fe && fn) {
      const eT = Math.exp(fe[0] + fe[1] * xt);
      const nT = Math.exp(fn[0] + fn[1] * xt);
      c = clamp((100 * eT) / nT, 0.01, 99.99);
    }
  }

  return [p, l, r, c];
}

function forecastCase(c: any): { trophies: { id: number; q: number[] }[] } {
  const levels: number[] = c.levels;
  const targetMs = Date.parse(c.target);
  const atMs = Date.parse(c.at);
  const targetHours = (targetMs - atMs) / 3600000;
  const xt = Math.log1p((targetMs - T0) / 3600000);
  const reads: Read[] = Object.values(c.history ?? {});
  const snaps = dedup(reads);
  const nT = (c.trophies as any[]).length;
  const NC = 4;

  // Last observed rate per trophy (for banding + the 100% special case).
  const lastRate: number[] = [];
  const allHundred: boolean[] = [];
  for (let ti = 0; ti < nT; ti++) {
    let lr = 50;
    let hundred = snaps.length > 0;
    for (const s of snaps) {
      const v = s.rates[ti];
      if (v === null || v === undefined) continue;
      lr = v;
      if (v !== 100) hundred = false;
    }
    lastRate.push(lr);
    allHundred.push(hundred);
  }
  const band: number[] = lastRate.map(bandOf);

  // Multi-horizon rolling-origin backtest, scored per band.
  const losses: number[][] = Array.from({ length: 4 }, () => new Array(NC).fill(0));
  const wsum: number[][] = Array.from({ length: 4 }, () => new Array(NC).fill(0));
  const nOrigins = Math.max(0, snaps.length - 2);
  // Blend residuals per band per horizon, collected for the spread fit.
  type Resid = { b: number; hi: number; meds: number[]; actual: number };
  const residList: Resid[] = [];
  const sumHours: number[] = HORIZONS.map(() => 0);
  const cntHours: number[] = HORIZONS.map(() => 0);
  for (let k = 1; k <= snaps.length - 2; k++) {
    const wOrigin = nOrigins >= 4 && k >= snaps.length - 3 ? 2 : 1;
    for (let hi = 0; hi < HORIZONS.length; hi++) {
      const [step, wH] = HORIZONS[hi];
      const j = k + step;
      if (j >= snaps.length) continue;
      sumHours[hi] += snaps[j].hours - snaps[k].hours;
      cntHours[hi] += 1;
      const xNext = Math.log1p(snaps[j].hours);
      for (let ti = 0; ti < nT; ti++) {
        const actual = snaps[j].rates[ti];
        if (actual === null || actual === undefined) continue;
        const meds = candidates(snaps, k, ti, xNext);
        const b = band[ti];
        for (let cj = 0; cj < NC; cj++) {
          let loss = 0;
          for (const tau of levels) loss += pinball(actual, meds[cj], tau);
          losses[b][cj] += (wOrigin * wH * loss) / levels.length;
          wsum[b][cj] += wOrigin * wH;
        }
        residList.push({ b, hi, meds, actual });
      }
    }
  }

  // Band blend weights (softmax temp 0.5); uniform when no origins.
  const weights: number[][] = [];
  for (let b = 0; b < 4; b++) {
    if (wsum[b][0] <= 0) {
      weights.push(new Array(NC).fill(1 / NC));
      continue;
    }
    const ls = losses[b].map((v, j) => v / Math.max(1, wsum[b][j]));
    const ex = ls.map((v) => Math.exp(-v / TEMP));
    const s = ex.reduce((a, v) => a + v, 0);
    weights.push(ex.map((v) => v / s));
  }

  // Blend residuals per band per horizon -> horizon-calibrated spread.
  const residBH: number[][][] = Array.from({ length: 4 }, () =>
    HORIZONS.map(() => [] as number[]),
  );
  for (const r of residList) {
    let blendLogit = 0;
    for (let cj = 0; cj < NC; cj++) blendLogit += weights[r.b][cj] * logit(r.meds[cj]);
    residBH[r.b][r.hi].push(logit(r.actual) - blendLogit);
  }
  const horizonHours = HORIZONS.map((_, hi) =>
    cntHours[hi] ? sumHours[hi] / cntHours[hi] : (hi + 1) * 0.5,
  );
  const spreadBand: number[] = [];
  for (let b = 0; b < 4; b++) {
    // Longest horizon with enough residuals (for the sqrt fallback).
    let sdLong = 0;
    let hLong = 0;
    for (let hi = HORIZONS.length - 1; hi >= 0; hi--) {
      if (residBH[b][hi].length >= 3) {
        sdLong = sd(residBH[b][hi]);
        hLong = horizonHours[hi];
        break;
      }
    }
    // Power-law fit log(sd) ~ a + slope*log(hours), slope in [0, 1].
    const xs: number[] = [];
    const ys: number[] = [];
    for (let hi = 0; hi < HORIZONS.length; hi++) {
      const s = sd(residBH[b][hi]);
      if (residBH[b][hi].length >= 3 && s > 1e-6) {
        xs.push(Math.log(horizonHours[hi]));
        ys.push(Math.log(s));
      }
    }
    const fit = xs.length >= 2 ? ols(xs, ys) : null;
    let spread: number;
    if (fit) {
      const slope = clamp(fit[1], 0, 1);
      spread = Math.exp(fit[0] + slope * Math.log(Math.max(targetHours, 1)));
    } else if (hLong > 0 && sdLong > 0) {
      spread = sdLong * Math.sqrt(targetHours / hLong);
    } else {
      spread = SPREAD_FLOOR;
    }
    spreadBand.push(clamp(Math.max(SPREAD_FLOOR, spread), 0, SPREAD_CAP));
  }

  // Final forecast on all snapshots.
  const upto = snaps.length - 1;
  const out: { id: number; q: number[] }[] = [];
  for (let ti = 0; ti < nT; ti++) {
    const id = (c.trophies[ti] as any).id as number;
    if (allHundred[ti]) {
      out.push({ id, q: [97, 99, 100, 100, 100] });
      continue;
    }
    const meds = candidates(snaps, upto, ti, xt);
    const w = weights[band[ti]];
    let blendLogit = 0;
    for (let cj = 0; cj < NC; cj++) blendLogit += w[cj] * logit(meds[cj]);
    const spread = spreadBand[band[ti]];
    const qs = Z.map((z) => {
      const v = 100 * sigmoid(blendLogit + z * spread);
      return Math.round(clamp(v, 0, 100) * 10000) / 10000;
    });
    // Enforce non-decreasing (rounding can only tie, but be safe).
    for (let i = 1; i < qs.length; i++) if (qs[i] < qs[i - 1]) qs[i] = qs[i - 1];
    out.push({ id, q: qs });
  }
  out.sort((a, b) => a.id - b.id);
  return { trophies: out };
}

function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? "/output";
  mkdirSync(outDir, { recursive: true });
  const files = readdirSync(inDir).filter((f) => f.endsWith(".in"));
  if (!files.length) throw new Error(`no .in files in ${inDir}`);
  for (const f of files) {
    const c = JSON.parse(readFileSync(join(inDir, f), "utf8"));
    const result = forecastCase(c);
    const outName = f.replace(/\.in$/, ".out");
    writeFileSync(join(outDir, outName), JSON.stringify(result) + "\n");
  }
}

main();
