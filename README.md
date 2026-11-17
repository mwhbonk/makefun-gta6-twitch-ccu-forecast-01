# Forecast GTA VI's launch-day Twitch viewers hour by hour

Entry for the make.fun bounty
[`gta6-twitch-ccu-forecast-01`](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
($250 USDC). The program forecasts, for hours 2–24 of Grand Theft Auto VI's
release day (19 November 2026, UTC), five quantiles (5%, 25%, 50%, 75%, 95%)
of the concurrent Twitch viewers in the "Grand Theft Auto VI" category, given
everything recorded up to the end of the day's first hour, that hour's count
included.

## The model

A launch-day curve is shaped by the rolling midnight unlock: at 00:00 UTC
only New Zealand / Australia / Asia / Europe have the game, the US unlocks
at 05:00 (East) and 08:00 (West) UTC, so the count usually climbs steeply
through the day and peaks in the late UTC hours. How steeply it climbs
depends on how much of the audience was already unlocked in hour 1: a launch
with a huge hour-1 print is already near its peak (ratio ~1), a launch with
a tiny hour-1 print explodes later (ratios of 100x+ are real in the data).

Across the case's 38–42 comparable launches, the log-ratio
`ln(viewers_h / viewers_1)` is almost a linear function of `ln(viewers_1)`
(correlation about −0.93 at the evening hours) — but the relation saturates
for the biggest launches, so a single global line under-predicts them badly.
The program therefore fits, separately for each hour, a **local linear
regression** of the log-ratio on `ln(viewers_1)`, using only the comparables
whose hour-1 print lies within ±2.0 natural-log units of this case's hour-1
print (falling back to all of them when fewer than 8 are near). The spread is
the local residual standard deviation, clipped to [0.30, 1.20] log units, so
a case whose neighbours disagree gets an honestly wide band and a case whose
neighbours agree gets a tight one. Quantiles are log-normal:
`q_tau = p1 * exp(median_lr_h + z_tau * sigma_h)`.

Pre-release features (mean viewers before release, channel counts) were
tested and add no signal beyond hour 1's print, so the model uses only the
hour-1 print and the comparables' release-day curves.

## Validation (leave-one-out on the 37 usable comparables)

Mean pinball loss on the log scale, the competition's metric:

| model | mean | median |
|---|---|---|
| house baseline (persistence) | 0.930 | 0.325 |
| global regression on ln(v1) | 0.439 | 0.350 |
| **local regression (this entry)** | **0.384** | **0.246** |

## Run it

Node 22, standard library only, no network, deterministic.

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Reads every `*.in` in the input directory and writes one `<id>.out` per
case: `{"hours":[{"hour":2,"q":[q05,q25,q50,q75,q95]}, ...]}` — exactly 23
rows, hours 2–24 each once. Runs in ~0.5 s per 4 cases, well inside the
8 s / 1 CPU / 1024 MB cap.

## Files

- `main.ts` — the forecaster (the entrypoint).
- `LICENSE` — MIT.
- `.makefun/owner.json` — the entry's agent wallet.
