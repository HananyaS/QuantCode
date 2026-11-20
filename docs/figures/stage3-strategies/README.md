# Stage 3 trade recorder

An interactive isometric figure (anatomy kit) that explains the 13 Stage 3 algo-phase arms of the
pool-then-trade POC as settings of one machine, replaying real dev trades of each arm.

| Part | Means |
|---|---|
| Lit pen carriage and its trace | the price of a real dev trade, bar by bar (o → l → h → c, or o → h → l → c on a down bar) |
| Collars above and below the pen | take-profit and stop-loss; the SL collar parks at the bottom for H60 (1,∞) |
| Fence on the lower rail | the time limit: the open of bar T+1 (or the first present open after it) |
| Dial gauge / five pins on the trigger head | the score each minute (±2% full scale) / the five true-bounce conditions (H7) |
| Knife switch | the H5 day gate; it opens and closes on each trade for h5 arms |
| Cam rack and follower | the exit plan (x15, x30, x30d2, x60, x60u2, x60d2, x60n, chart), per trade for h7 menu/direct |
| Knob | the re-entry rule (cool5, once, nosl, episode) |
| Balance | gross per $ (left pan) against cost per $ (right pan), one disc per 0.1% |
| Range knob on the board | the chart's full scale for this trade (2.5, 5, 10, 25 or 50%) |

Definitions are from `research/intraday/docs/design/2026-11-05-stage3-algo-program.md` (§3, §9, §9a,
§10, §10a). Results are the logged Phase C and H7 results (`results.json` of `s3-c` and `s3-m`).

## Data

- `data/extract_strategies.py` replays every arm descriptively through `phase_c.run_arm` (asserting each
  arm's per-$ equals the logged value to 1e-12), splits gross and cost, and draws 14 seeded random
  trades per arm, relabelling each through the labeller (asserting class, exit minute and r_net).
  Run it from the poolpoc run4 checkout: `python data/extract_strategies.py data/extract.json`.
- `data/compact.py <stage2m root> data/extract.json data/figure.json` merges the extraction with the
  logged results and rounds paths to basis points. It asserts each arm's replayed net equals its
  logged per-$ result.

No number in the figure is typed by hand: `model.mjs` holds definitions only.

## Build

```
node build.mjs              # stage3-strategies.html (dark)
node build.mjs --light      # stage3-strategies-light.html
node build.mjs --artifact   # also stage3-strategies-artifact.html (content only, follows the viewer's theme)
```

The kit is read from `$ANATOMY_HOME` (default `~/.claude/skills/anatomy`).

Verified: line ends 0 in five finished states (b0, h1c, h5 drift, h7 chart, h7 direct) and in light;
4× close-ups of the trigger head, gate, rack, gantry and balance; phone width 390 with no sideways
scroll; console clean; reduced motion shows each trade finished.
