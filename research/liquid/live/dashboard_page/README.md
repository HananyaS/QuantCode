# C3 dashboard page

The page files of the claude.ai dashboard (artifact `5ELZiTW9YF2qpXieq3yCFA`, a
"Data" dashboard). The dashboard's own store holds the live copy, under
`files/<name>`. These copies are the source to edit and write back with
ArtifactData.

| file | holds |
|---|---|
| `index.html` | all fixed text. Every value is a `data-draw` element stored as "—" (the one-writer rule) |
| `c3.css` | layout, built only on the dashboard's theme tokens |
| `c3-core.js` | the generic `val` drawing, the phase switch (waiting / running / stopped), tabs, print |
| `c3-data.js` | the Drive loaders (gzip+base64 decode) and the calculations |
| `c3-charts.js` | line, bar and intraday day charts, with tooltips |
| `c3-live.js`, `c3-research.js`, `c3-strategy.js` | one tab each |
| `harness.html` | a local test page that mocks `dash` (see below) |

To run the local test:
1. Build the dev files with `python -m research.liquid.live.dashboard --dev OUT`.
2. Bundle them, a payload and `index.html` into `test/harness-data.js`
   (`window.__DATA`, `__LIVE`, `__PAGE`).
3. Open `harness.html?live=fake&tab=research&w=400`.
