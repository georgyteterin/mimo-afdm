# mimo-afdm

BER simulator for OFDM and AFDM: Python, YAML configs, JSON results.

All commands run from the repository root.

## Setup

```bash
conda activate <env>    # python 3.12, numpy, scipy, matplotlib, pyyaml, pytest
pytest -q               # all tests should pass
```

## 1. Describe the simulation

A sim config is a yaml file in `configs/sim/` with one or more tests:

```yaml
defaults: &common
  mod_order: 4
  grid: {num_sc: 64, num_blocks: 8, scs_hz: 15000}
  channel:
    paths:
      - {delay_us: 10, avg_gain_db: -3.0}
      - {delay_us: 5, avg_gain_db: -6.0}
    max_doppler_hz: 1500
    doppler_mode: jakes        # or fixed: doppler_hz / doppler_norm per path
    fading: rayleigh           # or none
  noise:
    ebn0_db: [0, 2, 4, 6, 8, 10]
    count_cp_energy: false
  seed: 0                      # null: random, the drawn seed is logged and saved
  num_trials: 1000

tests:
  ofdm_mmse:
    <<: *common
    waveform: {type: ofdm, cp_len: 0.25}
    equalizer: {type: mmse_full}       # transparent, mmse_full, mmse_diag
  afdm_mmse:
    <<: *common
    waveform: {type: afdm, cp_len: 0.25}
    equalizer: {type: mmse_full}
```

Tests with the same `seed` get the same bits, channel and noise at each Eb/N0
point, so waveforms are compared on identical realizations.

Put drafts in `configs/sim/local/` (any depth): git ignores them.

## 2. Run

Two runners write the same results, pick by the job:

| | `runners.ber_vs_ebn0` | `runners.ber_vs_ebn0_parallel` |
|---|---|---|
| use for | debugging, quick checks | long runs, smooth curves |
| cores | one process, breakpoints work | `--workers` (default: 90% of cores) |
| log | every Eb/N0 point | progress, ETA, summary |
| trials per point | exactly `num_trials` | stops at `--target-err` errors (default 1000), `num_trials` is the maximum |
| Eb/N0 grid | `noise.ebn0_db` | `noise.ebn0_db` or `noise.adaptive` |

```bash
python -m runners.ber_vs_ebn0 --cfg configs/sim/fading_test.yaml --test afdm_mmse
python -m runners.ber_vs_ebn0_parallel --cfg configs/sim/long/fading_test.yaml
```

- `--test name1 name2`: run only these tests (all if omitted).
- `--out-dir`: results root (default `results`).
- parallel only: `--workers N`, `--target-err N` (`0` turns early stop off),
  `--chunk-trials N` (trials per task, default 50).

Both runners end with a BER summary table and the time per trial (`ms/trial`).

### Adaptive Eb/N0 grid (parallel runner only)

Replace `ebn0_db` with `adaptive`; see `configs/sim/long/fading_adaptive.yaml`:

```yaml
noise:
  adaptive:
    start_db: 0
    stop_db: 40
    num_coarse_pts: 9          # first pass
    min_step_db: 0.5           # grid step and finest refinement, >= 0.01
    max_log10_ber_step: 0.5    # refine where BER drops by more than 0.5 decade
    min_ber: 1.0e-5            # the curve ends at the first point below it
    max_rounds: 4
  count_cp_energy: false
```

## 3. Results

`results/<cfg path under configs/sim>/<test>.json`, e.g.
`configs/sim/long/fading_test.yaml` -> `results/long/fading_test/afdm_mmse.json`.

Each file holds the full derived config, `ebn0_db`, `ber`, `num_err`, `num_bits`,
the git commit (and whether the tree had uncommitted changes), time and versions.
A rerun overwrites the file; rename it by hand to keep an old one.

## 4. Plot

After the first run of a sim config, the runner creates
`configs/plots/<same path>.yaml` with one curve per test. It is never overwritten:
edit labels, colors, markers and line styles there. If you add a test later,
the runner warns that it has no curve in the plot config.

```bash
python -m plotters.plot_ber --plot-cfg configs/plots/long/fading_test.yaml
```

Allowed keys and values: `configs/plots/REFERENCE.md`. A curve whose result file
does not exist yet is skipped with a warning. Drafts go to `configs/plots/local/`.

Quick look without a plot config:

```bash
python -m plotters.plot_ber results/fading_test --ci --theory qpsk_rayleigh
```

The window has a save button; the plotter itself writes no files.

## 5. Check

```bash
pytest -q          # ~3 s
ruff check .
```

VS Code on Windows with conda: if tests crash on matrix products
(`0xc06d007f`), add `"python.envFile": "${workspaceFolder}/.env"` to
`.vscode/settings.json` and a `.env` file with
`PATH=<env path>\Library\bin;<env path>;${env:PATH}`.
