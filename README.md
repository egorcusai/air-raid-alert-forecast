# Where Are Air-Raid Alerts Predictable? — A Regional Study (Ukraine)

A short-horizon **risk classifier** for air-raid alerts in Ukraine, built as a
two-day prototype. For a chosen region it answers:

> **Will a new air-raid alert begin in this region within the next 3 hours?**

But the more interesting question — and the one this project actually answers —
turned out to be **comparative**:

> **In *which* regions are alerts predictable from alert history alone, and does
> knowing a region's geographic neighbours help?**

The answer has a clean geographic logic, reproduces a published finding in our
own data, and is reported honestly — modest numbers included.

---

## TL;DR — the finding

| | |
|---|---|
| **Most predictable** | Western / peripheral oblasts (Lviv, Zhytomyr: ROC-AUC ~0.66) |
| **Least predictable** | Strategic direct targets (Kyiv City: ROC-AUC ~0.55, barely > random) |
| **Spatial features help most** | Small western oblasts threats *cross* (Ivano-Frankivsk +0.039, Ternopil +0.029 AUC) |
| **Spatial features don't help** | Direct targets hit without warning (Kyiv City +0.001) |

**Interpretation:** alerts propagate geographically. A western oblast gets
advance signal as a threat traverses the country, so its alerts follow a
learnable rhythm. A high-priority target like Kyiv is struck directly with little
lead time, so from a single-region view its alerts look closer to noise. This
matches the peer-reviewed result that a region's alert status
[depends heavily on its adjacent regions](https://arxiv.org/abs/2411.14625) — and
we show *where* that dependence is strong vs weak.

---

## Quick start

```bash
git clone <this-repo>
cd air-raid-forecast
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_all.py                  # full pipeline + multi-region comparison
# python run_all.py "Kyiv City"    # focus a single region
```

Outputs land in `output/`: `metrics_temporal.json`, `metrics_spatial.json`,
`region_comparison.csv`, `results.png`.

---

## Why these design choices

| Decision | Choice | Why |
|---|---|---|
| Task | Binary classification, 3h ahead | A risk flag is actionable and *validatable*; exact-minute forecasting is false precision. |
| Granularity | Hourly grid, per region | Clean, regular index; one row = (region, hour). |
| Models | Logistic Regression + Random Forest | Interpretable baseline first. (LR often *wins* here — see results.) |
| Validation | **Time-based** train/val/test | Random splits leak the future into the past in time series. |
| Metric focus | Recall (+ full suite) | A missed alert costs more than a false alarm. |

---

## Data

**Source:** [`Vadimkin/ukrainian-air-raid-sirens-dataset`](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
`volunteer_data_en.csv` (public, daily-updated, ~101k records, Feb 2022 to present,
oblast-level, UTC). Schema: `region`, `started_at`, `finished_at`, `naive`.

Three verified data decisions:

1. **Volunteer, not official, dataset.** The official set switched from oblast to
   raion (district) level in Dec 2025 — confirmed by the official statistics
   provider — which would change the geographic unit mid-series. The volunteer
   set stays oblast-level throughout.
2. **Permanent sirens dropped.** Luhansk (3 records) and Crimea have continuous,
   unlogged sirens; a model "predicting" them earns fake recall.
3. **`naive` flag handled surgically.** ~5% of rows have an estimated end-time
   (`start + 30min`). We keep the **event** (real, so valid for target & occurrence
   features) but exclude it from **duration** features (its duration is fabricated).

---

## Features (all strictly historical)

**Temporal:** `alerts_last_3h/6h/24h`, `avg_duration_24h` (naive-excluded),
`hour_of_day`, `day_of_week`, `month`.

**Spatial** (neighbour activity, from `neighbours.py` adjacency map):
`nbr_alerts_last_1h/3h/6h`, `nbr_active_last_3h` (# distinct neighbours active).

Every rolling feature is computed on a **one-hour-shifted history**, so hour H's
own (and all future) activity is never included.

---

## Leakage audit — the core guarantee

`src/leakage_audit.py` runs three automated checks; all pass:

1. **Reconstruction independence** — rebuild features after deleting all data
   past a cutoff; past features must be byte-identical. **Max diff 0.00e+00 over
   18,836 hours.** Proof of no future peek.
2. **Target is genuinely future** — independent recomputation by timestamp
   search; **0 mismatches**.
3. **No feature trivially encodes the target** — max |corr| = **0.16**.

---

## Validation & metrics

Chronological split — train (oldest 68%) / val (next 12%, threshold tuning only)
/ test (most-recent 20%, untouched until final eval). Tuning the threshold on a
separate validation block avoids tuning-on-test, a subtler leak we caught and
fixed mid-build.

We report precision, recall, F1, ROC-AUC and the confusion matrix, weighting
recall and using `class_weight="balanced"`.

---

## Results

Best-of-(LR, RF) ROC-AUC on the held-out recent test period:

| Region | Pos. rate | AUC temporal | AUC +spatial | Spatial gain | Recall |
|---|---|---|---|---|---|
| Zhytomyrska | 0.16 | 0.656 | **0.663** | +0.007 | 0.59 |
| Lvivska | 0.19 | 0.657 | 0.652 | -0.005 | **0.76** |
| Kharkivska | 0.82 | 0.612 | 0.626 | +0.014 | 0.87 |
| Dnipropetrovska | 0.80 | 0.606 | 0.605 | -0.001 | 0.88 |
| Vinnytska | 0.10 | 0.593 | 0.602 | +0.009 | 0.32 |
| Ternopilska | 0.05 | 0.573 | 0.602 | **+0.029** | 0.40 |
| Kyivska | 0.33 | 0.592 | 0.592 | 0.000 | 0.77 |
| Ivano-Frankivska | 0.05 | 0.539 | 0.578 | **+0.039** | 0.46 |
| Zaporizka | 0.61 | 0.562 | 0.558 | -0.004 | 0.66 |
| **Kyiv City** | 0.19 | 0.547 | 0.548 | +0.001 | 0.47 |

Three honest observations:

- **Geography of predictability is real.** Peripheral west > direct targets. Kyiv
  City is near-random; Lviv/Zhytomyr are meaningfully predictable.
- **Spatial features help exactly where theory says they should** — small western
  oblasts threats traverse — and not where they don't (direct targets).
- **Logistic Regression often beats Random Forest** here. With few, largely
  linear signals, RF overfits the train period and generalizes worse — a useful
  reminder that "fancier" isn't "better." We report both.

High-`pos_rate` frontline regions (Kharkiv, Dnipro) show high F1 partly because
alerts are almost always active there; ROC-AUC and the confusion matrix keep that
honest, which is why we never lead with accuracy.

---

## Limitations

- **Alert history only** — no military-activity, weather, or strategic data.
- **First-order adjacency** — no distance weighting or threat-direction modelling.
- **Crowd-sourced data** — possible gaps; ~5% estimated end-times.
- **Non-stationarity** — war dynamics shift; the time-split tests on recent data
  but patterns still drift.

---

## Roadmap

1. **Directional / distance-weighted spatial features** — threats have a
   trajectory; undirected neighbour counts are a first approximation.
2. **Per-region tuned thresholds** for an operational recall target.
3. **Gradient boosting** once richer spatial features exist.
4. **Live inference** via the [alerts.in.ua](https://alerts.in.ua) API
   (out of scope here for reproducibility).

---

## Project structure

```
air-raid-forecast/
├── run_all.py              # one-command reproduction (+ comparison)
├── requirements.txt
├── src/
│   ├── data_loader.py      # download, clean, filter region
│   ├── features.py         # hourly grid, historical windows, target, spatial toggle
│   ├── spatial_features.py # leakage-safe neighbour-activity features
│   ├── neighbours.py       # oblast adjacency map
│   ├── leakage_audit.py    # 3 automated leakage checks  <-- read this
│   ├── models.py           # time-split, LR + RF, recall-weighted eval
│   ├── compare_regions.py  # multi-region comparison study
│   └── plot_results.py     # confusion matrix + ROC
└── output/
    ├── metrics_temporal.json
    ├── metrics_spatial.json
    ├── region_comparison.csv
    └── results.png
```

---

## Disclaimer

Selection-task prototype on public data. **Not** a safety system — always follow
official air-raid warnings.
