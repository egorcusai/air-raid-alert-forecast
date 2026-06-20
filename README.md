# Air-Raid Alert Risk Classification — Ukraine

A short-horizon **risk classifier** for air-raid alerts in Ukraine, built as a
two-day prototype. For a chosen region, it answers a single, operationally
useful question:

> **Will a new air-raid alert begin in this region within the next 3 hours?**

This is deliberately a **binary classification** problem, not exact-time
forecasting. The reasoning behind that choice — and several other decisions — is
documented below, because the *process* matters more than the headline number.

---

## TL;DR — what to look at

- **`run_all.py`** reproduces the entire pipeline in one command.
- **`src/leakage_audit.py`** is the most important file: it *proves* the
  features don't see the future. All three checks pass.
- **Honest result:** with occurrence-count + calendar features alone, the model
  is only marginally better than random (ROC-AUC ≈ 0.54 on a held-out, most-recent
  test period). **This is a real finding, not a bug** — see
  [Results & honest assessment](#results--honest-assessment).

---

## Quick start

```bash
git clone <this-repo>
cd air-raid-forecast
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python run_all.py            # defaults to "Kyiv City"
# python run_all.py "Lvivska oblast"   # any region label works
```

Outputs land in `output/` (`metrics.json`, `results.png`).

---

## Problem framing

| Decision | Choice | Why |
|---|---|---|
| Task type | Binary classification | A 3-hour risk flag is actionable and *validatable*. Exact-minute forecasting is false precision and near-impossible to verify in a prototype. |
| Horizon | Next 3 hours | Long enough to be useful for planning, short enough to be a meaningful signal. |
| Granularity | Hourly grid, per region | One row = (region, hour). Clean, regular time index. |
| Target | `1` if any alert *starts* in hours H+1…H+3 | Unambiguous, reconstructable from raw data. |

---

## Data

**Source:** [`Vadimkin/ukrainian-air-raid-sirens-dataset`](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset),
file `volunteer_data_en.csv` (public, updated daily). ~101k alert records,
Feb 2022 → present, oblast level. All timestamps UTC.

Schema: `region`, `started_at`, `finished_at`, `naive`.

**Three data decisions worth highlighting** (each verified against the source
README, not assumed):

1. **Volunteer set, not official.** The *official* dataset switched from
   oblast-level to raion (district) level in Dec 2025, which would silently
   change the geographic unit mid-series. The volunteer set stays oblast-level
   throughout, giving a consistent target.
2. **Permanent sirens excluded.** Luhansk (3 records total) and Crimea have
   continuous, unlogged sirens. A model "predicting" a permanent alert earns
   fake recall, so these regions are dropped.
3. **`naive` flag handled surgically.** ~5% of records have an *estimated*
   end-time (`started_at + 30min`). We **keep the event** (it really happened →
   valid for the target and occurrence features) but **exclude it from
   duration-derived features** (its duration is fabricated). Trust the event,
   distrust the guessed duration.

---

## Features (all strictly historical)

| Feature | Definition |
|---|---|
| `alerts_last_3h` | Count of alert starts in the 3h **before** the current hour |
| `alerts_last_6h` | …last 6h |
| `alerts_last_24h` | …last 24h |
| `avg_duration_24h` | Mean *observed* alert duration over the last 24h (naive excluded) |
| `hour_of_day` | 0–23 (known in advance) |
| `day_of_week` | 0–6 |
| `month` | 1–12 |

Every rolling feature is computed on a **one-hour-shifted history**, so the
current hour's own activity is never included.

---

## Leakage audit — the core guarantee

For a time-series classifier, the single biggest risk is **future leakage**:
a feature accidentally containing information from the window it's trying to
predict. `src/leakage_audit.py` runs three independent checks:

1. **Reconstruction independence.** Rebuild features after *deleting all future
   data* past a cutoff. Every past feature must be byte-identical. Result:
   **0.00e+00 max difference across 18,836 hours.** If any feature peeked ahead,
   this would be non-zero.
2. **Target is genuinely future.** Independently recompute the label by direct
   timestamp search and confirm it matches. **0 mismatches.**
3. **No feature trivially encodes the target.** Max |correlation| = **0.16** —
   reassuringly low. A near-1.0 correlation would signal accidental label copy.

---

## Validation

**Time-based split — never random.** A random split would leak future into past.
We split chronologically:

```
train: 2022-02-25 → 2025-01-30   (oldest 68%)
val:   2025-01-30 → 2025-08-07   (next 12%, threshold tuning only)
test:  2025-08-07 → 2026-06-18   (most-recent 20%, untouched until final eval)
```

The **validation block** exists so the decision threshold is chosen *without
ever looking at test* — tuning on test is itself a form of leakage we explicitly
avoid.

---

## Metrics — why recall is privileged

Accuracy is misleading on imbalanced data (only ~17–19% of hours are positive).
We report **precision, recall, F1, ROC-AUC, and the confusion matrix**, and we
weight toward **recall**: operationally, a *missed* alert (false negative) is far
more costly than a false alarm (false positive). Models use
`class_weight="balanced"` for the same reason.

---

## Results & honest assessment

Held-out test period (most recent ~10 months), region = Kyiv City:

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression @0.5 | 0.21 | 0.47 | 0.29 | **0.55** |
| Random Forest @0.5 | 0.22 | 0.39 | 0.28 | 0.54 |

**The honest headline: these features barely beat random (ROC-AUC ≈ 0.54).**
That is the main finding, and it's worth more than an inflated number would be:

- A leakage-free pipeline on this task *should* produce modest metrics. Anyone
  reporting 95%+ accuracy here almost certainly has a leak.
- Feature importance points at `avg_duration_24h`, `month`, and `hour_of_day` —
  i.e. weak seasonal/diurnal rhythm — rather than short-term momentum.
- **The likely missing signal is spatial:** air-raid alerts propagate
  geographically (a threat over one oblast precedes alerts in neighbours). Our
  single-region features can't see that. This is the #1 next step.

---

## Limitations

- **No spatial features.** Single-region only; cross-region propagation (the
  probable dominant signal) is not modelled.
- **No external signals.** No data on actual military activity, weather, or
  strategic context — only the alert history itself.
- **Volunteer data caveats.** Crowd-sourced; ~5% estimated end-times; possible
  reporting gaps.
- **Stationarity assumption.** War dynamics shift; patterns from 2022 may not
  hold in 2026. The time-split partly captures this (we test on recent data).

---

## Roadmap (next steps, in priority order)

1. **Spatial features** — counts of active/recent alerts in neighbouring
   oblasts. Most likely to move ROC-AUC meaningfully.
2. **Per-region models + comparison table** — the code is already
   region-parameterised; loop over oblasts.
3. **Sequence models** — once spatial features exist, test gradient boosting /
   a small temporal model against the linear baseline.
4. **Live inference** — the [alerts.in.ua](https://alerts.in.ua) API for
   real-time scoring (kept out of scope here for reproducibility).

---

## Project structure

```
air-raid-forecast/
├── run_all.py            # one-command reproduction
├── requirements.txt
├── src/
│   ├── data_loader.py    # download, clean, filter region
│   ├── features.py       # hourly grid, historical windows, target
│   ├── leakage_audit.py  # 3 automated leakage checks  <-- read this
│   ├── models.py         # time-split, LogReg + RF, recall-weighted eval
│   └── plot_results.py   # confusion matrix + ROC
└── output/
    ├── metrics.json
    └── results.png
```

---

## Disclaimer

Built as a selection-task prototype using public data. **Not** a safety system;
do not rely on it for real-world decisions. Always follow official air-raid
warnings.
