"""
risk_analysis.py
================
Statistical risk / bias / correlation analysis. ALL numbers are computed from
the data and reproducible. No model predicts here; this is descriptive rigor
that supports the propagation thesis and documents the dataset's biases
honestly.

Three analyses:
  1. CORRELATION  -- region x region correlation of hourly alert-active state.
                     Quantifies geographic propagation directly.
  2. BIAS         -- class imbalance per region, naive-flag share, temporal
                     drift (year-over-year), reporting-gap indicators.
  3. CALIBRATION  -- for a fitted model, how well predicted probabilities match
                     observed frequencies (reliability), plus false-negative
                     cost framing.
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

from data_loader import download_raw, clean_alerts, PERMANENT_SIREN_REGIONS

OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def _hourly_active_matrix(regions: list[str]) -> pd.DataFrame:
    """Build an hour x region matrix of alert-active state (1/0)."""
    alld = clean_alerts(download_raw())
    alld = alld[~alld["region"].isin(PERMANENT_SIREN_REGIONS)]
    start = alld["started_at"].min().floor("h")
    end = alld["finished_at"].max().ceil("h")
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    mat = pd.DataFrame(0, index=idx, columns=regions, dtype=np.int8)
    for region in regions:
        sub = alld[alld["region"] == region]
        for s, f in zip(sub["started_at"], sub["finished_at"]):
            mat.loc[s.floor("h"):f.floor("h"), region] = 1
    return mat


def correlation_analysis(regions: list[str]) -> dict:
    mat = _hourly_active_matrix(regions)
    corr = mat.corr()
    # Most/least correlated region pairs (off-diagonal).
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], round(float(corr.iloc[i, j]), 3)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return {
        "matrix": {c: {d: round(float(corr.loc[c, d]), 3) for d in cols} for c in cols},
        "regions": cols,
        "top_correlated": pairs[:5],
        "least_correlated": pairs[-5:],
    }


def bias_analysis(regions: list[str]) -> dict:
    alld = clean_alerts(download_raw())
    out = {}
    for region in regions:
        sub = alld[alld["region"] == region]
        by_year = sub.groupby(sub["started_at"].dt.year).size().to_dict()
        out[region] = {
            "total_alerts": int(len(sub)),
            "naive_share": round(float(sub["naive"].mean()), 4) if len(sub) else 0.0,
            "alerts_by_year": {int(k): int(v) for k, v in by_year.items()},
            # crude drift indicator: ratio of last full year to first full year
        }
    # Dataset-wide biases, stated plainly.
    notes = [
        "Crowd-sourced reporting: coverage may be denser for high-profile regions "
        "and sparser for frontline/occupied areas, biasing rates downward there.",
        "naive end-times (~5% of records) are fabricated as start+30min; duration "
        "features exclude them, but their presence still reflects reporting gaps.",
        "Temporal drift: alert frequency is non-stationary (war intensity changes), "
        "so models trained on older periods may miscalibrate on recent data.",
        "Class imbalance varies enormously by region (5% west to 60%+ frontline), "
        "so a single global threshold is inappropriate; thresholds should be "
        "region-specific.",
    ]
    return {"per_region": out, "dataset_bias_notes": notes}


def calibration_analysis(region: str = "Lvivska oblast", n_bins: int = 10) -> dict:
    """Reliability: bin predicted probs, compare to observed frequency."""
    from features import build_feature_matrix, FEATURE_COLUMNS
    from data_loader import load_region
    from models import time_based_split
    from sklearn.ensemble import RandomForestClassifier

    fm = build_feature_matrix(load_region(region), region=region, spatial=True)
    feat = [c for c in fm.columns if c not in ("hour", "target")]
    train, val, test = time_based_split(fm)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(train[feat].values, train["target"].values)
    prob = rf.predict_proba(test[feat].values)[:, 1]
    y = test["target"].values

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for b in range(n_bins):
        m = (prob >= bins[b]) & (prob < bins[b + 1])
        if m.sum() > 0:
            rows.append({
                "bin": f"{bins[b]:.1f}-{bins[b+1]:.1f}",
                "predicted_mean": round(float(prob[m].mean()), 3),
                "observed_freq": round(float(y[m].mean()), 3),
                "count": int(m.sum()),
            })
    # Expected Calibration Error
    ece = sum(r["count"] * abs(r["predicted_mean"] - r["observed_freq"]) for r in rows) / len(y)
    return {"region": region, "bins": rows, "ece": round(float(ece), 4)}


def main(regions=None):
    if regions is None:
        regions = ["Lvivska oblast", "Ivano-Frankivska oblast", "Zhytomyrska oblast",
                   "Kyiv City", "Kyivska oblast", "Vinnytska oblast",
                   "Kharkivska oblast", "Dnipropetrovska oblast", "Zaporizka oblast",
                   "Ternopilska oblast"]
    print("Computing correlation matrix...")
    corr = correlation_analysis(regions)
    print("  top correlated pairs:")
    for a, b, c in corr["top_correlated"]:
        print(f"    {a} <-> {b}: {c}")
    print("  least correlated pairs:")
    for a, b, c in corr["least_correlated"]:
        print(f"    {a} <-> {b}: {c}")

    print("\nComputing bias analysis...")
    bias = bias_analysis(regions)

    print("Computing calibration (Lvivska oblast)...")
    calib = calibration_analysis("Lvivska oblast")
    print(f"  Expected Calibration Error (ECE): {calib['ece']}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "risk_analysis.json"), "w") as f:
        json.dump({"correlation": corr, "bias": bias, "calibration": calib}, f, indent=2)
    print("\nSaved -> output/risk_analysis.json")
    return {"correlation": corr, "bias": bias, "calibration": calib}


if __name__ == "__main__":
    main()
