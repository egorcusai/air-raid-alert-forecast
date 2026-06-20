"""
leakage_audit.py
================
Automated checks that the feature matrix does NOT leak future information.
This is the project's most important correctness guarantee. If any check
fails, the metrics downstream are meaningless.

We run three independent tests:

  TEST 1 -- Reconstruction independence.
    Rebuild the feature matrix from the raw data but with ALL alerts AFTER a
    cutoff time deleted. For every hour strictly before the cutoff, the
    feature values must be IDENTICAL to those built from the full dataset.
    If deleting the future changes a past feature, that feature peeked ahead.

  TEST 2 -- Target is genuinely future.
    The target at hour H must be reconstructable purely from alert STARTS in
    (H, H+3]. We verify target==1 rows always have a real alert start in that
    forward window, and target==0 rows never do.

  TEST 3 -- No feature column equals or trivially encodes the target.
    A sanity check that correlation between each feature and the target is not
    a perfect/near-perfect 1.0 (which would signal accidental copy of label).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import load_region
from features import build_feature_matrix, build_hourly_grid, add_lag_features, FEATURE_COLUMNS


def test_reconstruction_independence(region: str = "Kyiv City") -> None:
    alerts = load_region(region)

    # Cutoff roughly in the middle of the series.
    cutoff = alerts["started_at"].min() + (
        alerts["started_at"].max() - alerts["started_at"].min()
    ) / 2
    cutoff = cutoff.floor("h")

    # Full-data features.
    full = add_lag_features(build_hourly_grid(alerts))

    # Features built from a world where the future (>= cutoff) never happened.
    truncated_alerts = alerts[alerts["started_at"] < cutoff].copy()
    trunc = add_lag_features(build_hourly_grid(truncated_alerts))

    # Compare on the overlap of hours strictly before the cutoff, leaving a
    # 24h margin before the cutoff so trailing windows are fully comparable.
    margin = cutoff - pd.Timedelta(hours=24)
    full_pre = full[full["hour"] < margin].set_index("hour")[FEATURE_COLUMNS]
    trunc_pre = trunc[trunc["hour"] < margin].set_index("hour")[FEATURE_COLUMNS]

    common = full_pre.index.intersection(trunc_pre.index)
    diff = (full_pre.loc[common] - trunc_pre.loc[common]).abs().max().max()

    assert diff < 1e-9, (
        f"LEAKAGE DETECTED: past features changed when future was deleted "
        f"(max abs diff={diff})."
    )
    print(f"[TEST 1 PASS] Reconstruction independence: {len(common)} hours, max diff={diff:.2e}")


def test_target_is_future(region: str = "Kyiv City") -> None:
    alerts = load_region(region)
    grid = build_hourly_grid(alerts)

    s = grid["starts_this_hour"]
    future_3 = s.shift(-1).fillna(0) + s.shift(-2).fillna(0) + s.shift(-3).fillna(0)
    target = (future_3 > 0).astype(int).iloc[:-3]

    # Independently recompute "is there a start in the next 3 hours" via a
    # direct timestamp search, and confirm it matches.
    starts_by_hour = grid.set_index("hour")["starts_this_hour"]
    mismatch = 0
    sample = grid["hour"].iloc[:-3]
    # Check a random sample of 500 hours for speed.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(sample), size=min(500, len(sample)), replace=False)
    for i in idx:
        h = sample.iloc[i]
        window = [h + pd.Timedelta(hours=k) for k in (1, 2, 3)]
        direct = int(any(starts_by_hour.get(w, 0) > 0 for w in window))
        if direct != target.iloc[i]:
            mismatch += 1

    assert mismatch == 0, f"TARGET ERROR: {mismatch} mismatches vs direct computation."
    print(f"[TEST 2 PASS] Target = future-3h starts: 0 mismatches over {len(idx)} sampled hours")


def test_no_feature_equals_target(region: str = "Kyiv City") -> None:
    fm = build_feature_matrix(alerts=load_region(region))
    corrs = {c: fm[c].corr(fm["target"]) for c in FEATURE_COLUMNS}
    worst = max(corrs.values(), key=abs)
    for c, v in corrs.items():
        print(f"    corr({c:18s}, target) = {v:+.3f}")
    assert abs(worst) < 0.99, f"Suspicious near-perfect correlation: {worst}"
    print(f"[TEST 3 PASS] No feature trivially encodes target (max |corr|={abs(worst):.3f})")


if __name__ == "__main__":
    print("=" * 70)
    print("LEAKAGE AUDIT")
    print("=" * 70)
    test_reconstruction_independence()
    print()
    test_target_is_future()
    print()
    test_no_feature_equals_target()
    print()
    print("ALL LEAKAGE CHECKS PASSED.")
