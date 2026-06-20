"""
models.py
=========
Trains and evaluates two baseline classifiers on the 3-hour-ahead air-raid
task, using a TIME-BASED train/test split.

Why time-based split (not random)
----------------------------------
This is a time series. A random split would put future hours in the training
set and past hours in the test set, letting the model learn from the future
to predict the past -- a classic leakage that inflates metrics and never
survives real deployment. We train on the earlier portion of the timeline and
test on the most recent portion, exactly mirroring how the model would be used.

Why these two models
--------------------
- Logistic Regression: fast, fully interpretable baseline. If a simple linear
  model already captures most of the signal, that's worth knowing before
  reaching for anything heavier.
- Random Forest: captures non-linear interactions (e.g. "late night AND high
  recent activity"), and gives feature-importance for free.

Why recall-weighted evaluation
-------------------------------
Operationally, a MISSED alert (false negative) is far more costly than a false
alarm (false positive). So we report the full metric suite but pay special
attention to recall, and we tune the decision threshold to favour it rather
than blindly using 0.5.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data_loader import load_region
from features import FEATURE_COLUMNS, build_feature_matrix

TEST_FRACTION = 0.2  # most-recent 20% of the timeline used for testing
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


def time_based_split(fm: pd.DataFrame, test_fraction: float = TEST_FRACTION,
                     val_fraction: float = 0.15):
    """Split chronologically into train / validation / test.

    The validation block (carved from the END of the training period) is used
    ONLY to choose the decision threshold. The test block (most recent data)
    is never touched until final evaluation. This prevents tuning-on-test,
    which is itself a form of leakage.
    """
    fm = fm.sort_values("hour").reset_index(drop=True)
    cut_test = int(len(fm) * (1 - test_fraction))
    cut_val = int(cut_test * (1 - val_fraction))
    train = fm.iloc[:cut_val]
    val = fm.iloc[cut_val:cut_test]
    test = fm.iloc[cut_test:]
    print(
        f"[split] train={len(train)} ({train['hour'].min()} -> {train['hour'].max()})\n"
        f"        val  ={len(val)} ({val['hour'].min()} -> {val['hour'].max()})\n"
        f"        test ={len(test)} ({test['hour'].min()} -> {test['hour'].max()})"
    )
    return train, val, test


def _evaluate(name, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_true * 0 + y_pred)  # ensure shape
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "model": name,
        "threshold": round(float(threshold), 3),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return metrics


def pick_recall_threshold(y_true, y_prob, min_precision=0.5):
    """Choose the lowest threshold that still keeps precision >= min_precision,
    maximising recall subject to not flooding the user with false alarms.
    Falls back to 0.5 if no threshold satisfies the precision floor.
    """
    best_t, best_recall = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 81):
        y_pred = (y_prob >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_recall, best_t = r, t
    return best_t


def run(region: str = "Kyiv City"):
    fm = build_feature_matrix(load_region(region))
    train, val, test = time_based_split(fm)

    X_train, y_train = train[FEATURE_COLUMNS].values, train["target"].values
    X_val, y_val = val[FEATURE_COLUMNS].values, val["target"].values
    X_test, y_test = test[FEATURE_COLUMNS].values, test["target"].values

    # Scale features for Logistic Regression (RF is scale-invariant but we
    # reuse the scaled matrix for LR only).
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []

    # --- Logistic Regression -------------------------------------------------
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_s, y_train)
    # Threshold chosen on VALIDATION, evaluated on TEST.
    t_lr = pick_recall_threshold(y_val, lr.predict_proba(X_val_s)[:, 1])
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    results.append(_evaluate("LogisticRegression@0.5", y_test, lr_prob, 0.5))
    results.append(_evaluate("LogisticRegression@tuned", y_test, lr_prob, t_lr))

    # --- Random Forest -------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    t_rf = pick_recall_threshold(y_val, rf.predict_proba(X_val)[:, 1])
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results.append(_evaluate("RandomForest@0.5", y_test, rf_prob, 0.5))
    results.append(_evaluate("RandomForest@tuned", y_test, rf_prob, t_rf))

    importances = dict(
        sorted(
            zip(FEATURE_COLUMNS, (round(float(v), 4) for v in rf.feature_importances_)),
            key=lambda kv: kv[1], reverse=True,
        )
    )

    # --- Report --------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"RESULTS  (region={region}, test positive rate={y_test.mean():.1%})")
    print("=" * 72)
    print(f"{'model':32s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'roc_auc':>8s}")
    for m in results:
        print(f"{m['model']:32s} {m['precision']:6.3f} {m['recall']:7.3f} "
              f"{m['f1']:6.3f} {m['roc_auc']:8.3f}")
    print("\nRandom Forest feature importance:")
    for k, v in importances.items():
        print(f"    {k:18s} {v:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(
            {"region": region, "test_positive_rate": round(float(y_test.mean()), 4),
             "results": results, "rf_feature_importance": importances},
            f, indent=2,
        )
    print(f"\nSaved -> output/metrics.json")
    return results, importances


if __name__ == "__main__":
    run("Kyiv City")
