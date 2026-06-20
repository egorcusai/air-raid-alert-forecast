"""
compare_regions.py
==================
Runs the full pipeline across several regions and builds a comparison table.
Headline question: WHERE are air-raid alerts predictable, and does geographic
neighbour information help?

Outputs output/region_comparison.csv and prints a ranked table.
"""
from __future__ import annotations
import os, warnings
import pandas as pd
warnings.filterwarnings("ignore")

from data_loader import load_region
from features import build_feature_matrix
from models import time_based_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

REGIONS = [
    "Lvivska oblast", "Ivano-Frankivska oblast", "Ternopilska oblast",  # far west
    "Vinnytska oblast", "Zhytomyrska oblast",                            # central-west
    "Kyiv City", "Kyivska oblast",                                       # capital
    "Dnipropetrovska oblast", "Kharkivska oblast", "Zaporizka oblast",   # east/front
]
OUT = os.path.join(os.path.dirname(__file__), "..", "output")


def _fit_eval(fm, spatial_cols_present):
    feat = [c for c in fm.columns if c not in ("hour", "target")]
    tr, va, te = time_based_split(fm)
    Xtr, ytr = tr[feat].values, tr["target"].values
    Xte, yte = te[feat].values, te["target"].values
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)
    lr_p = lr.predict_proba(sc.transform(Xte))[:, 1]
    rf_p = rf.predict_proba(Xte)[:, 1]
    best_auc = max(roc_auc_score(yte, lr_p), roc_auc_score(yte, rf_p))
    best_name = "LR" if roc_auc_score(yte, lr_p) >= roc_auc_score(yte, rf_p) else "RF"
    best_prob = lr_p if best_name == "LR" else rf_p
    pred = (best_prob >= 0.5).astype(int)
    return {
        "lr_auc": round(roc_auc_score(yte, lr_p), 3),
        "rf_auc": round(roc_auc_score(yte, rf_p), 3),
        "best_model": best_name,
        "best_auc": round(best_auc, 3),
        "recall": round(recall_score(yte, pred, zero_division=0), 3),
        "precision": round(precision_score(yte, pred, zero_division=0), 3),
        "f1": round(f1_score(yte, pred, zero_division=0), 3),
        "pos_rate": round(float(yte.mean()), 3),
    }


def main():
    rows = []
    for region in REGIONS:
        try:
            alerts = load_region(region)
        except ValueError:
            print(f"skip {region}: not found")
            continue
        fm_t = build_feature_matrix(alerts, region=region, spatial=False)
        fm_s = build_feature_matrix(alerts, region=region, spatial=True)
        rt = _fit_eval(fm_t, False)
        rs = _fit_eval(fm_s, True)
        rows.append({
            "region": region,
            "pos_rate": rt["pos_rate"],
            "auc_temporal": rt["best_auc"],
            "auc_spatial": rs["best_auc"],
            "spatial_gain": round(rs["best_auc"] - rt["best_auc"], 3),
            "best_model": rs["best_model"],
            "recall": rs["recall"],
            "precision": rs["precision"],
            "f1": rs["f1"],
        })
        print(f"done {region}: temporal={rt['best_auc']} spatial={rs['best_auc']}")

    df = pd.DataFrame(rows).sort_values("auc_spatial", ascending=False)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "region_comparison.csv"), index=False)
    print("\n" + "=" * 88)
    print("REGION COMPARISON  (ROC-AUC, best of LR/RF, on most-recent held-out test)")
    print("=" * 88)
    print(df.to_string(index=False))
    print(f"\nSaved -> output/region_comparison.csv")
    return df


if __name__ == "__main__":
    main()
