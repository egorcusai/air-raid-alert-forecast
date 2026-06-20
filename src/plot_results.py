"""plot_results.py -- render confusion matrix + ROC curve for the best model."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix
from data_loader import load_region
from features import FEATURE_COLUMNS, build_feature_matrix
from models import time_based_split

OUT = os.path.join(os.path.dirname(__file__), "..", "output")

def main(region="Kyiv City"):
    fm = build_feature_matrix(load_region(region))
    train, val, test = time_based_split(fm)
    Xtr, ytr = train[FEATURE_COLUMNS].values, train["target"].values
    Xte, yte = test[FEATURE_COLUMNS].values, test["target"].values
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                class_weight="balanced", random_state=42, n_jobs=-1).fit(Xtr, ytr)
    prob = rf.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    ax[0].imshow(cm, cmap="Blues")
    ax[0].set_xticks([0, 1]); ax[0].set_yticks([0, 1])
    ax[0].set_xticklabels(["No alert", "Alert"]); ax[0].set_yticklabels(["No alert", "Alert"])
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
    ax[0].set_title(f"Confusion Matrix -- {region} (RF @0.5)")
    for i in range(2):
        for j in range(2):
            ax[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    fpr, tpr, _ = roc_curve(yte, prob)
    ax[1].plot(fpr, tpr, label="Random Forest")
    ax[1].plot([0, 1], [0, 1], "--", color="gray", label="Random (0.5)")
    ax[1].set_xlabel("False Positive Rate"); ax[1].set_ylabel("True Positive Rate")
    ax[1].set_title("ROC Curve"); ax[1].legend()
    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "results.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved -> {os.path.normpath(path)}")

if __name__ == "__main__":
    main()
