"""run_all.py -- one-command reproduction of the full pipeline.

Usage:
    python run_all.py [REGION]      # default: "Kyiv City"

Steps: load -> features -> leakage audit -> train/evaluate -> plot.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from leakage_audit import (
    test_reconstruction_independence,
    test_target_is_future,
    test_no_feature_equals_target,
)
import models, plot_results

def main():
    region = sys.argv[1] if len(sys.argv) > 1 else "Lvivska oblast"
    print("\n### STEP 1-3: LEAKAGE AUDIT ###")
    test_reconstruction_independence(region)
    test_target_is_future(region)
    test_no_feature_equals_target(region)
    print("\n### STEP 4: TRAIN + EVALUATE (temporal, then +spatial) ###")
    models.run(region, spatial=False)
    models.run(region, spatial=True)
    print("\n### STEP 5: PLOTS ###")
    plot_results.main(region)
    print("\n### STEP 6: MULTI-REGION COMPARISON ###")
    import compare_regions
    compare_regions.main()
    print("\nDONE. See output/ for metrics, comparison table, and plots.")

if __name__ == "__main__":
    main()
