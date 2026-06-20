"""
temporal_patterns.py
====================
Descriptive seasonality analysis of air-raid risk across four cycles:

    DAILY   -> by hour of day      (0-23)
    WEEKLY  -> by day of week       (Mon-Sun)
    MONTHLY -> by month of year     (Jan-Dec)
    YEARLY  -> by calendar year     (2022-2026)

For each bucket we report the ALERT OCCUPANCY RATE: the fraction of hours in
that bucket during which an air-raid alert was active in the region. This is a
descriptive (not predictive) layer -- it explains *why* the model's hour_of_day
and month features carried signal, and it is leakage-irrelevant because it is
not used as a feature; it is analysis of the historical record.

Outputs:
  output/patterns.json  -> consumed by the dashboard UI
  output/patterns.png   -> heatmaps / bar charts for the repo
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_region, download_raw, clean_alerts, PERMANENT_SIREN_REGIONS

OUT = os.path.join(os.path.dirname(__file__), "..", "output")
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _active_hourly(alerts: pd.DataFrame) -> pd.DataFrame:
    """Expand alert intervals onto an hourly grid; mark each hour active (1)
    if any alert overlapped it. Returns DataFrame[hour, active]."""
    start = alerts["started_at"].min().floor("h")
    end = alerts["finished_at"].max().ceil("h")
    grid = pd.DataFrame({"hour": pd.date_range(start, end, freq="h", tz="UTC")})
    active = np.zeros(len(grid), dtype=np.int8)
    hour_index = {h: i for i, h in enumerate(grid["hour"])}

    for s, f in zip(alerts["started_at"], alerts["finished_at"]):
        h0 = s.floor("h")
        h1 = f.floor("h")
        # mark every hour bucket the interval touches
        for h in pd.date_range(h0, h1, freq="h"):
            idx = hour_index.get(h)
            if idx is not None:
                active[idx] = 1
    grid["active"] = active
    grid["hour_of_day"] = grid["hour"].dt.hour
    grid["day_of_week"] = grid["hour"].dt.dayofweek
    grid["month"] = grid["hour"].dt.month
    grid["year"] = grid["hour"].dt.year
    return grid


def compute_patterns(region: str) -> dict:
    alerts = load_region(region)
    grid = _active_hourly(alerts)

    daily = grid.groupby("hour_of_day")["active"].mean()
    weekly = grid.groupby("day_of_week")["active"].mean()
    monthly = grid.groupby("month")["active"].mean()
    yearly = grid.groupby("year")["active"].mean()

    return {
        "region": region,
        "overall_active_rate": round(float(grid["active"].mean()), 4),
        "daily": {int(k): round(float(v), 4) for k, v in daily.items()},
        "weekly": {DOW[int(k)]: round(float(v), 4) for k, v in weekly.items()},
        "monthly": {MON[int(k) - 1]: round(float(v), 4) for k, v in monthly.items()},
        "yearly": {int(k): round(float(v), 4) for k, v in yearly.items()},
        "peak_hour": int(daily.idxmax()),
        "quietest_hour": int(daily.idxmin()),
        "peak_month": MON[int(monthly.idxmax()) - 1],
    }


def plot_patterns(region: str, pat: dict) -> str:
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Air-raid alert seasonality — {region}", fontsize=14, fontweight="bold")

    # Daily
    hrs = list(range(24))
    ax[0, 0].bar(hrs, [pat["daily"].get(h, 0) for h in hrs], color="#4B9EFF")
    ax[0, 0].set_title("Daily — by hour of day (UTC)")
    ax[0, 0].set_xlabel("Hour"); ax[0, 0].set_ylabel("Alert-active rate")

    # Weekly
    ax[0, 1].bar(DOW, [pat["weekly"][d] for d in DOW], color="#22C55E")
    ax[0, 1].set_title("Weekly — by day of week")
    ax[0, 1].set_ylabel("Alert-active rate")

    # Monthly
    ax[1, 0].bar(MON, [pat["monthly"][m] for m in MON], color="#A78BFA")
    ax[1, 0].set_title("Monthly — by month")
    ax[1, 0].set_ylabel("Alert-active rate")
    ax[1, 0].tick_params(axis="x", rotation=45)

    # Yearly
    yrs = sorted(pat["yearly"].keys())
    ax[1, 1].bar([str(y) for y in yrs], [pat["yearly"][y] for y in yrs], color="#FBBF24")
    ax[1, 1].set_title("Yearly — by calendar year")
    ax[1, 1].set_ylabel("Alert-active rate")

    plt.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "patterns.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return os.path.normpath(path)


def main(regions=None):
    if regions is None:
        regions = ["Kyiv City", "Lvivska oblast", "Kharkivska oblast"]
    all_pat = {}
    for r in regions:
        pat = compute_patterns(r)
        all_pat[r] = pat
        print(f"\n[{r}] overall active rate = {pat['overall_active_rate']:.1%}")
        print(f"    peak hour={pat['peak_hour']:02d}:00 UTC  quietest={pat['quietest_hour']:02d}:00  "
              f"peak month={pat['peak_month']}")
        print(f"    yearly: " + "  ".join(f"{y}:{v:.0%}" for y, v in sorted(pat['yearly'].items())))

    # Plot the first region in detail.
    plot_patterns(regions[0], all_pat[regions[0]])

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "patterns.json"), "w") as f:
        json.dump(all_pat, f, indent=2)
    print(f"\nSaved -> output/patterns.json, output/patterns.png")
    return all_pat


if __name__ == "__main__":
    import sys
    regs = sys.argv[1:] if len(sys.argv) > 1 else None
    main(regs)
