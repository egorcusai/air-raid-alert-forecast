"""
data_loader.py
==============
Loads the Ukrainian air-raid siren dataset (volunteer, oblast-level) and
prepares the raw alert-interval table for feature engineering.

Source of truth
---------------
Vadimkin/ukrainian-air-raid-sirens-dataset, file `volunteer_data_en.csv`.
Confirmed schema (verified against the repo on 2026-06-20):

    region        str   oblast / city name, English
    started_at    str   UTC timestamp, alert start  (e.g. 2022-02-25 16:36:22+00:00)
    finished_at   str   UTC timestamp, alert end
    naive         bool  True  -> end-time was NOT observed; it was estimated
                              as started_at + 30 min. Duration is unreliable.
                        False -> end-time observed; duration trustworthy.

Design decisions (documented for reproducibility / audit)
---------------------------------------------------------
1. We use the *volunteer* dataset, not the official one. The official set
   switched from oblast-level to raion (district) level in Dec 2025, which
   would silently change the geographic unit mid-series. The volunteer set
   stays oblast-level throughout, giving a consistent target definition.

2. We drop regions with permanent sirens that are not properly logged as
   discrete events (Luhanska oblast has 3 records total; Crimea is absent).
   A model "predicting" a permanent siren earns fake recall and teaches us
   nothing, so these regions are excluded from modelling.

3. The `naive` flag is preserved untouched here. It is consumed downstream
   in features.py, where duration-derived features exclude naive rows while
   the alert *event itself* (used to build the target) is kept. Rationale:
   the occurrence of the alert is real and observed; only its duration is a
   guess, so we distrust the duration but trust the event.
"""

from __future__ import annotations

import os
import pandas as pd

# Raw CSV on GitHub. Pinned to `main`; the repo updates daily, so a run today
# pulls history up to today. For a frozen, fully-reproducible run, replace
# `main` with a specific commit SHA.
DATA_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_en.csv"
)

# Regions excluded from modelling: permanent / unlogged sirens (see note 2).
PERMANENT_SIREN_REGIONS = {"Luhanska oblast", "Crimea", "Autonomous Republic of Crimea"}

# Local cache so repeated runs don't re-download ~100k rows each time.
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw_alerts.csv")


def download_raw(force: bool = False) -> pd.DataFrame:
    """Download the raw CSV (or read the local cache if present).

    Parameters
    ----------
    force : bool
        If True, always re-download even when a cache file exists.

    Returns
    -------
    pd.DataFrame  with columns [region, started_at, finished_at, naive],
                  timestamps parsed to timezone-aware UTC datetimes.
    """
    cache = os.path.normpath(_CACHE_PATH)
    if os.path.exists(cache) and not force:
        df = pd.read_csv(cache)
    else:
        df = pd.read_csv(DATA_URL)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        df.to_csv(cache, index=False)

    # Parse timestamps. utc=True keeps everything tz-aware and comparable.
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True, errors="coerce")
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True, errors="coerce")

    # `naive` may arrive as string "True"/"False" from CSV; coerce to real bool.
    if df["naive"].dtype == object:
        df["naive"] = df["naive"].astype(str).str.strip().str.lower().map(
            {"true": True, "false": False}
        )

    return df


def clean_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Apply integrity filters that are independent of any chosen region.

    - Drop rows with unparseable / missing timestamps.
    - Drop rows where finished_at <= started_at (malformed intervals).
    - Drop permanent-siren regions.
    - Sort chronologically by start time.
    """
    n0 = len(df)

    df = df.dropna(subset=["started_at", "finished_at"]).copy()
    n_badts = n0 - len(df)

    df = df[df["finished_at"] > df["started_at"]].copy()
    n_badint = n0 - n_badts - len(df)

    df = df[~df["region"].isin(PERMANENT_SIREN_REGIONS)].copy()

    df = df.sort_values("started_at").reset_index(drop=True)

    # A duration in minutes is convenient downstream; for naive rows this is
    # the fabricated 30-min value, which features.py will exclude where needed.
    df["duration_min"] = (
        df["finished_at"] - df["started_at"]
    ).dt.total_seconds() / 60.0

    print(
        f"[clean_alerts] start={n0}  dropped_bad_timestamps={n_badts}  "
        f"dropped_bad_intervals={n_badint}  "
        f"after_region_filter={len(df)}"
    )
    return df


def load_region(region: str, force_download: bool = False) -> pd.DataFrame:
    """Full pipeline: download -> clean -> filter to a single region.

    Parameters
    ----------
    region : str
        Exact region label as it appears in the dataset, e.g. "Kyiv City".
        (Note: "Kyiv City" and "Kyivska oblast" are DIFFERENT zones.)
    force_download : bool
        Pass through to download_raw.

    Returns
    -------
    pd.DataFrame  of alert intervals for that region only.
    """
    df = download_raw(force=force_download)
    df = clean_alerts(df)

    available = set(df["region"].unique())
    if region not in available:
        raise ValueError(
            f"Region {region!r} not found. Available regions:\n"
            + "\n".join(sorted(available))
        )

    out = df[df["region"] == region].reset_index(drop=True)
    naive_share = out["naive"].mean() if len(out) else 0.0
    print(
        f"[load_region] region={region!r}  alerts={len(out)}  "
        f"span={out['started_at'].min()} -> {out['started_at'].max()}  "
        f"naive_share={naive_share:.1%}"
    )
    return out


if __name__ == "__main__":
    # Smoke test: load Kyiv City and show a few rows.
    sample = load_region("Kyiv City")
    print(sample.head(10).to_string())
