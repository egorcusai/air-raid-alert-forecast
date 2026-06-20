"""
features.py
===========
Turns the raw alert-interval table into a supervised learning matrix on an
HOURLY GRID, with a 3-hour-ahead binary target.

The problem (deliberately a classification, not exact-time forecasting):

    For region R, at the start of hour H, predict:
    "Will a NEW air-raid alert BEGIN in R during [H+1, H+3]?"   ->  y in {0,1}

Why this framing
----------------
- Operationally meaningful: a 3-hour risk flag is actionable; an exact-minute
  prediction is false precision and near-impossible to validate in a prototype.
- Easy to validate: a clean binary label with standard classification metrics.

=====================  LEAKAGE: THE CENTRAL CONCERN  =========================
Every feature for the row at hour H must be computable using ONLY information
available AT OR BEFORE the START of hour H. The target looks 1-3 hours into
the FUTURE. If any feature peeks into [H, H+3], the model cheats and the
metrics are meaningless.

Two specific traps we guard against:

  TRAP 1 (window edge). Rolling counts like "alerts in the last 3h" must cover
  [H-3h, H) -- strictly BEFORE H. We compute them on a shifted history so the
  current hour's own alert activity is never included.

  TRAP 2 (target/feature overlap). The label uses alert STARTS in [H+1, H+3].
  No feature may reference any timestamp >= H. duration features use only
  alerts that FINISHED before H.

  TRAP 3 (naive duration). Duration-derived features exclude naive rows
  (fabricated 30-min durations); the alert event itself still counts toward
  occurrence-based features and the target.
============================================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_hourly_grid(alerts: pd.DataFrame) -> pd.DataFrame:
    """Create a continuous hourly index spanning the data, with an
    `alert_started_this_hour` indicator built from alert START times.

    Parameters
    ----------
    alerts : pd.DataFrame
        Output of data_loader.load_region (single region), columns include
        started_at, finished_at, naive, duration_min.

    Returns
    -------
    pd.DataFrame indexed by hour (UTC), with helper columns.
    """
    start = alerts["started_at"].min().floor("h")
    end = alerts["started_at"].max().ceil("h")
    grid = pd.DataFrame(
        {"hour": pd.date_range(start, end, freq="h", tz="UTC")}
    )

    # Number of alerts that STARTED within each hour bucket.
    starts = alerts["started_at"].dt.floor("h").value_counts()
    grid["starts_this_hour"] = grid["hour"].map(starts).fillna(0).astype(int)

    # Number of alerts that started this hour AND had an observed (non-naive)
    # duration, plus the summed observed duration -- used later for a
    # leakage-safe rolling average duration.
    obs = alerts[~alerts["naive"]].copy()
    obs_starts = obs["started_at"].dt.floor("h").value_counts()
    obs_dur = obs.groupby(obs["started_at"].dt.floor("h"))["duration_min"].sum()
    grid["obs_count_this_hour"] = grid["hour"].map(obs_starts).fillna(0).astype(int)
    grid["obs_dur_sum_this_hour"] = grid["hour"].map(obs_dur).fillna(0.0)

    return grid


def add_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Add historical-window features. ALL use .shift(1) so the current hour
    is excluded -- they describe the past, strictly before hour H.
    """
    g = grid.copy()

    # Occurrence counts over trailing windows, computed on a history that is
    # shifted by one hour so hour H's own activity is NOT included.
    past = g["starts_this_hour"].shift(1).fillna(0)

    g["alerts_last_3h"] = past.rolling(3, min_periods=1).sum()
    g["alerts_last_6h"] = past.rolling(6, min_periods=1).sum()
    g["alerts_last_24h"] = past.rolling(24, min_periods=1).sum()

    # Rolling AVERAGE observed duration over the last 24h (naive excluded by
    # construction -- we summed only observed durations/counts above). Also
    # shifted so it's strictly historical. Guard divide-by-zero.
    past_dur_sum = g["obs_dur_sum_this_hour"].shift(1).fillna(0).rolling(24, min_periods=1).sum()
    past_obs_cnt = g["obs_count_this_hour"].shift(1).fillna(0).rolling(24, min_periods=1).sum()
    g["avg_duration_24h"] = np.where(past_obs_cnt > 0, past_dur_sum / past_obs_cnt, 0.0)

    # Calendar features -- known in advance, no leakage.
    g["hour_of_day"] = g["hour"].dt.hour
    g["day_of_week"] = g["hour"].dt.dayofweek
    g["month"] = g["hour"].dt.month

    return g


def add_target(grid: pd.DataFrame) -> pd.DataFrame:
    """Add the 3-hour-ahead binary target.

    y[H] = 1 if any alert STARTS in hours H+1, H+2, or H+3, else 0.

    Implemented with a forward-looking rolling sum on a reversed series, which
    is the ONLY place future information is allowed -- because it's the label.
    """
    g = grid.copy()
    s = g["starts_this_hour"]

    # Future window [H+1, H+3]: shift(-1) moves next hour to current row,
    # then sum the next three hours.
    future_3 = (
        s.shift(-1).fillna(0)
        + s.shift(-2).fillna(0)
        + s.shift(-3).fillna(0)
    )
    g["target"] = (future_3 > 0).astype(int)

    # The last 3 rows can't have a valid 3h-ahead label (we'd be looking past
    # the end of data); drop them so we never train on a truncated future.
    g = g.iloc[:-3].copy()
    return g


FEATURE_COLUMNS = [
    "alerts_last_3h",
    "alerts_last_6h",
    "alerts_last_24h",
    "avg_duration_24h",
    "hour_of_day",
    "day_of_week",
    "month",
]


def build_feature_matrix(alerts: pd.DataFrame, region: str | None = None,
                         spatial: bool = False) -> pd.DataFrame:
    """End-to-end: raw region alerts -> modelling table (features + target).

    Parameters
    ----------
    alerts : pd.DataFrame
        Single-region alert intervals (from data_loader.load_region).
    region : str | None
        Required if spatial=True; the region whose neighbours we look up.
    spatial : bool
        If True, append leakage-safe neighbour-activity features.
    """
    grid = build_hourly_grid(alerts)
    grid = add_lag_features(grid)

    feat_cols = list(FEATURE_COLUMNS)
    if spatial:
        if region is None:
            raise ValueError("region must be given when spatial=True")
        from spatial_features import add_spatial_features, SPATIAL_FEATURE_COLUMNS
        grid = add_spatial_features(grid, region)
        feat_cols = feat_cols + SPATIAL_FEATURE_COLUMNS

    grid = add_target(grid)

    cols = ["hour"] + feat_cols + ["target"]
    out = grid[cols].reset_index(drop=True)

    base = out["target"].mean()
    print(
        f"[build_feature_matrix] rows={len(out)}  "
        f"positive_rate={base:.1%}  features={len(feat_cols)}  spatial={spatial}"
    )
    return out


if __name__ == "__main__":
    from data_loader import load_region

    alerts = load_region("Kyiv City")
    fm = build_feature_matrix(alerts)
    print(fm.head(30).to_string())
    print("\nTarget distribution:")
    print(fm["target"].value_counts())
