"""
spatial_features.py
===================
Builds leakage-safe SPATIAL features: how much air-raid activity occurred in a
region's geographic NEIGHBOURS in the recent past. Motivated by the published
finding (arXiv:2411.14625) that a region's alert status depends heavily on its
adjacent regions -- alerts propagate across the map.

Leakage discipline is identical to the temporal features: every neighbour
window is computed on a one-hour-SHIFTED history, so hour H's features never
include hour H's own (or any future) neighbour activity.
"""
from __future__ import annotations
import pandas as pd
from data_loader import download_raw, clean_alerts, PERMANENT_SIREN_REGIONS
from neighbours import get_neighbours


def _region_hourly_starts(all_alerts: pd.DataFrame, region: str,
                          index: pd.DatetimeIndex) -> pd.Series:
    """Hourly count of alert starts for one region, reindexed onto `index`."""
    sub = all_alerts[all_alerts["region"] == region]
    counts = sub["started_at"].dt.floor("h").value_counts()
    return pd.Series(index=index, data=index.map(counts)).fillna(0)


def add_spatial_features(grid: pd.DataFrame, region: str) -> pd.DataFrame:
    """Add neighbour-activity features to a region's hourly grid.

    Features added:
      nbr_alerts_last_1h : total neighbour alert starts in the previous hour
      nbr_alerts_last_3h : ... previous 3 hours
      nbr_alerts_last_6h : ... previous 6 hours
      nbr_active_last_3h : how many distinct neighbours had >=1 alert in last 3h
    """
    g = grid.copy()
    hours = pd.DatetimeIndex(g["hour"])

    all_alerts = clean_alerts(download_raw())
    neighbours = [n for n in get_neighbours(region)
                  if n not in PERMANENT_SIREN_REGIONS]

    # Per-neighbour hourly start counts aligned to our grid.
    per_nbr = {n: _region_hourly_starts(all_alerts, n, hours) for n in neighbours}
    nbr_df = pd.DataFrame(per_nbr, index=hours)

    total = nbr_df.sum(axis=1)                 # total neighbour starts per hour
    active = (nbr_df > 0).sum(axis=1)          # distinct active neighbours per hour

    # Shift by 1 hour -> strictly historical, no peeking at current/future hour.
    total_past = total.shift(1).fillna(0)
    active_past = active.shift(1).fillna(0)

    g["nbr_alerts_last_1h"] = total_past.values
    g["nbr_alerts_last_3h"] = total_past.rolling(3, min_periods=1).sum().values
    g["nbr_alerts_last_6h"] = total_past.rolling(6, min_periods=1).sum().values
    g["nbr_active_last_3h"] = active_past.rolling(3, min_periods=1).max().values

    return g


SPATIAL_FEATURE_COLUMNS = [
    "nbr_alerts_last_1h",
    "nbr_alerts_last_3h",
    "nbr_alerts_last_6h",
    "nbr_active_last_3h",
]
