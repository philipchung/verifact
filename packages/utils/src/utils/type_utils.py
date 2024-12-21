from typing import Any

import pandas as pd


def convert_nan_nat_to_none(value: Any) -> Any:
    """Convert NaN and NaT values to None; all other values unchanged."""
    if pd.isna(value):  # detects np.nan, pd.NA, and pd.NaT
        return None
    else:
        return value


def convert_timestamps_to_iso(value: Any) -> Any:
    """Convert timestamps to ISO format string; all other values unchanged."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    else:
        return value


def convert_iso_to_timestamp(value: Any) -> Any:
    """Convert ISO formatted timestamps to pandas Timestamp; all other values unchanged."""
    if isinstance(value, str):
        try:
            value = pd.Timestamp.fromisoformat(value)
        except Exception:
            return value
    else:
        return value
