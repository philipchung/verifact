import pandas as pd

from .constants import NOT_SUPPORTED_OR_ADDRESSED, SUPPORTED


def binarize_verdicts(verdict: str) -> str:
    """Convert the original 3-label verdicts ("Supported", "Not Supported", "Not Addressed")
    to 2-label verdicts ("Supported", "Not Supported or Addressed")."""
    if pd.isna(verdict):
        return pd.NA
    elif verdict == SUPPORTED:
        return SUPPORTED
    else:
        return NOT_SUPPORTED_OR_ADDRESSED


def binarize_verdicts_in_df(df: pd.DataFrame, verdict_name: str = "verdict") -> pd.DataFrame:
    """Convert the original 3-label verdicts ("Supported", "Not Supported", "Not Addressed")
    to 2-label verdicts ("Supported", "Not Supported or Addressed")."""
    return df.assign(**{verdict_name: df[verdict_name].map(binarize_verdicts)})
