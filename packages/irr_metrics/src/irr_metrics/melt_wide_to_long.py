import pandas as pd


def melt_wide_to_long(human_verdicts_df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format human verdicts dataframe to long format, extracting the
    rater name and verdict for each proposition and exploding so there is one
    verdict per row."""
    rater_name = (
        human_verdicts_df.melt(
            id_vars=["proposition_id"],
            value_vars=["rater1", "rater2", "rater3"],
            var_name="source_rater_col",
            value_name="rater_name",
        )
        .set_index("proposition_id")
        .drop(columns="source_rater_col")
    )
    verdicts = (
        human_verdicts_df.melt(
            id_vars=["proposition_id"],
            value_vars=["verdict1", "verdict2", "verdict3"],
            var_name="source_verdict_col",
            value_name="verdict",
        )
        .set_index("proposition_id")
        .drop(columns="source_verdict_col")
    )
    all_propositions = pd.concat([rater_name, verdicts], axis=1)
    all_propositions = all_propositions.merge(
        human_verdicts_df[["proposition_id", "author_type", "proposition_type"]],
        left_index=True,
        right_on="proposition_id",
    )
    return all_propositions
