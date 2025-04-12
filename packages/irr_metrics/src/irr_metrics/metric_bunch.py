import itertools
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm
from utils import load_pandas, save_pandas

from irr_metrics.constants import VERDICT_LABELS
from irr_metrics.interrater import InterraterAgreement, MetricResult
from irr_metrics.type_utils import coerce_types

HUMAN_HYPERPARAM_COLS = ["author_type", "proposition_type"]
AI_HYPERPARAM_COLS = [
    "model",
    "author_type",
    "proposition_type",
    "fact_type",
    "top_n",
    "retrieval_method",
    "reference_format",
    "reference_only_admission",
]


class MetricBunch(BaseModel):
    """Container for storing computed interrater agreement metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(default="", description="Name of the MetricBunch.")
    strata_dfs: dict[Any, pd.DataFrame] = Field(
        default={},
        description="Dictionary of strata DataFrames. "
        "Each stratum dataframe corresponds to a specific rater or hyperparameter set. "
        "Each dataframe is used to compute metrics and corresponds to "
        "one row in the metrics dataframe.",
    )
    metrics: pd.DataFrame | None = Field(
        default=None,
        description="Dataframe of computed metrics for each stratum. "
        "Each row corresponds to a specific rater or hyperparameter set (stratum) "
        "For which the metric is computed across all propositions for that "
        "rater or hyperparameter set.",
    )

    ## Save/Load Methods

    @staticmethod
    def save_metric(
        stratum: str,
        metric_df: pd.DataFrame,
        save_dir: Path | str,
        name: str,
        metrics_dir: str = "metrics",
        **kwargs: Any,
    ) -> None:
        """Save a single metric result DataFrame to a file."""
        save_dir = Path(save_dir) / name / metrics_dir
        save_pandas(df=metric_df, filepath=save_dir / f"{stratum}.feather", **kwargs)

    @staticmethod
    def load_metric(
        stratum: str,
        save_dir: Path | str,
        name: str,
        metrics_dir: str = "metrics",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a single metric result DataFrame from a file."""
        save_dir = Path(save_dir) / name / metrics_dir
        return load_pandas(filepath=save_dir / f"{stratum}.feather", **kwargs)

    @staticmethod
    def save_stratum_data(
        stratum: str,
        stratum_df: pd.DataFrame,
        save_dir: Path | str,
        name: str,
        strata_data_dir: str = "strata_data",
        **kwargs: Any,
    ) -> None:
        """Save a single stratum DataFrame to a file."""
        save_dir = Path(save_dir) / name / strata_data_dir
        save_pandas(df=stratum_df, filepath=save_dir / f"{stratum}.feather", **kwargs)

    @staticmethod
    def load_stratum_data(
        stratum: str,
        save_dir: Path | str,
        name: str,
        strata_data_dir: str = "strata_data",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Load a single stratum DataFrame from a file."""
        save_dir = Path(save_dir) / name / strata_data_dir
        return load_pandas(filepath=save_dir / f"{stratum}.feather", **kwargs)

    def save(self, save_dir: Path | str, save_data: bool = True, **kwargs: Any) -> None:
        """Save the MetricBunch to files. `save_dir` should be a directory.

        Aggregate Metrics are saved to:
        {save_dir} / {self.name} / metrics.feather

        Individual Strata Metrics are saved to:
        {save_dir} / {self.name} / metrics / {stratum}.feather

        Individual Strata DataFrames are saved to:
        {save_dir} / {self.name} / strata_data / {stratum}.feather
        """
        save_dir = Path(save_dir) / self.name
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save Aggregated Metrics
        save_pandas(df=self.metrics, filepath=save_dir / "metrics.feather", **kwargs)
        # Save Individual Strata Metric
        for stratum, df in self.metrics.iterrows():
            MetricBunch.save_metric(
                stratum=stratum, metric_df=df, save_dir=save_dir, name=self.name, **kwargs
            )
        # Save Individual Strata Metrics
        if save_data:
            for stratum, df in self.strata_dfs.items():
                MetricBunch.save_stratum_data(
                    stratum=stratum, stratum_df=df, save_dir=save_dir, name=self.name, **kwargs
                )

    @classmethod
    def load(cls, save_dir: Path | str, name: str, load_data: bool = True, **kwargs: Any) -> Self:
        """Load the MetricBunch from files. `save_dir` should be a directory.

        Metrics are loaded from:
        {save_dir} / {name} / metrics / {stratum}.feather

        Strata DataFrames are loaded from:
        {save_dir} / {name} / {strata_data / {stratum}.feather
        """

        save_dir = Path(save_dir)
        if not save_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {save_dir}")
        elif save_dir.exists() and not any(save_dir.iterdir()):
            raise FileNotFoundError(f"Directory is empty: {save_dir}")

        # Load Individual Metrics
        metric_dfs: dict[str, pd.DataFrame] = {}
        try:
            metric_data = save_dir / name / "metrics"
            if metric_data.exists():
                for f in metric_data.iterdir():
                    if f.suffix == ".feather":
                        stratum = f.stem
                        metric_dfs[stratum] = MetricBunch.load_metric(
                            stratum=stratum, save_dir=save_dir, name=name, **kwargs
                        )
            # Concatenate Metric DataFrames for Each Stratum
            metrics = pd.concat(metric_dfs.values())
        except Exception:
            # If fail, load aggregated metrics
            metrics = load_pandas(filepath=save_dir / name / "metrics.feather", **kwargs)
        # Load Individual Strata Metrics
        strata_dfs: dict[str, pd.DataFrame] = {}
        if load_data:
            try:
                strata_data = save_dir / name / "strata_data"
                if strata_data.exists():
                    for f in strata_data.iterdir():
                        if f.suffix == ".feather":
                            stratum = f.stem
                            strata_dfs[stratum] = MetricBunch.load_stratum_data(
                                stratum=stratum, save_dir=save_dir, name=name, **kwargs
                            )
            except Exception:
                pass

        return MetricBunch(
            name=name,
            strata_dfs=strata_dfs,
            metrics=metrics,
        )

    ## Constructor Methods

    @classmethod
    def from_defaults(
        cls,
        rater_verdicts: pd.DataFrame,
        ground_truth: pd.DataFrame | None = None,
        verdict_labels: Sequence[str] = VERDICT_LABELS,
        rater_type: str = "human",
        rater_id_col: str = "stratum",
        stratify_cols: str | list[str] | None = ["author_type", "proposition_type", "fact_type"],
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        num_parallel_raters: int = 16,
        name: str | None = None,
        cache_dir: Path | str | None = None,
        force_recompute: bool = False,
        show_progress: bool = True,
        **kwargs,
    ) -> Self:
        """Create a MetricBunch from a DataFrame of Rater Verdicts
        (and optionally Ground Truth Labels). If ground truth is provided, each
        rater's verdicts are compared against the ground truth labels.
        If ground truth is not provided, each rater's verdicts are compared against
        each other for inter-rater agreement analysis.

        Args:
            rater_verdicts (pd.DataFrame): DataFrame of raters' verdicts.
            ground_truth (pd.DataFrame | None, optional): DataFrame of ground truth labels.
                If None, this method computes interrater agreement across all raters specified
                in `rater_verdicts`. If provided, this  method computes agreement metrics for
                each rater versus the ground truth labels. Defaults to None.
            verdict_labels (Sequence[str], optional): Labels for categorical verdicts.
                VERDICT_LABELS set includes "Supported", "Not Supported","Not Addressed".
                Other Options include BINARIZED_VERDICT_LABELS which includes
                "Supported" and "Not Supported or Addressed". Defaults to VERDICT_LABELS.
            rater_type (str, optional): "human" or "ai", modifies the associated metadata
                with each computed metric, adding AI-specific hyperparameters for "ai".
                Defaults to "human".
            rater_id_col (str, optional): Name of column in dataframe to identify a unique rater.
                Defaults to "stratum".
            stratify_cols (str | list[str], optional): Columns to use for stratifying the analysis.
                If multiple columns are provided, the cross-product of values from each column
                are used to stratify the analysis. Column values should be categorical or binary.
                Defaults to "author_type".
            metric (str, optional): "percent_agreement" or "gwet". Defaults to "percent_agreement".
            bootstrap_iterations (int | None, optional): Number of iterations for bootstrap
                resampling for confidence interval estimation. If None, no bootstrap resampling
                and confidence interval estimation is performed. Defaults to None.
            workers (int, optional): Number of parallel workers to use in bootstrap calculation.
                Defaults to 1.
            num_parallel_raters (int, optional): Number of raters' metrics to calculate in parallel.
                Defaults to 16.
            name (str, optional): Name of the MetricBunch. This is a label for organization and
                is also the name of the directory where the MetricBunch is saved. Defaults to None.
            cache_dir (Path | str | None, optional): Directory to save/load the MetricBunch.
                If None, no caching is performed. Defaults to None.
            force_recompute (bool, optional): If True, forces recomputation of metrics even if
                cached data exists. Defaults to False.
            show_progress (bool, optional): If True, display progress bars for computation.
                Defaults to True.

        Returns:
            Self: MetricBunch object with results computed for the given `metric`. Metrics represent
                the agreement between raters if `ground_truth` is None, or
                between raters and ground truth labels if `ground_truth` is provided. If multiple
        """

        if rater_verdicts is None or rater_verdicts.empty:
            raise ValueError("rater_verdicts is None or empty.")

        def _compute_agreement(
            verdicts: pd.DataFrame,
            ground_truth: pd.DataFrame | None = None,
            stratum: str | None = None,
        ) -> DataFrame:
            if ground_truth is None:
                # Compute Agreement Metric Between All Raters
                return MetricBunch.compute_interrater_agreement(
                    rater_verdicts=verdicts,
                    rater_type=rater_type,
                    rater_id_col=rater_id_col,
                    categories=verdict_labels,
                    rater_name=stratum,  # name for metric result index
                    metric=metric,
                    bootstrap_iterations=bootstrap_iterations,
                    workers=workers,
                    show_progress=False,
                )
            else:
                # Compute Agreement Metric of Rater vs. Ground Truth for All Raters
                return MetricBunch.compute_rater_verdict_agreement_vs_ground_truth(
                    rater_verdicts=verdicts,
                    ground_truth=ground_truth,
                    rater_type=rater_type,
                    rater_id_col=rater_id_col,
                    categories=verdict_labels,
                    metric=metric,
                    bootstrap_iterations=bootstrap_iterations,
                    workers=workers,
                    num_parallel_raters=num_parallel_raters,
                    show_progress=False,
                )

        # Define Strata and Corresponding Subset DataFrames
        strata_dfs: dict[str, pd.DataFrame] = MetricBunch.stratify_by_columns(
            df=rater_verdicts, columns=stratify_cols
        )

        # Compute or Load Agreement Metrics for Each Stratum
        metrics: dict[str, pd.DataFrame] = {}
        final_strata_dfs: dict[str, pd.DataFrame] = {}
        for stratum, stratum_verdicts in (
            pbar := tqdm(strata_dfs.items(), total=len(strata_dfs), disable=not show_progress)
        ):
            pbar.set_description(desc=f"Stratum: {stratum}")
            # If Agreement Metric previously computed for stratum, do not recompute
            if cache_dir is not None and not force_recompute:
                try:
                    # Attempt to Load Cached Metric
                    metrics[stratum] = MetricBunch.load_metric(
                        stratum=stratum,
                        save_dir=cache_dir,
                        name=name,
                    )
                    # Attempt to Load Cached Stratum Data
                    final_strata_dfs[stratum] = MetricBunch.load_stratum_data(
                        stratum=stratum,
                        save_dir=cache_dir,
                        name=name,
                    )
                    # If both succeed, skip recomputation and move to next stratum
                    continue
                except FileNotFoundError:
                    # If file not found, need to compute Agreement Metric
                    pass

            # Compute Agreement Metric for Stratum
            metric_df = _compute_agreement(
                verdicts=stratum_verdicts, ground_truth=ground_truth, stratum=stratum
            )
            metrics[stratum] = metric_df
            final_strata_dfs[stratum] = stratum_verdicts
            # Save Results to Cache
            if cache_dir is not None:
                MetricBunch.save_metric(
                    stratum=stratum,
                    metric_df=metric_df,
                    save_dir=cache_dir,
                    name=name,
                )
                MetricBunch.save_stratum_data(
                    stratum=stratum,
                    stratum_df=stratum_verdicts,
                    save_dir=cache_dir,
                    name=name,
                )

        # Concatenate Metric DataFrames for Each Stratum
        final_metrics = pd.concat(metrics.values())
        save_pandas(df=final_metrics, filepath=cache_dir / name / "metrics.feather", **kwargs)
        return cls(name=name, strata_dfs=final_strata_dfs, metrics=final_metrics)

    @staticmethod
    def stratify_by_columns(
        df: pd.DataFrame, columns: str | list[str] | None = None, **kwargs: Any
    ) -> dict[str, pd.DataFrame]:
        """Stratify a DataFrame by one or more columns. Returns a dictionary with keys
        of stratum tuples and values of the stratified DataFrame."""
        # Stratified DataFrame Dictionary
        strata_dfs: dict[str, pd.DataFrame] = {}
        # If No Stratify Columns, Create single stratum with all data
        if columns is None:
            strata_dfs["all"] = df
        else:
            # Otherwise, stratify with one or more categorical/binary columns
            if isinstance(columns, str):
                columns = [columns]

            # Create Mapping of all Columns and Categorical Values in Columns
            strata_map = {col: df[col].drop_duplicates().to_list() for col in columns}

            # Get Column Names and all Possible Value Combinations
            strata_map_keys = tuple(strata_map.keys())
            strata_map_value_tuples = list(itertools.product(*strata_map.values()))

            # Subset Dataframe for Each Stratification Group
            for stratum_tuple in strata_map_value_tuples:
                # Build Query String for Stratification
                query_components = []
                for k, v in zip(strata_map_keys, stratum_tuple):
                    match v:
                        case str():
                            query_components.append(f"{k} == '{v}'")
                        case int() | float() | bool():
                            query_components.append(f"{k} == {v}")
                        case _:
                            raise ValueError(
                                f"Cannot build query for stratum with variable {v} with {type(v)}."
                            )
                query_str = " & ".join(query_components)
                # Execute Query
                subset_df = df.query(query_str)
                # Coerce Types
                subset_df = coerce_types(subset_df)
                # Add to Stratified Dataframe Dictionary if not Empty
                if not subset_df.empty:
                    stratum_str = MetricBunch.dict_to_str(dict(zip(strata_map_keys, stratum_tuple)))
                    subset_df = subset_df.assign(stratum=stratum_str)
                    strata_dfs[stratum_str] = subset_df
        return strata_dfs

    @staticmethod
    def dict_to_str(dict: dict) -> str:
        return ",".join(f"{k}={v}" for k, v in dict.items())

    @staticmethod
    def categorical_agreement_metrics_for_verdicts(
        verdicts: pd.DataFrame,
        metric: str | tuple = "percent_agreement",
        categories: Sequence[str] | None = VERDICT_LABELS,
        rater_id_col: str = "rater_name",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> dict[str, MetricResult]:
        """Compute Categorical Agreement Metrics for a DataFrame of Verdicts."""
        ia = InterraterAgreement.from_defaults(
            data=verdicts,
            data_kind="categorical",
            categories=categories,
            rater_name=rater_id_col,
            item_name="proposition_id",
            values_name="verdict",
        )
        categorical_agreement = ia.categorical_agreement(
            metrics=metric,
            bootstrap_iterations=bootstrap_iterations,
            workers=workers,
            show_progress=show_progress,
        )
        return categorical_agreement

    @staticmethod
    def compute_interrater_agreement(
        rater_verdicts: pd.DataFrame,
        rater_type: str = "human",
        categories: Sequence[str] = VERDICT_LABELS,
        rater_id_col: str = "rater_name",
        rater_name: str | None = None,
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute Interrater Agreement Metrics for a cohort of raters.

        The rater name needs to be explicitly assigned or else it will default
        to "human_interrater".
        """
        result = MetricBunch.categorical_agreement_metrics_for_verdicts(
            verdicts=rater_verdicts,
            metric=metric,
            categories=categories,
            rater_id_col=rater_id_col,
            bootstrap_iterations=bootstrap_iterations,
            workers=workers,
            show_progress=show_progress,
        )
        agreement_metric = result.get(metric, None)
        # Create DataFrame of Agreement Metric (For consistent API)
        rater_name = rater_name or f"{rater_type}_interrater"
        metric_df = (
            pd.DataFrame([agreement_metric.model_dump()], index=[rater_name])
            .rename_axis(index="rater_name")
            .rename(columns={"name": "metric_name"})
        )
        # Add Display String
        metric_df = metric_df.assign(
            display_str=metric_df.apply(
                lambda row: MetricBunch.make_display_string(row=row, metric_name=metric),
                axis="columns",
            ).astype("string")
        )
        # Coerce Type
        metric_df = coerce_types(metric_df)
        return metric_df

    @staticmethod
    def compute_rater_verdict_agreement_vs_ground_truth(
        rater_verdicts: pd.DataFrame,
        ground_truth: pd.DataFrame,
        rater_type: str = "human",
        rater_id_col: str = "rater_name",
        categories: Sequence[str] = VERDICT_LABELS,
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        num_parallel_raters: int = 16,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute a Categorical Agreement Metrics for a cohort of raters vs. ground truth.
        Multiple raters agreement vs. ground truth are computed in parallel.

        The rater name from rater verdicts will become indices in the resulting metrics DataFrame.
        """
        # Compute Agreement Metrics for Each Rater
        metric_dict: dict[str, MetricResult | None] = {}
        rater_names = []
        futures = []
        with ProcessPoolExecutor(max_workers=num_parallel_raters) as executor:
            for rater_name, one_rater_verdicts in (
                pbar := tqdm(
                    rater_verdicts.groupby(rater_id_col),
                    total=len(rater_verdicts[rater_id_col].unique()),
                    disable=not show_progress,
                )
            ):
                pbar.set_description(desc=f"Human vs. {rater_name}")
                # Get Overlap Proposition Nodes Between Rater & Ground Truth
                # Then Create Concatenated Dataframe for Interrater Agreement Comparison
                id_overlap = list(
                    set(ground_truth.proposition_id) & set(one_rater_verdicts.proposition_id)
                )
                comparison_verdicts = pd.concat(
                    [ground_truth, one_rater_verdicts], axis="index"
                ).query(f"proposition_id in {id_overlap}")  # noqa: F841
                # Compute metrics
                future = executor.submit(
                    MetricBunch.categorical_agreement_metrics_for_verdicts,
                    verdicts=comparison_verdicts,
                    metric=metric,
                    categories=categories,
                    bootstrap_iterations=bootstrap_iterations,
                    workers=workers,
                    show_progress=show_progress,
                )
                futures.append(future)
                rater_names.append(rater_name)

            for _ in (
                pbar := tqdm(as_completed(futures), total=len(futures), disable=not show_progress)
            ):
                pbar.set_description(desc="Completion")

            for rater_name, future in zip(rater_names, futures, strict=False):
                agreement_metrics = future.result()
                metric_dict[rater_name] = agreement_metrics.get(metric, None)

        # Create DataFrame of Agreement Metrics
        index = metric_dict.keys()
        data = [x.model_dump() for x in metric_dict.values() if x is not None]
        metric_df = (
            pd.DataFrame(data, index=index)
            .rename_axis(index="rater_name")
            .rename(columns={"name": "metric_name"})
        )
        # Add Display String
        metric_df = metric_df.assign(
            display_str=metric_df.apply(
                lambda row: MetricBunch.make_display_string(row=row, metric_name=metric),
                axis="columns",
            ).astype("string")
        )
        # Add hyperparameters to the DataFrame
        if rater_type == "ai":
            intersection_cols = list(set(AI_HYPERPARAM_COLS) & set(rater_verdicts.columns))
            hyperparams_df = (
                rater_verdicts.drop_duplicates(subset=[rater_id_col])
                .set_index(rater_id_col)
                .loc[:, intersection_cols]
            )
        elif rater_type == "human":
            intersection_cols = list(set(HUMAN_HYPERPARAM_COLS) & set(rater_verdicts.columns))
            hyperparams_df = (
                rater_verdicts.drop_duplicates(subset=[rater_id_col])
                .set_index(rater_id_col)
                .loc[:, intersection_cols]
            )
        else:
            raise ValueError(f"Invalid rater_type: {rater_type}.")

        # Format and Join Hyperparameters
        metric_df = metric_df.join(hyperparams_df)

        # Coerce Types
        metric_df = coerce_types(metric_df)
        return metric_df

    @staticmethod
    def make_display_string(row: pd.Series, metric_name: str) -> str:
        """Create a string display for a metric row."""
        has_ci = not pd.isna(row.ci_lower) and not pd.isna(row.ci_upper)
        if metric_name == "percent_agreement":
            if has_ci:
                ci_lower = row.ci_lower * 100
                ci_upper = row.ci_upper * 100
                return f"{row.value:.1%} ({ci_lower:.1f}-{ci_upper:.1f}%)"
            else:
                return f"{row.value:.1%}"
        elif metric_name == "gwet":
            if has_ci:
                ci_lower = row.ci_lower
                ci_upper = row.ci_upper
                return f"{row.value:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"
            else:
                return f"{row.value:.2f}"
        else:
            return f"{row.value:.2f}"
