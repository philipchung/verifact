from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Self

import pandas as pd
from pandas.api.types import CategoricalDtype
from pandas.io.formats.style import Styler
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm
from utils import load_pandas, save_pandas

from irr_metrics import CATEGORIES, STRATA
from irr_metrics.interrater import AgreementMetric, InterraterAgreement

retrieval_method_dtype = CategoricalDtype(["Dense", "Hybrid", "Rerank"], ordered=True)
reference_context_dtype = CategoricalDtype(
    ["Score", "Absolute Time", "Relative Time"], ordered=True
)
retrieval_scope_dtype = CategoricalDtype(["Yes", "No"], ordered=True)


class MetricBunch(BaseModel):
    """Container for storing computed interrater agreement metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metric_name: str = Field(default="", description="Name of the metric.")
    all_samples: Any = Field(default=None, description="Metrics computed over all verdicts.")
    llm_claim: Any = Field(
        default=None,
        description="Metrics computed over author_type=LLM, proposition_type=claim verdicts.",
    )
    llm_sentence: Any = Field(
        default=None,
        description="Metrics computed over author_type=LLM, proposition_type=sentence verdicts.",
    )
    human_claim: Any = Field(
        default=None,
        description="Metrics computed over author_type=human, proposition_type=claim verdicts.",
    )
    human_sentence: Any = Field(
        default=None,
        description="Metrics computed over author_type=human, proposition_type=sentence verdict.",
    )

    ## Save/Load Methods

    def save(self, filepath: Path | str, **kwargs: Any) -> None:
        """Save the MetricBunch to files. `filepath` should be a directory.
        The files will be saved as feather files:
        * `all_samples` DataFrame is saved as `all_samples.feather`.
        * `llm_claim` DataFrame is saved as `llm_claim.feather`.
        * `llm_sentence` DataFrame is saved as `llm_sentence.feather`.
        * `human_claim` DataFrame is saved as `human_claim.feather`.
        * `human_sentence` DataFrame is saved as `human_sentence.feather`.
        """
        filepath = Path(filepath)
        # Save Metric Table
        if self.all_samples is not None:
            self.all_samples = MetricBunch.make_categorical_columns_string(self.all_samples)
            save_pandas(self.all_samples, filepath / "all_samples.feather", **kwargs)
        if self.llm_claim is not None:
            self.llm_claim = MetricBunch.make_categorical_columns_string(self.llm_claim)
            save_pandas(self.llm_claim, filepath / "llm_claim.feather", **kwargs)
        if self.llm_sentence is not None:
            self.llm_sentence = MetricBunch.make_categorical_columns_string(self.llm_sentence)
            save_pandas(self.llm_sentence, filepath / "llm_sentence.feather", **kwargs)
        if self.human_claim is not None:
            self.human_claim = MetricBunch.make_categorical_columns_string(self.human_claim)
            save_pandas(self.human_claim, filepath / "human_claim.feather", **kwargs)
        if self.human_sentence is not None:
            self.human_sentence = MetricBunch.make_categorical_columns_string(self.human_sentence)
            save_pandas(self.human_sentence, filepath / "human_sentence.feather", **kwargs)

    @classmethod
    def load(cls, filepath: Path | str, metric_name: str = "", **kwargs: Any) -> Self:
        """Load the MetricBunch from files. `filepath` should be a directory containing files:
        * `all_samples.feather` reconstitutes the `all_samples` DataFrame.
        * `llm_claim.feather` reconstitutes the `llm_claim` DataFrame.
        * `llm_sentence.feather` reconstitutes the `llm_sentence` DataFrame.
        * `human_claim.feather` reconstitutes the `human_claim` DataFrame.
        * `human_sentence.feather` reconstitutes the `human_sentence` DataFrame.
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Directory does not exist: {filepath}")
        elif filepath.exists() and not any(filepath.iterdir()):
            raise FileNotFoundError(f"Directory is empty: {filepath}")

        try:
            all_samples = load_pandas(filepath / "all_samples.feather", **kwargs)
            all_samples = MetricBunch.make_categorical_columns_categorical(all_samples)
        except FileNotFoundError:
            all_samples = None
        try:
            llm_claim = load_pandas(filepath / "llm_claim.feather", **kwargs)
            llm_claim = MetricBunch.make_categorical_columns_categorical(llm_claim)
        except FileNotFoundError:
            llm_claim = None
        try:
            llm_sentence = load_pandas(filepath / "llm_sentence.feather", **kwargs)
            llm_sentence = MetricBunch.make_categorical_columns_categorical(llm_sentence)
        except FileNotFoundError:
            llm_sentence = None
        try:
            human_claim = load_pandas(filepath / "human_claim.feather", **kwargs)
            human_claim = MetricBunch.make_categorical_columns_categorical(human_claim)
        except FileNotFoundError:
            human_claim = None
        try:
            human_sentence = load_pandas(filepath / "human_sentence.feather", **kwargs)
            human_sentence = MetricBunch.make_categorical_columns_categorical(human_sentence)
        except FileNotFoundError:
            human_sentence = None

        return MetricBunch(
            metric_name=metric_name,
            all_samples=all_samples,
            llm_claim=llm_claim,
            llm_sentence=llm_sentence,
            human_claim=human_claim,
            human_sentence=human_sentence,
        )

    @staticmethod
    def make_categorical_columns_string(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Categorical Columns to Strings."""
        for col, dtype in zip(
            ["retrieval_method", "reference_format", "reference_only_admission"],
            ["string", "string", "string"],
            strict=False,
        ):
            if col in df.columns:
                df = df.astype({col: dtype})
        return df

    @staticmethod
    def make_categorical_columns_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """Convert String Columns to Categorical."""
        for col, dtype in zip(
            ["retrieval_method", "reference_format", "reference_only_admission"],
            [retrieval_method_dtype, reference_context_dtype, retrieval_scope_dtype],
            strict=False,
        ):
            if col in df.columns:
                df = df.astype({col: dtype})
        return df

    ## Crosstab Creation & Export Methods

    def make_crosstab(
        self,
        stratum: str = "llm_claim",
        index: Sequence[str] = ("retrieval_method", "top_n"),
        columns: Sequence[str] = ("reference_format", "reference_only_admission"),
        values: str = "display_str",
        index_names: Sequence[str] = ("Retrieval Method", "Top N"),
        column_names: Sequence[str] = ("Reference Context Format", "Only Current Admission"),
    ) -> pd.DataFrame:
        """Create a crosstab of the metric values."""
        if stratum not in self.model_fields:
            raise ValueError(f"Invalid stratum: {stratum}.")
        # Pivot to get Cross Tabulation
        df = getattr(self, stratum)
        crosstab = df.pivot(
            index=index,
            columns=columns,
            values=values,
        ).rename_axis(index=index_names, columns=column_names)
        # Reorder Columns
        crosstab = crosstab.loc[
            :,
            [
                ("Score", "Yes"),
                ("Score", "No"),
                ("Absolute Time", "Yes"),
                ("Absolute Time", "No"),
                ("Relative Time", "Yes"),
                ("Relative Time", "No"),
            ],
        ]
        return crosstab

    def export_crosstab(
        self,
        stratum: str = "llm_claim",
        style: bool = True,
        filepath: Path | str | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | Styler:
        """Create a crosstab of the metric values and optionally save to file.
        `filepath` should be a directory to save the crosstab files.
        Individual files are saved by `stratum` name within the directory.
        """
        filepath = Path(filepath) if filepath is not None else None
        if stratum not in self.model_fields:
            raise ValueError(f"Invalid stratum: {stratum}.")

        # Make Cross Tabulation
        crosstab = self.make_crosstab(stratum=stratum, values="display_str")

        # Style Table as Heatmap
        if style:
            crosstab_value = self.make_crosstab(stratum=stratum, values="value")
            crosstab_styler = crosstab.style.background_gradient(
                cmap="Blues", axis=None, gmap=crosstab_value
            )
        # Save to File
        if filepath is not None:
            save_pandas(crosstab, filepath / f"{stratum}.feather", **kwargs)
            save_pandas(crosstab, filepath / f"{stratum}.csv", index=True, **kwargs)
            if style:
                save_pandas(crosstab_styler, filepath / f"{stratum}.xlsx", index=True, **kwargs)
        return crosstab_styler if style else crosstab

    def export_crosstabs(
        self,
        filepath: Path | str,
        style: bool = True,
        strata: Sequence[str] = STRATA,
        **kwargs: Any,
    ) -> None:
        """Export crosstabs for all strata to files."""
        filepath = Path(filepath)
        for stratum in strata:
            self.export_crosstab(filepath=filepath, stratum=stratum, style=style, **kwargs)

    ## Constructor Methods

    @classmethod
    def from_defaults(
        cls,
        rater_verdicts: pd.DataFrame,
        ground_truth_labels: pd.DataFrame | None = None,
        rater_type: str = "human",
        rater_id_col: str = "rater_name",
        strata: Sequence = STRATA,
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        num_parallel_raters: int = 16,
    ) -> Self:
        """Create a MetricBunch from a DataFrame of Rater Verdicts.

        Argument `rater_verdicts` is required.

        NOTE: The presence of `ground_truth_labels` argument changes the behavior
        of the agreement metrics computed and stored in the resulting MetricBunch.
        - If `ground_truth_labels` is provided, then agreement metrics for each rater
        versus the ground truth labels is computed for all propositions.
        - If `ground_truth_labels` is not provided, then we compute the inter-rater
        agreement across all raters over all propositions.
        """
        results: dict[str, Any] = {}
        # Loop through each stratum
        for stratum in (pbar1 := tqdm(strata)):
            pbar1.set_description(desc=f"Stratum: {stratum}")
            if ground_truth_labels is None:
                ## Compute Agreement Metrics Between All Raters (Interrater Agreement)
                # Select Stratum
                stratum_rater_verdicts = cls.select_strata(rater_verdicts, stratum)
                # Compute Agreement Metric Between All Raters
                results[stratum] = MetricBunch.compute_interrater_agreement(
                    rater_verdicts=stratum_rater_verdicts,
                    rater_type=rater_type,
                    categories=CATEGORIES,
                    metric=metric,
                    bootstrap_iterations=bootstrap_iterations,
                    workers=workers,
                )
            else:
                ## Compute Agreement Metrics for Each Rater vs. Ground Truth Reference
                # Select Stratum
                stratum_rater_verdicts = cls.select_strata(rater_verdicts, stratum)
                stratum_gt = cls.select_strata(ground_truth_labels, stratum)
                # Compute Agreement Metric of Rater vs. Ground Truth for All Raters
                results[stratum] = MetricBunch.compute_rater_verdict_agreement_vs_ground_truth(
                    rater_verdicts=stratum_rater_verdicts,
                    ground_truth=stratum_gt,
                    rater_type=rater_type,
                    rater_id_col=rater_id_col,
                    categories=CATEGORIES,
                    metric=metric,
                    bootstrap_iterations=bootstrap_iterations,
                    workers=workers,
                    num_parallel_raters=num_parallel_raters,
                )
        results["metric_name"] = metric
        return cls(**results)

    @classmethod
    def select_strata(cls, verdicts: pd.DataFrame, stratum: str) -> pd.DataFrame:
        """Convenience method to return verdicts only for a specific stratum."""
        match stratum:
            case "all_samples":
                return verdicts
            case "llm_claim":
                return verdicts.query("author_type == 'llm' & proposition_type == 'claim'")
            case "llm_sentence":
                return verdicts.query("author_type == 'llm' & proposition_type == 'sentence'")
            case "human_claim":
                return verdicts.query("author_type == 'human' & proposition_type == 'claim'")
            case "human_sentence":
                return verdicts.query("author_type == 'human' & proposition_type == 'sentence'")
            case _:
                raise ValueError(f"Invalid stratum: {stratum}")

    @staticmethod
    def categorical_agreement_metrics_for_verdicts(
        verdicts: pd.DataFrame,
        metric: str | tuple = "percent_agreement",
        categories: Sequence[str] | None = CATEGORIES,
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> dict[str, AgreementMetric]:
        """Compute Categorical Agreement Metrics for a DataFrame of Verdicts."""
        ia = InterraterAgreement.from_defaults(
            data=verdicts,
            data_kind="categorical",
            categories=categories,
            rater_name="rater_name",
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
        categories: Sequence[str] = CATEGORIES,
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute Interrater Agreement Metrics for a cohort of raters."""
        result = MetricBunch.categorical_agreement_metrics_for_verdicts(
            verdicts=rater_verdicts,
            metric=metric,
            categories=categories,
            bootstrap_iterations=bootstrap_iterations,
            workers=workers,
            show_progress=show_progress,
        )
        agreement_metric = result.get(metric, None)
        # Create DataFrame of Agreement Metric (For consistent API)
        metric_df = (
            pd.DataFrame([agreement_metric.model_dump()], index=[f"{rater_type}_interrater"])
            .rename_axis(index="rater_name")
            .rename(columns={"name": "metric_name"})
        ).astype(
            {
                "metric_name": str,
                "value": float,
                "ci_lower": float,
                "ci_upper": float,
                "p_value": float,
            }
        )
        # Add Display String
        metric_df = metric_df.assign(
            display_str=metric_df.apply(
                lambda row: MetricBunch.make_display_string(row=row, metric_name=metric),
                axis="columns",
            ).astype("string")
        )
        return metric_df

    @staticmethod
    def compute_rater_verdict_agreement_vs_ground_truth(
        rater_verdicts: pd.DataFrame,
        ground_truth: pd.DataFrame,
        rater_type: str = "human",
        rater_id_col: str = "rater_name",
        categories: Sequence[str] = CATEGORIES,
        metric: str = "percent_agreement",
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        num_parallel_raters: int = 16,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Compute a Categorical Agreement Metrics for a cohort of raters vs. ground truth.
        Multiple raters agreement vs. ground truth are computed in parallel."""
        # Compute Agreement Metrics for Each Rater
        metric_dict: dict[str, AgreementMetric | None] = {}
        rater_names = []
        futures = []
        with ProcessPoolExecutor(max_workers=num_parallel_raters) as executor:
            for rater_name, one_rater_verdicts in (
                pbar := tqdm(rater_verdicts.groupby(rater_id_col))
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

            for _ in (pbar := tqdm(as_completed(futures), total=len(futures))):
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
        ).astype(
            {
                "metric_name": str,
                "value": float,
                "ci_lower": float,
                "ci_upper": float,
                "p_value": float,
            }
        )
        # If rater_type is AI, add hyperparameters to the DataFrame
        if rater_type == "ai":
            hyperparams_df = (
                rater_verdicts.drop_duplicates(subset=[rater_id_col])
                .set_index(rater_id_col)
                .loc[
                    :, ["retrieval_method", "top_n", "reference_format", "reference_only_admission"]
                ]
            )
            hyperparams_df = hyperparams_df.assign(
                retrieval_method=hyperparams_df.retrieval_method.apply(
                    lambda x: x.replace("_", " ").title()
                ).astype(retrieval_method_dtype),
                reference_format=hyperparams_df.reference_format.apply(
                    lambda x: x.replace("_", " ").title()
                ).astype(reference_context_dtype),
                reference_only_admission=hyperparams_df.reference_only_admission.apply(
                    lambda x: "Yes" if x else "No"
                ).astype(retrieval_scope_dtype),
            )
            metric_df = metric_df.join(hyperparams_df)
        # Add Display String
        metric_df = metric_df.assign(
            display_str=metric_df.apply(
                lambda row: MetricBunch.make_display_string(row=row, metric_name=metric),
                axis="columns",
            ).astype("string")
        )
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
