import itertools
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Protocol, Self

import krippendorff
import numpy as np
import pandas as pd
import pingouin as pg
import scipy.stats as stats
from irrCAC.raw import CAC
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm


class MetricResult(BaseModel):
    """Container for metric values with confidence intervals, p-values, and bootstrap results."""

    name: str = Field(..., description="Name of the metric.")
    value: float = Field(..., description="Value of the metric.")
    # Confidence Interval
    confidence_level: float | None = Field(default=None, description="Confidence interval level.")
    ci_lower: float | None = Field(default=None, description="Lower bound of confidence interval.")
    ci_upper: float | None = Field(default=None, description="Upper bound of confidence interval.")
    # P-value
    p_value: float | None = Field(default=None, description="P-value of the metric.")
    # Bootstrap Iterations
    bootstrap_iterations: int | None = Field(
        default=None, description="Number of bootstrap iterations."
    )
    bootstrap_values: list[float] | None = Field(default=None, description="Bootstrap estimates.")

    def __str__(self) -> str:
        if self.ci_upper and self.ci_lower:
            return (
                f"{self.name}: "
                f"{self.value:.3f} "
                f"({self.confidence_level:.0%}CI: {self.ci_lower:.3f}, {self.ci_upper:.3f}, "
                f"p-value: {self.p_value:.3f})"
            )
        else:
            return f"{self.name}: {self.value:.3f}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def compute_bootstrap_ci_lower(
        bootstrap_values: list[float], confidence_level: float = 0.95
    ) -> float:
        ci_lower = np.percentile(bootstrap_values, (1 - confidence_level) / 2 * 100)
        return float(ci_lower)

    @staticmethod
    def compute_bootstrap_ci_upper(
        bootstrap_values: list[float], confidence_level: float = 0.95
    ) -> float:
        ci_upper = np.percentile(bootstrap_values, (1 + confidence_level) / 2 * 100)
        return float(ci_upper)

    @staticmethod
    def compute_bootstrap_p_value(metric_value: float, bootstrap_values: list[float]) -> float:
        # Compute Standard Error
        standard_error = np.std(bootstrap_values, ddof=1)
        # If standard error is too small to represent, set to a small value
        if standard_error == 0.0:
            standard_error = 1e-15
        # Compute p-value using standard error and observed metric
        z_score = metric_value / standard_error  # Assuming the null hypothesis value is zero
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
        return p_value


class PairwiseFunction(Protocol):
    __name__: str

    def __call__(self, *, ratings: pd.DataFrame, r1: str, r2: str) -> Any: ...


class MetricFunction(Protocol):
    __name__: str

    def __call__(self, *, ratings: pd.DataFrame, **kwargs: Any) -> MetricResult: ...


class InterraterAgreement(BaseModel):
    """Compute interrater agreement using variety of interrater agreement metrics.
    This is a wrapper around the `irrCAC`, `Krippendorff`, `pingouin` libraries.

    This class transforms the raw long format `data` DataFrame into a ratings table.
    The ratings table has is a N x M matrix
    - N number of raters
    - M number of items
    NOTE: The input data convention for irrCAC and Krippendorff libraries are flipped.
    Krippendorff expects raters as rows and items as columns.
    irrCAC expects items as rows and raters as columns.
    We handle this appropriately in this class.

    The `irrCAC` library only handles interrater agreement for categorical scores and
    implements the following agreement metrics:
    - Fleiss Kappa
    - Gwet AC1/AC2
    - Krippendorff Alpha
    - Conger Kappa
    - Brennar-Pediger Agreement Coefficient
    Even though Krippendorff Alpha can be used to compute agreement for continuous-valued
    (interval) scores, the Krippendorff implementation in irrCAC does not support this.
    So for continuous-valued scores, we use the `Krippendorff` package which contains an
    implementation that supports both categorical-valued and continuous-valued scores.

    For confidence intervals & p-values, the irrCAC library computes these analytically using
    the variance formulas for each of the interrater agreement metrics. However, there is no
    variance provided by the Krippendorff package. So we provide an option to bootstrap the
    metric values to estimate confidence intervals and p-values.

    The `pingouin` library is used to compute Intraclass Correlation Coefficient (ICC) and
    requires a specific data format. In our adapter method, we melt the ratings table
    to provide the correct data format. The `pingouin` library also provides analytically
    determined confidence intervals and p-values for ICC. We also provide a bootstrap option
    to compute confidence intervals and p-values for ICC.

    NOTE: All rudimentary metric calculations are implemented with @staticmethod
    to avoid pickling the entire class object when using multiprocessing for bootstrapping.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: pd.DataFrame = Field(
        ...,
        description="Raw data DataFrame in long format with each item as a row.",
    )
    ratings: pd.DataFrame = Field(
        ..., description="Ratings table with raters as rows and items as columns."
    )
    categories: Sequence[Any] | None = Field(
        default=None,
        description="List of unique categories for the agreement items. "
        "None if items are continuous-valued variables and not categorical.",
    )
    label2id: dict[str, int] | None = Field(
        default=None,
        description="Mapping of category names. None if items are continuous-valued variables.",
    )
    rater_name: str = Field(
        default="rater_name",
        description="Name of the column in `data` DataFrame containing rater IDs.",
    )
    item_name: str = Field(
        default="proposition_id",
        description="Name of the column in `data` DataFrame containing item IDs.",
    )
    values_name: str = Field(
        default="verdict",
        description="Name of the column in `data` DataFrame containing rating values.",
    )

    @classmethod
    def from_defaults(
        cls,
        data: pd.DataFrame,
        data_kind: str = "categorical",
        categories: Sequence[Any] | None = None,
        rater_name: str = "rater_name",
        item_name: str = "proposition_id",
        values_name: str = "verdict",
    ) -> Self:
        ratings = cls._make_ratings_table(
            df=data, rater_name=rater_name, item_name=item_name, values_name=values_name
        )
        if data_kind == "categorical":
            if categories is None:
                categories = sorted(ratings.stack().unique().tolist())
            label2id = {label: i for i, label in enumerate(categories)}
        else:
            # Values for continuous-valued data
            categories = None
            label2id = None
        return cls(
            data=data,
            ratings=ratings,
            categories=categories,
            label2id=label2id,
            rater_name=rater_name,
            item_name=item_name,
            values_name=values_name,
        )

    @staticmethod
    def _make_ratings_table(
        df: pd.DataFrame | None = None,
        rater_name: str = "rater_name",
        item_name: str = "proposition_id",
        values_name: str = "verdict",
    ) -> pd.DataFrame:
        """Transform raw data table into ratings table with raters as rows and items as columns."""
        if df is None:
            ratings = None
        else:
            ratings = df.pivot(columns=item_name, index=rater_name, values=values_name)
        return ratings

    ## Pairwise Rater Count and Agreement Matrix
    @staticmethod
    def pair_count(ratings: pd.DataFrame, r1: str, r2: str) -> int:
        if r1 == r2:
            return pd.NA
        else:
            subset_ratings = ratings.loc[[r1, r2], :].dropna(axis="columns")
            if subset_ratings.empty:
                return pd.NA
            else:
                return subset_ratings.shape[1]

    @staticmethod
    def pair_percent_agreement(ratings: pd.DataFrame, r1: str, r2: str) -> MetricResult:
        if r1 == r2:
            return pd.NA
        else:
            subset_ratings = ratings.loc[[r1, r2], :].dropna(axis="columns")
            if subset_ratings.empty:
                return pd.NA
            else:
                return InterraterAgreement._compute_percent_agreement(ratings=subset_ratings)

    @staticmethod
    def pair_gwet(ratings: pd.DataFrame, r1: str, r2: str) -> MetricResult:
        if r1 == r2:
            return pd.NA
        else:
            subset_ratings = ratings.loc[[r1, r2], :].dropna(axis="columns")
            if subset_ratings.empty:
                return pd.NA
            else:
                return InterraterAgreement._compute_cac_gwet(ratings=subset_ratings)

    def pairwise_matrix(
        self,
        ratings: pd.DataFrame | None = None,
        fn: PairwiseFunction | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Pairwise matrix with counts of ratings shared between any 2 raters."""
        if fn is None:
            raise ValueError("Must provide a pairwise function to compute pairwise matrix.")
        ratings = self.ratings if ratings is None else ratings
        rater_names = ratings.index.tolist()

        rater_combinations = list(itertools.product(rater_names, rater_names))
        if workers <= 1:
            pair_ct: dict[tuple[str, str], Any] = {}
            for r1, r2 in tqdm(
                rater_combinations,
                total=len(rater_combinations),
                desc=f"Computing {fn.__name__}",
                disable=not show_progress,
            ):
                pair_ct[(r1, r2)] = fn(ratings=ratings, r1=r1, r2=r2)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit futures
                futures: dict[tuple[str, str], Any] = {}
                for r1, r2 in rater_combinations:
                    future = executor.submit(fn, ratings=ratings, r1=r1, r2=r2)
                    futures[(r1, r2)] = future
                # Wait for all to complete
                for _ in tqdm(
                    as_completed(futures.values()),
                    desc=f"Computing {fn.__name__}",
                    total=len(futures),
                    disable=not show_progress,
                ):
                    pass
                # Unpack results from futures
                pair_ct = {k: future.result() for k, future in futures.items()}

        pair_mat = (
            pd.Series(data=pair_ct)
            .rename_axis(index=["Rater 1", "Rater 2"])
            .unstack(level="Rater 1")
        )
        return pair_mat

    def pairwise_count_matrix(
        self,
        ratings: pd.DataFrame | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self.pairwise_matrix(
            ratings=ratings,
            fn=InterraterAgreement.pair_count,
            workers=workers,
            show_progress=show_progress,
        )

    def pairwise_percent_agreement_matrix(
        self,
        ratings: pd.DataFrame | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self.pairwise_matrix(
            ratings=ratings,
            fn=InterraterAgreement.pair_percent_agreement,
            workers=workers,
            show_progress=show_progress,
        )

    def pairwise_gwet_matrix(
        self,
        ratings: pd.DataFrame | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        return self.pairwise_matrix(
            ratings=ratings,
            fn=InterraterAgreement.pair_gwet,
            workers=workers,
            show_progress=show_progress,
        )

    ## Bootstrap Metric Computation Helper Methods
    def compute_metric_fn(
        self,
        metric_fn: MetricFunction,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Generic dispatch to bootstrap vs. no bootstrap for `metric_fn`.
        Progress bar is only available when computing via bootstrap."""
        ratings = self.ratings if ratings is None else ratings
        if bootstrap_iterations is None:
            return metric_fn(ratings=ratings, **kwargs)
        else:
            return self._bootstrap(
                ratings=ratings,
                metric_fn=metric_fn,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )

    @staticmethod
    def _resample_and_compute_metric(
        ratings: pd.DataFrame,
        metric_fn: MetricFunction,
        bootstrap_fraction: float,
        random_state: int,
        **kwargs: Any,
    ) -> MetricResult:
        # Bootstrap Resample Rating Items
        boot_ratings = ratings.sample(
            frac=bootstrap_fraction, replace=True, random_state=random_state, axis=1
        )
        # Rename Items to Avoid Duplicate Item Names
        col_axis_name = boot_ratings.columns.name
        boot_ratings.columns = pd.Series(list(range(len(boot_ratings.columns))), name=col_axis_name)
        # Compute Agreement Metric
        return metric_fn(boot_ratings, **kwargs)

    def _bootstrap(
        self,
        ratings: pd.DataFrame,
        metric_fn: MetricFunction,
        metric_name: str | None = None,
        bootstrap_iterations: int = 1000,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Bootstrap the `metric_fn` to estimate confidence intervals and p-values.
        All args & kwargs are passed through to the `metric_fn`.

        `metric_fn` should accept argument `ratings` (pd.DataFrame) and
        return an AgreementMetric object. The `value` field of the AgreementMetric
        object is unpacked and will be used for bootstrap statistics calculation.

        `metric_fn` should be a function or static method (not be an instance method)
        to avoid the entire object being pickled & ensure multiprocessing works
        with appropriate speed/efficiency.
        """
        # Compute Metric Value with All Data
        agreement_metric = metric_fn(ratings, **kwargs)
        metric_value = agreement_metric.value
        metric_name = metric_name or agreement_metric.name

        # Compute Metric Values on Bootstrap Samples
        if workers <= 1:
            boot_metric_values = []
            for i in tqdm(
                range(bootstrap_iterations),
                desc=f"Bootstrap {metric_name}",
                disable=not show_progress,
            ):
                boot_agreement_metric = InterraterAgreement._resample_and_compute_metric(
                    ratings=ratings,
                    metric_fn=metric_fn,
                    bootstrap_fraction=bootstrap_fraction,
                    random_state=i,
                    **kwargs,
                )
                boot_metric_value = boot_agreement_metric.value
                boot_metric_values.append(boot_metric_value)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit futures
                futures = []
                for i in range(bootstrap_iterations):
                    future = executor.submit(
                        InterraterAgreement._resample_and_compute_metric,
                        ratings=ratings,
                        metric_fn=metric_fn,
                        bootstrap_fraction=bootstrap_fraction,
                        random_state=i,
                        **kwargs,
                    )
                    futures.append(future)
                # Wait for all to complete
                for _ in tqdm(
                    as_completed(futures),
                    desc=f"Bootstrap {metric_name}",
                    total=bootstrap_iterations,
                    disable=not show_progress,
                ):
                    pass
                # Unpack results from futures
                boot_agreement_metrics = [future.result() for future in futures]
                boot_metric_values = [metric.value for metric in boot_agreement_metrics]

        # Estimate Confidence Interval
        ci_lower = float(np.percentile(boot_metric_values, (1 - confidence_level) / 2 * 100))
        ci_upper = float(np.percentile(boot_metric_values, (1 + confidence_level) / 2 * 100))

        # Compute Standard Error
        standard_error = float(np.std(boot_metric_values, ddof=1))
        # If standard error is too small to represent, set to a small value
        if standard_error == 0.0:
            standard_error = 1e-15

        # Compute p-value using standard error and observed metric
        z_score = metric_value / standard_error  # Assuming the null hypothesis value is zero
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

        return MetricResult(
            name=metric_name,
            value=metric_value,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            p_value=p_value,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_values=boot_metric_values,
        )

    ## Categorical Label Agreement Metrics

    @staticmethod
    def _compute_percent_agreement(
        ratings: pd.DataFrame,
    ) -> MetricResult:
        """Compute percent agreement for a ratings table.
        This considers all pairwise comparisons of raters and is the number of agreeing
        pair of raters divided by the total number of pairwise comparisons for each item.

        If raters do not have a rating on the item, the item is excluded from the calculation.
        """
        # Get all pairwise comparisons
        rater_pairs = list(itertools.combinations(ratings.index.to_list(), 2))
        # Pairwise Comparisons
        pairwise_agree_vectors = {}
        for r1, r2 in rater_pairs:
            # Get ratings matrix for only the pair of raters; drop non-overlapping items
            sub_ratings = ratings.loc[[r1, r2], :].dropna(axis="columns")
            if sub_ratings.empty:
                continue
            pairwise_agree_vectors[(r1, r2)] = sub_ratings.loc[r1, :] == sub_ratings.loc[r2, :]

        # Each row is an item, each column is a rater pair, value is whether raters agree
        pairwise_agree_df = pd.DataFrame(pairwise_agree_vectors)

        def item_percent_agreement(row: pd.Series) -> float:
            """Computes percent agreement for single item.
            Drops all rater pairs with NaN values."""
            non_nan_agreement_pairs = row.dropna()
            return float(non_nan_agreement_pairs.sum()) / len(non_nan_agreement_pairs)

        # Compute percent agreement for each item, then take average
        item_percent_agreement_df = pairwise_agree_df.apply(item_percent_agreement, axis="columns")
        overall_percent_agreement = item_percent_agreement_df.mean()
        return MetricResult(
            name="percent_agreement",
            value=overall_percent_agreement,
        )

    def percent_agreement(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute percent agreement for a ratings table."""
        return self.compute_metric_fn(
            metric_fn=self._compute_percent_agreement,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    @staticmethod
    def _compute_cac_fleiss(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Fleiss' Kappa for a ratings table."""
        cac = CAC(ratings=ratings.T, **kwargs)
        fleiss = cac.fleiss()
        return MetricResult(
            name="fleiss",
            value=fleiss["est"]["coefficient_value"],
            confidence_level=0.95,
            ci_lower=fleiss["est"]["confidence_interval"][0],
            ci_upper=fleiss["est"]["confidence_interval"][1],
            p_value=fleiss["est"]["p_value"],
        )

    def cac_fleiss(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Fleiss' Kappa for a ratings table."""
        return self.compute_metric_fn(
            metric_fn=self._compute_cac_fleiss,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    @staticmethod
    def _compute_cac_krippendorff(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Krippendorff's Alpha for a ratings table.
        This irrCAC implementation supports only categorical-valued scores."""
        cac = CAC(ratings=ratings.T, **kwargs)
        krippendorff = cac.krippendorff()
        return MetricResult(
            name="krippendorff",
            value=krippendorff["est"]["coefficient_value"],
            confidence_level=0.95,
            ci_lower=krippendorff["est"]["confidence_interval"][0],
            ci_upper=krippendorff["est"]["confidence_interval"][1],
            p_value=krippendorff["est"]["p_value"],
        )

    def cac_krippendorff(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Krippendorff's Alpha for a ratings table.
        This irrCAC implementation supports only categorical-valued scores."""
        return self.compute_metric_fn(
            metric_fn=self._compute_cac_krippendorff,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    @staticmethod
    def _compute_cac_gwet(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Gwet's AC1 for a ratings table."""
        cac = CAC(ratings=ratings.T, **kwargs)
        gwet = cac.gwet()
        return MetricResult(
            name="gwet_AC1",
            value=gwet["est"]["coefficient_value"],
            confidence_level=0.95,
            ci_lower=gwet["est"]["confidence_interval"][0],
            ci_upper=gwet["est"]["confidence_interval"][1],
            p_value=gwet["est"]["p_value"],
        )

    def cac_gwet(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Gwet's AC1 for a ratings table."""
        return self.compute_metric_fn(
            metric_fn=self._compute_cac_gwet,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    @staticmethod
    def _compute_cac_conger(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Conger's Kappa for a ratings table."""
        cac = CAC(ratings=ratings.T, **kwargs)
        conger = cac.conger()
        return MetricResult(
            name="conger",
            value=conger["est"]["coefficient_value"],
            confidence_level=0.95,
            ci_lower=conger["est"]["confidence_interval"][0],
            ci_upper=conger["est"]["confidence_interval"][1],
            p_value=conger["est"]["p_value"],
        )

    def cac_conger(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Conger's Kappa for a ratings table."""
        return self.compute_metric_fn(
            metric_fn=self._compute_cac_conger,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    @staticmethod
    def _compute_cac_brennar_pediger(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Brennar-Prediger Agreement Coefficient for a ratings table."""
        cac = CAC(ratings=ratings.T, **kwargs)
        bp = cac.bp()
        return MetricResult(
            name="brennar_pediger",
            value=bp["est"]["coefficient_value"],
            confidence_level=0.95,
            ci_lower=bp["est"]["confidence_interval"][0],
            ci_upper=bp["est"]["confidence_interval"][1],
            p_value=bp["est"]["p_value"],
        )

    def cac_brennar_pediger(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Brennar-Prediger Agreement Coefficient for a ratings table."""
        return self.compute_metric_fn(
            metric_fn=self._compute_cac_brennar_pediger,
            ratings=ratings,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_fraction=bootstrap_fraction,
            confidence_level=confidence_level,
            workers=workers,
            show_progress=show_progress,
            **kwargs,
        )

    def categorical_agreement(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        metrics: Sequence[str] = ("percent_agreement", "gwet"),
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        """Convenience method to compute all agreement metrics for categorical labels."""
        if isinstance(metrics, str):
            metrics = [metrics]

        agreement_metrics: dict[str, MetricResult] = {}
        if "percent_agreement" in metrics:
            agreement_metrics["percent_agreement"] = self.percent_agreement(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "fleiss" in metrics:
            agreement_metrics["fleiss"] = self.cac_fleiss(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "gwet" in metrics:
            agreement_metrics["gwet"] = self.cac_gwet(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "krippendorff" in metrics:
            agreement_metrics["krippendorff"] = self.cac_krippendorff(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "conger" in metrics:
            agreement_metrics["conger"] = self.cac_conger(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "brennar_pediger" in metrics:
            agreement_metrics["brennar_pediger"] = self.cac_brennar_pediger(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        return agreement_metrics

    ## Continuous Label Agreement Metrics

    @staticmethod
    def _compute_concordance_correlation_coefficient(
        ratings: pd.DataFrame | None = None,
        y_true: Sequence | None = None,
        y_pred: Sequence | None = None,
    ) -> MetricResult:
        """Concordance correlation coefficient.
        Either provide `ratings` argument with 2 raters as rows and items as columns,
        or compare arrays from 2 raters using `y_true` and `y_pred` arguments.

        Code derived from:
        https://rowannicholls.github.io/python/statistics/agreement/concordance_correlation_coefficient.html
        """
        # Format input data
        if isinstance(ratings, pd.DataFrame):
            if ratings.shape[0] != 2:
                raise ValueError(
                    "Concordance Correlation Coefficient can only be computed for a "
                    "Ratings table with 2 raters."
                )
            else:
                # NOTE: Flip for CCC with items as rows, raters as cols
                df = ratings.T
        else:
            # Raw data
            dct = {"y_true": y_true, "y_pred": y_pred}
            df = pd.DataFrame(dct)
        # Use Dataframe format to remove NaN items
        df = df.dropna()
        # Pearson product-moment correlation coefficients
        y_true = df.iloc[:, 0].to_numpy()
        y_pred = df.iloc[:, 1].to_numpy()
        cor = np.corrcoef(y_true, y_pred)[0][1]
        # Means
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        # Population variances
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        # Population standard deviations
        sd_true = np.std(y_true)
        sd_pred = np.std(y_pred)
        # Calculate CCC
        numerator = 2 * cor * sd_true * sd_pred
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
        ccc = numerator / denominator
        return MetricResult(
            name="CCC",
            value=ccc,
        )

    def concordance_correlation_coefficient(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Concordance Correlation Coefficient for a ratings table.
        This implementation supports both continuous-valued scores and can only
        compare 2 raters (`ratings` must be a 2xN dataframe).
        """
        ratings = self.ratings if ratings is None else ratings
        if bootstrap_iterations is None:
            return self._compute_concordance_correlation_coefficient(ratings=ratings, **kwargs)
        else:
            return self._bootstrap(
                ratings=ratings,
                metric_name="CCC",
                metric_fn=self._compute_concordance_correlation_coefficient,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )

    @staticmethod
    def _compute_intraclass_correlation_coefficient(
        ratings: pd.DataFrame,
        rater_name: str = "rater_name",
        values_name: str = "score",
        icc_type: str = "ICC3",
    ) -> MetricResult:
        """Computes Intraclass Correlation Coefficient (ICC) for a ratings table.
        Valid ICC Types are:
        - "ICC1": Measure reliability of individual scores across raters.
            Uses a one-way random-effects ANOVA model.
        - "ICC2": Measure how well raters with random effects provide
            consistent ratings for subjects. Used to generalize results to other
            potential raters and interested in their agreement.
            Uses a two-way random-effects ANOVA model. Raters are treated as random.
        - "ICC3": Measures agreement among a specific fixed set of raters and you are
            not interested in generalizing to others.
            Uses a two-way mixed-effects ANOVA model. Raters are treated as fixed.
        - "ICC1k": Sames as ICC1, but for each item will take the average of scores
            across all raters and then compute ICC. This is used to measure consistency
            of average value of ratings.
        - "ICC2k": Same as ICC2, but for each item will take the average of scores
            across all raters and then compute ICC. This is used to measure consistency
            of average value of ratings.
        - "ICC3k": Same as ICC3, but for each item will take the average of scores
            across all raters and then compute ICC. This is used to measure consistency
            of average value of ratings.

        Default ICC type is ICC3.
        """
        # Melts ratings table into long format for pingouin ICC computation
        ratings_long = (
            ratings.dropna(axis="columns")
            .reset_index(drop=False)
            .melt(
                id_vars=rater_name,
                value_name=values_name,
            )
        )
        icc_df = pg.intraclass_corr(
            data=ratings_long,
            targets="subject_id",
            raters=rater_name,
            ratings="score",
            nan_policy="omit",
        )
        icc = icc_df.loc[icc_df.Type == icc_type].squeeze()
        return MetricResult(
            name="ICC",
            value=icc["ICC"],
            confidence_level=0.95,
            ci_lower=icc["CI95%"][0],
            ci_upper=icc["CI95%"][1],
            p_value=icc["pval"],
        )

    def intraclass_correlation_coefficient(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs,
    ) -> MetricResult:
        """Compute Intraclass Correlation Coefficient (ICC) for a ratings table."""
        ratings = self.ratings if ratings is None else ratings
        if bootstrap_iterations is None:
            return self._compute_intraclass_correlation_coefficient(ratings=ratings, **kwargs)
        else:
            return self._bootstrap(
                ratings=ratings,
                metric_name="ICC",
                metric_fn=self._compute_intraclass_correlation_coefficient,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )

    @staticmethod
    def _compute_krippendorff_alpha(
        ratings: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Krippendorff's Alpha for a ratings table."""
        label2id = kwargs.pop("label2id", None)
        # Ratings are categorical. Convert categorical labels using label mapping
        # Assumes "level_of_measurement" is "nominal" unless otherwise specified.
        if label2id is not None:
            # Convert labels to numpy array
            # NOTE: rows are raters and columns are items that are rated
            data = ratings.map(lambda x: label2id[x] if pd.notna(x) else x).to_numpy(
                dtype="float32", na_value=np.nan
            )
            kwargs["reliability_data"] = data
            kwargs["value_domain"] = list(label2id.values())
            kwargs["level_of_measurement"] = kwargs.get("level_of_measurement", "nominal")
        # Ratings are not categorical. Use raw ratings data.
        # Assumes "level_of_measurement" is "interval" unless otherwise specified.
        else:
            data = ratings.to_numpy(dtype="float32", na_value=np.nan)
            kwargs["reliability_data"] = data
            kwargs["level_of_measurement"] = kwargs.get("level_of_measurement", "interval")
        alpha = krippendorff.alpha(**kwargs)
        return MetricResult(
            name="krippendorff_alpha",
            value=alpha,
        )

    def krippendorff_alpha(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute Krippendorff's Alpha for a ratings table.
        This implementation supports both categorical and continuous-valued scores.
        """
        ratings = self.ratings if ratings is None else ratings
        kwargs["label2id"] = kwargs.get("label2id", self.label2id)
        if bootstrap_iterations is None:
            return self._compute_krippendorff_alpha(ratings=ratings, **kwargs)
        else:
            return self._bootstrap(
                ratings=ratings,
                metric_name="krippendorff_alpha",
                metric_fn=self._compute_krippendorff_alpha,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )

    def continuous_agreement(
        self,
        ratings: pd.DataFrame | None = None,
        bootstrap_iterations: int | None = None,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        metrics: Sequence[str] = ("icc", "ccc", "krippendorff"),
        show_progress: bool = True,
        **kwargs: Any,
    ) -> dict[str, MetricResult]:
        """Convenience method to compute all agreement metrics for continuous labels."""
        if isinstance(metrics, str):
            metrics = [metrics]

        agreement_metrics: dict[str, MetricResult] = {}
        if "icc" in metrics:
            agreement_metrics["icc"] = self.intraclass_correlation_coefficient(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "ccc" in metrics:
            agreement_metrics["ccc"] = self.concordance_correlation_coefficient(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        if "krippendorff" in metrics:
            agreement_metrics["krippendorff"] = self.krippendorff_alpha(
                ratings=ratings,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_fraction=bootstrap_fraction,
                confidence_level=confidence_level,
                workers=workers,
                show_progress=show_progress,
                **kwargs,
            )
        return agreement_metrics
