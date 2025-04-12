from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Protocol, Self

import numpy as np
import pandas as pd
import scipy.stats as stats
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from irr_metrics.interrater import MetricResult
from irr_metrics.type_utils import metric_category_dtype, verdict_label_dtype


class ClassificationMetricFunction(Protocol):
    __name__: str

    def __call__(
        self, *, y_true: list, y_pred: list, labels: list, **kwargs: Any
    ) -> MetricResult | list[MetricResult] | dict[str, MetricResult]: ...


class ClassificationMetrics(BaseModel):
    """Compute Sensitivity, Specificity, PPV, NPV, TP, FP, FN, TN, Support for each class
    in a classification task.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="", description="Name of the MetricBunch.")
    rater_verdicts: list = Field([], description="Labels to compare against ground truth.")
    ground_truth: list = Field([], description="Ground truth labels.")
    verdict_labels: list = Field([], description="List of unique labels for classification task.")
    metrics: pd.DataFrame | None = Field(
        default=None,
        description="Dataframe of computed metrics. Each row corresponds to a metric. "
        "Each column corresponds to a label (verdict category).",
    )

    @classmethod
    def from_defaults(
        cls,
        rater_verdicts: Sequence,
        ground_truth: Sequence,
        verdict_labels: Sequence,
        bootstrap_iterations: int | None = None,
        workers: int = 1,
        show_progress: bool = True,
    ) -> Self:
        "Compute Classification Metrics from Rater Verdicts and Ground Truth."
        metric_result: dict[str, dict[str, MetricResult]] = (
            ClassificationMetrics.compute_sens_spec_ppv_npv(
                y_true=ground_truth,
                y_pred=rater_verdicts,
                labels=verdict_labels,
                bootstrap_iterations=bootstrap_iterations,
                workers=workers,
                show_progress=show_progress,
            )
        )
        metrics = pd.DataFrame(metric_result)
        return cls(
            rater_verdicts=list(rater_verdicts),
            ground_truth=list(ground_truth),
            verdict_labels=list(verdict_labels),
            metrics=metrics,
        )

    def metrics_table(self, fmt: str = "") -> pd.DataFrame:
        "Return metrics table with columns for each metric and labels as rows."
        # Convert Metrics DataFrame to Long Format
        metrics_s = self.metrics.stack()
        metrics_df = pd.DataFrame(
            metrics_s.apply(lambda x: x.model_dump()).tolist(), index=metrics_s.index
        ).reset_index(names=["metric_name", "verdict_label"])

        def make_format_str_for_row(row: pd.Series, fmt: str = fmt) -> str:
            "Create formatted string for display column."
            if row.metric_name in ["TP", "FP", "FN", "TN", "Support"]:
                return f"{int(row.value)} ({int(row.ci_lower)}, {int(row.ci_upper)})"
            else:
                if not fmt or not isinstance(fmt, str):
                    fmt = ".2f"
                return f"{row.value:{fmt}} ({row.ci_lower:{fmt}}, {row.ci_upper:{fmt}})"

        # Add Display String Column
        metrics_df = metrics_df.assign(
            display_str=metrics_df.apply(make_format_str_for_row, axis="columns")
        )
        # Reorder Columns
        metrics_df = metrics_df.loc[
            :,
            [
                "verdict_label",
                "metric_name",
                "value",
                "ci_lower",
                "ci_upper",
                "p_value",
                "bootstrap_iterations",
                "bootstrap_values",
                "display_str",
            ],
        ]
        # Assign Data Types & Sort Values
        metrics_df = metrics_df.astype(
            {
                "verdict_label": verdict_label_dtype,
                "metric_name": metric_category_dtype,
                "value": "Float64",
                "ci_lower": "Float64",
                "ci_upper": "Float64",
                "p_value": "Float64",
                "bootstrap_iterations": "Int64",
                "bootstrap_values": "object",
                "display_str": "string",
            }
        ).sort_values(by=["verdict_label", "metric_name"])
        return metrics_df

    @staticmethod
    def _compute_sens_spec_ppv_npv(
        y_true: Sequence,
        y_pred: Sequence,
        labels: Sequence,
    ) -> dict[str, dict[str, float | int]]:
        "Compute Sensitivity, Specificity, PPV, NPV, TP, FP, FN, TN, Support for each class."
        output: dict[str, dict[str, float | int]] = {}
        # Confusion matrix whose i-th row and j-th column entry indicates the number of samples
        # with true label being i-th class and predicted label being j-th class.
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        # Sens/Spec/PPV/NPV for each class
        for i, label in enumerate(labels):
            # Extract TP, FP, FN, TN from Confusion Matrix
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            # Support
            support = tp + fn
            # Compute Sensitivity (TPR, Recall), Specificity (TNR), PPV (Precision), NPV
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            ppv = tp / (tp + fp)
            npv = tn / (tn + fn)
            output[label] = {
                "TPR": tpr,
                "TNR": tnr,
                "PPV": ppv,
                "NPV": npv,
                "TP": int(tp),
                "FP": int(fp),
                "FN": int(fn),
                "TN": int(tn),
                "Support": int(support),
            }
        return output

    @staticmethod
    def _resample_and_compute_metric(
        y_true: pd.Series,
        y_pred: pd.Series,
        metric_fn: ClassificationMetricFunction,
        bootstrap_fraction: float,
        random_state: int,
        **kwargs: Any,
    ) -> MetricResult:
        # Bootstrap Resample y_true and y_pred
        boot_y_true = y_true.sample(
            frac=bootstrap_fraction, replace=True, random_state=random_state
        )
        boot_y_pred = y_pred.sample(
            frac=bootstrap_fraction, replace=True, random_state=random_state
        )
        # Compute Agreement Metric
        return metric_fn(y_true=boot_y_true, y_pred=boot_y_pred, **kwargs)

    @staticmethod
    def compute_sens_spec_ppv_npv(
        y_true: list | pd.Series,
        y_pred: list | pd.Series,
        labels: list,
        bootstrap_iterations: int = 1000,
        bootstrap_fraction: float = 1.0,
        confidence_level: float = 0.95,
        workers: int = 1,
        show_progress: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        "Bootstrap sample original data and compute Sens/Spec/PPV/NPV for each class."
        # Convert y_true and y_pred to Series if not already
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true, name="y_true")
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, name="y_pred")

        # Compute Metrics on All Data
        observed_result: dict[str, dict[str, float | int]] = (
            ClassificationMetrics._compute_sens_spec_ppv_npv(
                y_true=y_true, y_pred=y_pred, labels=labels
            )
        )

        # Compute Metric Values on Bootstrap Samples
        if workers <= 1:
            boot_metric_results: list[dict[str, dict[str, float | int]]] = []
            for i in tqdm(
                range(bootstrap_iterations),
                desc="Bootstrap Sens/Spec/PPV/NPV",
                disable=not show_progress,
            ):
                boot_metric_result: dict[str, dict[str, float | int]] = (
                    ClassificationMetrics._resample_and_compute_metric(
                        y_true=y_true,
                        y_pred=y_pred,
                        labels=labels,
                        metric_fn=ClassificationMetrics._compute_sens_spec_ppv_npv,
                        bootstrap_fraction=bootstrap_fraction,
                        random_state=i,
                        **kwargs,
                    )
                )
                boot_metric_results.append(boot_metric_result)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit futures
                futures = []
                for i in range(bootstrap_iterations):
                    future = executor.submit(
                        ClassificationMetrics._resample_and_compute_metric,
                        y_true=y_true,
                        y_pred=y_pred,
                        labels=labels,
                        metric_fn=ClassificationMetrics._compute_sens_spec_ppv_npv,
                        bootstrap_fraction=bootstrap_fraction,
                        random_state=i,
                        **kwargs,
                    )
                    futures.append(future)
                # Wait for all to complete
                for _ in tqdm(
                    as_completed(futures),
                    desc="Bootstrap Sens/Spec/PPV/NPV",
                    total=bootstrap_iterations,
                    disable=not show_progress,
                ):
                    pass
                # Unpack results from futures
                boot_metric_results: list[dict[str, dict[str, float | int]]] = [
                    future.result() for future in futures
                ]

        # Unpack Each of the Metrics for each Label
        boot_tpr = defaultdict(list)
        boot_tnr = defaultdict(list)
        boot_ppv = defaultdict(list)
        boot_npv = defaultdict(list)
        boot_tp = defaultdict(list)
        boot_fp = defaultdict(list)
        boot_fn = defaultdict(list)
        boot_tn = defaultdict(list)
        boot_support = defaultdict(list)
        for boot_metric_result in boot_metric_results:
            for label in labels:
                boot_tpr[label] += [boot_metric_result[label]["TPR"]]
                boot_tnr[label] += [boot_metric_result[label]["TNR"]]
                boot_ppv[label] += [boot_metric_result[label]["PPV"]]
                boot_npv[label] += [boot_metric_result[label]["NPV"]]
                boot_tp[label] += [boot_metric_result[label]["TP"]]
                boot_fp[label] += [boot_metric_result[label]["FP"]]
                boot_fn[label] += [boot_metric_result[label]["FN"]]
                boot_tn[label] += [boot_metric_result[label]["TN"]]
                boot_support[label] += [boot_metric_result[label]["Support"]]

        def compute_ci(values: list[float], confidence_level: float) -> tuple[float, float]:
            """Estimate Confidence Interval"""
            ci_lower = float(np.percentile(values, (1 - confidence_level) / 2 * 100))
            ci_upper = float(np.percentile(values, (1 + confidence_level) / 2 * 100))
            return ci_lower, ci_upper

        def compute_p_value(values: list[float], observed_value: float) -> float:
            """Compute p-value using standard error and observed metric"""
            # Compute Standard Error
            standard_error = float(np.std(values, ddof=1))
            # If standard error is too small to represent, set to a small value
            if standard_error == 0.0:
                standard_error = 1e-15

            # Compute p-value using standard error and observed metric
            z_score = observed_value / standard_error  # Assuming the null hypothesis value is zero
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
            return p_value

        # Create MetricResults for each Metric & Label
        metric_results: dict[str, dict[str, MetricResult]] = {}
        for label in labels:
            if label not in metric_results:
                metric_results[label] = {}

            for metric_name, boot_metric_values in zip(
                ["TPR", "TNR", "PPV", "NPV", "TP", "FP", "FN", "TN", "Support"],
                [
                    boot_tpr,
                    boot_tnr,
                    boot_ppv,
                    boot_npv,
                    boot_tp,
                    boot_fp,
                    boot_fn,
                    boot_tn,
                    boot_support,
                ],
            ):
                # For Single Metric, Get Observed Value & Boot Values
                observed_value = observed_result[label][metric_name]
                boot_values = boot_metric_values[label]
                # Compute Confidence Intervals & p-value
                ci_lower, ci_upper = compute_ci(boot_values, confidence_level)
                p_value = compute_p_value(boot_values, observed_value)
                # Create MetricResult
                metric_results[label][metric_name] = MetricResult(
                    name=metric_name,
                    value=observed_value,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    confidence_level=confidence_level,
                    p_value=p_value,
                    bootstrap_iterations=bootstrap_iterations,
                    bootstrap_values=boot_values,
                )
        return metric_results
