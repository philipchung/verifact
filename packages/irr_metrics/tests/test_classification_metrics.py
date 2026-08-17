import math

import pandas as pd
import pytest
from irr_metrics import ClassificationMetrics, MetricBunch

LABELS = ["Supported", "Not Supported", "Not Addressed"]
Y_TRUE = [
    "Supported",
    "Supported",
    "Not Supported",
    "Not Supported",
    "Not Addressed",
    "Not Addressed",
]
Y_PRED = [
    "Supported",
    "Not Supported",
    "Not Supported",
    "Not Addressed",
    "Not Addressed",
    "Supported",
]


def test_classification_metrics_multiclass_values() -> None:
    results = ClassificationMetrics.compute_classification_metrics(
        y_true=Y_TRUE,
        y_pred=Y_PRED,
        labels=LABELS,
        bootstrap_iterations=4,
        workers=1,
        show_progress=False,
    )

    assert results["overall"]["mcc"].value == pytest.approx(0.25)
    for label in LABELS:
        metrics = results[label]
        assert metrics["tpr"].value == pytest.approx(0.5)
        assert metrics["tnr"].value == pytest.approx(0.75)
        assert metrics["ppv"].value == pytest.approx(0.5)
        assert metrics["npv"].value == pytest.approx(0.75)
        assert metrics["tp"].value == 1
        assert metrics["fp"].value == 1
        assert metrics["fn"].value == 1
        assert metrics["tn"].value == 3
        assert metrics["support"].value == 2


def test_metric_bunch_exposes_mcc_and_per_class_metrics() -> None:
    proposition_ids = [f"p{i}" for i in range(len(Y_TRUE))]
    ground_truth = pd.DataFrame(
        {
            "proposition_id": proposition_ids,
            "rater_name": "human_gt",
            "verdict": Y_TRUE,
        }
    )
    predictions = pd.DataFrame(
        {
            "proposition_id": proposition_ids,
            "rater_name": "judge",
            "verdict": Y_PRED,
        }
    )

    results = MetricBunch.classification_metrics_for_verdicts(
        verdicts=pd.concat([ground_truth, predictions], ignore_index=True),
        categories=LABELS,
        bootstrap_iterations=4,
        workers=1,
        show_progress=False,
    )

    assert results["mcc"].value == pytest.approx(0.25)
    assert results["s-tpr"].value == pytest.approx(0.5)
    assert results["ns-ppv"].value == pytest.approx(0.5)
    assert results["na-tnr"].value == pytest.approx(0.75)
    assert not math.isnan(results["mcc"].ci_lower)


def test_metric_bunch_rejects_nonoverlapping_propositions() -> None:
    ground_truth = pd.DataFrame({"proposition_id": ["ground-truth"], "verdict": ["Supported"]})
    predictions = pd.DataFrame({"proposition_id": ["prediction"], "verdict": ["Supported"]})

    with pytest.raises(ValueError, match="No overlapping proposition IDs"):
        MetricBunch.classification_metrics_for_verdicts(
            verdicts=predictions,
            ground_truth=ground_truth,
            categories=LABELS,
            bootstrap_iterations=1,
            show_progress=False,
        )
