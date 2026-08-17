from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

_RUN_LEGACY_PARITY = os.environ.get("VERIFACT_RUN_LEGACY_PARITY") == "1"
_MODEL_DIRECTORY_MAP = {
    "Llama-8B": "verifact_llama3_1_8B",
    "Llama-70B": "verifact_llama3_1_70B",
    "R1-8B": "verifact_deepseek_r1_distill_llama_8B",
    "R1-70B": "verifact_deepseek_r1_distill_llama_70B",
    "Gemma3-12B": "verifact_gemma3_12B",
    "Gemma3-27B": "verifact_gemma3_27B",
    "Qwen3-32B": "verifact_qwen3_32B",
    "Qwen3-30B-A3B-Instruct": "verifact_qwen3_30B-A3B-Instruct",
    "Qwen3-30B-A3B-Thinking": "verifact_qwen3_30B-A3B-Thinking",
}
_PARITY_COLUMNS = [
    "subject_id",
    "author_type",
    "proposition_type",
    "fact_type",
    "retrieval_method",
    "top_n",
    "reference_format",
    "reference_only_admission",
    "deduplicate_text",
    "verdict",
    "reason",
    "reasoning_chain",
    "reasoning_final_answer",
    "text",
]

SCRIPT = Path(__file__).resolve().parents[1] / "release_data.py"
SPEC = importlib.util.spec_from_file_location("analysis_release_data", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT}")
release_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_data
SPEC.loader.exec_module(release_data)


class ReleaseDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.release_dir = Path(self.temporary_directory.name)
        (self.release_dir / "verifact").mkdir()
        (self.release_dir / "propositions").mkdir()

        self.raters = pd.DataFrame(
            [
                {
                    "model": "Gemma-3-12B",
                    "rater_alias": "v1.1.0-rater-001",
                    "rater_name": (
                        "model=Gemma-3-12B,fact_type=claim,retrieval_method=rerank,"
                        "top_n=50,reference_format=absolute_time,"
                        "reference_only_admission=True,deduplicate_text=True"
                    ),
                    "fact_type": "claim",
                    "retrieval_method": "rerank",
                    "top_n": 50,
                    "reference_format": "absolute_time",
                    "reference_only_admission": True,
                    "deduplicate_text": True,
                },
                {
                    "model": "Llama-8B",
                    "rater_alias": "v1.1.0-rater-002",
                    "rater_name": (
                        "model=Llama-8B,fact_type=sentence,retrieval_method=dense,"
                        "top_n=100,reference_format=relative_time,"
                        "reference_only_admission=False,deduplicate_text=False"
                    ),
                    "fact_type": "sentence",
                    "retrieval_method": "dense",
                    "top_n": 100,
                    "reference_format": "relative_time",
                    "reference_only_admission": False,
                    "deduplicate_text": False,
                },
            ]
        )
        self.verdicts = pd.DataFrame(
            [
                {
                    "model": "Gemma-3-12B",
                    "proposition_id": "synthetic-p1",
                    "rater_alias": "v1.1.0-rater-001",
                    "verdict": "Supported",
                    "reason": "Synthetic reason one",
                    "reasoning_chain": "",
                    "reasoning_final_answer": "",
                    "reference_id": "synthetic-ref-a",
                },
                {
                    "model": "Llama-8B",
                    "proposition_id": "synthetic-p2",
                    "rater_alias": "v1.1.0-rater-002",
                    "verdict": "Not Addressed",
                    "reason": "Synthetic reason two",
                    "reasoning_chain": "",
                    "reasoning_final_answer": "",
                    "reference_id": "synthetic-ref-b",
                },
            ]
        )
        self.propositions = pd.DataFrame(
            [
                {
                    "proposition_id": "synthetic-p1",
                    "text": "Synthetic proposition one",
                    "subject_id": 101,
                    "author_type": "human",
                    "proposition_type": "claim",
                },
                {
                    "proposition_id": "synthetic-p2",
                    "text": "Synthetic proposition two",
                    "subject_id": 202,
                    "author_type": "llm",
                    "proposition_type": "sentence",
                },
            ]
        )
        self.human_verdicts = self.propositions.drop(columns="subject_id").assign(
            human_gt=["Supported", "Not Supported"]
        )
        self.references = pd.DataFrame(
            {
                "reference_id": ["synthetic-ref-a", "synthetic-ref-b"],
                "reference": [" alpha\tbeta\n gamma ", "single"],
            }
        )
        self._write_release()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_release(self) -> None:
        self.raters.to_csv(self.release_dir / "verifact/rater_configurations.csv", index=False)
        self.verdicts.to_parquet(self.release_dir / "verifact/verdicts.parquet", index=False)
        self.references.to_parquet(
            self.release_dir / "verifact/reference_payloads.parquet", index=False
        )
        self.propositions.to_csv(self.release_dir / "propositions/propositions.csv.gz", index=False)
        self.human_verdicts.to_csv(
            self.release_dir / "propositions/human_verdicts.csv.gz", index=False
        )

    def test_annotations_restore_historical_contract_and_filter_models(self) -> None:
        annotations = release_data.load_release_annotations(self.release_dir, models=["Gemma3-12B"])

        self.assertEqual(len(annotations), 1)
        row = annotations.iloc[0]
        self.assertEqual(row["model"], "Gemma3-12B")
        self.assertEqual(row["reference_format"], "absolute time")
        self.assertEqual(
            row["rater_name"],
            "model=Gemma3-12B,fact_type=claim,retrieval_method=rerank,top_n=50,"
            "reference_format=absolute_time,reference_only_admission=True,"
            "deduplicate_text=True",
        )
        self.assertEqual(row["text"], "Synthetic proposition one")
        self.assertEqual(row["subject_id"], 101)
        self.assertIn("reference_id", annotations.columns)
        self.assertIn("rater_alias", annotations.columns)
        self.assertNotIn("reference", annotations.columns)
        self.assertEqual(str(annotations["model"].dtype), "string")
        self.assertEqual(str(annotations["subject_id"].dtype), "Int64")
        self.assertEqual(str(annotations["reference_only_admission"].dtype), "boolean")

    def test_annotations_accept_release_model_name(self) -> None:
        annotations = release_data.load_release_annotations(self.release_dir, models="Gemma-3-12B")
        self.assertEqual(annotations["model"].tolist(), ["Gemma3-12B"])

    def test_annotations_push_model_filter_into_parquet_read(self) -> None:
        original_load_pandas = release_data.load_pandas
        calls: list[dict] = []

        def capture_load(*args, **kwargs):
            calls.append(kwargs)
            return original_load_pandas(*args, **kwargs)

        with patch.object(release_data, "load_pandas", side_effect=capture_load):
            release_data.load_release_annotations(self.release_dir, models=["Gemma3-12B"])

        self.assertEqual(calls[0]["filters"], [("model", "in", ["Gemma-3-12B"])])
        self.assertEqual(calls[0]["columns"], release_data._VERDICT_COLUMNS)

    def test_annotations_reject_unknown_and_duplicate_model_aliases(self) -> None:
        with self.assertRaisesRegex(release_data.ReleaseDataError, "Unknown model"):
            release_data.load_release_annotations(self.release_dir, models=["Synthetic-1B"])
        with self.assertRaisesRegex(release_data.ReleaseDataError, "duplicate aliases"):
            release_data.load_release_annotations(
                self.release_dir, models=["Gemma-3-12B", "Gemma3-12B"]
            )

    def test_annotations_reject_unresolved_rater_key(self) -> None:
        self.verdicts.loc[0, "rater_alias"] = "missing-rater"
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "unresolved"):
            release_data.load_release_annotations(self.release_dir)

    def test_annotations_reject_unresolved_proposition_key(self) -> None:
        self.verdicts.loc[0, "proposition_id"] = "missing-proposition"
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "unresolved"):
            release_data.load_release_annotations(self.release_dir)

    def test_annotations_reject_duplicate_rater_alias(self) -> None:
        self.raters.loc[1, "rater_alias"] = self.raters.loc[0, "rater_alias"]
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "duplicate rater_alias"):
            release_data.load_release_annotations(self.release_dir)

    def test_annotations_reject_duplicate_verdict_key(self) -> None:
        duplicate = self.verdicts.iloc[[0]].assign(model="Llama-8B")
        self.verdicts = pd.concat([self.verdicts, duplicate], ignore_index=True)
        self._write_release()
        with self.assertRaisesRegex(
            release_data.ReleaseDataError, "duplicate proposition_id/rater_alias"
        ):
            release_data.load_release_annotations(self.release_dir)

    def test_annotations_reject_missing_required_column(self) -> None:
        self.verdicts = self.verdicts.drop(columns="reason")
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "required columns"):
            release_data.load_release_annotations(self.release_dir)

    def test_annotations_reject_duplicate_proposition_key(self) -> None:
        self.propositions = pd.concat([self.propositions, self.propositions.iloc[[0]]])
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "duplicate proposition_id"):
            release_data.load_release_annotations(self.release_dir)

    def test_ground_truth_has_analysis_ready_schema_and_types(self) -> None:
        ground_truth = release_data.load_release_ground_truth(self.release_dir)

        self.assertEqual(
            ground_truth.columns.tolist(),
            [
                "proposition_id",
                "text",
                "author_type",
                "proposition_type",
                "rater_name",
                "verdict",
            ],
        )
        self.assertEqual(ground_truth["rater_name"].tolist(), ["human_gt", "human_gt"])
        self.assertEqual(str(ground_truth["verdict"].dtype), "string")

    def test_ground_truth_rejects_null_label(self) -> None:
        self.human_verdicts.loc[0, "human_gt"] = None
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "null human_gt"):
            release_data.load_release_ground_truth(self.release_dir)

    def test_reference_lengths_use_python_semantics_across_batches(self) -> None:
        with patch.object(release_data, "_PARQUET_BATCH_SIZE", 1):
            lengths = release_data.load_reference_lengths(self.release_dir)

        lengths = lengths.set_index("reference_id")
        self.assertEqual(lengths.loc["synthetic-ref-a", "reference_word_count"], 3)
        self.assertEqual(lengths.loc["synthetic-ref-a", "reference_char_count"], 19)
        self.assertEqual(lengths.loc["synthetic-ref-b", "reference_word_count"], 1)
        self.assertEqual(str(lengths["reference_word_count"].dtype), "Int64")

    def test_reference_lengths_filter_and_validate_requested_ids(self) -> None:
        lengths = release_data.load_reference_lengths(
            self.release_dir, reference_ids=["synthetic-ref-b"]
        )
        self.assertEqual(lengths["reference_id"].tolist(), ["synthetic-ref-b"])

        with self.assertRaisesRegex(release_data.ReleaseDataError, "duplicate values"):
            release_data.load_reference_lengths(
                self.release_dir,
                reference_ids=["synthetic-ref-b", "synthetic-ref-b"],
            )
        with self.assertRaisesRegex(release_data.ReleaseDataError, "missing requested"):
            release_data.load_reference_lengths(
                self.release_dir, reference_ids=["missing-reference"]
            )

    def test_reference_lengths_reject_duplicate_payload_ids(self) -> None:
        self.references = pd.concat([self.references, self.references.iloc[[0]]])
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "duplicate reference_id"):
            release_data.load_reference_lengths(self.release_dir)

    def test_reference_lengths_reject_null_payload(self) -> None:
        self.references.loc[0, "reference"] = None
        self._write_release()
        with self.assertRaisesRegex(release_data.ReleaseDataError, "null values"):
            release_data.load_reference_lengths(self.release_dir)

    def test_attach_payloads_preserves_order_index_and_input(self) -> None:
        annotations = pd.DataFrame(
            {"reference_id": ["synthetic-ref-b", "synthetic-ref-a", "synthetic-ref-b"]},
            index=pd.Index([8, 3, 5], name="source_row"),
        ).astype({"reference_id": "string"})
        original = annotations.copy()

        with patch.object(release_data, "_PARQUET_BATCH_SIZE", 1):
            attached = release_data.attach_reference_payloads(annotations, self.release_dir)

        pd.testing.assert_frame_equal(annotations, original)
        self.assertEqual(attached.index.tolist(), [8, 3, 5])
        self.assertEqual(attached.index.name, "source_row")
        self.assertEqual(
            attached["reference"].tolist(),
            ["single", " alpha\tbeta\n gamma ", "single"],
        )
        self.assertEqual(len(attached), len(annotations))

    def test_attach_payloads_handles_empty_and_rejects_invalid_input(self) -> None:
        empty = pd.DataFrame({"reference_id": pd.Series(dtype="string")})
        attached = release_data.attach_reference_payloads(empty, self.release_dir)
        self.assertEqual(attached.columns.tolist(), ["reference_id", "reference"])
        self.assertTrue(attached.empty)

        with self.assertRaisesRegex(release_data.ReleaseDataError, "must contain"):
            release_data.attach_reference_payloads(pd.DataFrame(), self.release_dir)
        with self.assertRaisesRegex(release_data.ReleaseDataError, "already contains"):
            release_data.attach_reference_payloads(
                pd.DataFrame({"reference_id": ["synthetic-ref-a"], "reference": ["old"]}),
                self.release_dir,
            )

    @unittest.skipUnless(
        _RUN_LEGACY_PARITY,
        "set VERIFACT_RUN_LEGACY_PARITY=1 to compare local private legacy results",
    )
    def test_all_nine_models_match_release_scoped_legacy_columns(self) -> None:
        release_root = os.environ.get("VERIFACT_PARITY_RELEASE_DIR")
        legacy_root = os.environ.get("VERIFACT_PARITY_LEGACY_RESULTS_DIR")
        if not release_root or not legacy_root:
            self.fail(
                "VERIFACT_PARITY_RELEASE_DIR and VERIFACT_PARITY_LEGACY_RESULTS_DIR "
                "must be set when legacy parity is enabled"
            )

        for model, model_directory in _MODEL_DIRECTORY_MAP.items():
            with self.subTest(model=model):
                release = release_data.load_release_annotations(Path(release_root), models=[model])
                legacy_path = Path(legacy_root) / model_directory / "score_reports/verdicts.feather"
                legacy = release_data.load_pandas(
                    legacy_path,
                    columns=["proposition_id", "rater_name", *_PARITY_COLUMNS],
                )
                if legacy.index.name == "proposition_id":
                    legacy = legacy.reset_index()
                if "proposition_id" not in legacy.columns:
                    self.fail(f"Legacy results have no proposition_id: {legacy_path}")
                legacy["rater_name"] = "model=" + model + "," + legacy["rater_name"]
                legacy = legacy.astype({"proposition_id": "string", "rater_name": "string"})

                key = ["proposition_id", "rater_name"]
                if release.duplicated(key).any() or legacy.duplicated(key).any():
                    self.fail(f"Comparison keys are not unique for {model}")
                release = release.set_index(key).sort_index().loc[:, _PARITY_COLUMNS]
                legacy = legacy.set_index(key)
                missing = release.index.difference(legacy.index)
                self.assertEqual(len(missing), 0, f"Missing legacy keys for {model}")
                legacy = legacy.loc[release.index, _PARITY_COLUMNS]

                text_columns = [
                    "reason",
                    "reasoning_chain",
                    "reasoning_final_answer",
                ]
                legacy[text_columns] = legacy[text_columns].fillna("")
                pd.testing.assert_frame_equal(
                    release,
                    legacy,
                    check_dtype=False,
                    obj=f"release-scoped legacy parity for {model}",
                )


if __name__ == "__main__":
    unittest.main()
