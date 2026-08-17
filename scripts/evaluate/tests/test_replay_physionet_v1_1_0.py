from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "replay_physionet_v1_1_0.py"
SPEC = importlib.util.spec_from_file_location("replay_physionet_v1_1_0", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to import {SCRIPT}")
replay = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay
SPEC.loader.exec_module(replay)

RUN_SCRIPT = Path(__file__).resolve().parents[1] / "run_verifact.py"
RUN_SPEC = importlib.util.spec_from_file_location("run_verifact_profile_test", RUN_SCRIPT)
if RUN_SPEC is None or RUN_SPEC.loader is None:
    raise RuntimeError(f"Unable to import {RUN_SCRIPT}")
run_verifact = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run_verifact
RUN_SPEC.loader.exec_module(run_verifact)

REPO_ROOT = Path(__file__).resolve().parents[3]


class FakeJudge:
    async def a_evaluate(self, *, texts, references, proposition_ids, **kwargs):
        del texts, references, kwargs
        verdicts = [
            SimpleNamespace(
                proposition_id=proposition_id,
                verdict="Supported",
                reason="Synthetic replay reason",
                reasoning_chain="",
                reasoning_final_answer="",
            )
            for proposition_id in proposition_ids
        ]
        return SimpleNamespace(verdicts=verdicts)


class ReplayFixtureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.release_dir = Path(self.temporary_directory.name) / "release"
        self.output_dir = Path(self.temporary_directory.name) / "outputs"
        (self.release_dir / "verifact").mkdir(parents=True)
        (self.release_dir / "propositions").mkdir()
        self.profile_path = Path(self.temporary_directory.name) / "model.env"
        self.profile_path.write_text(
            "\n".join(
                [
                    "VERIFACT_MODEL=Test-Model",
                    "VERIFACT_OUTPUT_SUBDIR=test_model",
                    "IS_REASONING_MODEL=false",
                    "LLM_MODEL_NAME=org/test-model",
                    "TOKENIZER_MODEL_NAME=org/test-tokenizer",
                    "LLM_MAX_MODEL_LEN=20000",
                    "MAIN_VLLM_USE_V1=1",
                    'EXTRA_ARGS="--enable-prefix-caching"',
                ]
            )
            + "\n"
        )
        self._write_release()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_release(self) -> None:
        pd.DataFrame(
            [
                {
                    "model": "Test-Model",
                    "judge_model_name": "org/test-model",
                    "tokenizer_model_name": "org/test-tokenizer",
                    "is_reasoning_model": False,
                    "structured_output_model_name": "",
                    "structured_output_tokenizer_name": "",
                }
            ]
        ).to_csv(self.release_dir / "verifact/model_metadata.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "Test-Model",
                    "rater_alias": "v1.1.0-rater-001",
                    "fact_type": "claim",
                    "retrieval_method": "rerank",
                    "top_n": 50,
                    "reference_format": "absolute_time",
                    "reference_only_admission": True,
                    "deduplicate_text": True,
                }
            ]
        ).to_csv(self.release_dir / "verifact/rater_configurations.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "Test-Model",
                    "author_type": "llm",
                    "proposition_type": "claim",
                    "fact_type": "claim",
                    "retrieval_method": "rerank",
                    "top_n": 50,
                    "reference_format": "absolute_time",
                    "reference_only_admission": True,
                    "deduplicate_text": True,
                    "publication_role": "main",
                }
            ]
        ).to_csv(self.release_dir / "verifact/model_configuration_matrix.csv", index=False)
        pd.DataFrame(
            [
                {
                    "model": "Test-Model",
                    "proposition_id": "p1",
                    "rater_alias": "v1.1.0-rater-001",
                    "reference_id": "ref-1",
                    "verdict": "Supported",
                },
                {
                    "model": "Test-Model",
                    "proposition_id": "p2",
                    "rater_alias": "v1.1.0-rater-001",
                    "reference_id": "ref-2",
                    "verdict": "Not Addressed",
                },
            ]
        ).to_parquet(self.release_dir / "verifact/verdicts.parquet", index=False)
        pd.DataFrame(
            [
                {
                    "proposition_id": "p1",
                    "text": "First proposition",
                    "subject_id": 101,
                    "author_type": "llm",
                    "proposition_type": "claim",
                },
                {
                    "proposition_id": "p2",
                    "text": "Second proposition",
                    "subject_id": 101,
                    "author_type": "llm",
                    "proposition_type": "claim",
                },
            ]
        ).to_csv(self.release_dir / "propositions/propositions.csv.gz", index=False)
        pd.DataFrame(
            [
                {"reference_id": "ref-1", "reference": "Reference one"},
                {"reference_id": "ref-2", "reference": "Reference two"},
            ]
        ).to_parquet(self.release_dir / "verifact/reference_payloads.parquet", index=False)

    def test_profile_and_release_metadata_must_match(self) -> None:
        profile = replay.load_model_profile(self.profile_path)
        metadata = replay.load_model_metadata(self.release_dir, profile.model)

        replay.validate_profile_against_metadata(profile, metadata)
        self.assertEqual(profile.judge_model_name, "org/test-model")

        bad_metadata = metadata | {"judge_model_name": "org/wrong-model"}
        with self.assertRaisesRegex(replay.ReplayConfigurationError, "does not match"):
            replay.validate_profile_against_metadata(profile, bad_metadata)

    def test_manifest_uses_only_released_verdict_keys_and_filters(self) -> None:
        manifest = replay.load_replay_manifest(
            self.release_dir,
            model="Test-Model",
            subject_ids=[101],
            top_n=[50],
            limit=1,
        )

        self.assertEqual(manifest["proposition_id"].tolist(), ["p1"])
        self.assertEqual(manifest["expected_verdict"].tolist(), ["Supported"])
        self.assertEqual(manifest["publication_role"].tolist(), ["main"])

    def test_reference_join_is_bounded_to_selected_ids(self) -> None:
        manifest = replay.load_replay_manifest(self.release_dir, model="Test-Model", limit=1)
        joined = replay.attach_reference_payloads(manifest, self.release_dir)

        self.assertEqual(joined["reference"].tolist(), ["Reference one"])
        self.assertEqual(len(joined), len(manifest))

    def test_replay_writes_comparable_output_and_resume_validates_keys(self) -> None:
        manifest = replay.load_replay_manifest(self.release_dir, model="Test-Model")
        manifest = replay.attach_reference_payloads(manifest, self.release_dir)

        first = asyncio.run(
            replay.replay_manifest(
                manifest,
                judge=FakeJudge(),
                output_dir=self.output_dir,
                resume=True,
            )
        )
        second = asyncio.run(
            replay.replay_manifest(
                manifest,
                judge=FakeJudge(),
                output_dir=self.output_dir,
                resume=True,
            )
        )

        self.assertEqual(first, {"completed_groups": 1, "skipped_groups": 0, "completed_rows": 2})
        self.assertEqual(second, {"completed_groups": 0, "skipped_groups": 1, "completed_rows": 0})
        output_path = next(self.output_dir.rglob("*.parquet"))
        output = pd.read_parquet(output_path)
        self.assertEqual(output["reference_id"].tolist(), ["ref-1", "ref-2"])
        self.assertEqual(output["verdict_matches_release"].tolist(), [True, False])

    def test_missing_reference_is_rejected(self) -> None:
        references_path = self.release_dir / "verifact/reference_payloads.parquet"
        pd.read_parquet(references_path).head(1).to_parquet(references_path, index=False)
        manifest = replay.load_replay_manifest(self.release_dir, model="Test-Model")

        with self.assertRaisesRegex(replay.ReplayConfigurationError, "Missing 1"):
            replay.attach_reference_payloads(manifest, self.release_dir)

    def test_replay_rejects_judge_output_with_changed_keys(self) -> None:
        class WrongKeyJudge(FakeJudge):
            async def a_evaluate(self, **kwargs):
                report = await super().a_evaluate(**kwargs)
                report.verdicts[0].proposition_id = "wrong-proposition"
                return report

        manifest = replay.load_replay_manifest(self.release_dir, model="Test-Model")
        manifest = replay.attach_reference_payloads(manifest, self.release_dir)

        with self.assertRaisesRegex(RuntimeError, "proposition IDs"):
            asyncio.run(
                replay.replay_manifest(
                    manifest,
                    judge=WrongKeyJudge(),
                    output_dir=self.output_dir,
                    resume=True,
                )
            )


class PublishedModelProfileTests(unittest.TestCase):
    EXPECTED_MODELS = {
        "Gemma-3-12B": ("google/gemma-3-12b-it", "google/gemma-3-12b-it", False),
        "Gemma-3-27B": ("google/gemma-3-27b-it", "google/gemma-3-27b-it", False),
        "Llama-70B": (
            "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
            False,
        ),
        "Llama-8B": (
            "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            False,
        ),
        "Qwen-3-30B-A3B-Instruct": (
            "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
            False,
        ),
        "Qwen-3-30B-A3B-Thinking": (
            "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
            "Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
            True,
        ),
        "Qwen-3-32B": ("Qwen/Qwen3-32B-FP8", "Qwen/Qwen3-32B-FP8", False),
        "R1-70B": (
            "casperhansen/deepseek-r1-distill-llama-70b-awq",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            True,
        ),
        "R1-8B": (
            "casperhansen/deepseek-r1-distill-llama-8b-awq",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            True,
        ),
    }

    def test_all_published_model_profiles_have_exact_runtime_identity(self) -> None:
        profiles = [
            replay.load_model_profile(path)
            for path in sorted((REPO_ROOT / "configs/inference").glob("*.env"))
        ]

        self.assertEqual({profile.model for profile in profiles}, set(self.EXPECTED_MODELS))
        for profile in profiles:
            self.assertEqual(
                (
                    profile.judge_model_name,
                    profile.tokenizer_model_name,
                    profile.is_reasoning_model,
                ),
                self.EXPECTED_MODELS[profile.model],
            )

        qwen = {profile.model: profile for profile in profiles}
        self.assertFalse(qwen["Qwen-3-32B"].enable_thinking)
        self.assertTrue(qwen["Qwen-3-30B-A3B-Thinking"].enable_thinking)


class ModelFactoryTests(unittest.TestCase):
    def test_qwen_thinking_flag_reaches_openai_request_body(self) -> None:
        from rag.components import models

        with (
            patch.object(models, "OpenAILike") as openai_like,
            patch.dict("os.environ", {"LLM_CHAT_TEMPLATE_ENABLE_THINKING": "false"}, clear=False),
        ):
            models.get_llm(
                model_name="Qwen/Qwen3-32B-FP8",
                api_base="http://llm.localhost/v1/",
                context_window=16000,
            )

        kwargs = openai_like.call_args.kwargs
        self.assertEqual(
            kwargs["additional_kwargs"]["extra_body"]["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_non_qwen_model_does_not_receive_thinking_flag(self) -> None:
        from rag.components import models

        with patch.object(models, "OpenAILike") as openai_like:
            models.get_llm(
                model_name="google/gemma-3-12b-it",
                api_base="http://llm.localhost/v1/",
                context_window=40000,
            )

        self.assertNotIn("extra_body", openai_like.call_args.kwargs["additional_kwargs"])


class FullPipelineProfileTests(unittest.TestCase):
    def test_profile_overrides_host_process_model_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "model.env"
            profile.write_text(
                "\n".join(
                    [
                        "VERIFACT_MODEL=Qwen-3-32B",
                        "VERIFACT_OUTPUT_SUBDIR=verifact_qwen3_32B",
                        "IS_REASONING_MODEL=false",
                        "LLM_MODEL_NAME=Qwen/Qwen3-32B-FP8",
                        "TOKENIZER_MODEL_NAME=Qwen/Qwen3-32B-FP8",
                    ]
                )
                + "\n"
            )
            with patch.dict("os.environ", {"LLM_MODEL_NAME": "old/default"}, clear=False):
                values = run_verifact.activate_model_profile(profile)

                self.assertEqual(values["IS_REASONING_MODEL"], "false")
                self.assertEqual(run_verifact.os.environ["LLM_MODEL_NAME"], "Qwen/Qwen3-32B-FP8")

    def test_profile_is_authoritative_for_full_pipeline_reasoning_mode(self) -> None:
        profile = {"IS_REASONING_MODEL": "true"}

        self.assertTrue(run_verifact.resolve_reasoning_mode(False, profile))
        self.assertFalse(run_verifact.resolve_reasoning_mode(False, None))


if __name__ == "__main__":
    unittest.main()
