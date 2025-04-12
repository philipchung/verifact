# Scripts to Compute Human Clinican Inter-Rater Agreement

The scripts focus on the first round of annotation where 3 clinicians annotate each propositions in `VeriFact-BHC` dataset. We analyze the interrater agreement across these propositions and specific subsets:

1. All Propositions
2. Propositions where at least one rater assigned a `Supported` verdict
3. Propositions where at least one rater assigned a `Not Supported` verdict
4. Propositions where at least one rater assigned a `Not Addressed` verdict

Two types of inter-rater agreement metrics are computed:

1. Percent Agreement
2. Gwet's AC1

## Running Computation

Metric computation uses 1000 bootstrap interations to estimate confidence intervals.

```sh
# Compute Percent Agreement
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-All" --metric=percent_agreement --proposition-strata=all --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-Supported" --metric=percent_agreement --proposition-strata=supported --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-NotSupported" --metric=percent_agreement --proposition-strata=not_supported --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-NotAddressed" --metric=percent_agreement --proposition-strata=not_addressed --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-AllBinarized" --metric=percent_agreement --proposition-strata=all_binarized --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-SupportedBinarized" --metric=percent_agreement --proposition-strata=supported_binarized --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="PA-NotSupportedOrAddressed" --metric=percent_agreement --proposition-strata=not_supported_or_addressed --workers=64 --bootstrap-iterations=1000

# Compute Gwet's AC1
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-All" --metric=gwet --proposition-strata=all --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-Supported" --metric=gwet --proposition-strata=supported --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-NotSupported" --metric=gwet --proposition-strata=not_supported --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-NotAddressed" --metric=gwet --proposition-strata=not_addressed --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-AllBinarized" --metric=gwet --proposition-strata=all_binarized --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-SupportedBinarized" --metric=gwet --proposition-strata=supported_binarized --workers=64 --bootstrap-iterations=1000;\
uv run python scripts/analysis_interrater/0_compute_interrater_agreement.py --run-name="Gwet-NotSupportedOrAddressed" --metric=gwet --proposition-strata=not_supported_or_addressed --workers=64 --bootstrap-iterations=1000
```
