# LLM Judge

This package contains all code for running LLM-as-a-Judge.

`JudgeSingleSubject` contains code to perform evaluation on a single Subject ID. Wrapped within is `ReferenceContextMaker` which retrieves facts from the EHR and forms a reference context, as well as `Judge` which is the actual LLM-as-a-Judge performing the evluation.

`JudgeCohort` defines a grid search over a cohort of `JudgeSingleSubjects`.
