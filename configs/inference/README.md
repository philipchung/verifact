# Manuscript inference profiles

These files are Docker Compose and replay-CLI overlays for the nine LLM judges represented in the
PhysioNet VeriFact-BHC v1.1.0 release. They contain no credentials.

Each profile pins:

- the release-facing model name (`VERIFACT_MODEL`);
- the exact judge and tokenizer identifiers recorded in `verifact/model_metadata.csv`;
- whether the judge uses the reasoning-plus-auxiliary-LLM path;
- model context length and vLLM launch arguments;
- explicit Qwen thinking behavior where it changes the manuscript model identity.

Use a profile after the machine-specific `.env` so its model settings take precedence:

```sh
docker compose --env-file .env \
  --env-file configs/inference/llama-70b.env \
  up -d traefik llm-main
```

Reasoning profiles require `llm-aux`. GPU assignments and memory utilization remain
machine-specific settings in `.env`; adjust those settings to the available hardware without
changing the published model or tokenizer identity.

The profiles describe the historical runtime configuration and are validated statically in this
repository. Model startup and full inference still require compatible NVIDIA GPUs and access to
the gated model repositories.
