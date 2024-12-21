# Services

Custom docker containers and configurations:
* embed: Infinity embedding server with some files replaced for BGE-M3 compatibility to yield sparse and dense embedding in single API call
* rerank: Infinity embedding server with some files to reduce memory footprint
* vllm: vLLM monitoring configurations using grafana and prometheus

Instructions for Use:
Build each Dockerfile and use local embed and rerank service. Additional files are copied into the Dockerfile to override original files.