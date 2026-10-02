# ops/runpod — the RunPod Serverless worker's build context

The Docker build context of AI-Hub's ComfyUI worker for RunPod Serverless (image
profile: Qwen-Image 2.1). Stack pins equal the Thunder bootstrap's, the node list is a
subset of `ops/thunder-nodes.default.txt`; `tests/test_runpod_worker.py` pins both.

The image is built on RunPod from a PRIVATE worker repo — never from ai-hub. Operator steps:

1. `ops/runpod/sync.sh <checkout of the private worker repo>` — copies this directory and
   writes `worker.json` (the version every job reports back).
2. In that checkout: `git add -A && git commit && git push`.
3. `gh release create v<N>` — the release starts the (billed) build on RunPod's GitHub
   integration.
4. Point the Serverless endpoint at the new release and attach the network volume
   (models under `/runpod-volume/models/<folder>/`, see `extra_model_paths.yaml`).

`handler.py` (the RunPod entry point) lives next to these files and is copied by `sync.sh`.
