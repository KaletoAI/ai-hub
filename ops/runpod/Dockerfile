# AI-Hub's RunPod Serverless worker — image profile (Qwen-Image 2.1).
# Generated build context: ops/runpod/ in ai-hub, copied into the private worker repo by
# ops/runpod/sync.sh. Stack pins = the Thunder bootstrap's (tests/test_runpod_worker.py).
# Our own code; nothing copied from runpod-workers/worker-comfyui (AGPL-3.0).
FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

ARG PY_VER=3.13
ARG TORCH_VER=2.11.0
ARG TORCHVISION_VER=0.26.0
ARG TORCHAUDIO_VER=2.11.0
ARG TORCH_CUDA=cu130
ARG COMFY_COMMIT=1d61dcc35c35541388c0001bacc7703db14e8bea
ARG COMFY_URL=https://github.com/comfyanonymous/ComfyUI

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_INSTALL_DIR=/usr/local/bin VENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && uv venv --python "${PY_VER}" --seed "${VENV}"
RUN uv pip install --python "${VENV}/bin/python" \
        "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" "torchaudio==${TORCHAUDIO_VER}" \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
RUN git clone --quiet "${COMFY_URL}" /comfyui \
    && git -C /comfyui checkout --quiet "${COMFY_COMMIT}" \
    && rm -rf /comfyui/.git \
    && uv pip install --python "${VENV}/bin/python" -r /comfyui/requirements.txt
COPY nodes.image.txt install-nodes.sh /opt/gw/
RUN /opt/gw/install-nodes.sh /opt/gw/nodes.image.txt /comfyui "${VENV}/bin/python"
RUN uv pip install --python "${VENV}/bin/python" runpod websockets
COPY extra_model_paths.yaml /comfyui/extra_model_paths.yaml
COPY gw_placeholder.png /comfyui/input/gw_placeholder.png
COPY handler.py worker.json /opt/gw/
WORKDIR /opt/gw
CMD ["python", "-u", "/opt/gw/handler.py"]
