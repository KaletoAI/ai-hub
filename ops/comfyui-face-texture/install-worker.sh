#!/usr/bin/env bash
# Portable CPU fallback. Never activate or install into ComfyUI's own venv.
set -euo pipefail

PACKAGE_DIR="$(dirname -- "$(readlink -f -- "$0")")"
VENV="$PACKAGE_DIR/.venv"
MODELS="$PACKAGE_DIR/models"
if [[ $# -gt 1 ]]; then
    echo "Usage: bash $0 [python3.12-executable]" >&2
    exit 2
fi
PYTHON312="${1:-${PYTHON312:-}}"
UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]] && command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
fi

if [[ ! -x "$VENV/bin/python" ]]; then
    if [[ -n "$PYTHON312" ]]; then
        "$PYTHON312" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Worker requires Python 3.12"'
        "$PYTHON312" -m venv "$VENV"
    elif [[ -n "$UV_BIN" ]]; then
        "$UV_BIN" venv --python 3.12 "$VENV"
    elif command -v python3.12 >/dev/null 2>&1; then
        python3.12 -m venv "$VENV"
    else
        echo "Python 3.12 is required. Install uv or pass an existing Python 3.12 executable: bash $0 /path/to/python3.12" >&2
        exit 2
    fi
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing worker .venv must use Python 3.12"; assert sys.prefix != sys.base_prefix, "Refusing installation outside a virtual environment"'
if [[ -n "$UV_BIN" ]]; then
    "$UV_BIN" pip install --python "$VENV/bin/python" -r "$PACKAGE_DIR/requirements-worker.txt"
else
    "$VENV/bin/python" -m pip install -r "$PACKAGE_DIR/requirements-worker.txt"
fi

# MediaPipe's AudioRecord supports sounddevice being absent. This worker uses
# vision only: importing sounddevice would initialize host PortAudio/PulseAudio
# even though no audio is requested, and fails on headless GPU servers.
# Remove it only from this dedicated CPU environment, never from ComfyUI.
if [[ -n "$UV_BIN" ]]; then
    "$UV_BIN" pip uninstall --python "$VENV/bin/python" sounddevice
else
    "$VENV/bin/python" -m pip uninstall -y sounddevice
fi

mkdir -p "$MODELS"
# Model assets are versioned official MediaPipe releases, verified even when
# an operator has already copied the weights into place.
"$VENV/bin/python" - "$MODELS" <<'PY'
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import urllib.request

model_dir = Path(sys.argv[1])
models = (
    ("face_landmarker.task", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    ("selfie_multiclass.tflite", "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite", "c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0"),
)
for name, url, expected in models:
    target = model_dir / name
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        print(f"Verified {name}")
        continue
    fd, temporary = tempfile.mkstemp(prefix=f".{name}.", dir=model_dir)
    os.close(fd)
    temporary = Path(temporary)
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Model hash mismatch for {name}: expected {expected}, received {actual}")
        temporary.replace(target)
        print(f"Downloaded and verified {name}")
    finally:
        temporary.unlink(missing_ok=True)
PY
printf 'FaceTextureOnly CPU worker installed in %s. ComfyUI dependencies were not changed.\n' "$VENV"
