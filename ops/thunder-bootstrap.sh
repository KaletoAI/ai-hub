#!/usr/bin/env bash
# The ComfyUI part of a managed host's setup (Thunder Compute GPU VM, Ubuntu container,
# user `ubuntu`): runs only when a ComfyUI service is attached — at the host's first
# start right AFTER ops/host-bootstrap.sh (which stops the template's own ComfyUI
# autostart, reports the models and node packs the template brought, and provides
# `flock` and ~/.local/bin/uv), or later when a ComfyUI service is attached to a running
# host. The gateway streams this file over SSH and reads its output:
#
#   ssh … -- ubuntu@<ip> bash -s -- <comfy_commit> [<nodes_file>]   (script on stdin)
#
# <nodes_file> defaults to ~/.gw-nodes.txt, which the controller uploads first — one
# custom-node pack per line (see ops/thunder-nodes.default.txt for the two forms).
# Lines starting with `GW:` are the machine-readable part of the output:
#
#   GW:PHASE <name>                   entering a phase (the log is shown in the panel)
#   GW:NODE_FAIL <name> <reason>      one node pack failed; the bootstrap goes on
#   GW:SMOKE ok | GW:SMOKE fail <a,b> CUDA matmul + the 3D extensions' imports
#   GW:DONE                           finished; ~/start-comfy.sh is ready to run
#
# Exit status: 0 done, 2 usage, 3 smoke test failed (the instance stays up for
# diagnosis), anything else = a step that cannot be skipped failed (see the last
# GW:PHASE and the "bootstrap: failed in phase" line).
#
# Adaptive, because what Thunder's `comfy-ui` template ships (ComfyUI version, venv)
# is only visible on the instance: an existing ComfyUI is reused and pinned
# to <comfy_commit>, an existing venv is kept only if it matches the k12-gpu build the
# extension wheels are compiled for (Python 3.13, torch 2.11.0+cu130) — otherwise a new
# one is built. Re-running on the same VM is safe (every step checks what is there).
#
# Everything runs inside main(), called on the LAST line: bash reads a script from a
# pipe as it executes it, so any child that reads stdin (pip, git, an install.py) would
# otherwise swallow the rest of this file. main also gets </dev/null for the same reason.
set -euo pipefail
set -E

COMFY_URL=https://github.com/comfyanonymous/ComfyUI
REGISTRY_API=https://api.comfy.org
PY_VER=3.13
TORCH_VER=2.11.0
TORCHVISION_VER=0.26.0
TORCHAUDIO_VER=2.11.0
TORCH_CUDA=cu130

PHASE=init
NODE_FAILS=()
NODE_DIRS=()
NODE_REASON=

phase() { PHASE=$1; echo "GW:PHASE $1"; }
die() { echo "bootstrap: $*" >&2; exit 1; }
node_fail() {
  NODE_FAILS+=("$1:$2")
  echo "GW:NODE_FAIL $1 $2"
}

# parse_node_line <line> → sets NODE_KIND (git|registry), NODE_NAME, NODE_SRC, NODE_REV.
# Returns 0 for a node, 1 for a blank/comment line, 2 for a line that is neither form.
# (ops/host-bootstrap.sh carries a byte-identical copy for its template-node report.)
#   <https-url>@<commit>       NODE_NAME = repo name without .git
#   registry:<id>@<version>    NODE_NAME = <id>
# NODE_NAME becomes a directory under custom_nodes/, so it is held to plain characters
# (no `..`, no `/`) and the pin to characters that are safe in a URL path.
parse_node_line() {
  local line=$1
  NODE_KIND= NODE_NAME= NODE_SRC= NODE_REV=
  line=${line%$'\r'}
  line="${line#"${line%%[![:space:]]*}"}"
  line="${line%"${line##*[![:space:]]}"}"
  [ -n "$line" ] || return 1
  [ "${line:0:1}" != "#" ] || return 1
  case $line in *@*) ;; *) return 2 ;; esac
  NODE_SRC=${line%@*}
  NODE_REV=${line##*@}
  if [[ $NODE_SRC == registry:* ]]; then
    NODE_KIND=registry
    NODE_NAME=${NODE_SRC#registry:}
  else
    NODE_KIND=git
    [[ $NODE_SRC == https://* ]] || return 2
    NODE_NAME=${NODE_SRC%/}
    NODE_NAME=${NODE_NAME##*/}
    NODE_NAME=${NODE_NAME%.git}
  fi
  [[ $NODE_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || return 2
  [[ $NODE_REV =~ ^[A-Za-z0-9][A-Za-z0-9._+-]*$ ]] || return 2
  return 0
}

# git_pin <dir> <url> <rev>: point origin at <url>, make <rev> present, check it out.
# --force because the pin is the point: a template's local edits must not survive it.
git_pin() {
  local dir=$1 url=$2 rev=$3
  if git -C "$dir" remote get-url origin >/dev/null 2>&1; then
    git -C "$dir" remote set-url origin "$url" || return 1
  else
    git -C "$dir" remote add origin "$url" || return 1
  fi
  git -C "$dir" fetch --quiet origin || return 1
  if ! git -C "$dir" cat-file -e "$rev^{commit}" 2>/dev/null; then
    # a commit on no branch head (GitHub serves fetch-by-sha)
    git -C "$dir" fetch --quiet origin "$rev" || return 1
  fi
  git -C "$dir" checkout --quiet --force "$rev" || return 1
  git -C "$dir" submodule update --init --recursive --quiet || return 1
}

venv_ok() {  # venv_ok <python>: 0 when it is the build the extension wheels expect
  [ -x "$1" ] || { echo "no interpreter at $1" >&2; return 1; }
  "$1" - "$PY_VER" "$TORCH_VER" "$TORCH_CUDA" <<'PY'
import sys
want_py, want_torch, want_cuda = sys.argv[1:4]
have = "%d.%d" % sys.version_info[:2]
if have != want_py:
    sys.exit(f"python {have}, want {want_py}")
import torch
base, _, local = torch.__version__.partition("+")
if base != want_torch or local != want_cuda:
    sys.exit(f"torch {torch.__version__}, want {want_torch}+{want_cuda}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False")
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
PY
}

make_venv() {  # make_venv <dir>: Python $PY_VER venv with pip in it
  local dir=$1 py
  if py=$(command -v "python$PY_VER") && "$py" -m venv "$dir"; then
    return 0
  fi
  # no system python$PY_VER (or no ensurepip for it): uv (installed by the host
  # bootstrap) fetches a standalone build into ~/.local/share/uv — inside $HOME, so it
  # is part of every snapshot
  rm -rf "$dir"
  if [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "no ~/.local/bin/uv (ops/host-bootstrap.sh installs it — did it run?)" >&2
    return 1
  fi
  "$HOME/.local/bin/uv" venv --quiet --seed --python "$PY_VER" "$dir" || return 1
}

# build_venv <dir>: new venv with the pinned torch. 1 = no venv/torch install,
# 2 = installed but not the expected CUDA build (reason on stdout).
build_venv() {
  local dir=$1 py=$1/bin/python why
  make_venv "$dir" || return 1
  "$py" -m pip install --quiet --upgrade pip setuptools wheel || return 1
  "$py" -m pip install --quiet "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
    "torchaudio==$TORCHAUDIO_VER" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    || return 1
  if ! why=$(venv_ok "$py" 2>&1); then
    echo "venv: ${why##*$'\n'}"
    return 2
  fi
  echo "new venv: $why"
}

# registry_install <id> <version> <dir>: the Comfy registry's zip of that exact version.
# The API answers a version record whose `downloadUrl` is the zip (read with the venv's
# python — jq is not on the image). A marker file makes a re-run skip the download.
registry_install() {
  local id=$1 ver=$2 dir=$3 meta url tmp
  if [ -f "$dir/.gw-registry" ] && [ "$(cat "$dir/.gw-registry")" = "$id@$ver" ]; then
    return 0
  fi
  if ! meta=$(curl -fsSL --retry 3 "$REGISTRY_API/nodes/$id/versions/$ver"); then
    NODE_REASON=registry-api; return 1
  fi
  if ! url=$(printf '%s' "$meta" | "$PY" -c \
      'import json,sys; print(json.load(sys.stdin)["downloadUrl"])'); then
    NODE_REASON=no-downloadUrl; return 1
  fi
  tmp=$(mktemp -d) || { NODE_REASON=tmp; return 1; }
  if ! curl -fsSL --retry 3 -o "$tmp/node.zip" "$url"; then
    rm -rf "$tmp"; NODE_REASON=download; return 1
  fi
  # zipfile.extractall drops absolute paths and `..` components
  if ! "$PY" - "$tmp/node.zip" "$tmp/x" <<'PY'
import os, sys, zipfile
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
entries = os.listdir(sys.argv[2])
# registry zips carry the pack at the root; tolerate one wrapping folder
if len(entries) == 1 and os.path.isdir(os.path.join(sys.argv[2], entries[0])):
    inner = os.path.join(sys.argv[2], entries[0])
    os.rename(inner, sys.argv[2] + ".inner")
    os.rmdir(sys.argv[2])
    os.rename(sys.argv[2] + ".inner", sys.argv[2])
PY
  then
    rm -rf "$tmp"; NODE_REASON=unzip; return 1
  fi
  rm -rf "$dir" && mv "$tmp/x" "$dir" && printf '%s\n' "$id@$ver" >"$dir/.gw-registry" \
    || { rm -rf "$tmp"; NODE_REASON=install; return 1; }
  rm -rf "$tmp"
}

git_install() {  # git_install <url> <rev> <dir>
  local url=$1 rev=$2 dir=$3
  if [ -e "$dir" ] && [ ! -e "$dir/.git" ]; then
    rm -rf "$dir"  # a template copy without history cannot be pinned
  fi
  if [ ! -e "$dir" ]; then
    if ! git clone --quiet "$url" "$dir"; then
      rm -rf "$dir"; NODE_REASON=clone; return 1
    fi
  fi
  git_pin "$dir" "$url" "$rev" || { NODE_REASON=checkout-$rev; return 1; }
}

install_nodes() {
  local line rc dir
  # fd 3, not stdin: pip/git/curl in the loop body must not read the list
  while IFS= read -r -u 3 line || [ -n "$line" ]; do
    rc=0
    parse_node_line "$line" || rc=$?
    if [ "$rc" -eq 1 ]; then continue; fi
    if [ "$rc" -ne 0 ]; then
      node_fail "$(printf '%s' "$line" | tr -c 'A-Za-z0-9._:/@-' '_' | cut -c1-80)" malformed-line
      continue
    fi
    dir="$CUI/custom_nodes/$NODE_NAME"
    echo "node $NODE_NAME @ $NODE_REV"
    NODE_REASON=
    if [ "$NODE_KIND" = registry ]; then
      registry_install "$NODE_NAME" "$NODE_REV" "$dir" || { node_fail "$NODE_NAME" "$NODE_REASON"; continue; }
    else
      git_install "$NODE_SRC" "$NODE_REV" "$dir" || { node_fail "$NODE_NAME" "$NODE_REASON"; continue; }
    fi
    NODE_DIRS+=("$dir")
    if [ -f "$dir/requirements.txt" ]; then
      (cd "$dir" && "$PY" -m pip install --quiet -r requirements.txt) \
        || node_fail "$NODE_NAME" requirements
    fi
  done 3<"$NODES"
}

# Compiled CUDA extensions of the 3D packs. Neither comes from requirements.txt:
#  - ComfyUI-Trellis2 (and its GGUF fork) ship prebuilt wheels per torch build under
#    wheels/Linux/Torch<digits>/ (Torch2110 = torch 2.11.0) — installed --no-deps so a
#    wheel cannot pull another torch;
#  - ComfyUI-Hunyuan3d-2-1 builds two from source (hy3dpaint/…, nvcc from CUDA 13.0).
# SMOKE_MODS is what the smoke test must be able to import. It STARTS with the fixed
# baseline — the modules the Trellis2/Hunyuan3D node code imports on k12-gpu — so a
# pack that failed to install cannot drop its modules from the test and let the
# bootstrap end in `GW:SMOKE ok`. On top: every wheel a pack ships for ANY torch build
# (a pack with no wheel for OUR torch fails the smoke test instead of the first 3D job).
SMOKE_BASELINE="cumesh o_voxel flex_gemm nvdiffrast.torch nvdiffrec_render custom_rasterizer custom_rasterizer_kernel mesh_inpaint_processor"
declare -A SMOKE_MODS=()
for _m in $SMOKE_BASELINE; do SMOKE_MODS[$_m]=1; done
unset _m
install_extensions() {
  local d name w mod pair sub tt=${TORCH_VER//./} pytag=cp${PY_VER//./}
  local -a whls
  if ! command -v nvcc >/dev/null && [ -x /usr/local/cuda/bin/nvcc ]; then
    export PATH="/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
  fi
  shopt -s nullglob
  for d in "${NODE_DIRS[@]}"; do
    name=${d##*/}
    for w in "$d"/wheels/Linux/*/*.whl; do
      mod=${w##*/}; mod=${mod%%-*}
      [ "$mod" = nvdiffrast ] && mod=nvdiffrast.torch
      SMOKE_MODS[$mod]=1
    done
    whls=("$d"/wheels/Linux/Torch"$tt"/*-"$pytag"-"$pytag"-linux_x86_64.whl)
    if [ "${#whls[@]}" -gt 0 ]; then
      "$PY" -m pip install --quiet --no-deps "${whls[@]}" || node_fail "$name" wheels
    elif [ -d "$d/wheels/Linux" ]; then
      # not a failure by itself: another pack (or this one's install.py) may provide
      # the same modules — the smoke test imports them and decides
      echo "$name ships no wheels for Torch$tt/$pytag"
    fi
    for pair in hy3dpaint/custom_rasterizer:custom_rasterizer_kernel \
                hy3dpaint/DifferentiableRenderer:mesh_inpaint_processor; do
      sub=${pair%%:*}; mod=${pair##*:}
      [ -f "$d/$sub/setup.py" ] || continue
      SMOKE_MODS[$mod]=1
      [ "$mod" = custom_rasterizer_kernel ] && SMOKE_MODS[custom_rasterizer]=1
      if "$PY" -c "import torch, $mod" 2>/dev/null; then continue; fi
      (cd "$d/$sub" && "$PY" -m pip install --quiet --no-build-isolation .) \
        || node_fail "$name" "build:$mod"
    done
  done
  shopt -u nullglob
}

# ComfyUI-Manager runs a pack's install.py after its requirements; so do we (the
# Trellis2-GGUF one patches cumesh/o_voxel, hence AFTER the extension wheels).
run_install_scripts() {
  local d
  for d in "${NODE_DIRS[@]}"; do
    [ -f "$d/install.py" ] || continue
    echo "install.py of ${d##*/}"
    (cd "$d" && timeout 1800 "$PY" install.py) || node_fail "${d##*/}" install.py
  done
}

# stop_comfy_processes <comfy dir>: our start loop, and every `main.py` whose working
# directory is inside <comfy dir> (the template's ComfyUI runs from there, whatever
# python and flags it was started with).
stop_comfy_processes() {
  local cui=$1 p cwd
  pkill -f '[s]tart-comfy[.]sh' || true
  for p in $(pgrep -f 'main\.py' || true); do
    cwd=$(readlink -f "/proc/$p/cwd" 2>/dev/null || true)
    case $cwd in
      "$cui"|"$cui"/*) kill "$p" 2>/dev/null || true ;;
    esac
  done
  pkill -f '[m]ain[.]py --listen 127[.]0[.]0[.]1 --port 8188' || true
}

write_start_script() {
  # The interpreter is fixed at write time (whichever venv was chosen); everything
  # else is resolved when the script runs.
  {
    echo '#!/usr/bin/env bash'
    printf 'COMFY_PY=%q\n' "$PY"
    cat <<'START_COMFY_EOF'
# Written by ops/thunder-bootstrap.sh. Supervised by nothing but itself:
# ComfyUI-Manager's reboot (the gateway's restart action and auto_restart) exits the
# process and expects a wrapper to start it again — without the loop ComfyUI would stay
# dead after the first restart. One instance per VM (the lock); fd 9 is closed for the
# child so a killed wrapper cannot leave the lock held by an orphaned ComfyUI.
# Loopback only: the gateway reaches it through its SSH tunnel, and Thunder's port
# forwarding is public without auth. --disable-cuda-malloc like k12-gpu: Thunder's GPU
# layer does not implement every CUDA call, cudaMallocAsync is the riskiest default.
set -u
exec 9>"$HOME/.start-comfy.lock"
flock -n 9 || exit 0
export HF_HOME="$HOME/hf-cache"
cd "$HOME/ComfyUI" || exit 1
while :; do
  "$COMFY_PY" main.py --listen 127.0.0.1 --port 8188 --disable-cuda-malloc \
    9>&- >>"$HOME/comfy.log" 2>&1
  sleep 2
done
START_COMFY_EOF
  } >"$HOME/start-comfy.sh"
  chmod +x "$HOME/start-comfy.sh"
}

smoke_test() {  # prints "ok" or "fail <a,b>" on stdout, details on stderr
  "$PY" - "${!SMOKE_MODS[@]}" <<'PY'
import importlib, sys
fails = []
try:
    import torch
    a = torch.randn(1024, 1024, device="cuda")
    (a @ a).sum().item()
    torch.cuda.synchronize()
except Exception as e:
    print(f"smoke: torch/cuda: {e!r}", file=sys.stderr)
    fails.append("torch-cuda")
for mod in sorted(sys.argv[1:]):
    try:
        importlib.import_module(mod)
        print(f"smoke: import {mod} ok", file=sys.stderr)
    except Exception as e:
        print(f"smoke: import {mod}: {e!r}", file=sys.stderr)
        fails.append(mod)
print("ok" if not fails else "fail " + ",".join(fails))
PY
}

# Last words on every exit path that got this far: the node failures matter most
# exactly when the smoke test failed and the instance stays up.
finish() {  # finish <smoke result>
  local smoke=$1
  mkdir -p "$HOME/hf-cache"
  if [ "${#NODE_FAILS[@]}" -gt 0 ]; then
    echo "node steps failed (${#NODE_FAILS[@]}): ${NODE_FAILS[*]}"
  fi
  echo "GW:SMOKE $smoke"
  if [ "$smoke" != ok ]; then exit 3; fi
  echo "GW:DONE"
}

main() {
  local COMMIT=${1:-} d why smoke rc
  NODES=${2:-$HOME/.gw-nodes.txt}
  if ! [[ $COMMIT =~ ^[0-9a-fA-F]{40}$ ]]; then
    # full sha only: the fetch-by-sha fallback (a commit on no branch head) needs it
    echo "usage: bash -s -- <comfy_commit (full 40-hex sha)> [<nodes_file>]" >&2
    exit 2
  fi
  [ -f "$NODES" ] || die "node list $NODES not found (the controller uploads it first)"
  export GIT_TERMINAL_PROMPT=0 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  trap 'rc=$?; echo "bootstrap: failed in phase $PHASE (line $LINENO, exit $rc)" >&2' ERR

  phase locate
  CUI=
  for d in "$HOME/ComfyUI" /workspace/ComfyUI /opt/ComfyUI; do
    if [ -f "$d/main.py" ]; then CUI=$(readlink -f "$d"); break; fi
  done
  if [ -z "$CUI" ] || [ "$CUI" != "$(readlink -f "$HOME/ComfyUI")" ]; then
    # ~/ComfyUI is what the backend's paths and start-comfy.sh use; a leftover there
    # (a half clone, a dangling link) is moved aside, never deleted
    if [ -e "$HOME/ComfyUI" ] || [ -L "$HOME/ComfyUI" ]; then
      mv "$HOME/ComfyUI" "$HOME/ComfyUI.gw-moved-$(date +%s)"
    fi
    if [ -z "$CUI" ]; then
      echo "no ComfyUI on the image, cloning"
      git clone --quiet "$COMFY_URL" "$HOME/ComfyUI"
      CUI=$(readlink -f "$HOME/ComfyUI")
    else
      echo "template ComfyUI at $CUI, linking ~/ComfyUI to it"
      ln -s "$CUI" "$HOME/ComfyUI"
    fi
  fi
  echo "ComfyUI: $CUI"

  phase stop
  # Before the checkout, any ComfyUI running from this checkout: our own loop (a re-run
  # after a smoke failure that was started by hand) and a template ComfyUI that came
  # back after the host bootstrap stopped it — that one would win the 8188 bind race
  # against start-comfy.sh. A checkout and pip under a running ComfyUI break both. The
  # template's rc-file autostart guard is the host bootstrap's (ops/host-bootstrap.sh).
  stop_comfy_processes "$CUI"

  phase checkout
  if [ ! -e "$CUI/.git" ]; then git -C "$CUI" init --quiet; fi
  git_pin "$CUI" "$COMFY_URL" "$COMMIT" || die "cannot check out ComfyUI $COMMIT"
  echo "ComfyUI at $(git -C "$CUI" rev-parse HEAD)"

  phase venv
  PY=
  local -a unsuitable=()
  for d in "$CUI/venv" "$CUI/.venv"; do
    [ -e "$d" ] || continue
    if why=$(venv_ok "$d/bin/python" 2>&1); then
      PY="$d/bin/python"; echo "keeping $d ($why)"; break
    fi
    echo "not using $d: ${why##*$'\n'}"
    unsuitable+=("$d")
  done
  if [ -z "$PY" ]; then
    # Built at its final path (a venv does not survive being renamed: console-script
    # shebangs name it), with the old one parked beside it until the new one works —
    # a failed build puts the template's venv back for the diagnosis.
    rm -rf "$CUI/venv.gw-old"
    if [ -e "$CUI/venv" ]; then mv "$CUI/venv" "$CUI/venv.gw-old"; fi
    rc=0
    build_venv "$CUI/venv" || rc=$?
    if [ "$rc" -ne 0 ]; then
      rm -rf "$CUI/venv"
      if [ -e "$CUI/venv.gw-old" ]; then mv "$CUI/venv.gw-old" "$CUI/venv"; fi
      if [ "$rc" -eq 1 ]; then die "cannot build the venv (Python $PY_VER or the torch install failed)"; fi
      finish "fail torch-cuda"
    fi
    PY="$CUI/venv/bin/python"
    # only now: the template's venvs, built for another torch — the wheels would not
    # load in them, and they would be paid for in every snapshot
    rm -rf "$CUI/venv.gw-old"
    for d in "${unsuitable[@]}"; do
      if [ "$d" != "$CUI/venv" ]; then rm -rf "$d"; fi
    done
  fi
  "$PY" -m pip install --quiet setuptools wheel
  # From here on no pip call may move torch: a node requirement that wants another
  # torch fails that node instead of silently breaking every compiled extension.
  printf 'torch==%s\ntorchvision==%s\ntorchaudio==%s\n' \
    "$TORCH_VER" "$TORCHVISION_VER" "$TORCHAUDIO_VER" >"$CUI/.gw-constraints.txt"
  export PIP_CONSTRAINT="$CUI/.gw-constraints.txt" UV_CONSTRAINT="$CUI/.gw-constraints.txt"
  (cd "$CUI" && "$PY" -m pip install --quiet -r requirements.txt)

  phase nodes
  mkdir -p "$CUI/custom_nodes"
  install_nodes

  phase extensions
  install_extensions

  phase node-install-scripts
  run_install_scripts

  phase start-script
  write_start_script

  phase smoke
  smoke=$(smoke_test) || smoke="fail smoke-script"
  smoke=${smoke##*$'\n'}
  finish "$smoke"
}

if [ "${GW_BOOTSTRAP_LIB:-}" != 1 ]; then
  main "$@" </dev/null
fi
