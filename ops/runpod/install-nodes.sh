#!/usr/bin/env bash
# Installs the RunPod worker's custom-node packs at build time:
#   install-nodes.sh <nodes file> <comfy dir> <python>
# Lines: <https-git-url>@<commit> | registry:<id>@<version> (ops/thunder-nodes.default.txt
# forms). Unlike the Thunder bootstrap there is no GW: protocol — a failed pack FAILS THE
# BUILD, because an image that silently lacks a pack is a worker that fails every job.
# Our own code; nothing copied from runpod-workers/worker-comfyui (AGPL-3.0).
set -euo pipefail
NODES=$1 CUI=$2 PY=$3
REGISTRY_API=https://api.comfy.org

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

die() { echo "install-nodes: $*" >&2; exit 1; }

while IFS= read -r line || [ -n "$line" ]; do
  rc=0
  parse_node_line "$line" || rc=$?
  [ "$rc" -eq 1 ] && continue
  [ "$rc" -eq 0 ] || die "malformed line: $line"
  dir="$CUI/custom_nodes/$NODE_NAME"
  echo "node $NODE_NAME @ $NODE_REV"
  if [ "$NODE_KIND" = registry ]; then
    url=$(curl -fsSL --retry 3 "$REGISTRY_API/nodes/$NODE_NAME/versions/$NODE_REV" \
          | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["downloadUrl"])') \
      || die "$NODE_NAME: registry lookup failed"
    tmp=$(mktemp -d)
    curl -fsSL --retry 3 -o "$tmp/node.zip" "$url" || die "$NODE_NAME: download failed"
    "$PY" - "$tmp/node.zip" "$dir" <<'PY' || die "$NODE_NAME: unzip failed"
import os, sys, zipfile
dst = sys.argv[2]
zipfile.ZipFile(sys.argv[1]).extractall(dst)      # drops absolute paths and `..`
entries = os.listdir(dst)
if len(entries) == 1 and os.path.isdir(os.path.join(dst, entries[0])):
    inner = os.path.join(dst, entries[0])           # tolerate one wrapping folder
    os.rename(inner, dst + ".inner")
    os.rmdir(dst)
    os.rename(dst + ".inner", dst)
PY
    rm -rf "$tmp"
  else
    git clone --quiet "$NODE_SRC" "$dir" || die "$NODE_NAME: clone failed"
    git -C "$dir" fetch --quiet origin "$NODE_REV" 2>/dev/null || true
    git -C "$dir" checkout --quiet --force "$NODE_REV" || die "$NODE_NAME: no commit $NODE_REV"
    git -C "$dir" submodule update --init --recursive --quiet || die "$NODE_NAME: submodules"
    rm -rf "$dir/.git"
  fi
  if [ -f "$dir/requirements.txt" ]; then
    (cd "$dir" && "$PY" -m pip install --no-cache-dir -r requirements.txt) \
      || die "$NODE_NAME: requirements"
  fi
  if [ -f "$dir/install.py" ]; then
    (cd "$dir" && "$PY" install.py) || die "$NODE_NAME: install.py"
  fi
done <"$NODES"
