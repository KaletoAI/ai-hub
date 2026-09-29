#!/usr/bin/env bash
# One-time setup of a MANAGED HOST (a provider's GPU VM, today Thunder Compute: Ubuntu
# container, user `ubuntu`), whatever services are attached to it. It runs on the first
# start of every host — and again on a start from a snapshot taken before it finished —
# BEFORE any service's own setup (ops/thunder-bootstrap.sh for a ComfyUI service). The
# gateway streams this file over SSH and reads its output:
#
#   ssh … -- ubuntu@<ip> bash -s -- [<nodes_file>]   (script on stdin)
#
# <nodes_file> defaults to ~/.gw-nodes.txt, which the controller uploads first when a
# ComfyUI service will be bootstrapped next; without it every pack under the template's
# custom_nodes/ is reported. Lines starting with `GW:` are the machine-readable part of
# the output (the same protocol as ops/thunder-bootstrap.sh):
#
#   GW:PHASE <name>                   entering a phase (the log is shown in the panel)
#   GW:TEMPLATE_NODE <dir>            a custom_nodes/ pack the template brought (not in
#                                     the node list; reported, never deleted)
#   GW:UNKNOWN_MODEL <rel>\t<bytes>   a model the TEMPLATE brought (not in any manifest —
#                                     delete it before the first snapshot or pay for it
#                                     in every snapshot)
#   GW:DONE                           finished
#
# Exit status: 0 done, anything else = a step failed (see the last GW:PHASE and the
# "host-bootstrap: failed in phase" line).
#
# What it does, and why it is not the ComfyUI script's business: a provider template
# may start its own services (Thunder's `comfy-ui` template starts a ComfyUI listening
# on every interface — on a vLLM-only host just as much), may ship model files that
# every snapshot would then carry, and may lack the tools the gateway's remote commands
# rely on (`flock` for the model sync's transfer lock and the start scripts, `uv` for a
# Python the image does not have). No service is installed here. Re-running on the same
# VM is safe (every step checks what is there).
#
# Everything runs inside main(), called on the LAST line: bash reads a script from a
# pipe as it executes it, so any child that reads stdin (curl | sh, apt-get) would
# otherwise swallow the rest of this file. main also gets </dev/null for the same reason.
set -euo pipefail
set -E

PHASE=init

phase() { PHASE=$1; echo "GW:PHASE $1"; }
die() { echo "host-bootstrap: $*" >&2; exit 1; }

# parse_node_line <line> → sets NODE_KIND (git|registry), NODE_NAME, NODE_SRC, NODE_REV.
# Returns 0 for a node, 1 for a blank/comment line, 2 for a line that is neither form.
# (Byte-identical to the copy in ops/thunder-bootstrap.sh — a script streamed over ssh
# cannot source a shared file; tests/test_thunder_scripts.py pins the two equal.)
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

# find_template_comfy: the template's ComfyUI checkout (resolved), or nothing. The
# places Thunder's templates (and the ComfyUI bootstrap's ~/ComfyUI link) use.
find_template_comfy() {
  local d
  for d in "$HOME/ComfyUI" /workspace/ComfyUI /opt/ComfyUI; do
    if [ -f "$d/main.py" ]; then readlink -f "$d"; return 0; fi
  done
}

# disable_rc_autostart <rcfile>: comment out the template's `start-comfyui` hook.
# Only when an ACTIVE line is left, so a re-run neither rewrites the file nor
# overwrites the first backup (the one holding the original) with an edited copy.
disable_rc_autostart() {
  local rc=$1
  if [ -f "$rc" ] && grep -q '^[^#]*start-comfyui' "$rc"; then
    sed -i.gw-bak '/start-comfyui/{/^[[:space:]]*#/!s/^/# gw-disabled: /}' "$rc"
    echo "disabled the template autostart in $rc (backup ${rc}.gw-bak)"
  fi
}

# stop_template_autostart [<comfy dir>]: the template may start ComfyUI itself,
# listening on every interface. Stop it (and a start loop left on a re-run) and keep it
# from coming back on the next login.
stop_template_autostart() {
  local cui=${1:-} p
  pkill -f 'start-comfy' || true
  if [ -n "$cui" ]; then
    for p in $(pgrep -f 'main\.py' || true); do
      if [ "$(readlink -f "/proc/$p/cwd" 2>/dev/null || true)" = "$cui" ]; then
        kill "$p" 2>/dev/null || true
      fi
    done
    pkill -f "$cui/main.py" || true
  fi
  disable_rc_autostart "$HOME/.bashrc"
  disable_rc_autostart "$HOME/.profile"
}

# inventory_models <comfy dir>: every file over 1 MB under models/ came with the
# template — `GW:UNKNOWN_MODEL models/<rel>\t<bytes>`, a count on stderr.
inventory_models() {
  local cui=$1 out
  [ -d "$cui/models" ] || { echo "0 template model file(s) found" >&2; return 0; }
  if ! out=$(find "$cui/models/" -type f -size +1M -printf 'GW:UNKNOWN_MODEL models/%P\t%s\n'); then
    echo "inventory of $cui/models incomplete (find failed)" >&2
  fi
  if [ -n "$out" ]; then printf '%s\n' "$out"; fi
  echo "$(printf '%s' "$out" | grep -c '^GW:' || true) template model file(s) found" >&2
}

# report_template_nodes <custom_nodes dir> [<nodes file>]: every pack dir the node list
# does not name came with the template — `GW:TEMPLATE_NODE <dir>`. Reported, never
# deleted; the likely case is the template's own ComfyUI-Manager under another
# spelling, i.e. two Managers once ours is installed. No list (a host without a ComfyUI
# service to bootstrap) → every pack is the template's.
report_template_nodes() {
  local cn=$1 list=${2:-} line rc d
  local -A listed=()
  if [ -n "$list" ] && [ -f "$list" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      rc=0
      parse_node_line "$line" || rc=$?
      if [ "$rc" -eq 0 ]; then listed[$NODE_NAME]=1; fi
    done <"$list"
  fi
  [ -d "$cn" ] || return 0
  for d in "$cn"/*/; do
    [ -d "$d" ] || continue
    d=${d%/}; d=${d##*/}
    [ "$d" = __pycache__ ] && continue
    if [ -z "${listed[$d]:-}" ]; then echo "GW:TEMPLATE_NODE $d"; fi
  done
}

# ensure_tools: `flock` (the model sync's transfer lock, every start script's
# one-instance guard) and `uv` at ~/.local/bin/uv (a standalone Python for a venv the
# image cannot build — inside $HOME, so it is part of every snapshot).
ensure_tools() {
  if ! command -v flock >/dev/null; then
    echo "flock missing — installing util-linux"
    if command -v sudo >/dev/null && sudo -n true 2>/dev/null; then
      sudo -n env DEBIAN_FRONTEND=noninteractive apt-get update -qq || true
      sudo -n env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq util-linux || true
    fi
    command -v flock >/dev/null || die "flock is missing and could not be installed (util-linux)"
  fi
  if [ ! -x "$HOME/.local/bin/uv" ]; then
    echo "installing uv into ~/.local/bin"
    curl -LsSf https://astral.sh/uv/install.sh \
      | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh \
      || die "cannot install uv"
  fi
  echo "flock: $(command -v flock); uv: $("$HOME/.local/bin/uv" --version 2>/dev/null || echo '?')"
}

main() {
  local NODES=${1:-$HOME/.gw-nodes.txt} cui rc
  trap 'rc=$?; echo "host-bootstrap: failed in phase $PHASE (line $LINENO, exit $rc)" >&2' ERR

  phase locate
  cui=$(find_template_comfy)
  if [ -n "$cui" ]; then echo "template ComfyUI at $cui"; else echo "no ComfyUI on the image"; fi

  phase autostart
  stop_template_autostart "$cui"

  phase inventory
  # Before any service's setup: whatever sits in models/ now came with the template.
  if [ -n "$cui" ]; then
    inventory_models "$cui"
    report_template_nodes "$cui/custom_nodes" "$NODES"
  fi

  phase tools
  ensure_tools

  echo "GW:DONE"
}

if [ "${GW_BOOTSTRAP_LIB:-}" != 1 ]; then
  main "$@" </dev/null
fi
