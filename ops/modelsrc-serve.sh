#!/usr/bin/env bash
# Read-only model source for AI-Hub: the SSH forced command on the model-share host.
# The gateway's LAN transfer key may do exactly what this script lets through and
# nothing else — it is the whole security boundary of that key.
#
# Install (operator, on the share host):
#   install -m 0755 ops/modelsrc-serve.sh /usr/local/bin/modelsrc-serve
#   user `modelsrc`, read access to the share (group/ACL); its login shell must be a
#   real shell (/bin/bash) — sshd runs a forced command THROUGH it, nologin runs nothing.
#   ~modelsrc/.ssh/authorized_keys:
#     command="/usr/local/bin/modelsrc-serve",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA… ai-hub
#   Share root: env MODELSRC_ROOT (default /mnt/xfs/shared/comfyui-models); the HF
#   cache is expected as a real directory `hf-cache/` inside it.
#
# The request comes from SSH_ORIGINAL_COMMAND (never from "$@"), in the quoting the
# gateway's shlex.quote produces: bare words of [A-Za-z0-9@%+=:,./_-], '…' single-
# quoted runs, and "…" runs without $ ` \ — anything else (;, |, $(…), unquoted
# spaces inside a word, control characters) is refused. Nothing is ever evaluated.
#
# Verbs:
#   list              every servable entry of the share, one per line, TAB-separated:
#                       F<TAB><rel><TAB><size in bytes>
#                       L<TAB><rel><TAB><link target exactly as stored (relative)>
#                     F = regular file. L = a symlink (the HF hub cache's
#                     snapshots/<rev>/<name> -> ../../blobs/<sha>) whose target text
#                     is `../`* followed by plain segments, stays inside the share,
#                     and names — without passing any further symlink — a regular
#                     file that is itself an F line of the same listing. So joining
#                     dirname(rel) with the target and normalising it always yields
#                     an F rel: recreating the link on the other side never dangles.
#                     Every other symlink (absolute, escaping, dangling, to a
#                     directory, to another link, to a denied path) is omitted.
#                     <rel> is relative to the share root, never contains TAB, LF,
#                     other control characters or `\` (such names are omitted), and
#                     passes the path rules below. Files come first, then links;
#                     no particular order within each. One `find` pass, no per-file
#                     subprocess (the share is ~1 TB / thousands of files).
#   cat <rel> <off>   the bytes of regular file <rel> from decimal byte offset <off>
#                     (0 = whole file; past the end = nothing) — resumable transfer.
#   sha256 <rel>      the lowercase hex SHA-256 of regular file <rel>, then LF.
#
# Path rules (list filters by them, cat/sha256 refuse on them): not empty, relative,
# no `\`, no control character, no empty segment, no segment starting with `.` (so
# no `.`/`..` and no dotfiles or dot-dirs), not `*.log`, not `hf-cache/token`, and
# under `hf-cache/` only `hf-cache/hub/…` (token files, xet/modules/download caches
# are not model data). cat/sha256 additionally require `realpath -e` of the path to be
# the path itself — i.e. no symlink anywhere on it, a link inside the tree included:
# the gateway recreates links from `list` and never fetches through one. The file is
# then opened once and checked again through the open descriptor, and only that
# descriptor is read (a swap after the check cannot redirect the read). An L line's
# link and target are both under `hf-cache/` or both outside it. `list` fails (1) when
# `hf-cache` itself is a symlink — it would otherwise silently lack the HF cache.
#
# Exit status: 0 ok; 2 refused (one short line on stderr, nothing on stdout);
# 1 failure — e.g. the share root is missing, or `list` could not read part of the
# tree ("list incomplete"): discard that listing, it is not the whole share.
set -euo pipefail
export LC_ALL=C

ROOT_DEFAULT=/mnt/xfs/shared/comfyui-models
MAX_CMD=4096

refuse() {
    printf 'modelsrc-serve: refused: %s\n' "$1" >&2
    exit 2
}

fail() {
    printf 'modelsrc-serve: %s\n' "$1" >&2
    exit 1
}

# Split $1 into WORDS the way a POSIX shell would for shlex.quote output — without a
# shell. Returns 1 on anything outside that language.
split_cmd() {
    local s=$1
    local n=${#s} i=0 c rest q w="" inword=0
    WORDS=()
    while (( i < n )); do
        c=${s:i:1}
        case $c in
            ' ')
                if (( inword )); then WORDS+=("$w"); w=""; inword=0; fi
                i=$((i + 1)) ;;
            "'")
                rest=${s:i+1}
                [[ $rest == *"'"* ]] || return 1
                q=${rest%%"'"*}
                w+=$q; inword=1
                i=$((i + ${#q} + 2)) ;;
            '"')
                rest=${s:i+1}
                [[ $rest == *'"'* ]] || return 1
                q=${rest%%'"'*}
                # inside "…" a shell would still expand these
                [[ $q != *[\$\`\\]* ]] || return 1
                w+=$q; inword=1
                i=$((i + ${#q} + 2)) ;;
            [A-Za-z0-9@%+=:,./_-])
                w+=$c; inword=1
                i=$((i + 1)) ;;
            *)
                return 1 ;;
        esac
    done
    if (( inword )); then WORDS+=("$w"); fi
    return 0
}

# The path rules (see header). Pure bash: list calls it per link.
path_ok() {
    local p=$1 seg
    local -a segs
    [[ -n $p ]] || return 1
    [[ $p != /* ]] || return 1
    [[ $p != *\\* ]] || return 1
    [[ $p != *[[:cntrl:]]* ]] || return 1
    [[ $p != */ && $p != *//* ]] || return 1
    IFS=/ read -r -a segs <<<"$p"
    for seg in "${segs[@]}"; do
        [[ -n $seg && $seg != .* ]] || return 1
    done
    [[ $p != *.log ]] || return 1
    [[ $p != hf-cache/token ]] || return 1
    if [[ $p == hf-cache/* && $p != hf-cache/hub/* ]]; then return 1; fi
    return 0
}

# A regular file at "$ROOT_REAL/$1" reached through no symlink at all.
plain_file() {
    local rel=$1 real
    real=$(realpath -e -- "$ROOT_REAL/$rel" 2>/dev/null) || return 1
    [[ $real == "$ROOT_REAL/$rel" && -f $real && ! -L $real ]]
}

# Decide one symlink found by find -P; prints its L line when it qualifies.
# Pure bash (builtin tests only) so thousands of HF snapshot links cost no process.
emit_link() {
    local rel=$1 tgt=$2 dir="" t seg cur
    local -a parts segs
    path_ok "$rel" || return 0
    [[ -n $tgt && $tgt != /* && $tgt != *\\* && $tgt != *[[:cntrl:]]* ]] || return 0
    [[ $tgt != */ && $tgt != *//* ]] || return 0
    if [[ $rel == */* ]]; then dir=${rel%/*}; fi
    parts=()
    if [[ -n $dir ]]; then IFS=/ read -r -a parts <<<"$dir"; fi
    # target text: `../` prefixes, then plain segments — a `..` after a plain segment
    # could climb out of a symlinked directory, which a lexical check would miss
    t=$tgt
    while [[ $t == ../* ]]; do
        (( ${#parts[@]} > 0 )) || return 0          # climbs out of the share
        unset 'parts[${#parts[@]}-1]'
        t=${t#../}
    done
    IFS=/ read -r -a segs <<<"$t"
    for seg in "${segs[@]}"; do
        [[ -n $seg && $seg != .* ]] || return 0
        parts+=("$seg")
    done
    t=""
    for seg in "${parts[@]}"; do t+=${t:+/}$seg; done
    path_ok "$t" || return 0
    # link and target on the same side of the models/ ↔ hf-cache/ split: the gateway
    # maps the two halves to two different roots on the VM, where a crossing link
    # would point at a place that file never lands
    if [[ $rel == hf-cache/* ]]; then
        [[ $t == hf-cache/* ]] || return 0
    else
        [[ $t != hf-cache/* ]] || return 0
    fi
    # no symlink on the way down to the target, and the target a regular file: the
    # kernel then resolves exactly what the text says, and it is an F line
    cur=$ROOT_REAL
    for seg in "${parts[@]}"; do
        cur+="/$seg"
        [[ ! -L $cur ]] || return 0
    done
    [[ -f $cur ]] || return 0
    printf 'L\t%s\t%s\n' "$rel" "$tgt"
}

do_list() {
    local rel tgt rc=0
    # global, not local: the EXIT trap runs after this function has returned
    LIST_TMP=$(mktemp) || fail "cannot create a temp file"
    trap 'rm -f -- "$LIST_TMP"' EXIT
    cd -- "$ROOT_REAL" || fail "cannot enter the share root"
    # find -P never enters a symlinked hf-cache/: the listing would silently lack the
    # whole HF cache and read as complete
    [[ ! -L hf-cache ]] || fail "hf-cache is a symlink; listing would omit the HF cache"
    # ONE pass: files go straight to stdout; links go NUL-separated to $LIST_TMP for the
    # bash check above. -P: never follow a link. Pruned: dot entries, names find
    # could not print on one line (control chars) or that the path rules refuse
    # (`\`), and everything under hf-cache/ except hf-cache/hub.
    command find -P . -mindepth 1 \
        \( -name '.*' -o -name '*[[:cntrl:]]*' -o -name '*\\*' \
           -o \( -path './hf-cache/*' ! -path './hf-cache/hub' ! -path './hf-cache/hub/*' \) \
        \) -prune \
        -o -name '*.log' \
        -o -type f -printf 'F\t%P\t%s\n' \
        -o -type l -fprintf "$LIST_TMP" '%P\0%l\0' \
        || rc=$?
    while IFS= read -r -d '' rel && IFS= read -r -d '' tgt; do
        emit_link "$rel" "$tgt"
    done <"$LIST_TMP"
    (( rc == 0 )) || fail "list incomplete (find exit $rc)"
}

# ---------------------------------------------------------------------------------

cmd=${SSH_ORIGINAL_COMMAND-}
(( ${#cmd} <= MAX_CMD )) || refuse "command too long"
[[ $cmd != *[[:cntrl:]]* ]] || refuse "control character"
split_cmd "$cmd" || refuse "not a plain quoted command"
(( ${#WORDS[@]} >= 1 )) || refuse "empty command"

verb=${WORDS[0]}
case $verb in
    list)   (( ${#WORDS[@]} == 1 )) || refuse "list takes no arguments" ;;
    cat)    (( ${#WORDS[@]} == 3 )) || refuse "usage: cat <rel> <offset>" ;;
    sha256) (( ${#WORDS[@]} == 2 )) || refuse "usage: sha256 <rel>" ;;
    *)      refuse "unknown verb" ;;
esac

if [[ $verb != list ]]; then
    rel=${WORDS[1]}
    path_ok "$rel" || refuse "path not allowed"
fi
if [[ $verb == cat ]]; then
    off=${WORDS[2]}
    # 18 digits: $((off + 1)) cannot overflow bash's 64-bit arithmetic
    [[ $off =~ ^[0-9]{1,18}$ ]] || refuse "offset must be a decimal byte count"
fi

ROOT=${MODELSRC_ROOT:-$ROOT_DEFAULT}
ROOT_REAL=$(realpath -e -- "$ROOT" 2>/dev/null) || fail "share root missing: $ROOT"
[[ -d $ROOT_REAL ]] || fail "share root is not a directory: $ROOT"

# The checked file, opened ONCE on fd 3 and re-verified through the open descriptor:
# between `plain_file` and a later open by name, the path could be swapped for a
# symlink out of the share (TOCTOU) — reading from fd 3 reads exactly what was checked.
open_checked() {
    local real
    plain_file "$rel" || refuse "not a regular file inside the share"
    exec 3<"$ROOT_REAL/$rel" || fail "cannot open $rel"
    real=$(realpath -e /proc/self/fd/3 2>/dev/null) || refuse "file changed while opening"
    [[ $real == "$ROOT_REAL/$rel" && -f /dev/fd/3 ]] \
        || refuse "file changed while opening"
}

case $verb in
    list)
        do_list ;;
    cat)
        open_checked
        exec tail -c "+$((10#$off + 1))" <&3 ;;
    sha256)
        open_checked
        sum=$(sha256sum <&3) || fail "cannot read $rel"
        printf '%s\n' "${sum%% *}" ;;
esac
