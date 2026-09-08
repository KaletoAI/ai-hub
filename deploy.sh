#!/usr/bin/env bash
set -euo pipefail

HOST="${DEPLOY_HOST:?Set DEPLOY_HOST=user@host (e.g. root@192.168.1.10)}"
DEST="/opt/ai-hub"
SERVICE="ai-hub"

cd "$(dirname "$(readlink -f "$0")")"

# Same exclude set for both transports. rsync wants trailing-slash dir
# patterns; tar wants the in-archive path (./foo) — hence two lists.
RSYNC_EXCLUDES=(
    # No trailing slash on .git: in a WORKTREE it is a file, not a directory, and
    # 'gitdir: …' pointing at a dev box has no business on prod.
    --exclude='.git'         --exclude='.claude/'
    # Lives ON the prod box and in no checkout — without these, --delete wipes it on
    # every deploy: agent/tool state, and the raw workflow originals .gitignore
    # deliberately keeps out of git ("keep raw originals out of git").
    --exclude='.remember/'   --exclude='.superpowers/'  --exclude='.agents/'
    --exclude='skills-lock.json*'
    --exclude='sample_comfyui_workflows/org/'
    --exclude='__pycache__/' --exclude='*.pyc'
    --exclude='.venv/'       --exclude='venv/'        --exclude='.env'
    --exclude='config.yaml'  --exclude='config.yaml.bak*'  --exclude='config.prod.yaml'
    --exclude='secret.key'   --exclude='store.db'      --exclude='store.db-wal' --exclude='store.db-shm' --exclude='store.db.bak*'
    --exclude='stats.db'     --exclude='stats.db-wal'  --exclude='stats.db-shm'
    --exclude='jobs.db'      --exclude='jobs.db-wal'   --exclude='jobs.db-shm'
    --exclude='jobs/'        --exclude='calls/'    --exclude='voiceref/'
    --exclude='deploy.sh'
)
TAR_EXCLUDES=(
    --exclude='./.git'        --exclude='./.claude'
    --exclude='./.remember'   --exclude='./.superpowers' --exclude='./.agents'
    --exclude='./skills-lock.json*'
    --exclude='./sample_comfyui_workflows/org'
    --exclude='__pycache__'   --exclude='*.pyc'
    --exclude='./.venv'       --exclude='./venv'        --exclude='./.env'
    --exclude='./config.yaml' --exclude='./config.yaml.bak*' --exclude='./config.prod.yaml'
    --exclude='./secret.key'  --exclude='./store.db'    --exclude='./store.db-wal' --exclude='./store.db-shm' --exclude='./store.db.bak*'
    --exclude='./stats.db'    --exclude='./stats.db-wal' --exclude='./stats.db-shm'
    --exclude='./jobs.db'     --exclude='./jobs.db-wal'  --exclude='./jobs.db-shm'
    --exclude='./jobs'        --exclude='./calls'   --exclude='./voiceref'
    --exclude='./deploy.sh'
)

if command -v rsync >/dev/null 2>&1; then
    echo "==> Syncing files to ${HOST}:${DEST} (rsync)"
    rsync -az --delete --human-readable --info=stats1,progress2 \
        "${RSYNC_EXCLUDES[@]}" ./ "${HOST}:${DEST}/"
else
    echo "==> rsync not found — falling back to tar-over-ssh (no --delete)"
    ssh "${HOST}" "mkdir -p '${DEST}'"
    tar czf - "${TAR_EXCLUDES[@]}" -C . . \
        | ssh "${HOST}" "tar xzf - -C '${DEST}'"
    echo "    synced (stale remote files are NOT removed without rsync)"
fi

echo "==> Ensuring venv + installing requirements + syncing systemd unit"
ssh "${HOST}" bash -se <<EOF
set -euo pipefail
cd "${DEST}"
if [ ! -d venv ]; then
    python3 -m venv venv
fi
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt

UNIT_SRC="${DEST}/${SERVICE}.service"
UNIT_DST="/etc/systemd/system/${SERVICE}.service"
if [ ! -f "\${UNIT_DST}" ] || ! cmp -s "\${UNIT_SRC}" "\${UNIT_DST}"; then
    echo "    unit file changed -> installing + daemon-reload"
    install -m 0644 "\${UNIT_SRC}" "\${UNIT_DST}"
    systemctl daemon-reload
    systemctl enable ${SERVICE} >/dev/null
fi
EOF

echo "==> Restarting ${SERVICE}"
ssh "${HOST}" "systemctl restart ${SERVICE}"

echo "==> Service status"
ssh "${HOST}" "systemctl is-active ${SERVICE} && systemctl status ${SERVICE} --no-pager --lines=5"

echo "==> Done."
