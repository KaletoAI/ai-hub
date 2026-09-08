# AI-Hub Rename, Repo Structure, Doc Review, Fresh-Install Test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the project `llm-gateway` → **AI-Hub** everywhere that is alive (code, unit, deploy, remotes, prod path, docs), move the tests into `tests/`, archive finished plans, verify the living docs against the code, and prove a README-only installation on the empty container 192.168.8.148 with chat completions over LocalAI and OpenRouter.

**Architecture:** Three phases that must stay separable. Phase 1 is one branch (`ai-hub-rename`) that is fully testable locally. Phase 2 is one maintenance step on the prod box .10 (move dir, swap unit, rename bare repo) plus the GitHub rename. Phase 3 is a clean-room install on .148 following the README literally; every README gap found there is a README fix committed on the spot.

**Tech Stack:** Python 3 / FastAPI / uvicorn, stdlib `unittest`, bash `deploy.sh` (rsync over SSH), systemd, `gh` CLI, git.

**Spec:** `docs/superpowers/specs/2026-09-08-ai-hub-rename-design.md`

## Global Constraints

- Display name **AI-Hub**; slug **`ai-hub`** for paths, unit, repos, remotes; `owned_by` strings become `"ai-hub (virtual)"` / `"ai-hub (image)"`.
- The 15 Python modules stay flat in the repo root; `main:app` is unchanged. No package.
- Historical documents under `docs/superpowers/` and `docs/archive/` keep the old name verbatim — never rewrite history.
- No functional change to the gateway in any task. If a task needs one, stop and report.
- Never commit `config.yaml`, `store.db`, `secret.key`, `stats.db*`, `jobs.db*`, `jobs/`, `voiceref/`, `*.key`. The OpenRouter key lives only in .148's `config.yaml` and is removed after Phase 3.
- Remotes are **two**: `github` (KaletoAI) and `lxc` (root@192.168.8.10). "Merge" means push to both. The former `origin` (Forgejo .110) was removed on 2026-09-08 and is not to be re-added.
- Prod work on .10 only when idle (`curl -s localhost:4000/health` shows no in-flight work); always a backup first; always compile-gate before deploy.
- Every commit message ends with:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Q4599AyXxrDNJhPLBvCuuy
  ```
- Test runner from Task 1 on: `venv/bin/python -m unittest discover -s tests -t .` — must stay green after every task.
- Doc review subagents (Task 6) run on **Opus** (`model: "opus"`), one per document.

---

## Phase 1 — Branch `ai-hub-rename`

### Task 1: Branch, `.gitignore`, tests into `tests/`

**Files:**
- Modify: `.gitignore`
- Move: `test_*.py` (20 files) → `tests/test_*.py`; Create: `tests/__init__.py`
- Modify: `tests/test_chain_mesh_param.py`, `tests/test_gen_backend_for.py`, `tests/test_cloud_editor.py`, `tests/test_err_text.py`, `tests/test_run_job_failover.py`, `tests/test_ws_progress.py` (the `_here` preambles), `tests/test_scheduler.py:149` (sample path)
- Modify: `CLAUDE.md:32-33`, `CLAUDE.md:508-509`, `CLAUDE.md:546` (runner commands + count wording)

**Interfaces:**
- Produces: the runner command `venv/bin/python -m unittest discover -s tests -t .` used by every later task.

- [ ] **Step 1: Create the branch**

```bash
cd /home/dev/projekte/llm-gateway
git checkout -b ai-hub-rename
```

- [ ] **Step 2: Extend `.gitignore`** — append:

```
# Agent/tool state and session artefacts
.claude/
.agents/
skills-lock.json
skills-lock.json.bak
# Never ship documents that landed in the samples folder by accident
sample_comfyui_workflows/*.pdf
```

- [ ] **Step 3: Move the tests**

```bash
mkdir tests && touch tests/__init__.py
git mv test_*.py tests/
```

- [ ] **Step 4: Run the suite from the new location — expect import failures**

Run: `venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -3`
Expected: errors — the six preamble files insert `tests/` into `sys.path` and cannot `import main`; `test_scheduler` cannot find `sample_comfyui_workflows`.

- [ ] **Step 5: Fix the six preambles.** In each of `tests/test_chain_mesh_param.py`, `tests/test_gen_backend_for.py`, `tests/test_cloud_editor.py`, `tests/test_err_text.py`, `tests/test_run_job_failover.py`, `tests/test_ws_progress.py` replace the line

```python
_here = os.path.dirname(os.path.abspath(__file__))
```
with
```python
_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root (tests/ is one level down)
```
`sys.path.insert(0, _here)` stays; `tests/test_run_job_failover.py:465` (`os.path.join(_here, "sample_comfyui_workflows")`) now resolves correctly by itself.

- [ ] **Step 6: Fix `tests/test_scheduler.py:149`**

```python
        d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_comfyui_workflows")
```

- [ ] **Step 7: Run the suite — expect green**

Run: `venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -2`
Expected: `Ran 386 tests` … `OK`. If a file fails on `import main` from its own module directory, check it uses the preamble pattern above; `test_admin_live.py` and `test_playground_files.py` import modules directly and rely on `-t .` putting the repo root on `sys.path`.

- [ ] **Step 8: Update CLAUDE.md commands.** Replace at `CLAUDE.md:32-33`:

```
  `anthropic_bridge.py`): `venv/bin/python -m unittest discover -p 'test_*.py'`.
```
with
```
  `anthropic_bridge.py`): `venv/bin/python -m unittest discover -s tests -t .`.
```
At `CLAUDE.md:508-509` replace `` `ls test_*.py` is the count of record `` with `` `ls tests/test_*.py` is the count of record ``. At `CLAUDE.md:546` replace `` `python -m unittest discover -p 'test_*.py'` `` with `` `python -m unittest discover -s tests -t .` ``. Also fix the two test docstrings that carry the absolute dev path (`tests/test_cloudtask.py:1`, `tests/test_meshy.py:1`, `tests/test_meshy_adapter.py:2`, `tests/test_tripo.py:1`, `tests/test_tripo_adapter.py:2`, `tests/test_chain_hooks.py:2`, `tests/test_err_text.py:10`): `/home/dev/projekte/llm-gateway/venv/bin/python -m unittest test_x` → `venv/bin/python -m unittest tests.test_x`.

- [ ] **Step 9: Commit**

```bash
git add -A .gitignore tests CLAUDE.md
git commit -m "repo: tests live in tests/, ignore agent state

Runner: venv/bin/python -m unittest discover -s tests -t . (top-level . keeps
import main working). The six chdir/sys.path preambles point one level up."
```

---

### Task 2: A test that pins the project name

**Files:**
- Create: `tests/test_project_name.py`

**Interfaces:**
- Produces: `OLD_NAME_PATTERN` and the exclusion list every later rename task must satisfy.

- [ ] **Step 1: Write the failing test**

```python
"""The project is AI-Hub. The old name must not survive in anything ALIVE — a stale
`llm-gateway` in deploy.sh points a deploy at a path that no longer exists, in the
unit at a WorkingDirectory that is gone, in the README at a clone URL that redirects.
None of that raises; it fails at the worst moment. History (docs/superpowers,
docs/archive) keeps the name of its time on purpose.

Run: venv/bin/python -m unittest tests.test_project_name -v
"""
import os
import re
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = re.compile(r"llm[-_ ]gateway", re.I)
# Historical records keep their name; the archive index and the migration note in
# the README/CLAUDE.md may mention the old name ONCE as "formerly".
EXCLUDE_DIRS = ("docs/superpowers/", "docs/archive/")
ALLOWED_MENTIONS = {
    "README.md": 1,           # "formerly llm-gateway" once, near the title
    "CLAUDE.md": 1,           # the migration note
    "tests/test_project_name.py": 99,
}


class ProjectName(unittest.TestCase):
    def test_the_old_name_is_gone_from_everything_alive(self):
        files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
        offenders = {}
        for f in files:
            if f.startswith(EXCLUDE_DIRS) or not f.endswith((".py", ".md", ".sh", ".service", ".yaml", ".txt", ".json")):
                continue
            try:
                with open(os.path.join(ROOT, f), encoding="utf-8") as fh:
                    n = len(OLD.findall(fh.read()))
            except UnicodeDecodeError:
                continue
            if n > ALLOWED_MENTIONS.get(f, 0):
                offenders[f] = n
        self.assertEqual(offenders, {}, f"old project name still alive in: {offenders}")

    def test_the_unit_and_deploy_agree_on_the_slug(self):
        with open(os.path.join(ROOT, "deploy.sh")) as fh:
            deploy = fh.read()
        self.assertIn('DEST="/opt/ai-hub"', deploy)
        self.assertIn('SERVICE="ai-hub"', deploy)
        self.assertTrue(os.path.exists(os.path.join(ROOT, "ai-hub.service")))
        with open(os.path.join(ROOT, "ai-hub.service")) as fh:
            unit = fh.read()
        self.assertIn("WorkingDirectory=/opt/ai-hub", unit)
        self.assertIn("/opt/ai-hub/venv/bin/uvicorn main:app", unit)
```

- [ ] **Step 2: Run it — expect FAIL** listing every file that still carries the old name (README.md, CLAUDE.md, main.py, admin.py, stats.py, deploy.sh, llm-gateway.service, config.example.yaml, sample README, several docs).

Run: `venv/bin/python -m unittest tests.test_project_name -v`

- [ ] **Step 3: Commit the test alone** (red is fine — it is the checklist for Tasks 3–5):

```bash
git add tests/test_project_name.py
git commit -m "test: the old project name must not survive in anything alive"
```

---

### Task 3: Rename in code, unit and deploy

**Files:**
- Modify: `main.py:702` (`"Starting LLM Gateway"` → `"Starting AI-Hub"`), `main.py:778` (`FastAPI(title="AI-Hub", …)`), `main.py:1871, 1881, 1891` (`owned_by` → `"ai-hub (virtual)"` / `"ai-hub (image)"`)
- Modify: `admin.py:180` (`<span class="brand">AI-Hub</span>`), `admin.py:561` (`<title>{_esc(title)} · AI-Hub</title>`)
- Modify: `stats.py:391` (`title="AI-Hub Stats"`), `stats.py:550` (`<h1>ai-hub</h1>`), `stats.py:619` (`"ai-hub stats"`), `stats.py:726` (`"ai-hub routing"`)
- Modify: `deploy.sh:5-6` (`DEST="/opt/ai-hub"`, `SERVICE="ai-hub"`) and every other `llm-gateway` inside it (the unit filename in the systemd sync block — read the whole script, `grep -n llm-gateway deploy.sh`)
- Move: `llm-gateway.service` → `ai-hub.service` (`Description=AI-Hub`, `WorkingDirectory=/opt/ai-hub`, `ExecStart=/opt/ai-hub/venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000`)
- Modify: `config.example.yaml:1` (`# AI-Hub — example config.`), `:4` (`see ai-hub.service`)

- [ ] **Step 1: Apply the code renames** (exact strings above; use `grep -n -i "llm-gateway\|llm_gateway\|LLM Gateway" main.py admin.py stats.py deploy.sh config.example.yaml` before and after — the after-grep must be empty).

- [ ] **Step 2: Rename the unit**

```bash
git mv llm-gateway.service ai-hub.service
sed -i 's/Description=LLM Gateway/Description=AI-Hub/; s#/opt/llm-gateway#/opt/ai-hub#g' ai-hub.service
```

- [ ] **Step 3: Compile-gate and run the suite**

Run: `venv/bin/python -m py_compile *.py && venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -2`
Expected: `test_project_name` still fails (docs pending), everything else green; `test_the_unit_and_deploy_agree_on_the_slug` passes.

- [ ] **Step 4: Render check** — the UI brand and title:

```bash
mkdir -p /tmp/adm-check && cd /tmp/adm-check && printf 'api_key: ""\nbackends: []\n' > config.yaml
PYTHONPATH=/home/dev/projekte/llm-gateway /home/dev/projekte/llm-gateway/venv/bin/python -c "
import admin; h = admin._page('Dashboard', '<p>x</p>', 'dashboard')
assert 'AI-Hub' in h and 'LLM Gateway' not in h, h[:400]; print('ui ok')"
cd /home/dev/projekte/llm-gateway
```

- [ ] **Step 5: Commit**

```bash
git add -A main.py admin.py stats.py deploy.sh ai-hub.service config.example.yaml
git commit -m "rename: the project is AI-Hub — code, unit, deploy

Slug ai-hub for the prod path, the systemd unit and deploy.sh; display name in the
console, the FastAPI title and the stats pages; owned_by in /v1/models."
```

---

### Task 4: Rename in the living documentation

**Files:**
- Modify: `README.md` (title line 1 → `# AI-Hub`, one line under it: `*(formerly `llm-gateway` — renamed 2026-09-08; the old GitHub URL redirects.)*`; Quick start clone URL → `https://github.com/KaletoAI/ai-hub.git` and `cd ai-hub`; Running & deploying section → `ai-hub.service`, `/opt/ai-hub`, `systemctl enable --now ai-hub`, `journalctl -u ai-hub -f`; every other occurrence)
- Modify: `CLAUDE.md` (line 1 heading area: add one migration note `Renamed from llm-gateway on 2026-09-08 (prod: /opt/ai-hub, unit ai-hub.service).`; every other occurrence)
- Modify: `sample_comfyui_workflows/README.md:20` (`## 1. Sicht AI-Hub`)
- Modify: `docs/mesh-client-spec.md`, `docs/host-coordination-plan.md`, `docs/anima-versa-integration.md`, `docs/tripo-api-v3-notes.md` — every `llm-gateway` / `LLM-Gateway` / `LLM Gateway` → `AI-Hub` (paths → `/opt/ai-hub`)

- [ ] **Step 1: Apply** — work file by file with `grep -n -i "llm-gateway\|llm_gateway\|LLM.Gateway" <file>`; in prose "the gateway" as a common noun stays (it is what the software is), only the NAME changes.

- [ ] **Step 2: Run the name test — expect PASS**

Run: `venv/bin/python -m unittest tests.test_project_name -v`
Expected: both tests PASS. If `test_the_old_name_is_gone_from_everything_alive` lists a `docs/*.md` file, fix it; if it lists a file that must legitimately mention the old name once more, extend `ALLOWED_MENTIONS` with a comment saying why.

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md sample_comfyui_workflows/README.md docs/*.md
git commit -m "docs: AI-Hub throughout the living documentation"
```

---

### Task 5: Archive finished plans, add a docs index

**Files:**
- Create: `docs/archive/README.md`
- Move: `docs/next-steps.md`, `docs/review-backlog-2026-07.md`, `docs/multimodal-gateway-plan.md`, `docs/voice-and-subtabs-plan.md`, `code-review-plan.md` → `docs/archive/`
- Create: `docs/README.md`

- [ ] **Step 1: Move**

```bash
mkdir -p docs/archive
git mv docs/next-steps.md docs/review-backlog-2026-07.md docs/multimodal-gateway-plan.md docs/voice-and-subtabs-plan.md docs/archive/
git mv code-review-plan.md docs/archive/code-review-plan-2026-07.md
```

- [ ] **Step 2: Write `docs/archive/README.md`** — one row per file: name, date (from its heading or first dated line), what it planned, why it is done (name the commit or feature that superseded it — `git log --oneline -S"<title words>"` finds it). Header: `# Archiv — erledigte Pläne und Reviews` and the sentence `Diese Dokumente sind Protokolle ihrer Zeit; sie behalten den damaligen Projektnamen (llm-gateway) und werden nicht mehr gepflegt.`

- [ ] **Step 3: Write `docs/README.md`**

```markdown
# docs/

| Ort | Inhalt | Pflege |
|---|---|---|
| `mesh-client-spec.md` | Client-Vertrag der Mesh-Aliase (Parameter, Lieferung, Rigging) | lebend |
| `host-coordination-plan.md` | GPU-Host-Koordination: Flags, VRAM-Free, Scheduler-Affinität | lebend |
| `tripo-api-v3-notes.md` | Tripo3D API V3 — technische Zusammenfassung | lebend |
| `anima-versa-integration.md` | AI-Hub als Image-Backend in anima-versa | lebend |
| `install-test-2026-09.md` | Protokoll der Frischinstallation auf einer leeren VM | Protokoll |
| `archive/` | erledigte Pläne/Reviews mit Index | eingefroren |
| `superpowers/specs/`, `superpowers/plans/` | Design-Specs und Umsetzungspläne je Feature (datiert) | eingefroren nach Umsetzung |

Die Quelle der Wahrheit für Betrieb und Konfiguration ist `../README.md`; für die
Architektur und ihre Invarianten `../CLAUDE.md`.
```

- [ ] **Step 4: Run suite + name test, commit**

```bash
venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -1
git add -A docs code-review-plan.md
git commit -m "docs: archive finished plans, add a docs index"
```

---

### Task 6: Doc review — one Opus subagent per document

**Files:**
- Modify (as findings dictate): `README.md`, `CLAUDE.md`, `config.example.yaml`, `docs/mesh-client-spec.md`, `docs/host-coordination-plan.md`, `docs/tripo-api-v3-notes.md`, `docs/anima-versa-integration.md`, `sample_comfyui_workflows/README.md`

- [ ] **Step 1: Dispatch eight subagents in parallel** (Agent tool, `model: "opus"`, `subagent_type: "general-purpose"`), one per file, with this brief (fill in the file):

```
You are reviewing ONE document of the AI-Hub repo at /home/dev/projekte/llm-gateway
(branch ai-hub-rename): <FILE>. The project was just renamed from llm-gateway to AI-Hub
and the tests moved to tests/. Your job: verify every checkable claim in this document
against the CODE, not against other documents. Checkable = endpoint paths and methods,
config keys and their defaults (grep config.example.yaml AND the code that reads them),
CLI commands (run them where harmless), file/dir paths, counts ("fifteen modules",
"twenty tests"), function/setting names, UI tab names (admin.py TABS/SUBTABS), header
names, status codes, and claims of the form "X does Y" where Y is observable in code.
Do NOT restyle prose, do NOT shorten explanations, do NOT touch docs/superpowers or
docs/archive. Output: a list of findings, each as {line, claim, what the code actually
does (file:line), proposed replacement text}. Only findings you VERIFIED; say "verified
correct" for sections you checked and found right. Do not edit files.
```

- [ ] **Step 2: Apply verified findings yourself** — read each proposed replacement against the cited code line before applying; reject anything not backed by a `file:line`.

- [ ] **Step 3: Re-run the suite and the name test; commit per document**

```bash
venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -1
git add <file> && git commit -m "docs(<file>): review against code — <one line of what was wrong>"
```

---

### Task 7: Phase 1 verification gate

- [ ] **Step 1: Compile + suite**

Run: `venv/bin/python -m py_compile *.py && venv/bin/python -m unittest discover -s tests -t . 2>&1 | tail -2`
Expected: `OK`, 388 tests (386 + 2 from Task 2).

- [ ] **Step 2: Name sweep** — must print nothing:

```bash
git grep -n -i "llm-gateway\|llm_gateway\|LLM Gateway" -- . ':!docs/superpowers' ':!docs/archive' ':!tests/test_project_name.py' | grep -v "formerly\|Renamed from"
```

- [ ] **Step 3: Test instance boots and serves /ui**

```bash
cd /tmp/adm-check && PYTHONPATH=/home/dev/projekte/llm-gateway timeout 20 /home/dev/projekte/llm-gateway/venv/bin/uvicorn --app-dir /home/dev/projekte/llm-gateway main:app --port 4999 >/tmp/aihub-boot.log 2>&1 &
sleep 4; curl -s localhost:4999/health | head -c 200; curl -s localhost:4999/ui | grep -o "AI-Hub" | head -1
```
Expected: health JSON, and `AI-Hub` in the page. Kill the instance afterwards.

- [ ] **Step 4: Commit anything left; the branch is ready for Phase 2.**

---

## Phase 2 — Prod migration, remotes, dev directory

### Task 8: Merge, GitHub rename, LXC bare repo, remotes

- [ ] **Step 1: Merge to master (fast-forward)**

```bash
git checkout master && git merge --ff-only ai-hub-rename
```

- [ ] **Step 2: Rename the GitHub repo**

```bash
gh repo rename ai-hub --repo KaletoAI/llm-gateway --yes
git remote set-url github git@github.com:KaletoAI/ai-hub.git
git ls-remote github HEAD | head -1        # must answer
```

- [ ] **Step 3: Rename the LXC bare repo** (on .10; harmless while the service runs)

```bash
ssh root@192.168.8.10 'mv /opt/llm-gateway.git /opt/ai-hub.git && ls -d /opt/ai-hub.git'
git remote set-url lxc root@192.168.8.10:/opt/ai-hub.git
git ls-remote lxc HEAD | head -1
```

- [ ] **Step 4: Push master to both**

```bash
git push github master && git push lxc master
git remote -v
```
Expected: exactly two remotes, both `ai-hub`.

---

### Task 9: Prod migration on .10 (one maintenance step)

**Preconditions:** .10 idle (`ssh root@192.168.8.10 "curl -s localhost:4000/health" | python3 -c "import sys,json; d=json.load(sys.stdin); print({k:v for k,v in d.items() if 'inflight' in k or 'parked' in k})"` shows nothing in flight); Task 8 done.

- [ ] **Step 1: Backup**

```bash
ssh root@192.168.8.10 'set -e; d=/root/ai-hub-migration-$(date +%Y%m%d-%H%M); mkdir -p $d; cd /opt/llm-gateway; cp -a store.db secret.key config.yaml jobs.db stats.db $d/ 2>/dev/null || true; ls -la $d; curl -s localhost:4000/health | python3 -c "import sys,json;print(\"backends before:\", len(json.load(sys.stdin)[\"backends\"]))"'
```
Record the backend count (expected 14).

- [ ] **Step 2: Stop, move, drop the venv, keep everything else**

```bash
ssh root@192.168.8.10 'set -e; systemctl stop llm-gateway; mv /opt/llm-gateway /opt/ai-hub; rm -rf /opt/ai-hub/venv; ls /opt/ai-hub | head -30'
```

- [ ] **Step 3: Deploy** (recreates the venv, installs the new unit, starts it)

```bash
cd /home/dev/projekte/llm-gateway && venv/bin/python -m py_compile *.py && DEPLOY_HOST=root@192.168.8.10 ./deploy.sh 2>&1 | tail -15
```
Expected: `active` for `ai-hub`.

- [ ] **Step 4: Retire the old unit**

```bash
ssh root@192.168.8.10 'systemctl disable llm-gateway 2>/dev/null; rm -f /etc/systemd/system/llm-gateway.service; systemctl daemon-reload; systemctl is-enabled ai-hub; systemctl is-active ai-hub'
```

- [ ] **Step 5: Prove it**

```bash
ssh root@192.168.8.10 'sleep 5; curl -s localhost:4000/health | python3 -c "import sys,json;print(\"backends after:\", len(json.load(sys.stdin)[\"backends\"]))"; journalctl -u ai-hub --since "-3 min" --no-pager | grep -ci "traceback\|error"; journalctl -u ai-hub --since "-3 min" --no-pager | grep -m1 "Starting"'
```
Expected: same backend count as Step 1, `0` errors, `Starting AI-Hub`. Then one real chat call through an alias (key from `/opt/ai-hub/config.yaml`), and `/ui/backends` renders with the `AI-Hub` brand.

**Rollback** (only if Step 5 fails and the cause is the move): `systemctl stop ai-hub; mv /opt/ai-hub /opt/llm-gateway; install -m 0644 <old unit from git show master~1:llm-gateway.service> /etc/systemd/system/; systemctl daemon-reload; systemctl start llm-gateway`.

---

### Task 10: Dev directory and memory

- [ ] **Step 1: Move the checkout** (from a shell OUTSIDE the directory)

```bash
cd /home/dev/projekte && mv llm-gateway ai-hub && cd ai-hub && git status --short | head -3
```

- [ ] **Step 2: Copy the Claude memory directory to the new project key**

```bash
cp -a /home/dev/.claude/projects/-home-dev-projekte-llm-gateway /home/dev/.claude/projects/-home-dev-projekte-ai-hub
```

- [ ] **Step 3: Update path-bearing memories** in `/home/dev/.claude/projects/-home-dev-projekte-ai-hub/memory/`: `project_prod_topology_localai.md` (`/opt/llm-gateway` → `/opt/ai-hub`, unit `ai-hub`), `feedback_check_prod_before_deploy.md`, `feedback_worktree_deploy_delete.md`, `project_vram_free_before_job_2026-09.md`, and any other `grep -l "llm-gateway" memory/*.md`. Add one line to `MEMORY.md`: `- [AI-Hub rename 2026-09-08](project_ai_hub_rename_2026-09.md) — prod /opt/ai-hub + unit ai-hub, GitHub KaletoAI/ai-hub, lxc /opt/ai-hub.git, dev ~/projekte/ai-hub; Forgejo origin removed` with a matching memory file.

- [ ] **Step 4: Tell the user to restart the Claude session in `~/projekte/ai-hub`** so the memory is loaded from the new key. Nothing else in Phase 2 depends on it.

---

## Phase 3 — Fresh install on 192.168.8.148

**Precondition:** `ssh root@192.168.8.148 true` works (the user installs the dev key via `pct exec` on the Proxmox host; the plan cannot do this).

### Task 11: Install exactly as the README says

**Files:**
- Create: `docs/install-test-2026-09.md` (protocol, appended as you go)
- Modify: `README.md` (every gap found)

- [ ] **Step 1: Snapshot the container**

```bash
ssh root@192.168.8.148 'grep PRETTY /etc/os-release; python3 --version; which git rsync systemctl pip3; python3 -c "import venv" && echo venv-ok; free -m | sed -n 2p; df -h / | tail -1'
```
Write the output into the protocol under `## Ausgangszustand`.

- [ ] **Step 2: Follow README "Quick start" literally.** Run each README line via ssh, in order, from a fresh shell. The first line that fails or presupposes something (`git` missing, `python3-venv` missing, `pip` missing) is a README gap: fix the README **now** (add the prerequisite line `apt install -y git python3 python3-venv` under Quick start with a sentence that says which Debian it was verified on), commit, and continue.

```bash
ssh root@192.168.8.148 'git clone https://github.com/KaletoAI/ai-hub.git && cd ai-hub && python3 -m venv venv && venv/bin/pip install -r requirements.txt 2>&1 | tail -3'
```

- [ ] **Step 3: Configure.** Read the OpenRouter key from the prod box (`ssh root@192.168.8.10 "grep -n -i openrouter -A6 /opt/ai-hub/config.yaml"`; if backends live in the store, `venv/bin/python -c "import store; ..."` to read the `openrouter` backend's key — decrypting is what `store` does). Write `/root/ai-hub/config.yaml` on .148 from `config.example.yaml` with:

```yaml
api_key: "sk-test-aihub"
backends:
  - name: localai-strix
    url: http://192.168.8.38:8080/v1
    type: openai
    local: true
  - name: localai-phoenix
    url: http://192.168.8.39:8080/v1
    type: openai
    local: true
  - name: openrouter
    url: https://openrouter.ai/api/v1
    type: openai
    api_key: "<from prod>"
    paid: true
virtual_models:
  chat-test:
    localai-strix: "<a model both LocalAI boxes list — read /v1/models on .38 first>"
    localai-phoenix: "<same>"
    openrouter: "openai/gpt-4o-mini"
```
Every key name above must be checked against `config.example.yaml` — if the README/example disagrees with what `load_config` reads (`grep -n "def load_config" -A60 main.py`), that is a doc finding: fix README/example, commit.

- [ ] **Step 4: Start**

```bash
ssh root@192.168.8.148 'cd ai-hub && nohup venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000 > /root/aihub.log 2>&1 & sleep 6; tail -5 /root/aihub.log'
```

---

### Task 12: Functional proof

- [ ] **Step 1: Health and models**

```bash
B=http://192.168.8.148:4000; K=sk-test-aihub
curl -s $B/health | python3 -c "import sys,json; d=json.load(sys.stdin); print([(b['name'], b.get('healthy')) for b in d['backends']])"
curl -s $B/v1/models -H "Authorization: Bearer $K" | python3 -c "import sys,json; ids=[m['id'] for m in json.load(sys.stdin)['data']]; print(len(ids), [i for i in ids if 'chat-test' in i or 'openrouter' in i][:5])"
```
Expected: three backends `True`; the alias plus prefixed OpenRouter models listed.

- [ ] **Step 2: Chat over LocalAI and over OpenRouter, streamed and not** — record the `x-gateway-backend` header each time:

```bash
for M in "localai-strix/<model>" "openrouter/openai/gpt-4o-mini" "chat-test"; do for S in false true; do
  echo "== $M stream=$S"; curl -s -D - $B/v1/chat/completions -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
   -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"Antworte mit einem Wort: Hauptstadt von Frankreich?\"}],\"stream\":$S,\"max_tokens\":20}" | grep -i "x-gateway-backend\|\"content\"\|data: \[DONE\]" | head -4; done; done
```

- [ ] **Step 3: Failover.** Edit `.148`'s `config.yaml`: point `localai-strix` and `localai-phoenix` URLs at `http://192.168.8.250:9/v1` (dead), save (hot reload), wait one `health_check_interval`, repeat the `chat-test` call: `x-gateway-backend` must say `openrouter`. Restore the URLs afterwards and confirm the header goes back to a LocalAI backend.

- [ ] **Step 4: Console.** `curl -s -o /dev/null -w "%{http_code}\n" $B/ui` → 200 (bootstrap: sign in with the api_key). Log in with the key, open `/ui/backends` and `/ui?tab=statistic` (find the exact paths in `admin.TABS`) and confirm the three backends and the calls from Step 2 appear. Use the CDP harness from memory `project_ui_verification_cdp` if a rendered check is needed; otherwise curl with the session cookie.

- [ ] **Step 5: Protocol.** Append every command and its trimmed answer to `docs/install-test-2026-09.md` under `## Nachweise`; list each README gap found and the commit that fixed it under `## Gefundene Doku-Fehler`.

---

### Task 13: Second path — `deploy.sh` against the container

- [ ] **Step 1: Stop the nohup instance on .148**, then deploy from the dev checkout:

```bash
ssh root@192.168.8.148 'pkill -f "uvicorn main:app" || true; rm -rf /root/ai-hub'
cd /home/dev/projekte/ai-hub && DEPLOY_HOST=root@192.168.8.148 ./deploy.sh 2>&1 | tail -12
```
Expected: venv created, `ai-hub.service` installed, `active`. (`config.yaml` is excluded by deploy.sh — copy the test config to `/opt/ai-hub/config.yaml` first or the service starts bootstrap-open; that is itself a README point: say so under "Running & deploying" if it is not said.)

- [ ] **Step 2: Prove via systemd**

```bash
ssh root@192.168.8.148 'systemctl is-active ai-hub; journalctl -u ai-hub --since "-2 min" --no-pager | grep -m1 "Starting"; curl -s localhost:4000/health | head -c 120'
```

- [ ] **Step 3: Clean the key.** Remove the OpenRouter key from `/opt/ai-hub/config.yaml` on .148 (replace with `"REMOVED"`) and note in the protocol that the container keeps a keyless install.

---

### Task 14: Close out

- [ ] **Step 1: Commit the protocol and README fixes; push both remotes**

```bash
cd /home/dev/projekte/ai-hub
git add docs/install-test-2026-09.md README.md docs/README.md
git commit -m "docs: fresh-install test on a clean Debian container — protocol + README fixes"
git push github master && git push lxc master
```

- [ ] **Step 2: Deploy the doc changes to prod** (docs only; harmless): `DEPLOY_HOST=root@192.168.8.10 ./deploy.sh` when idle — or skip if only docs changed and say so.

- [ ] **Step 3: Report** — what was renamed, what the install test found (each gap + fix), the two remotes, the new prod path, and the one manual step left for the user (restart the Claude session in `~/projekte/ai-hub`).
