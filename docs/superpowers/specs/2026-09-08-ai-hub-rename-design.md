# AI-Hub: Umbenennung, Repo-Struktur, Doku-Review, Frischinstallation — Design

Stand 2026-09-08. Beschlossen im Gespräch (Rename-Tiefe „vollständig“, Module bleiben
flach, Tests nach `tests/`, Archiv statt Löschen, Test-VM 192.168.8.148, lokale KI =
LocalAI .38/.39, OpenRouter-Key aus der Prod-Config).

## Ziel

Das Projekt `llm-gateway` heißt künftig **AI-Hub** (Slug `ai-hub`), das Repo ist
aufgeräumt (Tests in `tests/`, erledigte Pläne archiviert), die lebende Doku stimmt
nachweislich mit dem Code überein, und eine Installation nach README auf einer leeren
Debian-Maschine funktioniert — belegt durch eine echte Installation auf .148 mit
Chat-Completions über LocalAI und OpenRouter.

## Nicht-Ziele

- Kein Python-Package (`aihub/`): `main:app`, Hot-Reload und deploy.sh bleiben, wie sie sind.
- Keine Umbenennung in historischen Pläne/Specs unter `docs/superpowers/` — datierte
  Protokolle behalten den Namen ihrer Zeit.
- Kein neuer lokaler LLM-Server auf der Test-VM.
- Keine Funktionsänderung am Gateway.

## Phasen

Gestaffelt: erst ein Branch mit Rename + Struktur + Doku (lokal komplett testbar), dann
EIN Wartungsschritt auf .10, dann der VM-Test. So ist bei einem Fehler klar, welche
Hälfte ihn verursacht hat.

### Phase 1 — Branch `ai-hub-rename`: Rename + Struktur + Doku

**Namen.** Anzeige „AI-Hub“; Slug `ai-hub` für Pfade, Unit, Repos, Remotes;
`owned_by: "ai-hub (virtual|image)"` in `/v1/models` (API-sichtbar, von keinem
bekannten Client ausgewertet).

**Rename-Stellen im Repo** (Fundstellen `grep -i "llm-gateway\|llm_gateway\|LLM Gateway"`
außerhalb `docs/superpowers/`):

| Wo | Was |
|---|---|
| `README.md` | Titel, Fließtext, Quick start, Deploy-Abschnitt, Pfade |
| `CLAUDE.md` | Kopf, Run/develop, Deploy, Testkommandos (neuer Pfad `tests/`) |
| `admin.py:180`, `:561` | UI-Brand, `<title>`-Suffix „· AI-Hub“ |
| `main.py:702, 778, 1871, 1881, 1891` | Start-Log, FastAPI-Titel, `owned_by` |
| `stats.py:391, 550, 619, 726` | Stats-Titel/Überschriften |
| `deploy.sh` | `DEST=/opt/ai-hub`, `SERVICE=ai-hub`, Unit-Dateiname |
| `llm-gateway.service` → `ai-hub.service` | Description, WorkingDirectory, ExecStart |
| `sample_comfyui_workflows/README.md:20` | Überschrift |
| Testdateien-Docstrings | Run-Kommandos ohne absoluten Pfad (`venv/bin/python -m unittest tests.<modul>`) |
| lebende docs (`mesh-client-spec`, `host-coordination-plan`, `anima-versa-integration`, `tripo-api-v3-notes`) | Name/Pfade |

**Struktur.**

- `tests/` mit `__init__.py` und den 20 `test_*.py`. Runner:
  `venv/bin/python -m unittest discover -s tests -t .` (Top-Level `.`, damit `import main`
  weiter das Root sieht). Die sechs Präambeln mit `_here = dirname(abspath(__file__))`
  zeigen auf das Elternverzeichnis (`dirname(dirname(...))`); `test_scheduler` liest die
  Samples relativ dazu.
- `docs/archive/` mit `README.md` (Index: Datei, Datum, worum es ging, warum erledigt)
  für `next-steps.md`, `review-backlog-2026-07.md`, `multimodal-gateway-plan.md`,
  `voice-and-subtabs-plan.md`. `code-review-plan.md` (Root) ebenfalls dorthin, sofern
  Inhalt historisch; sonst löschen.
- `.gitignore` +`.claude/`, `.agents/`, `skills-lock.json*`, `*.pdf` unter
  `sample_comfyui_workflows/`.
- `docs/README.md` als Einstieg: was liegt wo (lebend / archive / superpowers).

**Doku-Review.** Ein Opus-Subagent pro Dokument (README, CLAUDE.md, config.example.yaml,
die vier lebenden docs, sample-README) mit dem Auftrag: jede prüfbare Behauptung gegen
den Code verifizieren (Endpunkte, Config-Schlüssel, Defaults, Pfade, Zählungen wie
„fünfzehn Module / zwanzig Tests“, Kommandos) und Abweichungen als konkrete Änderung
vorschlagen; Übernahme nach eigener Prüfung. Regel: Historie bleibt, lebende Doku zeigt
auf den Stand nach Phase 1.

**Verifikation Phase 1.** `py_compile *.py`; volle Suite aus `tests/` grün;
`grep -ri "llm-gateway\|llm_gateway\|LLM Gateway" --exclude-dir=docs/superpowers` liefert
nur bewusste Reste (Archiv-Index, Migrationshinweis); `deploy.sh` Dry-Run gegen .10
zeigt die erwarteten Umbenennungen; `/ui` rendert (Testinstanz).

### Phase 2 — Prod-Migration auf .10 (ein Wartungsschritt, wenn idle)

Reihenfolge, jeweils mit Prüfung:

1. Backup: `store.db`, `secret.key`, `config.yaml`, `jobs.db`, `stats.db` nach
   `/root/ai-hub-migration-<datum>/`.
2. `systemctl stop llm-gateway`.
3. `mv /opt/llm-gateway /opt/ai-hub`; `rm -rf /opt/ai-hub/venv` (Shebangs tragen den
   alten absoluten Pfad).
4. Bare-Repo: `mv /opt/llm-gateway.git /opt/ai-hub.git`.
5. Deploy vom Branch: `DEPLOY_HOST=root@192.168.8.10 ./deploy.sh` — legt venv neu an,
   synct `ai-hub.service`, `systemctl enable --now ai-hub`.
6. `systemctl disable llm-gateway`; alte Unit-Datei entfernen.
7. Nachweis: `/health` 200 mit derselben Backend-Zahl wie vorher (14), Journal ohne
   Traceback, Backends-Tab, ein Chat-Call.

Rollback: Service stoppen, `mv` zurück, alte Unit wieder aktivieren (Backup unangetastet).

**Remotes.**

- GitHub: `gh repo rename ai-hub` (KaletoAI/ai-hub; GitHub leitet die alte URL weiter).
- Forgejo (.110:3000, war `origin`): **deaktiviert** (2026-09-08, Kais Entscheidung) —
  Remote aus dem Dev-Clone entfernt, dort wird nichts umbenannt und nichts mehr gepusht.
- LXC: erledigt in Schritt 4.
- Dev-Clone: die zwei verbleibenden Remote-URLs (`github`, `lxc`) umstellen; Branch nach
  master mergen, auf beide pushen. „Merge“ heißt ab jetzt: Push auf **zwei** Remotes.

**Dev-Verzeichnis.** `~/projekte/llm-gateway` → `~/projekte/ai-hub`; das Claude-Memory-
Verzeichnis `~/.claude/projects/-home-dev-projekte-llm-gateway` wird nach
`-home-dev-projekte-ai-hub` kopiert (nicht verschoben, bis die neue Session es liest);
Memory-Einträge mit Pfaden (`/opt/llm-gateway`, Dev-Pfad) werden aktualisiert.

### Phase 3 — Frischinstallation auf 192.168.8.148

Als neuer Nutzer, exakt nach README (jede Abweichung = README-Fehler, der zurückfließt):

0. SSH-Zugang: Kais Dev-Key wird auf root@.148 hinterlegt (`ssh-copy-id`); der
   Container antwortet aktuell mit `Permission denied (publickey,password)`.
1. Voraussetzungen laut README installieren (python3, venv, git; was fehlt, steht dann
   in der README).
2. `git clone` von GitHub (`KaletoAI/ai-hub`), venv, `pip install -r requirements.txt`.
3. `config.yaml` aus `config.example.yaml`: Backends `localai-strix` (192.168.8.38),
   `localai-phoenix` (192.168.8.39), `openrouter` (Key aus der Prod-Config auf .10, nur
   in die VM-Config kopiert, nie ins Repo), ein Alias über beide Wege, `api_key` gesetzt.
4. Start per `uvicorn main:app`, dann Nachweise:
   - `GET /health` 200, alle drei Backends healthy;
   - `GET /v1/models` zeigt lokale + OpenRouter-Modelle + Alias;
   - `POST /v1/chat/completions` gegen ein LocalAI-Modell und gegen ein OpenRouter-Modell,
     je `stream:false` und `stream:true`;
   - Failover: Alias mit LocalAI zuerst, LocalAI-Backend im Config auf eine tote URL →
     Antwort kommt von OpenRouter, `x-gateway-backend` belegt es;
   - `/ui`: Login, Backends-Tab, Statistic-Tab zeigt die Calls.
5. Zweiter Weg: `deploy.sh` gegen .148 (`DEPLOY_HOST=root@192.168.8.148`) — Unit,
   venv, Restart, `/health` über systemd.
6. Ergebnisprotokoll in `docs/install-test-2026-09.md` (Kommandos, Antworten gekürzt,
   gefundene README-Lücken + ihre Fixes).

## Risiken

- **venv-Shebangs** nach `mv`: gelöst durch Neuanlage (deploy.sh macht das).
- **Store-Backend-URLs** sind absolut und unabhängig vom Pfad — unberührt.
- **Memory-Verzeichnis** hängt am Dev-Pfad: Kopie statt Move, damit nichts verloren geht.
- **OpenRouter-Key** auf der Test-VM: nur in deren `config.yaml`, nach dem Test entfernt.

## Erfolgskriterien

- Prod läuft als `ai-hub` unter `/opt/ai-hub` mit unveränderter Backend-Zahl und Store.
- Beide Remotes (GitHub, LXC) heißen `ai-hub`; Dev-Clone und Memory zeigen darauf.
- Suite grün aus `tests/`; kein unbeabsichtigtes `llm-gateway` mehr im lebenden Repo.
- README-Installation auf .148 ohne Rückfrage durchführbar; beide Backend-Wege liefern
  Chat-Completions, Failover belegt, Protokoll im Repo.
