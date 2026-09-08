# Frischinstallation auf einer leeren VM — Protokoll (2026-09-08)

Ziel: AI-Hub **exakt nach README** auf einem leeren Debian-Container installieren und mit
einer lokalen KI (llama-swap) und OpenRouter Chat-Completions belegen. Jede Stelle, an der
die README nicht reichte, ist unten unter „Gefundene Doku-Fehler" mit ihrem Fix vermerkt.
Spec: `superpowers/specs/2026-09-08-ai-hub-rename-design.md`, Phase 3.

## Ausgangszustand

Proxmox-LXC `<test-vm>`, frisch angelegt, nie benutzt:

```
PRETTY_NAME="Debian GNU/Linux 13 (trixie)"
Python 3.13.5
git= rsync= pip3=            ← nichts davon installiert
venv-ok                      ← python3 -m venv vorhanden
Mem: 512 MB total, 350 MB free
/: 2.0 GB, 458 MB belegt, 1.6 GB frei
```

Lokale KI: die LocalAI-Backends .38/.39 aus der Spec sind seit dem Morgen DOWN (LocalAI wird
laut Betriebsnotizen durch llama-swap abgelöst); der Test nutzt stattdessen
`llamaswap-strix` (`http://<llama-swap-1>:8080`) und `llamaswap-phoenix` (`http://<llama-swap-2>:8080`).
OpenRouter-Key: aus dem Prod-Store auf .10 gelesen und **nur** in die `config.yaml` der
Test-VM geschrieben (nach dem Test entfernt, siehe unten).

## Nachweise

### Schritt 1 — README „Quick start", Zeile 1

```
root@148:~# git clone https://github.com/KaletoAI/ai-hub.git
bash: git: command not found
```

**Doku-Fehler #1:** die README setzt `git`, `python3-venv` (für `pip` im venv) und optional
`rsync` (für `deploy.sh`) stillschweigend voraus. Auf einem minimalen Debian 13 fehlen alle
drei. → Fix: Voraussetzungs-Zeile im Quick start (siehe unten).

Nach `apt-get install -y git python3-venv`:

```
git=2.47.3   ensurepip=pip 25.1.1
git clone https://github.com/KaletoAI/ai-hub.git      → 1bda5c9
python3 -m venv venv && venv/bin/pip install -r requirements.txt
  → Successfully installed … faster-whisper-1.2.1 … uvicorn-0.52.4 … (pip exit=0)
  → venv 249 MB; Platte danach 892 MB belegt (1.2 GB frei); RAM unauffällig
```

Die Installation läuft mit 512 MB RAM und 2 GB Platte durch; `faster-whisper`/`ctranslate2`/
`onnxruntime` sind der Löwenanteil des venv.

### Schritt 2 — `config.yaml` und Start

`cp config.example.yaml config.yaml`, dann drei Backends (zwei llama-swap, OpenRouter mit
`paid: true` + `api_key`) und ein Alias `chat-test` über alle drei — alle Schlüsselnamen
stammen aus `config.example.yaml` (dort auskommentiertes OpenRouter-Beispiel mit dem Hinweis
„`/api` only — the gateway appends /v1/…"). Start exakt mit der README-Zeile
`venv/bin/uvicorn main:app --host 0.0.0.0 --port 4000`:

```
2026-09-08 14:57:45 INFO Starting AI-Hub
GET /health → {"status":"ok", backends: llamaswap-strix True, llamaswap-phoenix True, openrouter True}
```

### Schritt 3 — Modelle und Chat-Completions

```
GET /v1/models              → 447 ids; alias chat-test vorhanden; llamaswap-strix/…, openrouter/… gelistet
POST /v1/chat/completions   (Frage: „Hauptstadt von Frankreich?", max_tokens 16)
  llamaswap-strix/gemma-4-12b-it   stream=false → 200, x-gateway-backend: llamaswap-strix, "Paris", usage 27
  llamaswap-strix/gemma-4-12b-it   stream=true  → 200, 3 Chunks + [DONE], "Paris"
  openrouter/openai/gpt-4o-mini    stream=false → 200, x-gateway-backend: openrouter, "Paris", usage 21
  openrouter/openai/gpt-4o-mini    stream=true  → 200, 3 Chunks + [DONE], "Paris"
  chat-test                        stream=false → 200, x-gateway-backend: llamaswap-strix (unpaid zuerst), "Paris"
  chat-test                        stream=true  → 200, 3 Chunks + [DONE], "Paris"
```

### Schritt 4 — Failover

Beide llama-swap-URLs in `config.yaml` per `sed -i` auf tote Adressen gesetzt (Hot-Reload):

```
14:58:43 Detected change in config.yaml — reloading … Loaded 3 backend(s)
14:58:48 WARNING [llamaswap-strix] DOWN — All connection attempts failed   (phoenix ebenso)
POST chat-test → 200, x-gateway-backend: openrouter, "Rom"
```

Failover belegt. Die **Rückkehr** dagegen nicht: nach `cp config.yaml.good config.yaml`
(14:59:24) kam **kein zweiter Reload**, die lokalen Backends blieben DOWN, jeder Call ging
weiter zu OpenRouter. Nachgeprüft: weder ein In-place-`echo >>` noch ein zweites `sed -i`
wurde noch erkannt — nach der ersten Ersetzung der Datei (neuer Inode) war der Watch tot.
→ **Gateway-Bug**, siehe „Gefundene Fehler" #3. Für die weiteren Schritte wurde die Instanz
neu gestartet (danach wieder 3/3 healthy).

### Schritt 5 — Konsole

```
GET  /ui                → 303 /ui/login?next=/ui
POST /ui/login (key)    → 303 /ui/dashboard (Session-Cookie)
GET  /ui/backends       → 200, alle drei Backends gelistet
GET  /ui/statistic      → 200, LEER ;  /ui/jobs?sub=llm → 200, keine Calls
```

Leer, weil `stats.enabled` im Beispiel `false` ist und die Testconfig keinen `stats:`-Block
hatte — die Seite sagt das aber nicht (→ Doku-Fehler #2). Mit `stats: {enabled: true}` und
Neustart (der Schalter wird nur beim Start gelesen):

```
POST chat-test ×2 → "Blau", x-gateway-backend: llamaswap-strix
GET /ui/statistic   → chat-test / gemma-4-12b-it / llamaswap-strix sichtbar
GET /ui/jobs?sub=llm → 1 Zeile chat-test      (stats.db 24 KB angelegt)
```

## Gefundene Doku-Fehler und Bugs

| # | Fund | Fix |
|---|---|---|
| 1 | README Quick start setzt `git`, `python3-venv` (und für `deploy.sh` `rsync`) voraus, ohne es zu sagen; auf Debian 13 minimal fehlen alle drei. | Voraussetzungs-Zeile im Quick start (Debian/Ubuntu: `apt install -y git python3 python3-venv`), mit dem Hinweis, auf welchem System es geprüft wurde. |
| 2 | Statistic- und LLM-Calls-Seiten bleiben leer, solange `stats.enabled` (Default `false`) nicht gesetzt ist; weder README-Konsolentabelle noch die Seite selbst sagen das. | Satz in der README unter der Konsolentabelle: beide brauchen `stats.enabled: true` (Server-Tab oder config.yaml) und einen Neustart. Ein Leerzustand-Hinweis auf der Seite selbst wäre ein Codeänderung — notiert, nicht Teil dieses Tests. |
| 3 | **Bug:** Hot-Reload der `config.yaml` stirbt nach dem ersten Speichern, das die Datei ersetzt (`sed -i`, vim, die meisten Editoren): `awatch(CONFIG_PATH)` hängt am alten Inode. Danach wird keine Änderung mehr erkannt — auch kein Zurücksetzen. README/CLAUDE.md versprechen „hot-reloaded on save". Auf Prod unbemerkt, weil dort `backends: []` gilt und alles im Store liegt. | Watch auf das Verzeichnis mit Filter auf den Dateinamen + Regressionstest `tests/test_config_watch.py` (Commit siehe unten). |
| 4 | **deploy.sh:** die Wahl zwischen rsync und dem tar-Fallback prüft `rsync` nur LOKAL. Auf ein Ziel ohne rsync (frisches Debian) stirbt der Lauf mit `rsync error … (code 12)`, exit 12 — der Fallback, der genau dafür existiert, läuft nie. | `command -v rsync` zusätzlich per `ssh $HOST` auf dem Ziel prüfen, sonst tar-Pfad (Commit siehe unten). README nennt rsync jetzt unter den Voraussetzungen. |

### Schritt 6 — zweiter Weg: `deploy.sh` gegen die VM

```
DEPLOY_HOST=root@<test-vm> ./deploy.sh
rsync error: error in rsync protocol data stream (code 12) at io.c(232) [sender=3.2.7]   → exit 12
```

(→ Fund #4.) Zweiter Lauf mit dem korrigierten `deploy.sh`: siehe unten.

Mit korrigiertem `deploy.sh` (`e8360d7`):

```
==> rsync missing locally or on root@<test-vm> — falling back to tar-over-ssh (no --delete)
    synced (stale remote files are NOT removed without rsync)
==> Ensuring venv + installing requirements + syncing systemd unit
==> Restarting ai-hub → active
systemctl is-active ai-hub → active ; is-enabled → enabled
journal: Starting AI-Hub ; /health → 3/3 healthy ; /opt/ai-hub/tests + venv vorhanden
```

### Schritt 7 — Hot-Reload nach dem Fix (`c6bea7b`, systemd-Instanz auf .148)

Drei Speichervorgänge, drei Erkennungen — inklusive der Rückkehr, die vorher verloren ging:

```
sed -i (Datei-Ersetzung, tote URLs)   → Detected change #1 ; llamaswap-* DOWN
cp good config.yaml (In-place)        → Detected change #2 ; llamaswap-* wieder UP
sed -i (Key → "REMOVED")              → Detected change #3
```

### Aufräumen

OpenRouter-Key aus `/opt/ai-hub/config.yaml` entfernt (`api_key: "REMOVED"`), die Kopien unter
`/root/ai-hub/` und `/root/.or_key` gelöscht; auf .10 war die Übergabekopie sofort nach dem
Transfer gelöscht worden. Der Container behält eine lauffähige, schlüssellose Installation
(`ai-hub.service`, zwei llama-swap-Backends).

## Commits aus diesem Test

| Commit | Inhalt |
|---|---|
| `c6bea7b` | `config: hot reload must survive an editor's rename-save` — Watch auf das Verzeichnis + Filter, `watch_config(path, on_change)`, `tests/test_config_watch.py` (Fund #3) |
| `e8360d7` | `deploy: rsync must exist on BOTH ends — fall back to tar when the target lacks it` (Fund #4) |
| (dieser) | README: Voraussetzungen im Quick start (Fund #1), Statistic-Hinweis (Fund #2), dieses Protokoll |

## Ergebnis

Eine Installation exakt nach README funktioniert auf einem leeren Debian 13 mit 512 MB RAM,
sobald die drei Voraussetzungen installiert sind. Lokale KI (llama-swap) und OpenRouter
liefern Chat-Completions, streamed und nicht, direkt und über einen Alias; der Failover auf
den bezahlten Anbieter greift, die Konsole funktioniert. Der Test hat einen echten Bug
(Hot-Reload) und einen Deploy-Fehler (rsync-Erkennung) aufgedeckt; beide sind behoben und
auf derselben VM nachgewiesen.
