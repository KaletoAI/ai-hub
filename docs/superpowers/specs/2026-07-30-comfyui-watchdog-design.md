# ComfyUI-Watchdog: Stuck-Executor-Erkennung + Dienst-Neustart

Datum: 2026-07-30 · Status: freigegeben (Kai)

## Anlass

Am 30.07.2026 ab 06:27 fiel auf `k12-gpu` (192.168.1.37, RTX 3090 Passthrough) die
GPU vom PCIe-Bus (`NV_ERR_GPU_IS_LOST`); der ComfyUI-`prompt_worker` starb mit
`CUDA error: unspecified launch failure`. Der HTTP-Server von ComfyUI antwortete
weiter mit 200, `discover()` (nur `/object_info`) hielt das Backend daher für
gesund. Folge: jeder Media-Job lief in das 600-s-Poll-Timeout, geparkte Jobs
liefen in `park timeout` — über Stunden, ohne Alarm. Signatur im `/queue`-Snapshot:
`queue_running: []` bei gefüllter `queue_pending`, dauerhaft.

## Ziel

1. Der Gateway erkennt einen hängenden/toten ComfyUI-Executor innerhalb weniger
   Health-Zyklen und markiert das Backend unhealthy (Routing weicht aus bzw.
   scheitert schnell statt in 10-min-Timeouts).
2. Der Gateway kann den ComfyUI-Dienst neu starten: manuell per UI-Button,
   optional automatisch (opt-in, mit Cooldown).

Nicht-Ziel: Ein GPU-Bus-Verlust (heutiger Root Cause) ist durch einen
Dienst-Neustart nicht behebbar — das Backend bleibt dann schlicht unhealthy;
die Erkennung/Sichtbarkeit greift trotzdem.

## 1. Stuck-Detection (`adapters.py`, `ComfyUIAdapter`)

- `discover()` ruft zusätzlich `GET /queue` ab (gleicher Health-Tick, ein
  Request mehr).
- Adapter-interner Zustand über Discover-Zyklen: `(head_pending_id,
  running_leer_seit_ts)`.
- Stuck-Regel: `queue_pending` nicht leer UND `queue_running` leer UND derselbe
  Head-Pending-Prompt über ≥ 2 aufeinanderfolgende Checks UND Zustand hält
  ≥ `stuck_after_s` (Default 90) → `discover()` wirft `ComfyExecutorStuck`
  (Subklasse von `Exception` mit sprechender Message).
- Wirkung: `refresh_backend()` in `main.py` fängt das wie jeden Discovery-Fehler
  → Backend DOWN geloggt („executor stuck …“) und `backend_healthy=False`.
  Kein neuer Routing-Pfad nötig.
- Transiente Leerlauf-Momente zwischen Dequeues (Head wechselt, oder running
  nicht leer) setzen den Zustand zurück; Fehlalarme sind damit ausgeschlossen.
- Ein `/queue`-Fetch-Fehler für sich ist KEIN Stuck-Signal (dann greift ohnehin
  der normale Discovery-Fehlerpfad).

## 2. Restart-Aktion (`ComfyUIAdapter.restart()`)

- POST `{url}/manager/reboot` (ComfyUI-Manager-Extension). Der Prozess beendet
  sich; systemd (`Restart=always` bzw. `on-failure`) startet ihn neu.
- Ein Verbindungsabbruch/Timeout auf dem Reboot-Call ist der ERWARTETE
  Erfolgsfall (der Prozess stirbt mitten in der Antwort). Ein 404 heißt:
  Manager-Extension fehlt → Fehler an Aufrufer („ComfyUI-Manager nicht
  installiert“), kein Warte-Loop.
- Danach begrenztes Warten (Poll `/object_info`, bis 120 s) auf Wiederkehr;
  Ergebnis (`ok`/`timeout`) wird zurückgemeldet und geloggt. Der reguläre
  Health-Loop übernimmt das endgültige Gesundwerden.
- Adapter merkt sich `last_restart_ts` + letztes Ergebnis (für UI/Cooldown).

## 3. Auto-Restart (opt-in, `main.refresh_backend`)

- Neue Comfy-Backend-Felder (Store-Backend-Editor, wie `comfy_output_dir`):
  `auto_restart` (bool, Default aus), `restart_cooldown_s` (int, Default 600).
- Trigger: Discovery endet mit `ComfyExecutorStuck` UND `auto_restart` UND
  Cooldown abgelaufen → genau EIN `adapter.restart()` als `asyncio.create_task`
  (fire-and-forget, Log-Event). Bewusst nicht an `backend_inflight` gekoppelt —
  stuck heißt: nichts läuft; ein pending Gateway-Prompt ist ohnehin verloren
  (sein Poll läuft in Failover/Park).
- Bleibt der Executor danach stuck, bleibt das Backend unhealthy; der Cooldown
  verhindert eine Restart-Schleife.

## 4. Sichtbarkeit

- `/health`: Comfy-Backend-Eintrag erhält `exec_stuck: bool`,
  `last_restart: ts|null`, `last_restart_result: str|null`.
- Backends-Tab (`admin.py`): Statuszeile zeigt „executor stuck“; Button
  „Restart ComfyUI“ (POST-Route, nur `type: comfyui`, `_ui_guard`-gated);
  Backend-Editor bekommt Auto-Restart-Checkbox + Cooldown-Feld.
- Restart-Ereignisse (manuell + auto) im Gateway-Log.

## 5. Verifikation

Kein Testsuite-Setup im Repo (CLAUDE.md). Verifikation über:
- Mock-ComfyUI (Scratchpad-Script: `/object_info`, `/queue` mit steuerbarem
  Zustand, `/manager/reboot`) + lokale Gateway-Instanz: Stuck-Erkennung,
  Fehlalarm-Freiheit (Head-Wechsel), manueller Restart, Auto-Restart inkl.
  Cooldown, Recovery.
- `py_compile`-Gate vor Deploy; Deploy erst nach Freigabe (Prod-Instanz läuft).

## Infra-Voraussetzungen (auf .37, nach VM-Wiederkehr zu prüfen)

- ComfyUI-Manager installiert? (`/manager/reboot` erreichbar)
- `comfyui.service` mit `Restart=always` (sonst kommt der Prozess nach
  `/manager/reboot` nicht wieder).

## Bewusst weggelassen

- SSH-`restart_cmd`-Fallback (spätere Erweiterung, falls Manager-API nicht
  reicht).
- Automatik für GPU-Bus-Verlust (Host-Reboot ist manuelle Ops-Entscheidung).
