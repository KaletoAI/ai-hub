# Plan: Host-Tabelle + koordiniertes GPU-Handling (ComfyUI ↔ llama-swap)

Stand 2026-07-10. Recherche-Basis: Live-Incident-Analyse (llama-swap `/logs`
auf .37, `jobs.db`/`stats.db` auf prod, ComfyUI `/system_stats`) + Code-Scan
Routing/Dispatch (`main.py`) und Adapter (`adapters.py`). Zeilenangaben sind
Anker auf dem Stand nach dem Stream-Failover-Fix — bei Drift nach Symbolnamen
suchen.

**Reihenfolge:** Phase 1 → Phase 2 → Phase 3. Jede Phase ist einzeln
deploybar und ohne die nächste nützlich.

---

## Problem (verifiziert 2026-07-10)

`k12-gpu` (.37) und `evo-x2-gpu` (.34) fahren **llama-swap UND ComfyUI auf
derselben GPU** (.37: eine RTX 3090, 25 GB). Der Gateway kennt sie als zwei
unabhängige Backends (`type: openai` :8080 / `type: comfyui` :8188) und weiß
nicht, dass sie sich VRAM teilen. Belegter Ablauf:

1. Chat-Modell wird per llama-swap-TTL (120 s) entladen.
2. img2img-Job lädt das Bildmodell → ComfyUI **cached es dauerhaft im VRAM**
   (nur ~4 GB blieben frei); ComfyUI gibt nie von selbst frei.
3. Nächster Chat-Call → llama-swap-Reload → llama-server stirbt
   (exit 1 / SIGABRT) → 502 `unable to start process: upstream command
   exited prematurely but successfully`.

**Bereits gebaut (Sicherheitsnetz, deployed 2026-07-10):** genau dieses 502
ist jetzt failover-würdig (`_retryable_upstream_error` in `main.py`,
`_dispatch_over` läuft zum nächsten Kandidaten weiter; Streams liefern
Upstream-Fehler dank Early-Open als echte `Response` mit echtem Status).
Damit ist der Client-Impact weg, aber die Kollision selbst bleibt — der Plan
hier behebt die Ursache.

## Idee

Eine **Host-Tabelle**: Hosts sind die physischen Kisten; Backends werden
einem Host zugeordnet. Damit weiß der Gateway, welche Backends
zusammengehören, und kann pro Host **Policies** (den „Handler") anwenden:
Routing-Rücksicht + aktive VRAM-Freigabe.

---

## Phase 1 — Datenmodell + UI (umgesetzt: main.backend_host/rebuild_backends, store `hosts`, /health `hosts`, Backends-Tab)

**Store** (`store.py`): Settings-Key `hosts` nach dem Muster von
`voice_library`/`alias_park`: dict `name → {label, …policy-flags}`
(Flags kommen in Phase 2/3; Phase 1 legt nur Name/Label an).
Backend-Zuordnung: neues Feld `host` im Backend-JSON (Spalte bleibt, wie
alles Backend-Config, im `json`-Blob).

**Ableitung (keine Migration):** der Host wird bei jedem `rebuild_backends()`
frisch berechnet — explizites Feld `host`, sonst `urlparse(url).hostname`
(z. B. `192.168.8.37`), sonst der Backend-Name als Fallback (`main.backend_host`).
Gleiche IP = gleicher Host, ohne dass jemand klicken muss; damit sind k12-gpu
(openai+comfyui) und die evo-Paare sofort korrekt gruppiert. Nichts davon wird
gespeichert, sodass eine geänderte URL sofort regruppiert. In den Store kommt ein
`host` nur, wenn er im Backend-Editor eingetragen wird.

**main.py:** in `rebuild_backends()` zwei Maps ableiten:
`backend_hosts: bid → host` und `host_backends: host → [bids]`. Kein
Route-Index-Impact (Hosts ändern keine Kandidatenmenge in Phase 1).
`/health` bekommt die Host-Gruppierung dazu (Sichtbarkeit).

**UI** (`admin.py`, Backends-Tab): der Host steht in der Sub-Zeile jeder
Backend-Kachel (`· host …`); im Backend-Editor ein Freitextfeld mit
`<datalist>`-Vorschlägen (leer = aus der URL abgeleitet). Darunter, in derselben
linken Spalte, das Hosts-Panel `_hosts_panel`: eine Zeile je Box mit
ComfyUI-Backend (plus jede Box mit gespeicherter Meta, damit eine alte
Einstellung nie uneditierbar wird), mit Mitglieder-Liste, `shared`-Badge und
Badges für jeden vom Default abweichenden Flag-Wert. Ein „Add" gibt es nicht —
Mitgliedschaft wird am Backend gesetzt; `✎` öffnet über `?host=<name>` das
Host-Formular in der Detailspalte, das Label und alle vier Policy-Flags
(Phase 2 + 3) trägt.

## Phase 2 — Routing-Rücksicht („media-busy" Kandidaten ans Ende) (umgesetzt: main._media_busy_hosts + resolve_routes)

**Flag pro Host:** `avoid_llm_during_media` (default **an** für Hosts mit
llama-swap+comfy, sonst irrelevant).

**Mechanik:** `resolve_routes()` (main.py) bewertet pro Kandidat die Live-Flags
(healthy/busy/draining). Ein Chat-Kandidat, dessen Host gerade rendert, wird
**nicht entfernt, sondern ans Ende der Ready-Liste sortiert**. „Rendert" heißt:
`_media_busy_hosts()` liest den Live-Zähler `backend_inflight` für die
ComfyUI-Backends (`comfyui:`-Präfix) und schlägt deren Host in `backend_hosts`
nach — ein laufender Cloud-Task zählt nicht, der belegt keine lokale GPU.
Sortiert wird nach `scheduler.order_ready`, stabil, sodass die Scheduler-Ordnung
innerhalb beider Gruppen erhalten bleibt. Ergebnis:

- Gibt es Alternativen, gewinnen die — die Kollision entsteht gar nicht.
- Gibt es keine, wird der Host trotzdem versucht (best effort); crasht der
  Load, greift das Failover aus dem Sicherheitsnetz, am Ende steht das
  echte 502.

Bewusst NICHT als harter Skip: sortieren statt filtern hält die
Parking-Semantik unangetastet (media-busy ≠ busy; es wird nicht geparkt,
nur umsortiert).

## Phase 3 — Aktive VRAM-Freigabe (umgesetzt: main._free_comfy_vram / _unload_host_llms; siehe Nachtrag 2026-09-05)

Zwei Host-Flags, beide einzeln schaltbar:

1. **`comfy_free_after_job`** (default an für geteilte Hosts): wenn ein
   Media-Job auf dem Host **endet** (done/failed/cancelled), feuert der
   Gateway fire-and-forget `POST {comfy_url}/free`
   `{"unload_models": true, "free_memory": true}` — ComfyUI gibt das VRAM
   frei, der nächste LLM-Load hat Platz. Hook: Job-Abschlusspfad in
   `main.py` (dort, wo der Job-Status finalisiert wird), NICHT im Adapter —
   der Adapter bleibt protokoll-dumm. Kompromiss: unmittelbar
   aufeinanderfolgende Media-Jobs laden das Bildmodell neu (~Sekunden);
   akzeptabel, das Flag kann sonst aus bleiben.
   Auch dieser Free ist inzwischen ein beobachteter: aus Sicht des Job-Pfads
   fire-and-forget (`asyncio.create_task`), im Call selbst aber derselbe
   `_comfy_free(settle_s=30)` mit Nachposten alle 2 s wie beim Free vor dem Job —
   ein einzelner POST landet meist im `gc.collect()`-Fenster des Workers und geht
   verloren. Ein `abort_when` bricht das Nachposten ab, sobald ein neuer Job das
   Backend beansprucht hat: ein Re-Post unter einem gerade gestarteten Prompt
   entlädt genau die Modelle, die er lädt.
2. **`llm_unload_before_media`** (default aus): vor dem Submit eines
   Comfy-Workflows auf dem Host das LLM aktiv entladen, damit die
   Generation nicht ihrerseits an VRAM-Mangel scheitert. **Umgesetzt** in
   `main._unload_host_llms` (Endpoint auf k12-gpu verifiziert: llama-swap
   `GET /unload` → 200, andere Server ignorieren ihn): best effort, 8 s Timeout,
   über alle `openai:`-Geschwister des Hosts, direkt nach dem Claim. Flag bleibt
   default aus — die llama-swap-TTL (120 s) entlädt ohnehin meist zuerst.

Aufwand: mittel (Comfy-`/free`-Hook klein; llama-swap-Unload je nach
Endpoint). Risiko: gering — beide Aktionen sind idempotente Aufräum-Calls,
fire-and-forget mit Log, nie im Request-Pfad blockierend.

### Nachtrag 2026-09-05 — der Free gehört VOR den Job

Flag 1 stellt die Frage zum falschen Zeitpunkt: wenn ein Job endet, ist noch
nicht entschieden, was als Nächstes kommt, also ist jede Antwort geraten. Beim
**Claim** ist sie bekannt. Deshalb ist die Hauptregel jetzt
`comfy_free_before_job` (default an für JEDE ComfyUI-Box, auch dedizierte):
sobald ein Job ein Backend beansprucht (`main._claim_gen_backend`), wird
`POST /free` **awaited** ausgeführt, wenn der **Modellsatz** des Requests ein
**anderer** ist als der zuletzt dort geladene. Der Modellsatz ist nicht der Alias
(der war es zuerst — falsch in beide Richtungen: `trellis2_high`/`_low` laden
dasselbe `TRELLIS.2-4B` und wurden bei jedem Wechsel entladen, ein per Mapping
gewähltes Modell unter EINEM Alias nie), sondern `scheduler.model_set_key`: die
Gewichtsnamen der Loader-Nodes nach Pins/Mapping/LoRAs
(`ComfyUIAdapter.model_set_key`), Alias nur als Fallback. Gleicher Satz → Cache
bleibt; die Alias-Affinität aus `scheduler.designated_taker` bleibt davon
unberührt (sie steuert Zuteilung, nicht das Freigeben). Verglichen wird dabei nicht
der Affinitäts-Key `backend_last_key`, sondern `backend_vram_key` — was die GPU
nachweislich HÄLT, geschrieben aus dem Ergebnis des Free (`_comfy_free` liefert
ein Urteil), nie aus dem Versuch: ein fehlgeschlagener oder wegen eines laufenden
Jobs übersprungener Free hinterlässt `None`, und `None` (unbekannt, gemischt,
Gateway-Neustart — der leert nie ComfyUIs VRAM) zählt als Änderung. „Awaited“
heißt außerdem beobachtet: ComfyUIs `POST /free` setzt nur zwei Flags und
antwortet sofort; der Worker liest sie erst nach seinem nächsten `q.get()`, und
das `notify` geht verloren, während er nach einem Prompt im `gc.collect()`
steckt — genau dann postet der Gateway-Poll. `_comfy_free(settle_s=…)` pollt
deshalb alle 0,5 s `torch_vram_total` aus `/system_stats` und postet alle 2 s
nach, bis der Pool unter die Zielmarke `max(20 % des Ausgangswerts, 256 MiB)`
gefallen ist (sonst nach 30 s Urteil False). Ein Pool, der schon vorher ≤ 256 MiB
hält, gilt als leer (Urteil True); sind die `/system_stats` nicht lesbar, bleibt
es beim alten Fire-and-forget. Review 2026-09-08.

Zwei Dinge, die Flag 1 nicht abdecken konnte:
- **Chain-Stages.** Stage 1 und Stage 2 sind zwei Workflows, oft auf DEMSELBEN
  Backend, ohne Job-Ende dazwischen — vorher lief dort nie ein Free. Gemessen
  2026-09-05 (Job `cc604da29e0e`, Meshy → `mesh-mia` auf k12-gpu): Stage 2 starb
  an `torch.OutOfMemoryError: Tried to allocate 20.00 MiB`, weil der ComfyUI-
  Prozess 21,4 von 23,5 GiB hielt. Der Rig-Node (Make-It-Animatable) läuft in
  einem EIGENEN venv/Prozess und kann diesen Cache prinzipiell nicht mitbenutzen.
- **Free über den Gateway-Neustart hinweg** (siehe oben).

Flag 1 bleibt, aber nur noch für seinen einen eigenen Zweck: eine geteilte Box
soll auch dann freigeben, wenn gar kein Media-Job folgt, sonst scheitert der
nächste llama-swap-Load. Es überspringt den Free zusätzlich dann, wenn der
Scheduler diesem Backend als Nächstes einen Job zuteilen würde, der genau den
Modellsatz will, den die GPU nachweislich hält: gefragt wird
`_designated_gen_waiter` (die EINE Stelle, an der die Zuteilung berechnet wird —
overdue > Typ-Affinität > ältester), und verglichen wird gegen
`backend_vram_key`, nicht gegen den zuletzt gelaufenen Alias. Ein unbekannter
Stand (`None`) gibt unabhängig von Wartenden frei. Ein Scan der ganzen Queue nach
Alias wäre falsch und war es: ein Wartender, der dieses Backend gar nicht nehmen
kann (nach einer fehlgeschlagenen Chain-Stage ausgeschlossen, anderswo
force-gepinnt), unterdrückte den Free dauerhaft — und damit gab auf einer
geteilten Box nichts mehr die GPU frei (Review 2026-09-08).

Pure Entscheidung: `scheduler.free_vram_before_job(last_key, next_key,
others_inflight, enabled)`, getestet in `tests/test_scheduler.py`
(`TestFreeVramBeforeJob`, `TestModelSetKey`) — beide Fehler-
richtungen sind still (zu eifrig = unerklärter Reload, gar nicht = OOM im Node).

## Nicht-Ziele

- Kein VRAM-Messen/-Budgetieren (keine verlässliche Quelle über beide
  Welten; die Policies brauchen es nicht).
- Kein Cross-Host-Scheduling von LLM- gegen Media-Last (Parking + Prioritäten
  existieren und reichen).
- Keine ComfyUI-/llama-swap-Konfigänderung auf den Boxen (deren Setup bleibt
  wie es ist; der Gateway arbeitet drumherum).

## Offene Fragen

1. ~~llama-swap-Unload-Endpoint verifizieren~~ — auf .37 (k12-gpu) erledigt:
   `GET /unload` → 200, andere Server ignorieren ihn. Für .34 nicht nachgemessen,
   das Flag ist dort aus.
2. Sollen `dx10-01/-02` (falls je eine Comfy-Instanz dazukommt) dieselben
   Defaults bekommen? (Host-Tabelle macht das später zum No-Brainer.)
3. Dashboard-Gruppierung nach Host (optional, jederzeit nachrüstbar).
