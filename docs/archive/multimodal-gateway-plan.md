# Multimodal Gateway — Detailplan

Status: Entwurf v1. Erweitert den bestehenden LLM-Gateway um **Bild-, Video- und
TTS-Generierung** über dieselbe Mapping-/Routing-Idee, mit einem **VRAM-Koordinator**
für GPU-geteilte Backends. Quelle der ComfyUI-Logik: `anima-verse` (wird in den
Gateway gezogen, sodass der Gateway der *einzige* Dispatcher auf jede GPU ist).

Leitprinzipien (aus dem Bestand übernommen):
- OpenAI-kompatible Front-API + internes Mapping auf Backends (wie heute bei LLM).
- Hot-reload-Config, modulare Globals, Failover über Priority.
- Minimalismus: kein Plugin-Loader-Framework, sondern ein In-Repo-Adapter-Registry.
  Das Projekt wächst kontrolliert von „zwei Dateien" zu einem kleinen Paket.

---

## 1. Architektur-Spine

```
HTTP  →  Input-Adapter   →  Normalized Request  →  Router            →  VRAM-Koordinator   →  Backend-Adapter  →  Dispatch
         (Plugin-Achse A)   (modalitätsfrei)       (Capability+Alias)   (Queue+Swap+Evict)    (Plugin-Achse B)
                                                          │                     │
                                                    Job-Store (Status + Ergebnis-Cache, TTL)
```

Zwei **Plugin-Achsen** (austauschbar), zwei **protokoll-freie Kerne** (fest):

| Schicht | Rolle | Beispiele |
|---|---|---|
| Input-Adapter (A) | HTTP-Shape → Normalized Request, und Normalized Response → HTTP-Shape | `openai-chat`, `openai-images`, `openai-edits`, `openai-speech`, `native` |
| **Router** (Kern) | Capability-Match + Alias-Auflösung + Failover-Reihenfolge | — |
| **VRAM-Koordinator** (Kern) | Queue pro GPU-Domain, Swap-Lock, Eviction, Recovery | — |
| Backend-Adapter (B) | Normalized Request → Backend-Protokoll, Dispatch, optional `free()`/`vram_domain()` | `openai`, `comfyui`, `a1111`, `localai` |

Input-Adapter und Backend-Adapter sind **unabhängig**: ein OpenAI-Request kann auf
ComfyUI landen (über Übersetzung), ein nativer Request auf ein LLM. Genau wie heute
`/v1/responses → /v1/chat/completions` (`main.py:583`).

---

## 2. Normalized Request / Response

Ein einziges internes Modell für **alle** Modalitäten (Text, Bild, Video, Audio).

```python
NormalizedRequest:
  task:        "chat" | "text2img" | "img2img" | "inpaint" | "img2video" | "tts" | ...
  alias:       str            # das "model"-Feld; resolved per-backend (s. §4)
  inputs:                     # modalitätsabhängige Nutzlast
    prompt, negative_prompt:  str
    text:                     str            # TTS
    reference_images:         list[bytes|url]
    init_image, mask:         bytes|url      # img2img / inpaint
    audio_ref:                bytes|url      # voice clone
  params:                     # normalisierte Knöpfe (s. unten)
    width, height, steps, cfg, seed, sampler, scheduler, denoise
    loras:                    list[{name, weight}]
    extra:                    dict           # workflow-spezifischer Passthrough
  output:
    n:                        int
    format:                   "png" | "jpg" | "mp4" | "wav" | ...
    mode:                     "sync" | "async"
    ttl_s:                    int            # wie lange Ergebnis abholbar bleibt
```

`params.extra` ist der Passthrough-Kanal in die generischen ComfyUI-Injektoren
(`string_inputs` / `float_inputs` / `boolean_inputs` / `lora_inputs`), die der
anima-verse-Adapter heute schon liest. Damit muss das Normalized-Modell nicht jeden
workflow-spezifischen Knopf kennen.

```python
NormalizedResponse:
  job_id:      str
  status:      "queued" | "running" | "done" | "failed" | "expired"
  results:     list[{kind, mime, ref|inline_b64, meta}]   # Bilder/Video/Audio
  usage:       {tokens|frames|seconds, cost_usd}
  error:       optional
```

**Begründung der Param-Schicht:** Die Workflows sind bereits als parametrisierte
Templates gebaut — benannte `PrimitiveInt`/`PrimitiveStringMultiline`-Nodes plus
generische Inputs. Die „Input-API" ist also *normalisiertes Param-Dict + die Knöpfe,
die ein Workflow freilegt*. Wir erfinden nichts, wir normalisieren das Bestehende.

---

## 3. Input-Adapter (Front-APIs)

### 3a. Native API (Ground Truth — volle Fidelity)
```
POST /v1/generations              # universell: task im Body, sync oder async
GET  /v1/jobs/{id}                # Status + Ergebnis-Refs
GET  /v1/jobs/{id}/result/{n}     # rohe Bytes (Bild/Video/Audio)
DELETE /v1/jobs/{id}              # früh aufräumen
```
Ein `POST /v1/generations` mit `output.mode`:
- `sync`  → blockt bis `done`/Timeout, liefert Ergebnis inline (oder 202 + job_id bei Timeout).
- `async` → liefert sofort `{job_id, status:"queued"}`; Client holt per `GET /v1/jobs/{id}` ab.

### 3b. OpenAI-kompatible Shims (Reichweite — dünn, übersetzen in native)
| Endpoint | Deckt ab | Übersetzung |
|---|---|---|
| `POST /v1/images/generations` | text2img | OpenAI-Body → `task:text2img`; `model`→alias; `size`→w/h |
| `POST /v1/images/edits` | img2img / inpaint | `image`+`mask` → `inputs.init_image`/`mask` |
| `POST /v1/audio/speech` | TTS | `input`→`inputs.text`; `voice`→alias/voice-ref |
| (kein Standard) | Video | nur native API |

Die Shims sind reine Übersetzer (wie `responses_to_chat`/`chat_to_responses`,
`main.py:612`/`:679`). OpenAI-Schema ist dünn (kein steps/cfg/sampler/lora) → die
fehlenden Knöpfe bekommen Defaults aus dem Alias/Workflow.

---

## 4. Mapping-Engine (Router)

Erweitert die bestehende Alias-Maschinerie (`virtual_models`, `alias_entry()`
`main.py:313`) von „Alias → real_model" auf „Alias → Workflow + Bindings".

```yaml
virtual_models:
  "flux":                          # ein Bild-Alias
    ct452-comfy:
      task:     text2img
      workflow: text2img_workflow_flux1_api.json
      model:    flux1-dev-Q8.gguf
      priority: 1
    evo-comfy:
      task:     text2img
      workflow: text2img_workflow_flux1_api.json
      model:    flux1-schnell-Q4.gguf
      priority: 5                   # langsamer/iGPU → niedrigere Prio
```

**Capability-Match:** Ein Backend ist Kandidat nur, wenn es
`task` + `workflow` + `model` bereitstellt (Discovery, §6) — die multimodale
Verallgemeinerung des heutigen `real in backend_models`-Checks (`main.py:388`).

`alias_entry()` löst künftig auf `(task, workflow, model, priority_override)` statt
nur `(real_model, priority)`. Die vier Config-Shapes bleiben erhalten.

**Affinity-Routing (neu, billig):** In `get_routes_for` wird die Kandidaten-Sortierung
um einen *Warm-Bonus* ergänzt — ein Backend, dessen GPU-Domain das Modell schon
resident hat, gewinnt gegen eines, das erst swappen müsste. Multi-Node fällt damit
fast geschenkt raus (Routing statt VRAM-Pool).

---

## 5. Job-Modell & Ergebnis-Cache

Jobs sind die natürliche Form für lange Generierungen (Flux ~67 s; Video länger).

**Lifecycle:** `queued → running → done | failed`, danach `done → expired` (TTL).

**Store:**
- Metadaten (job_id, task, alias, backend, status, timing, usage, cost) in SQLite —
  erweitert `stats.py` (gleiche WAL-DB, zero-dep) um eine `jobs`-Tabelle.
- Ergebnis-Bytes auf Disk unter `jobs/<id>/<n>.<ext>` + Manifest. Kein Base64 in der DB.
- **TTL-Pruning** wiederverwendet `stats.prune_loop()` (`stats.py:127`): abgelaufene
  Jobs → Dateien löschen + Zeile markieren. `output.ttl_s` pro Job, Default aus Config.

**Abholung:** `GET /v1/jobs/{id}` (Status + Refs), `GET /v1/jobs/{id}/result/{n}` (Bytes).
Sync-Modus = intern submit + auf `done` warten (Timeout-bounded), dann inline ausliefern —
aber das Ergebnis bleibt **trotzdem** bis TTL abholbar. So ist „sync" nur ein Warte-Shim
über demselben Job-Store; nichts geht doppelt.

---

## 6. Backend-Adapter-Interface

```python
class BackendAdapter(ABC):
    async def health(self) -> bool
    async def discover(self) -> Capabilities      # {tasks, workflows, models, pricing}
    async def dispatch(self, req: NormalizedRequest, on_progress=None) -> NormalizedResponse
    def vram_domain(self) -> Optional[DomainId]   # None = nicht GPU-gemanagt
    async def free(self) -> None                  # optional; Default: NotSupported
```

Registry: `ADAPTERS = {"openai": OpenAIAdapter, "comfyui": ComfyUIAdapter, ...}`,
Auswahl per `type:`-Feld im Backend (Default `openai` → **kein** Verhaltenswechsel
für heutige Backends).

- **OpenAIAdapter:** kapselt das heutige `proxy()` (`main.py:498`) + `/v1/models`-Discovery.
- **ComfyUIAdapter:** Port aus `anima-verse/app/skills/image_backends.py:646`
  (`POST /prompt` → poll `/history/{id}` → `GET /view`), aber:
  - sync `requests` → **async `httpx`** (passt in die Event-Loop, kein Thread-Pool).
  - Node-Injektion per class/title (`_find_node_by_*`) übernehmen.
  - Discovery: `/object_info` (Checkpoints/UNets/LoRAs) + vorhandene Workflow-Dateien.
  - `free()` → `POST /free` inkl. Signature-Skip (`image_backends.py:860`).

---

## 7. VRAM-Koordinator (das Herzstück)

Eine **Contention-Domain** = `(host, gpu_index)`. Mehrere Backends teilen sich eine
Domain; nur eines ist resident.

**Speichermodell (entschieden):** Die Domain ist *immer* eine **VRAM-Budget-
Semaphore** — mehrere Workloads dürfen gleichzeitig resident sein, solange
`Σ Footprints ≤ Budget`. **Mutex (hart exklusiv) ist der Spezialfall „Budget lässt
genau einen zu".** PoC (Phase 2) setzt Budgets konservativ → verhält sich wie Mutex,
aber die Maschinerie ist schon allgemein. Co-Residency = später Footprints messen
(Discovery §13e: Modell laden, VRAM-Delta) und Budget anheben. OOM/Reset-Risiko wird
damit ein **Tuning-Knopf, kein Rewrite**. Begründung: Strix Halo Unified Memory macht
LLM+Flux-Co-Residency real (96 GB UMA), aber gfx1151 ist unter Speicherdruck instabil
→ konservativ starten, scharf schalten wenn Footprint-Zahlen vertrauenswürdig.

### 7a. State-Machine pro Domain
```
IDLE ──load──▶ RESIDENT(x) ──same-workload requests──▶ (bleibt)
   ▲              │
   │          evict/drain
   │              ▼
   └──── SWAPPING ──load──▶ RESIDENT(y)
                   │
            timeout│
                   ▼
                 STUCK ──recovery──▶ RESETTING ──ok──▶ IDLE
```

### 7b. Locking-Ebenen (drei, bewusst getrennt)
1. **Swap-Lock** (`asyncio.Lock` pro Domain) — nur während Evict+Load gehalten, *nicht*
   über den ganzen Request. Löst „67 s die GPU sperren".
2. **Residency-State** — welcher Workload hält VRAM. Treibt Affinity-Routing (§4).
3. **Per-Backend-Inflight** (Bestand, `backend_inflight` `main.py:83`) — Parallelität
   *innerhalb* des residenten Workloads (z. B. ComfyUI-Queue-Tiefe).

### 7c. Queue & Fairness
- Eine **FIFO-Queue pro Domain**; ein Worker drained sie.
- **Min-Residency-Quantum:** ein geladener Workload behält die GPU für ≥`T` s oder
  ≥`N` Jobs, bevor er für den konkurrierenden weicht → verhindert LLM↔Image-Thrashing
  (jeder Swap kostet Ladezeit).
- **Batch-Drain:** alle wartenden Jobs *desselben* Workloads vor dem nächsten Swap
  abarbeiten.
- Serialisierung (Entscheidung des Users): **parken**, wenn kein alternatives
  warmes/freies Backend existiert; **spillen** (Bestand, `main.py:383`), wenn doch.
  Single-GPU-PoC → immer parken (503 wäre unbrauchbar bei 67-s-Bildern).

### 7d. Eviction & Aktivierung — Control-Plane (getrennt vom Datenpfad)
Pro Backend ein `evict`/`activate`-Deskriptor, eine von mehreren Strategien:
```yaml
evict:  { http: "/free" }                 # ComfyUI, LocalAI
evict:  { systemd: "llama@gpu0", host: evo }   # vanilla llama.cpp/vLLM: Prozess stoppen
evict:  { command: "ssh evo 'rocm-smi ...'" }  # generischer Fallback
```
Der Koordinator kennt nur: `evict(backend) -> awaitable, das auflöst, wenn VRAM frei
ist, mit Timeout`. Implementierung dahinter ist Plugin (HTTP/systemd/SSH).

- **llama.cpp/vLLM können nicht self-evicten** → hier zwingend systemd/SSH.
- SSH ist **opt-in pro Host**, privilegierter Remote-Exec → bewusst getrennte
  Control-Plane, nicht in den HTTP-Dispatch gemischt.

### 7e. Verifikation („frei" glauben wir nicht)
`/free`-200 ≠ VRAM ist runter. Vor `RESIDENT(y)` ein **Readback**:
- Primär: `app/core/beszel.py` aus anima-verse, falls es Host/GPU-VRAM schon pollt
  (zu prüfen) → fertige Quelle ohne SSH-`rocm-smi` bei jedem Swap.
- Fallback: `rocm-smi`/`nvidia-smi` via Control-Plane.

### 7f. Recovery
- Jede Evict/Load-Transition hat **Timeout** → bei Hänger Domain → `STUCK`,
  Swap-Lock **immer** freigeben, Job auf anderen Node spillen (falls vorhanden).
- `STUCK` → Recovery-Hook (`systemd restart`, `rocm-smi --gpureset`) → `RESETTING` →
  re-probe → `IDLE`.
- **gfx1151-Reset ist ein legitimer State**, kein Crash — die Domain-Machine kennt ihn.

---

## 8. Config / Topologie

Neuer `topology:`-Block **in `config.yaml`** (erbt Hot-reload gratis), optional via
`!include inventory.yaml` aus einer separaten Datei mergebar.

```yaml
topology:
  hosts:
    evo:   { control: { ssh: "root@192.168.x.x" } }
    ct452: { }                       # dedizierter Bild-Host, kein Contention
  domains:
    evo:gpu0:   { host: evo,   gpu: 0, vram_mb: 96000 }   # Strix Halo unified
    ct452:gpu0: { host: ct452, gpu: 0 }

backends:
  - name: ct452-comfy
    type: comfyui
    url: http://ct452:8188
    domain: ct452:gpu0
    evict: { http: "/free" }
  - name: evo-llama
    type: openai
    url: http://evo:8080
    domain: evo:gpu0
    evict: { systemd: "llama", host: evo }
  - name: evo-comfy
    type: comfyui
    url: http://evo:8188
    domain: evo:gpu0                 # teilt Domain mit evo-llama → Contention!
    evict: { http: "/free" }
```

Backends **ohne** `domain` sind nicht GPU-gemanagt → Koordinator no-op (heutiges
Verhalten, Cloud-APIs, dedizierte Hosts).

---

## 9. Observability

- **Stats-Tab:** Jobs (task, backend, dauer, kosten) zusätzlich zu LLM-Calls.
- **Routing-Tab:** pro Domain den State (`RESIDENT(x)`, `SWAPPING`, queue-tiefe,
  letzter Swap, STUCK/RESETTING) — die multimodale Erweiterung der heutigen
  busy/shadowed-Anzeige.
- `/health`: Domain-States im Snapshot (`routing_snapshot()` `main.py:429`).

---

## 10. Phasenplan (Abhängigkeits-Reihenfolge)

| Phase | Inhalt | Ziel-Host | De-Risk |
|---|---|---|---|
| **0 — Refactor** | Adapter-Interface (A+B) einziehen, `OpenAIAdapter` kapselt heutiges `proxy()`. NormalizedRequest intern. **Kein** Verhaltenswechsel. | — | Seams ohne Risiko |
| **1 — Bild ohne Contention** | `ComfyUIAdapter` (async-Port), native `/v1/generations` + Job-Store + Ergebnis-Cache (sync+async+TTL). text2img zuerst. **Kein** VRAM-Koordinator. | **CT452** (dediziert) | Bild-Pfad end-to-end |
| **2 — VRAM-Koordinator** | Domain-State-Machine, Swap-Lock, Queue, Quantum, `/free`+systemd-evict, Readback. ComfyUI ↔ LLM auf **einer** iGPU. | **EVO** (PoC) | Das harte Herzstück isoliert |
| **3 — Multi-Node** | Affinity-Routing + Cross-Node-Failover/Spill, Topologie aus inventory. | + **K12** | Routing statt VRAM-Pool |
| **4 — Breite** | img2img/inpaint, img2video, TTS-Adapter/Workflows; OpenAI-Shims (`/v1/images/*`, `/v1/audio/speech`). | alle | Modalitäten-Fan-out |
| **5 — Hardening** | gfx1151-Reset, Stuck/Deadlock-Timeouts, Recovery-Hooks, Dashboard-States. | alle | Produktionsreife |

Schnitt-Logik: **Refactor → Bild läuft (ohne Contention) → Contention auf einem Host →
Multi-Host → Breite → Hardening.** Jede Phase ist für sich lauffähig und testbar
(curl gegen Endpoints, wie im Bestand).

**Parallel-Track UI/Store (§13):** Phasen 0–2 bootstrappen weiter aus YAML — die
Config-Komplexität ist da noch tragbar. Der Store-als-Wahrheit + Admin-UI-Track
landet ab **Phase 3** (Multi-Node = wo Topologie + viele Mappings YAML sprengen).
Discovery-Endpoints (§13e) entstehen ohnehin schon in Phase 1 (ComfyUI-Discovery) —
die UI konsumiert sie nur, baut sie nicht neu.

| Phase | UI/Store-Anteil |
|---|---|
| 0–1 | Discovery-Endpoints als JSON-API (für später + Debugging). YAML bleibt Wahrheit. |
| 2 | Read-only Domain-States im Routing-Tab (Bestand erweitert). |
| 3 | Store wird Wahrheit; Admin-UI (HTMX) für Wiring + Validierung; YAML-Import/Export. |
| 4 | Dry-Run-Validierung pro Mapping; Discovery-gestützte Auswahllisten. |

### Phase 3 (vorgezogen) — UI erste Scheibe: ✅ umgesetzt & verifiziert (gegen Stub)

Vorgezogen, weil der User vor einer UI nicht testet (Feedback-Schleife). Eine
*minimale, aber durchgängige* Generierungs-Konsole unter **`/ui`** (am Hauptserver
gemountet), plain server-rendered (kein JS/Build/Deps; GET-Forms → kein
`python-multipart`).

- **`store.py`** (neu): schreibbarer Generierungs-Alias-Store (SQLite), Bootstrap-
  Import aus `image_models`, danach Source of Truth (§13b). `get_gen_routes` liest
  von hier (Fallback Config).
- **`admin.py`** (neu): Übersicht (Aliase), **Workflow registrieren → introspizieren
  → Mapping editieren** (vorbefüllt via `suggest_mapping`), **Playground** (Generierung
  ausführen → Bild anzeigen). `register(app)` statt `include_router` (Letzteres ist in
  der hiesigen starlette-Version 1.3.1 kaputt).
- `main.py`: `run_generation`-Kern (geteilt von HTTP-Endpoint + Playground); Store-
  Init/Bootstrap im Lifespan; `admin.register(app)` + `admin.bind(...)`.

Verifiziert (Stub + TestClient): Übersicht, Introspektion (Node-Tabelle + Mapping
node 2 vorbefüllt), Save (explizites Mapping persistiert), Playground-Generierung
(Job done), Bildabruf (PNG). Real-uvicorn: `/ui` mountet, graceful „not enabled"
ohne `image_models`.

**Bewusst deferred (Phase-3-Härtung):** Admin-Auth/Multi-User (UI aktuell offen),
HTMX-Partial-Updates + Live-Validierung, Backend-/Topologie-Verwaltung, Dry-Run.

**Erweiterung (verifiziert gegen echtes CT452 + Flux):**
- **Reitersystem** (`admin.py` `TABS`): Backends · Input · Mapping · Playground ·
  Statistic · Users · Server. Backends/Input/Mapping/Playground mit Inhalt;
  Statistic/Users/Server als vorgesehene Stubs.
- **Registrierung v2** (User-Vorgabe „API-JSON + Mapping, Gateway besitzt sie"):
  API-JSON **einfügen/Share-Pfad** → `workflow_json` im Store (unabhängig von
  GUI-Änderungen) → Auto-Suggest Request-Mapping + Auto-Detect Modell-Slots →
  Edit mit **Discovery-Dropdowns** aus `/object_info`, die stale Modellnamen
  flaggen (`fixed`-Bindings). POST dependency-frei via `parse_qs`.
- Echte Bilder durch beide Pfade erzeugt (Flux2 ~66–71 s); Auto-Mapping trifft die
  realen `input_*`-Workflows; „von der Instanz laden" via `/history` als gültiger
  Importweg bestätigt.

### Phase 1 — Status: ✅ umgesetzt & verifiziert (gegen Stub)

`adapters.py`: `ComfyUIAdapter` (`type: comfyui`) — `discover()` via `/object_info`,
`generate()` als async Submit (`POST /prompt`) → Poll (`/history/{id}`) → Fetch
(`/view`); **mapping-getriebene** Injektion: `_apply_mapping(wf, mapping, values)` setzt pro
logischem Param `workflow[node].inputs[field]` — **konventionsfrei**, funktioniert mit
jedem Workflow. Die Bindungstabelle kommt explizit aus dem `image_models`-Candidate
(`mapping:`) bzw. später aus der Registrierungs-UI; fehlt sie, liefert `suggest_mapping()`
(die alte Konventionsheuristik, jetzt nur noch **Auto-Vorschlag/Fallback**) eine
Default-Bindung. Explizit gewinnt immer. `params.extra` ist generischer Passthrough für
workflow-spezifische Knöpfe. `BackendAdapter.dispatch` ist jetzt Default-Raise, neuer `generate()`-
Hook; `NormalizedRequest` um `task/inputs/params/output/workflow` erweitert; neue
`GenBlob`/`GenOutput`.

Neue Datei **`jobs.py`**: eigenständiger Job-Store (SQLite-Metadaten + Blobs auf
Disk, `owner` schon dabei für Phase-3-Multi-User, TTL-Prune). Lifecycle
queued→running→done|failed→(expired).

`main.py`: `image_models`-Resolver (`get_gen_routes`, **bewusst getrennt** vom LLM-
Router), native **`POST /v1/generations`** (sync-wait + async/202), Retrieval
**`GET /v1/jobs/{id}`** + **`/result/{n}`** (FileResponse); Job-Store-Init + Prune
im `lifespan` (auto-on bei `image_models`). `config.example.yaml` dokumentiert
`type: comfyui`, `image_models`, `jobs`.

Verifiziert (Stub-ComfyUI): Injektion, discover (2 Modelle), generate (PNG-Bytes,
Inflight→0), Job-Store CRUD + Disk-Persistenz, voller HTTP-Pfad (sync 200 / async
202→done / Result-Fetch / unknown-model 503). **Noch nicht:** echtes CT452, img2img/
video/tts, OpenAI-Shims, VRAM-Koordinator (Phasen 2/4).

### Phase 0 — Status: ✅ umgesetzt & verifiziert

Neue Datei **`adapters.py`**: `BackendAdapter` (ABC), `OpenAIAdapter` (1:1-Umzug
von `proxy()` + dem `/v1/models`-Discovery-Cluster), `AdapterContext` (DI, hält
Adapter import-cycle-frei und hot-reload-sicher), `NormalizedRequest` (Phase-0-dünn,
trägt den OpenAI-Body verbatim), `Capabilities`, Registry `ADAPTERS` + `make_adapter`.

`main.py`: `proxy()` und der Discovery-Helper-Block entfernt (−101 Zeilen netto);
`refresh_backend()` delegiert Discovery an `adapter.discover()` (Status/Logging/State
bleiben hier); `route()` dispatcht via `backend_adapters[name].dispatch(req)`;
`backend_adapters` wird bei Start + jedem Config-Reload via `build_backend_adapters()`
neu gebunden. `/v1/responses` bleibt unangetastet (Input-Achse, Phase 1+).

Verifiziert: Boot + `/health`/`/v1/models`/Auth-401/503-Routing identisch; isolierter
Adapter-Test grün für Non-Stream + **Streaming** (Inflight-Dekrement erst im `finally`
nach Drain), Stats-Recording mit korrekten Token-Counts, Header-Stripping. **Kein
Verhaltenswechsel** im LLM-Pfad.

---

## 11. Offene Punkte

**Entschieden (eingearbeitet):**
- **Co-Residency** → VRAM-Budget-Semaphore, Mutex als Spezialfall, PoC konservativ (§7).
- **anima-verse** → bleibt unangetastet/separat; rein operative Topologie-Regel:
  Gateway muss alleiniger Dispatcher der von ihm koordinierten GPU sein (§12).
- **YAML-Rolle** → Bootstrap-Import + manueller Export, Store ist Wahrheit (§13b).
- **Auth** → Multi-User-Daten-Ebene + Single-Admin + Job-Ownership (§13c).

**Noch offen (für später, bewusst nicht jetzt entschieden):**
1. **inventory.yaml-Schema** — eigene Datei vs. `topology:`-Block; wird mit Store
   ohnehin zum Bootstrap-/Export-Format (§13b).
2. **Exakte Evict-Contracts** — llama-swap-TTL vs. `/unload`; LocalAI load/unload-URLs;
   `comfy-free-then-exec` (Skript oder HTTP?).
3. **beszel als Readback-Quelle** — pollt es schon GPU-VRAM? (zu prüfen)
4. **Progress-Streaming** — ComfyUI hat WebSocket-Progress; an Clients durchreichen
   (SSE) oder reicht Job-Polling?

---

## 13. Admin-UI & Config-as-Data

Mit Topologie, Domains, Evict-Deskriptoren und Workflow-Bindings ist
hand-editiertes YAML nicht mehr administrierbar. Source of Truth verschiebt sich,
**aber** der Discovery-Reichtum macht die UI kleiner als sie klingt.

### 13a. Deklariert vs. abgeleitet
- **Hand-registriert (nicht discoverbar):** Backend-URLs, Credentials, SSH/Control-
  Plane-Zugänge, Host-Namen. Wenige Felder.
- **Discovered:** Modelle/Checkpoints/LoRAs, Workflow-Param-Surface, Workflow-
  Modellbedarf, GPU/VRAM-Topologie, Capability, Pricing.
- **Gewired + validiert (der eigentliche UI-Job):** Alias → Workflow + Binding,
  Backend → Domain, Evict-Strategie. Klein, aber fehleranfällig.

→ Die UI ist primär eine **Wiring-/Validierungs-Oberfläche über Discovery-Fakten**,
kein Formular-Friedhof.

### 13b. Source of Truth (entschieden)
**Store wird alleiniger Writer-of-Record; YAML = Bootstrap-Import + manueller
Export.** Bei leerem Store seedet YAML einmalig; danach ist der Store Wahrheit, YAML
wird zur Laufzeit ignoriert. Export ist eine separate „dump current state"-Aktion
(→ git-commit für Review/Disaster-Recovery). **Kein** bidirektionaler Dual-Betrieb
(zwei Schreiber = Round-Trip-/Merge-Hölle) und **kein** GitOps (würde „UI
administriert" widersprechen). Nebeneffekt: Hot-reload wird *einfacher* — ein
Store-Write triggert Reload in-process, `watch_config_loop()` (File-Watch) entfällt;
nur die Startup-Ausnahmen (`stats.enabled`, Ports) liest der Bootstrap weiter.

### 13c. Auth & Multi-User (entschieden)
Zwei getrennte Auth-Ebenen:
- **Daten-Ebene → Multi-User.** Statt eines geteilten `api_key` eine `users`-Tabelle
  (id, name, key-hash, role, quota, enabled) im Store. Daten-Auth löst Bearer-Token →
  User-Identität auf (ersetzt `check_auth` `main.py:304`). Rückwärtskompatibel: ein
  einzelner konfigurierter `api_key` bleibt als „Default-User" gültig.
- **Admin-Ebene → Single-Admin.** Eigener Admin-Credential (Rolle `admin`), getrennt
  vom Daten-Key, gated UI + Admin-API. Keine Admin-Delegation („nicht multi-admin").

**Job-Ownership ist Security-Pflicht, nicht Kür:** der abholbare Ergebnis-Cache (§5)
ist per `job_id` adressierbar → ohne Besitzer-Prüfung kann jeder fremde Generierungen
abholen (ID-Raten = Daten-Leak). Daher `jobs.owner`-Spalte; `GET /v1/jobs/{id}` prüft
Eigentum. Nebengewinn: `source` (`main.py:486`) wird von fälschbarem `X-Source`-Header
zu **authentifizierter Identität** → vertrauenswürdige Stats/Kosten pro User, plus
Per-User-Quota/Rate-Limit als natürliche Erweiterung. Landet mit Store/UI in Phase 3.

### 13d. UI-Technologie
**HTMX (eine statische JS-Datei, kein Build-Step).** Partial-Updates, Python
rendert weiter HTML, Validierung server-seitig mit Feld-Re-Render. Vereinbar mit
„zero *Python*-deps". Ort: der Dashboard-Server (Port 4001) wächst zur Admin-
Konsole — aber raus aus `stats.py` in ein eigenes `admin.py` (sonst „self-contained
stats" verletzt). Alternativen verworfen: SPA (Build-Step, bricht Minimalismus),
reines plain-HTML (Discovery/Live-Validierung ohne JS zu zäh).

### 13e. Discovery-Mechaniken
| Discovery | Quelle | Liefert |
|---|---|---|
| Backend-Typ | `/v1/models` vs ComfyUI `/system_stats` vs A1111 | `type:` auto |
| Modelle/Checkpoints/LoRAs/VAE | ComfyUI `/object_info`, OpenAI `/v1/models` | Auswahllisten |
| Workflow-Param-Surface | Workflow-JSON → benannte `Primitive*`-Nodes | freigelegte Knöpfe |
| Workflow-Modellbedarf | Loader-Nodes (`UNETLoader`, `CheckpointLoaderSimple`, `CLIPLoader`, `VAELoader`, `Lora*`) | benötigte Dateien |
| GPU/Host-Topologie | Control-Plane `rocm-smi`/`nvidia-smi` | Domains + VRAM auto |
| VRAM-Footprint | Modell laden, Delta messen | Co-Residency-Budget |
| Capability | aus Workflow-Task | text2img/img2video/tts |
| Pricing | `/v1/models` (Bestand) | Kosten |

Schlüssel-Kombi: **Param-Surface + Modellbedarf** → beim Wiren eines Alias sofort
„Workflow legt w/h/steps/prompt + 2 LoRA-Slots frei, braucht `flux1-dev.gguf` +
`t5xxl.safetensors` — auf `ct452-comfy` ✓, auf `evo-comfy` fehlt `t5xxl` ✗".

### 13f. Validierungs-Mechaniken
- Routing-Erreichbarkeit: jeder Alias ≥1 routebares Backend (Error).
- Mapping-Integrität: Workflow existiert/parst; benötigte Modelle present.
- Param-Kompatibilität: API-Knöpfe ⊆ Workflow-Surface.
- Domain-Referenzen existieren + passen zur entdeckten GPU.
- Control-Plane-Live-Test: `/free`→200? systemd-Unit da? SSH connectet? Readback live?
- Kollisionen: `alias_model_conflicts()` (`main.py:395`) als UI-Befund.
- **Dry-Run:** Mini-Test-Job durch die ganze Kette als grüner Haken pro Mapping.
- Severity Error vs Warning (verallgemeinert `covered`/`shadowed` des Routing-Tabs).

---

## 12. Risiken

- **Swap-Latenz dominiert UX** auf der geteilten iGPU → Quantum/Batch-Drain sind nicht
  optional, sondern Kern.
- **SSH-Control-Plane** = Security-Fläche + Single Point of Failure für Recovery.
- **Refactor-Regression** (Phase 0) im LLM-Pfad → Adapter muss `proxy()` 1:1 kapseln,
  inkl. Streaming-Inflight-Dekrement im `finally` (`main.py:524`).
- **Fremder Dispatcher auf koordinierter GPU** (z. B. anima-verse, das separat bleibt)
  hebelt den Koordinator aus — er sieht die fremde VRAM-Belegung nicht → OOM. Operative
  Regel, kein Code-Thema: pro koordinierter GPU ist der Gateway alleiniger Dispatcher;
  Mitnutzer zeigen ihre Backend-URL auf den Gateway *oder* nutzen unkoordinierte GPUs.
- **Job-Result-Cache als Leak-Fläche** ohne Ownership → durch Multi-User + `jobs.owner`
  entschärft (§13c); bei Implementierung nicht vergessen.
