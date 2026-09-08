# Next Steps — ausgearbeiteter Plan

Stand 2026-06-26: Chat/LLM-Gateway ist auf Prod (`192.168.8.10:4000`) produktiv,
multi-user, authentifiziert, store-getrieben. Bild/Multimodal: Mechanismus (Phase
0/1) vorhanden, Verdrahtung läuft (User). Dieser Plan ordnet die offenen Arbeiten
in ausführbare Schritte — Aufwand (S/M/L), Risiko, Ziel-Datei, Verifikation.

Reihenfolge-Empfehlung am Ende.

> **Status 2026-06-28 — ERLEDIGT seit Planerstellung:** A1 · A2 · A3 · E1 · E2 · E3 ·
> E4 · F0–F5 (Call-Parking sync+async) · C4 (OpenAI-Image-Endpoints) · G1 (Image-Job-
> Viewer) · **G2** (LoRA-Discovery-Dropdowns + bounded number inputs). Generation-
> Parking (busy→queue), `reconcile_orphans()`. **Multi-User** scharf: /v1/models nach
> Allow-List gefiltert, Allow-List inkl. ganzer Backends, `?type=` Filter, Key-Generator.
> **LoRA**: Cascade in freie Slots, **LoRA-bewusstes Backend-Routing**, Discovery von
> installierten LoRAs, `GET /v1/generations/{alias}/loras`. **Inpaint**: `LoadImageMask`
> als Upload, per-Slot `required` (kein 8×8), OpenAI-`mask`-Feld. **Bild-Pipeline-
> Härtung** (war „D"): Job-Cancel + ComfyUI-Interrupt, kurzer Read-Timeout, lesbare
> ComfyUI-Fehler. Pin-Schutz (API kann gepinnte Nodes nicht überschreiben).
> Doku komplett überarbeitet (README/CLAUDE/config/anima-versa).
>
> **Status 2026-07-01 — seither umgebaut/ergänzt:**
> - **Media-Pipeline:** Generierung liefert **Bild/Video/Audio** end-to-end — Adapter
>   scannt alle ComfyUI-Output-Keys (SaveVideo etc.), `kind`+`mime` durch die API,
>   `<video>`/`<audio>` im Playground/Job-Detail. Tabs umbenannt **Image → Media**.
> - **Parking neu (ersetzt F-Design):** Sync-Park ist jetzt **Default** (kein Client-Feld),
>   faire FIFO-Queue, **per-Alias `park_s`** (+ globaler Default 60s im Server-Tab,
>   `max_parked`), „Parked calls"-Panel im Dashboard, Timeout → **503** + `Retry-After`.
>   Das alte `park:"sync"/"async"`-Client-Feld ist **entfernt**.
> - **Async = offizieller OpenAI-Responses-Background-Mode:** `POST /v1/responses
>   {background:true}` → `resp_…` queued → `GET /v1/responses/{id}` poll → `POST …/cancel`.
>   non-background `/v1/responses` sync-parkt jetzt auch; `X-Park-Mode` wird gestrippt.
>   Streaming-Reconnect für Background bewusst vertagt (poll-only).
> - **Konsole:** Recent-LLM-Calls (running+5min) + eigener **LLM-Calls**-Tab, farbige
>   Backend-Typ-Chips, `req·1h`-Spalte, klick-sortierbare Tabellen, Offline-Backends
>   im Dashboard ausgeblendet.
>
> **Offene Spitze:** C1 (weitere img2img/inpaint-Aliase verdrahten — Mechanismus steht) ·
> C2 (img2video: Mechanismus steht, WAN-Workflow user-verdrahtet) · C3 (TTS/Audio —
> Mechanismus steht, Workflow offen) · B (VRAM-Koordinator, nur bei GPU-Co-Residency) ·
> D4 (Audit-Log, Master-Key-Rotation) · llama-swap Load-/Switch-Erkennung (via
> `/api/events`, vertagt).

---

## A. Bestand härten (billig, kein neuer Host nötig)

> Das sind die in der Review als „technische Lücken" gelisteten Punkte. Sie waren
> **bewusst deferred** (CLAUDE.md + plan §11: „SSE-Translation not implemented",
> „Responses Phase 1+") solange der Multimodal/Multi-User-Track Vorrang hatte —
> **nicht blockiert, nur nicht eingeplant.** Jetzt einplanbar.

### A1 — Streaming-Token-/Kosten-Erfassung · S · Risiko niedrig
- **Problem:** Bei `stream:true` erfasst `record_call` 0 Tokens (Backends lassen
  `usage` in Stream-Chunks weg) → Stats/Kosten für Streaming-Verkehr unvollständig.
- **Lösung:** beim Forwarden eines Streams `stream_options:{include_usage:true}`
  in die Backend-Request injizieren (OpenAI-Standard → finaler Chunk trägt `usage`).
  Im Streaming-Generator von `OpenAIAdapter.dispatch()` den letzten `usage`-Chunk
  abgreifen und im `finally` an die Stats-Senke geben statt `0`.
- **Fallback:** Backends ohne `stream_options`-Support → bleibt 0 (graceful);
  optional grobe Schätzung (`len(text)//4`) hinter einem Flag.
- **Datei:** `adapters.py` (`OpenAIAdapter.dispatch`, Stream-Pfad), evtl.
  `main.py` Record-Aufruf. Kein DB-Schema-Change.
- **Verifik.:** `curl` mit `stream:true` → Stats-Tab zeigt Tokens > 0 + Kosten.

### A2 — `/v1/responses` adapter-routed · M · Risiko mittel
- **Problem:** Der Responses-Handler hat eine **eigene** Backend-Schleife (ruft
  Backends direkt), statt `backend_adapters[bid].dispatch()` wie `route()` →
  doppelte Routing-Logik, Inflight/Busy-Spill nicht konsistent.
- **Lösung:** nach `responses_to_chat()` die übersetzte Chat-Body durch denselben
  `route()/dispatch()`-Pfad schicken, Antwort via `chat_to_responses()` zurück →
  **eine** Routing-Quelle. `get_routes_for()` wird ohnehin schon geteilt.
- **Datei:** `main.py` (`/v1/responses`-Handler).
- **Verifik.:** Responses-Call routet/failover wie Chat; `/health`-Inflight zählt.

### A3 — Responses SSE-Streaming · M–L · Risiko mittel
- **Problem:** `stream:true` auf `/v1/responses` wird still gedowngradet; keine
  SSE-Übersetzung (CLAUDE.md).
- **Lösung:** Chat-Stream-Chunks → Responses-Event-SSE mappen
  (`response.output_text.delta`, `…completed` etc.). Baut auf **A2** auf.
- **Datei:** `main.py` (neuer SSE-Translator).
- **Aufwand-Hinweis:** aufwändigster der drei (Responses-Event-Schema ist
  verbose). **Nur** lohnend, wenn echte Responses-Streaming-Clients existieren.
- **Verifik.:** SSE-Client gegen `/v1/responses` mit `stream:true` bekommt Deltas.

---

## B. Phase 2 — VRAM-Koordinator (das Herzstück)

> **Trigger:** erst nötig, wenn ComfyUI und LLM sich **eine** GPU teilen
> (Co-Residency, iGPU). Solange Bild- und LLM-Backends auf **getrennten** Hosts
> laufen → überspringbar. De-Risk laut Plan: isoliert auf **EVO** als PoC,
> konservative Budgets (verhält sich wie Mutex). Bausteine (plan §7):

- **B1 — Domain-State-Machine pro GPU** (`idle / llm-resident / comfy-resident /
  swapping`). Read-only zuerst im Routing/Dashboard-Tab sichtbar machen.
- **B2 — Drei Lock-Ebenen** (bewusst getrennt): Swap-Lock (Mutex pro GPU),
  VRAM-Budget-Semaphore, Request-Inflight (existiert bereits).
- **B3 — Evict-Contracts konkretisieren** (offener Punkt §11.2): llama-swap-TTL
  vs. `/unload`; LocalAI load/unload-URLs; ComfyUI „free-then-exec" (Skript o. HTTP).
- **B4 — Queue & Fairness + Quantum** (Anti-Starvation beim Domain-Wechsel).
- **B5 — Readback/Verifikation** („frei glauben wir nicht", §7e): VRAM-Poll —
  klären ob **beszel** schon GPU-VRAM liefert (§11.3), sonst eigener Probe.
- **B6 — Recovery** (§7f): Stuck-Swap-Timeout, gfx1151-Reset-Hook.

---

## C. Phase 4 — Breite (Modalitäten-Fan-out)

- **C1 — img2img / inpaint:** ComfyUIAdapter hat den Bild-Upload-Pfad schon
  (`PLACEHOLDER_SENTINEL` / `UPLOAD_SENTINEL`) → nur Workflows + Mapping nötig.
- **C2 — img2video-Workflows.**
- **C3 — TTS-Adapter + Workflows.**
- **C4 — OpenAI-Shims:** `/v1/images/generations`, `/v1/images/edits`,
  `/v1/audio/speech` → übersetzen auf `adapter.generate()`. Reichweite für
  Standard-OpenAI-Clients ohne native `/v1/generations`.

---

## D. Phase 5 — Hardening (Produktionsreife)

- **D1 — Stuck/Deadlock-Timeouts** (Jobs + Swap-Lock).
- **D2 — Recovery-Hooks** (Backend-Flap, GPU-Reset).
- **D3 — Dashboard-Domain-States:** VRAM/Domain pro GPU sichtbar (erweitert den
  bestehenden Dashboard-Tab).
- **D4 — Auth-Härtung Reste:** per-User Rate-Limit (Quota-Feld existiert schon),
  Audit-Log, optional Master-Key-Rotation im /ui → Server.

---

## E. Statistik & Quota (User-getrieben, near-term)

> Befundlage geprüft am 2026-06-26 — was schon da ist und was fehlt.

### E1 — Cost-/Credit-Quota pro User durchsetzen · S–M
- **Stand:** Feld `quota_cost_month` existiert im User-Schema (`store.py`), wird
  aber **nicht durchgesetzt** — `gate_request()` prüft nur `quota_req_day`
  (Requests/Tag, In-Memory-Zähler). Auch **nicht im User-Formular** (nur req/day).
- **Mechanik-Hürde:** Kosten stehen erst **nach** dem Call fest. Cost-Quota =
  Pre-Call-Check der **Monatssumme**: `SELECT SUM(cost_usd) FROM calls WHERE
  source=<user> AND ts ≥ Monatsanfang` (stats führt `source`=User + `cost_usd`
  bereits) vs. `quota_cost_month` → 429/402. Streaming-Kosten brauchen **A1**,
  sonst zählt Stream-Verkehr als $0.
- **Datei:** `main.py` (`gate_request`), `admin.py` (User-Formular: Feld
  `quota_cost_month`).
- **Verifik.:** User mit `quota_cost_month=0.01` → nach 1–2 paid-Calls 429/402.

### E2 — Model-/Alias-Suche im Statistic-Tab · S
- **Stand:** Das alte Standalone-Dashboard (`stats.py`, :4001) hatte eine
  Filter-Box („filter aliases, models, hosts…"); der neue Admin-`statistic`-Tab
  rendert `stats.summary()` **ohne** Suche → verloren.
- **Lösung:** client-seitigen Filter (wie im alten Dashboard) über die
  by_model/by_backend/recent-Tabellen wieder einsetzen.
- **Datei:** `admin.py` (`statistic_page`).

### E3 — Recent calls: vollständig speichern + TTL-Prune (wie Bilder) · M
- **Stand:** nur `req_preview` (50+50 Zeichen, `_preview()`), keine Response.
  TTL-Prune existiert schon (`stats.prune_loop(retention_days)`).
- **Lösung:** vollen Request **(und Response)** speichern; bestehender Prune
  löscht nach Retention — „gleicher Mechanismus wie beim Bild".
- ⚠️ **Größe:** Vision-Calls tragen base64-Bilder → DB-Bloat. Daher wie `jobs.py`:
  große/binäre Bodies **on-disk** (Blob-Dir + TTL), nur Referenz in der DB; kleine
  Text-Bodies inline. UI: aufklappbarer Full-Body in „Recent calls".
- **Datei:** `stats.py` (Schema + Record + Prune + evtl. Blob-Dir), `admin.py`.

### E4 — Per-User-Statistik (Drilldown) · S–M
- **Stand:** `source` **ist** schon der authentifizierte User (`_source_of`), und
  es gibt bereits eine „By user / source"-Aggregat-Tabelle. Fehlt: Drilldown.
- **Lösung:** User-Auswahl/Filter → dessen Calls + Kosten/Tokens/Latenz gefiltert
  (eigene Detailansicht oder Filterparameter `?user=`). Baut auf vorhandenen Daten.
- **Datei:** `stats.py` (gefilterte Queries), `admin.py` (`statistic_page`).

---

## F. Call-Parking (Queue statt sofortigem Busy/503)

> **Problem:** Sind alle den Alias mappenden Backends am Inflight-Cap, filtert
> `get_routes_for()` sie alle raus → `route()` wirft sofort
> `503 "No healthy backend"`. Statt abzuweisen → den Call **parken**, bis ein Slot
> frei wird. Zwei Modi, Client wählt: **async** (Job-ID + Polling) und **sync**
> (HTTP offen halten bis Dispatch). Gilt für `/v1/chat/completions`,
> `/v1/completions`, `/v1/embeddings` (und später `/v1/responses`).

### F0 — „busy" sauber von „kein Backend" trennen · S
- Heute kann `route()` „alle busy" nicht von „kein Backend mappt/healthy"
  unterscheiden (beide → leere Kandidatenliste). Einführen: Kandidaten-Resolver
  liefert beides getrennt, z.B. `{ready:[...], busy:[...]}`. **Nur** wenn
  `ready` leer **und** `busy` nicht leer → parken; sonst weiter `503`.
- **Datei:** `main.py` (`get_routes_for` / neuer `resolve_candidates`).

### F1 — Park-Queue + Slot-Signal · M
- Der Inflight-Counter wird beim Completion dekrementiert (existiert bereits,
  inkl. Stream-`finally`). Dort ein **`asyncio.Condition`** (pro Backend oder
  global) feuern → wartende Parker aufwecken. **FIFO**-Queue (optional nach
  Priorität), mit `max_parked`-Obergrenze (darüber → `503`, kein unbegrenztes
  Backlog).
- **Datei:** `main.py` (Queue + Condition, Hook am Inflight-Dekrement),
  `adapters.py` (Dekrement-Stelle signalisiert).

### F2 — Sync-Modus · M
- Request blockiert auf der Condition bis (a) ein passender Backend frei wird →
  normaler Dispatch + Antwort, oder (b) `park_timeout_s` → `504`. Keine
  Persistenz nötig. Streaming im Park-Fall: erst **nach** Dispatch normal
  streamen (während des Wartens keine Bytes) — Detail beim Bau bestätigen.

### F3 — Async-Modus (Job-ID) · M–L
- Sofort `202 {job_id}` zurück. **Reuse `jobs.py`** (Lifecycle
  queued→running→done/failed→expired + TTL existiert schon für Bilder; `owner`
  für Multi-User auch). Ein Worker zieht geparkte Chat-Jobs, dispatcht bei freiem
  Slot, speichert die Completion als Job-Ergebnis. Abruf via `GET /v1/jobs/{id}`.
- **Generalisierung:** `jobs.py` ist heute bild-spezifisch (Datei-Artefakte). Für
  Chat: Task-Typ `chat` neben `image`, Ergebnis = JSON-Body (inline/Blob).
- **Datei:** `jobs.py` (Task-Typ `chat`), `main.py` (Park-Worker + 202-Pfad).

### F4 — Modus-Wahl + Default · S
- Client wählt per Request-Feld/Header, z.B. `{"park":"sync"|"async"|false}` oder
  `X-Park-Mode`. **Default** = heutiges Verhalten (kein Park → `503`) für
  Rückwärtskompatibilität; per-Alias/Server-Default im Store konfigurierbar.
- **Empfehlung:** Body-Feld `park` (sichtbar, loggbar) + Server-Default-Setting.
- **Datei:** `main.py`, `store.py` (Settings), `admin.py` (Server-Tab-Default).

### F5 — Beobachtbarkeit · S
- Geparkte Anzahl pro Backend/Alias in `/health` + Dashboard (erweitert den
  bestehenden busy/inflight-Block). Park-Wartezeit in den Stats erfassbar.

**Aufwand gesamt:** M–L. **De-Risk:** erst **F0+F1+F2** (Sync-Park hinter Flag,
ohne Job-Store), dann **F3** (Async/Job-Park). **Abhängig:** baut auf bestehender
`jobs.py` + Inflight-Mechanik; profitiert von **A1** für korrekte Park-Stats.

---

## G. Image-UX & Workflow-Features (nach C4)

### G1 — Image-Job-Viewer: Input + Output in der UI · M · ✅ ERLEDIGT (2026-06-27)
- **Wunsch (User):** gegenerierte Image-Jobs später — **innerhalb der Vorhaltezeit
  (Job-TTL)** — in der UI ansehen: **Input** (Prompt, negative_prompt, Params,
  **Referenzbilder**) **und Output** (die erzeugten Bilder).
- **Umgesetzt:** `jobs.set_inputs()`/`input_path()` persistieren Prompt/Params inline
  in `meta.inputs` + Referenzbilder als on-disk-Blobs (`in_<n>.<ext>`); `jobs.complete()`
  **merged** Meta jetzt (Inputs überleben den Completion-Write). `main.py`:
  `_job_view` liefert `inputs`+`input_images`, neuer Endpoint
  `GET /v1/jobs/{id}/input/{n}`. `admin.py`: Tab **Image Jobs** (`/ui/jobs` Liste +
  `/ui/job/{id}` Detail mit Input/Output-Thumbnails), Dashboard-Job-IDs verlinkt.
  Bonus-Fix: `_gen_image_slots` nutzt `include_busy=True` (Slots sind Workflow-
  Eigenschaft → Referenzbilder werden unter Last nicht mehr verworfen).
- Reuse: dieselbe TTL-Prune-Mechanik wie bei den Artefakten löscht alles mit.

### G2 — LoRA-Support im Workflow-Mapping · M · ✅ KERN ERLEDIGT (2026-06-27)
- **Stand:** Discovery-Dropdowns für LoRA-Namen sind über die generische Modell-Feld-
  Maschinerie abgedeckt — real bestätigt: `LoraLoader.lora_name` = COMBO[98] →
  `_value_control`/Playground rendern Dropdown + Stale-Flag (pinned, Override-Tabs,
  dynamische Request-Params). `_LOADER_HINTS` enthält `"lora"`, `.safetensors` triggert
  das Dropdown. Hart verdrahtet = `fixed`-Binding (nichts Neues), dynamisch = Feld
  promoten (→), API-Pass-through verifiziert (`applied:[…,'lora_02','strength_02',…]`).
  Neu (2026-06-27): „Available fields" flaggt Discovery-Felder mit `▾ N`-Badge.
- **Optionaler Rest:** bounded number inputs (default/min/max/step aus `/object_info`)
  für `strength_*`-Felder — heute nackte Zahlenfelder; descoped (berührt den aktiv
  genutzten Mapping-Editor), opt-in.
- **Bedarf (User):** ComfyUI-Workflow kann LoRAs — **eine hart verdrahtet**, weitere
  **dynamisch für User/Schnittstelle**.
- **Hart verdrahtete LoRA:** braucht **nichts Neues** — ein `fixed`-Binding pinnt
  den LoraLoader-Node (`{node, field:"lora_name", value:"x.safetensors"}` +
  `strength_model`/`strength_clip`), exakt wie vae/gguf/clip im „test"-Alias.
- **Dynamische User-LoRAs:** als **Mapping-Params** auf LoraLoader-Nodes exponieren,
  z.B. `lora1`→{node, field:"lora_name"}, `lora1_strength`→{node, field:"strength"}.
  Der Workflow braucht je dynamischem Slot einen LoraLoader (oder einen Lora-Stacker).
- **Discovery/Validierung:** LoRA-Namen kommen aus ComfyUI `/object_info`
  (LoraLoader `lora_name`-Enum) → die UI-Dropdowns (wie bei Modell-Feldern) bieten
  sie an und flaggen veraltete.
- **API-Übergabe (Design-Entscheidung):**
  - **OpenAI-Pfad:** die Bild-API hat **kein** LoRA-Feld → sauberster Weg sind
    **Alias-Presets** (je LoRA-Kombi ein Gen-Alias, Client wählt nur `model`). Optional
    ein Extra-Feld (`lora`) durchreichen für Clients, die Extras erlauben.
  - **Nativer `/v1/generations`-Pfad:** nimmt beliebige `params` → volle dynamische
    LoRA-Kontrolle ohne OpenAI-Zwang.
- **Datei:** Mapping-UI (`admin.py`) für LoRA-Param-Typ + Discovery-Dropdown; der
  native Pfad trägt die Werte schon (`_apply_mapping`).

---

## Empfohlene Reihenfolge

1. **A1** — sofort, billig, schließt die Stats-/Kosten-Lücke für Streaming
   (Voraussetzung für korrekte **E1** Cost-Quota).
2. **E2 + E4** — billige Quick-Wins (Suche zurück, per-User-Drilldown; Daten da).
3. **E1** — Cost-Quota scharf schalten (nach A1).
4. **F0 → F1 → F2** — Sync-Call-Parking (busy-trennen, Queue, sync warten).
5. **F3** — Async-Parking mit Job-ID (Job-Store generalisieren).
6. **E3** — volle Call-Speicherung + Blob-Prune (größtes Statistik-Stück).
7. **A2 → A3** — Responses sauber; A3 nur bei aktiven Responses-Streaming-Clients.
8. **B** — nur bei echtem Co-Residency-Bedarf (eine GPU für Bild+LLM); sonst skip.
9. **C** — Breite, sobald ein ComfyUI-Host produktiv verdrahtet ist.
10. **D** — laufend mitziehen.

**Deploy-Hinweis:** Jede Änderung hier trifft prod-deployten Code → nach Merge
`scp` + `systemctl restart` (siehe [[deploy.sh]] / Memory). Code-Pfade hot-reloaden
**nicht** (nur `config.yaml`), also Restart einplanen.
