# Code-Review-Plan

Ziel: besser lesbarer, klar strukturierter Code + Performance auf dem Request-Pfad.
Kein Bug-Hunt. Basis: Voll-Scan aller 7 Module am 2026-07-02 (Arbeitsstand inkl.
uncommitteter Reasoning-Änderungen — Zeilenangaben sind Anker, bei Drift nach
Symbolnamen suchen).

## Grundregeln (gelten für jede Session)

- **Vorher:** aktuellen Stand committen (Reasoning-Feature ist noch uncommittet).
  Eine Session = ein in sich geschlossener Commit; danach ist das Repo wieder
  deploybar.
- **Verhalten erhalten:** reine Refactors + Perf; keine API-/UI-Änderungen, außer
  die Session nennt sie explizit.
- **Projekt-Constraints:** kein Modul importiert `main` (DI über Callables),
  hot-reload-sicher (keine Config-Werte zur Importzeit cachen), keine neuen
  pip-Abhängigkeiten, `python-multipart`-frei bleiben.
- **Verifikation (kein Testsuite vorhanden):** `venv/bin/python -m py_compile *.py`
  → Instanz starten (nur die eine, wenn idle) → `curl /health` → 1 Chat-Call
  (stream + non-stream) → betroffene `/ui`-Tabs rendern. Sessions nennen zusätzlich
  eigene Checks.

## Übersicht

| # | Session | Dateien | Aufwand | Impact |
|---|---------|---------|---------|--------|
| 1 | ✅ 2026-07-02 (`5b54420`) Warm-up: Konstanten & Namen | main, adapters, store, reasoning, responses_bridge | S (~1h) | Lesbarkeit |
| 2 | ✅ 2026-07-02 (`8d928c5`) Shared HTTP-Client | adapters, main | M (~2h) | **Perf: größter Hebel** |
| 3 | ✅ 2026-07-02 (`0a837e3`) SQLite entblocken | store, stats, jobs, main | M (~2–3h) | **Perf unter Last** |
| 4 | ✅ 2026-07-02 (`fa59248`) Routing-Hotpath | main | M (~2h) | Perf pro Request |
| 5 | ✅ 2026-07-02 (`8f6e007`) `dispatch()` aufräumen + JSON-Passthrough | adapters | M (~2h) | Perf + Lesbarkeit |
| 6 | ✅ 2026-07-02 (`0930e02`) Responses-Bridge konsolidieren | main (→ responses_bridge.py) | M (~2–3h) | Struktur |
| 7 | ✅ 2026-07-02 (`4d5101a`) Generierungs-/Bilder-Pfad | main, jobs (→ openai_image_bridge.py) | M/L (~3h) | Lesbarkeit + Perf |
| 8 | ✅ 2026-07-02 (`31e06b8`) store.py: generisches CRUD | store, stats, jobs | M (~2h) | Struktur |
| 9 | ✅ 2026-07-02 (`680b75b`) admin.py: Dedup & Helfer | admin, main | M (~2–3h) | Lesbarkeit |
| 10 | ✅ 2026-07-02 (`e5288e7`) admin.py: Groß-Funktionen (Paket-Split separat) | admin | L | Struktur |
| 11 | ✅ 2026-07-02 (`2c4c29f`) Reasoning-Deltas in der Responses-Bridge | responses_bridge | S/M (~1–2h) | Feature-Lücke (Fund aus S6) |
| 12 | ✅ 2026-07-02 (`d40b3c0`) Playgrounds als echte API-Clients | admin, main, adapters | M (~2h) | Korrektheit (User-Vorgabe) |
| 13 | ✅ 2026-07-02 (`06768cf`) Reasoning-Default pro Alias | main, store, admin | S/M (~1–2h) | Feature (User-Vorgabe 2026-07-02) |

Reihenfolge ist Empfehlung: 2+3 zuerst, wenn Performance drängt; 1 als Einstieg.
Abhängigkeiten: 6 profitiert von 5 (läuft aber auch allein); alle anderen sind
unabhängig.

---

## Session 1 — Warm-up: Konstanten, Namen, Kleinkram (S)

Risikofreie Einzeledits, gut zum Reinkommen.

- Magic Numbers/Strings in benannte Konstanten heben: HTTP-Timeouts `300.0`
  (`adapters.py:313,341`, `main.py:1132`) und `5.0/8.0/20.0/30.0` (adapters,
  Discovery/Upload); `uuid4().hex[:24]`-Kürzung (5×, `main.py:986–1151`);
  `"resp_"`-Prefix + `len("resp_")`-Slicing (`main.py:1273–1375`);
  Default-Priorität `100` (`main.py:111,1428`); `Retry-After`-Werte.
- Namenskollision: zwei verschiedene `_coerce` (`adapters.py:457` vs
  `main.py:1749`) — eins umbenennen.
- `_cost_usd(backend_name=…)` bekommt tatsächlich eine `bid` (`main.py:748`) —
  Parameter umbenennen.
- Konvention „gateway-private `_`-Keys im Body" (setzt `main.py:871`, strippt
  `adapters.py:287`) an EINER Stelle dokumentieren (Kommentar bei
  `NormalizedRequest` oder im `route()`-Docstring).
- `store.py:246,265`: Parameter `type` shadowt das Builtin → `btype`.
- `reasoning.py:64,72,104`: Docstring sagt „COPY", aber auto/unsupported geben das
  Original zurück — Docstring präzisieren (oder konsequent kopieren).

**Verify:** Compile + 1 Chat-Call + `/ui/backends` rendern.

---

## Session 2 — Shared HTTP-Client (M) · größter Perf-Hebel

**Befund:** Jeder Forward baut einen frischen `httpx.AsyncClient` →
kein Connection-Pooling/Keep-alive, TLS-Handshake pro Request.
Stellen: `adapters.py:312,340` (dispatch stream/non-stream),
`adapters.py:718,740,772,818` (ComfyUI), `main.py:1132` (`run_chat`),
`main.py:1531`, `main.py:1997`. Korrekt gelöst ist es nur im Health-Loop
(`main.py:305`) und in der Lifespan-Discovery (`main.py:372`).

**Vorgehen:**
1. Einen prozessweiten, gepoolten Client in `main` anlegen (Lifespan: erzeugen /
   schließen), per `AdapterContext` injizieren (z. B. `http_client: Callable`);
   Default im Context bleibt „frischer Client" für Nicht-main-Konstruktionen.
2. Alle o. g. Stellen umstellen. Timeouts bleiben pro Aufruf (`client.post(...,
   timeout=…)`).
3. Streaming beachten: `client.stream(...)` mit geteiltem Client ist ok, der
   Client darf nur nicht pro Request geschlossen werden.
4. Nebenbei: Workflow-Deep-Copy `json.loads(json.dumps(...))`
   (`adapters.py:692`) → `copy.deepcopy`.

**Verify:** stream + non-stream Chat, ein Generation-Job, Failover simulieren
(einen Backend-Port falsch stellen → Connect-Error muss weiter failovern).

---

## Session 3 — SQLite: Event-Loop entblocken + Verbindungs-Policy (M)

**Befunde:**
- `store.py` komplett synchron, kein einziges `to_thread`; ebenso die Query-Seite
  von `stats` (`_q`, `summary`, `month_cost`) und `jobs`
  (`create/get/set_status/fail`). Aufgerufen aus async-Handlern:
  `main.py:496` (Quota!), `1413`, `1486`, `1642` … → blockiert unter Last alle
  parallelen Requests.
- Connection pro Aufruf, und `store._conn` (`store.py:178`) + `jobs._conn`
  (`jobs.py:70`) setzen `PRAGMA journal_mode=WAL` bei JEDEM Open (WAL ist
  persistent — einmal in `init` genügt, wie `stats.py:143` es korrekt macht).
  Drei Module, drei Transaktionsstile (autocommit vs. context-manager).
- `stats.month_cost` (`stats.py:124`) scannt pro Quota-Request per
  `SUM(...) WHERE source=? AND ts>?` ohne Index auf `source`.
- `jobs.complete` (`jobs.py:149`) öffnet 3 Connections pro Abschluss
  (Blob-Write → `_read_meta` → UPDATE); `set_inputs` ähnlich.

**Vorgehen:**
1. Einheitliche `_conn`-Policy in allen drei Modulen: WAL einmalig in `init`,
   ein Transaktionsstil.
2. Die aus async-Handlern erreichbaren Lese-/Schreibpfade über
   `asyncio.to_thread` führen — am saubersten dünne async-Wrapper in den Modulen
   (`async def aget(...): return await asyncio.to_thread(get, ...)`), Call-Sites
   in `main.py` umstellen. Hot-Path zuerst: `month_cost`, `store.get`,
   `jobs.create/get/set_status/fail`.
3. Index `idx_calls_source ON calls(source, ts)` per bestehendem
   Migrations-Muster in `stats.init`.
4. `jobs.complete`/`set_inputs`: Meta-Roundtrip in eine Connection falten.

**Verify:** Quota-User-Call (Kostenquote gesetzt), Generation-Job end-to-end,
`/ui/statistic` + Dashboard rendern; `PRAGMA journal_mode` einmalig prüfen.

---

## Session 4 — Routing-Hotpath: Scans reduzieren (M)

**Befunde:**
- `resolve_routes` (`main.py:571–606`) macht pro Request Voll-Scans:
  `enabled_backends()` frisch, dann pro Backend `alias_entry`/`backend_models`-
  Lookups; `split_backend_prefix` (`main.py:566`) scannt zusätzlich
  `any(b["name"]==prefix …)` — O(Backends) nur für den Prefix-Test.
- `_dispatch_over` mutiert das geteilte `body["model"]` über
  Failover-Kandidaten hinweg (`main.py:814`) — funktioniert, aber subtil.

**Vorgehen:**
1. Beim Config-Reload/Discovery einen Routen-Index vorberechnen (Backend-Name-Set
   für den Prefix-Test; alias → Kandidatenliste), sodass pro Request nur noch
   der Health-/Busy-Check läuft. Achtung: Invariante erhalten — zwischen
   `resolve_routes` und `inflight_inc` darf kein `await` liegen.
2. `_dispatch_over`: Body pro Kandidat flach kopieren statt shared-dict-Mutation.
3. **Optional/diskutieren:** Parking-Thundering-Herd (`_notify_slot_free`,
   `main.py:264` weckt ALLE Geparkten; jeder re-scant). Ist dokumentiert
   gewollt (FIFO-Fairness) — nur angehen, wenn Queues real groß werden.

**Verify:** `/health`-Routing-Snapshot vorher/nachher identisch (JSON diffen!),
Prefix-Call `backend/model`, Alias-Call, Park-Szenario (Backend mit
`max_concurrent:1` doppelt beschicken).

---

## Session 5 — `dispatch()` aufräumen + JSON-Passthrough (M)

**Befunde (`adapters.py:273–364`):**
- `OpenAIAdapter.dispatch` ist ~90 Zeilen; Stream- und Non-Stream-Zweig enden in
  fast identischen Blöcken (`cost_usd` + `record_call` + `inflight_dec` +
  `active_done`, Z. 330–335 vs. 355–363).
- Non-Stream: Body wird geparst (`resp.json()`), für Stats re-serialisiert und
  in `JSONResponse` NOCHMAL serialisiert — das Gateway transformiert den Body
  gar nicht.

**Vorgehen:**
1. Gemeinsamen `_record(...)`-Helper extrahieren; Stream/Non-Stream in zwei
   private Methoden splitten.
2. Non-Stream: Rohbytes durchreichen (`Response(resp.content, media_type=…)`),
   nur für `usage`/Stats einmal parsen. Response-Header (`x-reasoning-control`)
   beibehalten.

**Verify:** Chat stream + non-stream, Stats-Zeile hat Tokens/Kosten,
Fehlerstatus (4xx vom Backend) kommt unverändert durch.

---

## Session 6 — Responses-Bridge konsolidieren & extrahieren (M)

**Befunde:**
- Drei fast identische Responses-Shell-Builder: `chat_to_responses`
  (`main.py:1015`), `shell()` in `_responses_stream` (`main.py:1162`),
  `_resp_shell` (`main.py:1232`); der `output_text`-Join ist 3× kopiert.
- `responses_to_chat` (~65 Zeilen, `main.py:907`) mischt Filter, Tool-Übersetzung
  und Message-Loop.
- Zwei divergente Owner-Checks: `_bg_owner_check` (`main.py:1292`) vs.
  `_require_job_owner` (`main.py:1690`) — gleiche Absicht, andere Tests.
- Die Bridge (`main.py:884–1033, 1146–1240`) ist pur — fasst keine
  Gateway-Globals an → sauberster Extraktionskandidat.
- `/v1/responses` re-parst die dispatch-Response (`json.loads(bytes(resp.body))`,
  `main.py:1249,1344,1346`) — bis zu 4 Konversionen pro Call.

**Vorgehen:**
1. Einen Shell-Builder + einen `output_text`-Helper, drei Stellen darauf umziehen.
2. Owner-Check vereinheitlichen (eine Funktion, beide Call-Sites).
3. Extraktion nach `responses_bridge.py` (pure Funktionen, importiert kein
   `main` — passt ins Modulschema; CLAUDE.md ergänzen).
4. **Nur wenn Session 5 schon gelaufen ist:** dispatch für interne Aufrufer
   geparste Daten liefern lassen (Carrier statt Starlette-Response), damit die
   Bridge nicht re-parst. Sonst weglassen — Session bleibt auch ohne rund.

**Verify:** `/v1/responses` sync, `stream:true` (SSE-Events!), `background:true`
→ poll → cancel; Owner-Check mit Fremd-User.

---

## Session 7 — Generierungs-/Bilder-Pfad (M/L)

**Befunde:**
- `run_generation` (~90 Zeilen, `main.py:1576–1665`): Alias-Resolve, Force-Pin,
  LoRA-Mengenlogik, verschachtelte `_pick`/`build_req`-Closures, Job-Anlage,
  sync/async/park-Verzweigung — alles inline.
- `get_gen_routes` (je Aufruf: `store.get` + Backend-Scan) läuft pro
  Bilder-Request 4–5× (`main.py:1588,1611,1880,1883`).
- Force-Filter-Idiom 5× kopiert (`main.py:1549–1613`).
- Bilder-Shims sind fast pur (`main.py:1749–1865`) → extrahierbar.
- `jobs.get()` parst das komplette Manifest, nur um EINEN Dateinamen aufzulösen
  (`jobs.py:244,269` — auf dem Artifact-Serving-Pfad); doppelte
  ran_on/no-ran_on-UPDATE-Zweige (`jobs.py:166,185`).

**Vorgehen:**
1. `run_generation` zerlegen: `_resolve_gen_candidates(alias, force, loras)`
   (einmal rechnen, Ergebnis durchreichen — erledigt auch die 4–5×-Aufrufe),
   `_start_gen_job(...)`, Park-Logik bleibt.
2. Force-Filter als Helper.
3. Bilder-Shims nach `openai_image_bridge.py` (bekommt `jobs` +
   `get_gen_routes` injiziert).
4. `jobs.py`: gezielte Filename-Query statt Voll-`get()`; UPDATE-SQL einmal bauen.

**Verify:** `/v1/generations` sync + async + cancel, `/v1/images/generations`,
`/v1/images/edits` mit Referenzbild, LoRA-Routing (Backend ohne LoRA wird
übersprungen), `/ui/jobs`.

---

## Session 8 — store.py: generisches CRUD + Settings (M)

**Befunde:**
- Vier Entities mit fast identischem Boilerplate: gen_aliases (`store.py:203`),
  backends (`240`), users (`421`), chat_aliases (`455`) — gleicher
  Upsert/Select/Delete-Shape, nur Tabelle/PK/Decode-Fn anders.
- `_decode_backend`/`_decode_user` identisch bis aufs Feld (`store.py:233,414`);
  `upsert_backend`/`upsert_user` teilen den Encrypt-then-Store-Body.
- `get_settings()` liest + parst bei jedem Aufruf die ganze Tabelle; Mutatoren
  (`set_ip_alias` `store.py:354`, `set_alias_park` `375`) machen
  Read-Modify-Write des Gesamt-Blobs. (Entschärft: main cached die heißen
  Werte — trotzdem unnötig teuer für die UI-Pfade.)
- Migrations-Idiom dupliziert (`stats.py:146` / `store.py:123`);
  `locals().get("old")`-Trick im Backends-Rebuild (`store.py:138`).
- Fehlerbehandlung inkonsistent: stiller Blob-Write-Drop (`stats.py:202`),
  stilles `_read_meta` (`jobs.py:200`) vs. loggendes `decrypt_secret`.

**Vorgehen:**
1. Generischen Tabellen-Helper (`_crud(table, pk, decode=None)`) bauen, vier
   Entities darauf umziehen — Public-API der Funktionen unverändert lassen
   (Call-Sites in main/admin bleiben unangetastet).
2. Settings: einzelnen Key lesen/schreiben statt Gesamt-Blob.
3. Migrations-Helper (`_ensure_columns(conn, table, {...})`) für beide Module;
   `locals()`-Trick durch explizite Variable ersetzen.
4. Einheitliche Policy: still schlucken nur wo begründet, sonst `logger.warning`.

**Verify:** Alle CRUD-UI-Flows einmal durch (Backend anlegen/ändern/löschen,
User, Chat-Alias, Gen-Alias), Store-Datei vorher sichern; Neustart →
Seed/Migration läuft sauber.

---

## Session 9 — admin.py: Dedup & Helfer (M)

**Befunde (Duplikate):**
- Dashboard re-inlined `_JOB_TICK` (byte-identisch, `admin.py:2448` vs. `2106`),
  `_JOB_SCLS` (`2373` vs. `2111`) und die komplette Job-Zeile aus `jobs_page`
  (`2371–2385` vs. `2125–2146`).
- LLM-Call-Zeile doppelt: Dashboard `2408–2417` vs. `_recent_calls_table`
  `2472–2495` (Docstring behauptet Wiederverwendung — stimmt nicht).
  → Dabei auch die inkonsistenten Spalten-Shapes von `stats.recent/`
  `recent_since/summary/dashboard` vereinheitlichen (ein Row-Format).
- `llmcalls_page`/`statistic_page`: ~30 Zeilen Header (User-Picker, Suchfeld,
  Filter-JS) byte-gleich kopiert (`2506–2525` vs. `2535–2579`).
- Kleinere Mehrfach-Muster: Zwei-Spalten-Shell 6× handgebaut (`_cols`-Helper),
  „+ Add backend…"-Dropdown 2×, `_select` kann kein `onchange` → 7 Stellen
  bauen Selects von Hand, Form-Getter-Lambda 4× in 2 Varianten,
  `_form_multi()` fehlt (2 Handler re-parsen den Body wegen Checkbox-Listen).
- Perf (klein): `users_page` ruft `stats.summary()` 2× pro Render
  (`2838` via `_autoresolve_ips` + `2840`); `_alias_editor` holt `_object_info`
  sequenziell pro Backend (`1516`, `1600`) → `asyncio.gather`.
- `bind()` pflegt ~24 Callables in drei parallelen Listen (`admin.py:68–122`)
  → auf ein Registry-Dict umstellen (`_cb.update(overrides)`), bleibt
  import- und hot-reload-sicher.

**Vorgehen:** Helfer bauen (`_job_row`, `_call_row`, `_user_picker` +
`_FILTER_JS`, `_cols`, `_add_select`, `_select(..., onchange=)`, `_getter`,
`_form_multi`), dann Call-Sites umziehen; `bind()`-Registry zuletzt (mechanisch,
aber viele Stellen).

**Verify:** Jeden `/ui`-Tab einmal rendern (Dashboard, Jobs, LLM Calls,
Statistic, Users, Reasoning, Mapping-Editor), ein Save-Flow pro Formulartyp.

---

## Session 10 — admin.py: Groß-Funktionen zerlegen, optional Paket-Split (L)

**Befunde:**
- `_alias_editor` ~186 Zeilen (`admin.py:1501–1686`): vier Sub-Views + Inline-
  CSS/JS + Form in einer Funktion, Rückgabe-2-Tupel.
- `dashboard_page` ~149 Zeilen (`2307–2455`): fünf unabhängige Panels inline
  (nach Session 9 schon kleiner — Rest hier zerlegen).
- Datei 3147 Zeilen, aber saubere Sektions-Banner → Paket-Split nach
  `admin/` (`_components.py` + ein Modul pro Tab, `register()` aggregiert)
  ist machbar, da admin nichts aus `main` importiert und nur über `bind()`
  kommuniziert.

**Vorgehen:** Erst die zwei Funktionen in Panel-/Sub-View-Builder zerlegen.
Den Paket-Split als eigene Entscheidung behandeln (Aufwand rein durch Volumen;
Nutzen: Navigierbarkeit) — vorher absprechen.

**Verify:** Alle Tabs rendern, Mapping-Editor mit Workflow-JSON + Pins,
Deploy-Skript prüfen (erfasst es ein `admin/`-Verzeichnis? `deploy.sh` +
scp-Pfad anpassen!).

---

## Session 11 — Reasoning-Deltas in der Responses-Bridge (S/M) · Fund aus Session 6

**Befund (2026-07-02, live verifiziert):** Thinking-Modelle auf LocalAI (z. B.
`tool` = qwen3.5-9b-heretic) streamen ihre komplette Ausgabe als
`delta.reasoning`; `delta.content` bleibt `null`. `responses_stream()` (und auch
`chat_to_responses` non-stream, falls `message.reasoning` gesetzt ist) reicht nur
`content` weiter → über `/v1/responses` mit `stream:true` kommt bei diesen
Modellen **leerer Text** an (`output_text.done` mit `""`). Nicht-Thinking-Modelle
(z. B. `translator`) sind korrekt (Content-Deltas kommen durch).

**Vorgehen:**
1. `responses_stream()`: `delta.reasoning` als Responses-konforme
   Reasoning-Events emittieren (`response.reasoning_summary_text.delta` + ein
   `type:"reasoning"`-Output-Item im completed-Objekt), `delta.content` bleibt
   `output_text`. Clients ohne Reasoning-Support ignorieren die Events.
2. `chat_to_responses`: non-stream `message.reasoning`/`reasoning_content`
   analog als Reasoning-Item abbilden.
3. Optional (diskutieren): Fallback „reasoning in content falten", wenn der
   Client das per Flag will — oder auf die Reasoning-Rules verweisen
   (reasoning off erzwingt Content-only beim Modell selbst).

**Verify:** `/v1/responses stream:true` mit `tool` → Reasoning-Deltas kommen an,
`output_text` bleibt leer/korrekt getrennt; mit `translator` unverändert;
N8N/LangChain-Kompatibilität (unbekannte Events werden toleriert) gegenprüfen.

## Session 12 — Playgrounds als echte API-Clients (M) · User-Vorgabe 2026-07-02

**Vorgabe:** Playgrounds (aufgefallen beim Chat Playground) dürfen NICHTS an der
API vorbei machen. Sie sind zum Testen der API da und sollen sie deshalb auch
benutzen — am besten als eigener Client.

**Ist-Zustand:**
- Chat Playground → `run_chat()` postet direkt ans Backend, umgeht damit:
  Adapter-Dispatch, In-flight-Zähler/Busy, Parking, Reasoning-Toggle,
  `x-reasoning-control`, Auth/Quota — und bis `4493a57` auch die Stats
  (dort nur als Sonderfall nachgerüstet; wird mit dieser Session obsolet).
- Media Playground → ruft `run_generation()` in-process (immerhin derselbe Kern
  wie `/v1/generations`, aber ohne HTTP/Auth-Schicht).

**Vorgehen:**
1. `run_chat` durch einen echten Self-Request ersetzen: POST über den geteilten
   `http_client` an `http://127.0.0.1:<port>/v1/chat/completions` — durchläuft
   uvicorn, gate_request, Parking, Adapter, Stats wie jeder externe Client.
2. Auth/Attribution klären: Master-Key bzw. Admin-Session-Key mitsenden;
   Quelle als `playground` ausweisen (dediziertes Konzept statt `x-source`-Hack
   diskutieren — z. B. interner Playground-User).
3. Media Playground analog auf `POST /v1/generations` (+ Job-Polling über
   `/v1/jobs/{id}`) umstellen.
4. Sonderfall-Recording aus `run_chat` (4493a57) entfernen; `run_chat` wird zum
   dünnen Client oder verschwindet aus `main` (Playground-Client kann in admin
   leben — er braucht dann kein bind-Callable mehr, nur die Port-Info).

**Verify:** Playground-Call erscheint in LLM Calls (source playground), respektiert
Reasoning-Rules (`x-reasoning-control` sichtbar), parkt bei besetztem Backend,
zählt im In-flight/Dashboard; Media Playground erzeugt Job über die API-Route.

## Session 13 — Reasoning-Default pro Alias (S/M) · User-Vorgabe 2026-07-02

**Anforderung:** Reasoning soll zusätzlich **pro Alias** ein-/ausschaltbar sein,
sodass z. B. `tool` (reasoning off) und `tool-thinking` (reasoning on) auf
dasselbe Backend+Modell zeigen können.

**Design:**
1. Store: Settings-Map `alias_reasoning: {alias: "off"|"on"}` (fehlend = auto) —
   analog `alias_park` (get/set/rename-Helfer).
2. main: Cache wie `alias_park_s` (rebuild in `rebuild_virtual_models`);
   in `route()`/`responses()`: explizite Client-Angabe gewinnt, sonst greift der
   Alias-Default (`_normalize_reasoning(body) or alias_reasoning.get(alias)`).
   Die Auflösung auf den Mechanismus läuft weiter über die Reasoning-Rules
   (Modell×Backend) — der Alias-Default liefert nur das „off/on/auto".
3. UI: Dropdown (auto/on/off) im Chat-Alias-Editor (Mapping-Tab), neben dem
   Park-Feld; Save-Handler + Alias-Rename mitziehen.

**Verify:** Zwei Aliases auf dasselbe Modell, einer off/einer auto → Rule matcht →
`x-reasoning-control` unterscheidet sich; expliziter Client-`reasoning` übersteuert.

## Bewusst NICHT im Plan

- Parking-Wakeup-Design (FIFO-alle-wecken): dokumentiert gewollt, Queues sind
  klein — nur bei realem Bedarf (Session 4, optional).
- `stats._render`/`_render_routing` (HTML in stats.py, lazy `import main` in
  `routing()`): bekannter, dokumentierter Kompromiss; ein Umbau lohnt erst,
  wenn stats ohnehin angefasst wird.
- Handgebautes `parse_qs`/Multipart-Parsing in admin: absichtlich
  `python-multipart`-frei — bleibt.
- `str +=`-Akkumulatoren in admin: CPython optimiert das; nur bei den zwei
  100+-Zeilen-Tabellen im Zuge von Session 9 auf `"".join` umstellen.
