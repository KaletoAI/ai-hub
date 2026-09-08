# Plan: Voice-Funktion + Unterreiter-Muster (Playground)

Stand 2026-07-03. Recherche-Basis: zwei Voll-Scans (Dispatch-/Audio-Pfad in
`main.py`/`adapters.py`; Tab-/Playground-Struktur in `admin.py`) + Live-Checks
gegen prod. Zeilenangaben sind Anker auf dem Stand von Commit `d4367b2` — bei
Drift **nach Symbolnamen suchen**, nicht nach Zeilen.

**Zieltermin-Reihenfolge:** Teil A → Teil B (B1–B3 + B5) → B4/Upload/C nach Bedarf.

---

## Verifizierte Fakten (nicht erneut recherchieren)

Live gegen prod (192.168.8.10) bestätigt am 2026-07-03:

1. **Audio-Modelle sind bereits routbar.** `backend_models` von `localai-strix`
   enthält `omnivoice-cpp` und `qwen3-tts-cpp-customvoice` (via `/health`
   geprüft). Discovery filtert nur bei `chat_only`/`serverless_only`
   (`adapters.py: extract_models`, `_is_chat_model`) — lokale Backends setzen
   das nicht; LocalAI liefert ohnehin kein `type`-Feld. Bare model-id →
   `rebuild_route_index` indiziert sie → `resolve_routes` liefert das Backend.
   Routing ist **endpoint-agnostisch** (kein Guardrail nötig, solange die ids
   eindeutig audio sind).
2. **`/v1/audio/speech` existiert im Gateway noch nicht** (404).
3. **Binär-Passthrough funktioniert schon heute** in
   `OpenAIAdapter._dispatch_once` (`adapters.py`): `resp.json()` ist
   try/except-gewrappt (Binär → `{}`), der Body geht als `resp.content` mit dem
   **Backend-Content-Type** zurück (`media_type=resp.headers.get("content-type")`).
   Ein `audio/wav` käme heute korrekt beim Client an.
4. **Extra-Body-Felder werden verbatim geforwardet**: `_prepare`
   (`adapters.py`) strippt nur `_`-präfixierte Keys →
   `input`, `voice: "voices/kai-ref.wav"`, `params: {ref_text: …}` erreichen
   das Backend unverändert. Für den serverseitigen `voice`-Pfad ist **kein**
   neuer Code nötig.
5. Der Ziel-Aufruf (funktioniert direkt gegen localai-strix
   `http://192.168.8.38:8080`):
   ```bash
   curl http://192.168.8.38:8080/v1/audio/speech \
     -H "Content-Type: application/json" \
     -d '{"model":"omnivoice-cpp","input":"Der Text…",
          "voice":"voices/kai-ref.wav",
          "params":{"ref_text":"Exaktes Transkript der Referenz-Aufnahme."}}' \
     --output out.wav
   ```
6. **ComfyUI-Audio existiert bereits end-to-end** (job-basiert):
   `_MIME_BY_EXT` kennt wav/mp3/flac (`kind:"audio"`), `_fetch_outputs` scannt
   alle Output-Keys (SaveAudio wird erfasst), UI rendert `<audio controls>`
   (`_media_tag`). „Voice via ComfyUI" = normale Generation-Alias mit
   Audio-Workflow — **Mechanismus fertig**, nur Workflow+Mapping fehlen (User).

---

## Teil A — Wiederverwendbares Unterreiter-Muster (Start: Playground)

**Motivation:** Zu viele Top-Reiter. Generelles Design-Kriterium: Top-Reiter
können Kinder als Unterreiter gruppieren. Erste Anwendung: **ein** Reiter
„Playground" mit Unterreitern **Media / Chat / Voice** (heute: zwei separate
Top-Reiter `chatplay` + `playground`).

### Ist-Zustand (admin.py)

- `TABS` (~Z. 41): `(key, label)`-Liste; key == URL-Slug (`/ui/{key}`) == Marker
  für `_nav(active)` (~Z. 115, `class="on"` bei Match). `_page(title, body,
  active)` (~Z. 282) rendert die Nav.
- Playground-Routen in `register(app)` (~Z. 3480 ff.):
  - Chat: `/ui/chatplay` (GET `chatplay_page`), `/ui/chatplay/send` (POST `chatplay_send`)
  - Media: `/ui/playground` (GET `playground_page`), `/ui/playground/generate`
    (POST `generate`), `/ui/playground/result/{job_id}/{n}`,
    `/ui/playground/status/{job_id}`
- Beide Bodies sind fertige Builder: `_chatplay_body(vals, result_html)` und
  `_playground_body(aliases, vals, cand, result_html, oi, kept, poll_job)` —
  jeweils 2-spaltig (`.cols`: Form links, Result rechts).
- Präzedenz für Sub-View-Multiplex: `mapping_page` (Query-Param-getrieben,
  `?cedit=`/`?edit=`/`?new=` → if/elif auf Body-Builder). Es gibt **keine**
  bestehende Sub-Tab-Leiste und kein `.subnav`-CSS.
- CSS-Sprache für „aktiv": `nav a{…border-bottom:2px solid transparent}` +
  `nav a.on{color:#fff;border-bottom-color:#3b82f6}` (~Z. 134–140).
- ⚠️ Layout-Falle: `.cols` hat `height:100%`, jede `.col` scrollt selbst
  (~Z. 175–183). Die Sub-Tab-Leiste muss **oberhalb** des `.cols`-Blocks in den
  Body (sonst scrollt sie in einer Spalte weg).

### Umsetzung

1. **Registry + Helfer** (neben `TABS`):
   ```python
   SUBTABS = {"playground": [("media", "Media"), ("chat", "Chat"), ("voice", "Voice")]}

   def _subnav(parent: str, active_sub: str) -> str:
       subs = SUBTABS.get(parent) or []
       links = "".join(
           f'<a class="{"on" if k == active_sub else ""}" '
           f'href="/ui/{parent}?sub={k}">{_esc(lbl)}</a>' for k, lbl in subs)
       return f'<nav class="subnav">{links}</nav>'
   ```
   CSS (bei den nav-Regeln einsortieren, gleiche Sprache, sekundär):
   ```css
   .subnav{display:flex;gap:2px;margin:-6px 0 12px;border-bottom:1px solid #272b33}
   .subnav a{color:#9aa7b4;padding:8px 12px;text-decoration:none;border-bottom:2px solid transparent;font-size:13px}
   .subnav a:hover{color:#dce4ec;background:#1b1f27}
   .subnav a.on{color:#fff;border-bottom-color:#3b82f6}
   ```
2. **`TABS`**: `("chatplay", "Chat Playground")` + `("playground", "Media
   Playground")` ersetzen durch **ein** `("playground", "Playground")`.
3. **`playground_page` wird Parent-Dispatcher**: liest `sub =
   qp.get("sub") or "media"`;
   - `media` → bisheriger Body (`_playground_body`, unverändert),
   - `chat` → `_chatplay_body` (unverändert; vals aus Query wie
     `chatplay_page` es tut),
   - `voice` → neuer Body (Teil B5).
   Body = `_subnav("playground", sub) + <bisheriger Body>`;
   `_page(..., "playground")`.
4. **Rückwärtskompatibilität** (alle Routen bleiben registriert):
   - `chatplay_page` → 307-Redirect auf `/ui/playground?sub=chat` (einfachste
     Variante) — ODER weiter selbst rendern mit `active="playground"` +
     Sub-Leiste (weniger Umbau in `chatplay_send`, das nach POST denselben Body
     re-rendert). **Empfehlung: Redirect für die GET-Seite; `chatplay_send`
     rendert weiter selbst, nur `_page(..., "playground")` + `_subnav` davor.**
   - `generate` (Success-Redirect `/ui/playground?…`): `sub=media` in die Query
     aufnehmen (Default wäre eh media — explizit ist robuster).
   - `job_to_playground`-Redirect: landet per Default auf `sub=media` — ok.
   - `pgSwitch`-JS (Alias-Wechsel): baut `/ui/playground?model=…` — `sub=media`
     mit anhängen.
   - Fragment-Routen (`/ui/playground/status|result`) sind body-only — unberührt.

**Verify (A):** Compile → Instanz → Login-Cookie → `GET /ui/playground`,
`?sub=chat`, `?sub=voice` rendern (Sub-Leiste, richtige `on`-Markierung, Top-Tab
„Playground" aktiv); `POST /ui/chatplay/send` (Chat-Antwort erscheint, Leiste
bleibt); ein Media-Generate (Redirect trägt `sub=media`); `/ui/chatplay` →
Redirect. Alte Bookmarks funktionieren.

---

## Teil B — Voice via direktem Backend: `POST /v1/audio/speech`

**Konzept:** synchroner **Binär-Proxy** durch die bestehende Chat-Maschinerie —
kein Job, kein neues Routing. Auth/Allow-List/Quota (`gate_request`), Parking,
Failover, In-flight-Zählung kommen gratis, weil `route()` →
`_dispatch_or_park()` → `_dispatch_over()` → `OpenAIAdapter.dispatch()`
endpoint-agnostisch sind (`req.path` wird verbatim an `{backend.url}{path}`
gehängt).

### B1 — Endpoint (main.py) · S

Exakt das `embeddings`-Muster (`main.py` ~Z. 1259):

```python
@app.post("/v1/audio/speech")
async def audio_speech(request: Request, authorization: Optional[str] = Header(None)):
    """OpenAI-shaped TTS: routed like chat by body["model"] (bare id or alias);
    binary audio passes through untouched."""
    return await route("/v1/audio/speech", request, authorization)
```

**Streaming-Guard — Achtung, Ort:** `route()` parst den Body **selbst**
(`body = await request.json()`, main.py ~Z. 977) — der Endpoint-Handler kann
`stream` also NICHT vorher strippen. Der Guard gehört als pfad-bedingte Zeile
in `route()`, direkt neben das bestehende `body.pop("park", None)`:

```python
if path.startswith("/v1/audio/"):
    body.pop("stream", None)      # audio is a binary passthrough — never SSE
```

Sonst landet `stream:true` in `_dispatch_stream`, das `text/event-stream` +
`stream_options.include_usage` erzwingt → kaputt für Audio.
`_normalize_reasoning` ist für Audio harmlos (kein `reasoning`-Feld → None →
no-op), kann mitlaufen.

### B2 — Binär-sicherer Non-Stream-Dispatch (adapters.py) · S — der eine echte Fix

`_dispatch_once` (~Z. 415–441) generisch content-type-bewusst machen (nützt
jedem künftigen Binär-Passthrough, nicht nur Audio):

```python
ct = resp.headers.get("content-type", "")
is_texty = ct.startswith(("application/json", "text/"))
resp_json = {}
if is_texty:
    try: resp_json = resp.json()
    except Exception: pass
...
elapsed_ms = self._record(req, call, resp.status_code, in_tok, out_tok,
                          response_text=(resp.text if is_texty else None))
```

Warum: heute würde `response_text=resp.text` die WAV-Bytes charset-dekodieren
und als Müll-String in den Stats-Blob schreiben; `resp.json()` auf Multi-MB
Binär ist verschwendete Arbeit. Tokens/Kosten bleiben 0 (ok — Status/Dauer/
Backend werden geloggt; `record_call` verkraftet 0-Werte). `out.parsed_json={}`
liest nur die Responses-Bridge — irrelevant für Audio.

### B3 — Voice-Referenz v1: serverseitiger Pfad · 0 Code

`voice` (Pfad-String) + `params.ref_text` fließen verbatim durch (Fakt 4).
**v1 = genau der User-curl, nur gegen das Gateway** (Port 4000, mit
Authorization-Header). Upload einer Referenz-WAV ist bewusst **nicht** v1
(siehe „Später").

### B4 — Voice-Aliase (optional, empfohlen) · S–M

Ziel: Client sendet nur `model:"kai"` + `input`; das Gateway ergänzt
`voice`/`ref_text`. Design analog `alias_reasoning` (per-Alias-Default,
Session 13):

- Store: Settings-Map `alias_voice: {alias: {"voice": "...", "ref_text": "..."}}`
  (get/set-Helfer in `store.py` nach dem `alias_reasoning`-Muster).
- main: Cache-Rebuild in `rebuild_virtual_models()`; im
  `/v1/audio/speech`-Handler (NICHT generisch in `route()`): fehlt
  `body["voice"]` und der Alias hat einen Eintrag → Defaults einfüllen
  (explizite Client-Felder gewinnen immer). Der Alias selbst ist eine normale
  Chat-Alias (`virtual_models`) auf das Audio-Modell — Routing fertig.
- UI: 2 Felder im Chat-Alias-Editor (neben park/reasoning-Default) — nur
  anzeigen/pflegen, wenn gesetzt; kein eigener Tab.

### B5 — Voice-Sub-Tab im Playground · M

Analog `_chatplay_*` (echter API-Client via `_self_api`):

- **Form (links):** `model` (Datalist: v1 einfach alle Modelle + Aliase — es
  gibt keine Typ-Info; Vorbelegung `omnivoice-cpp`), `input` (Textarea,
  mehrzeilig), `voice` (Text, Pfad), `ref_text` (Textarea 2). POST
  `/ui/playground/voice`.
- **Handler `voiceplay_send`:** `_self_api(request, "POST",
  "/v1/audio/speech", json=body)` (Timeout 600s reicht; TTS kann Minuten
  dauern bei Modell-Load). Antwort:
  - 200 + `audio/*` → Bytes in einen per-User-Stash legen
    (`_voice_results[(user)] = (bytes, mime)` — Muster `_pg_images`), Body
    re-rendern mit `<audio controls src="/ui/playground/voice-audio">` +
    Download-Link (`download="out.wav"`).
  - Fehler → Status + Detail rendern (Muster `_chat_result_html`).
- **Route `GET /ui/playground/voice-audio`:** liefert den Stash-Inhalt
  (`Response(bytes, media_type=mime)`); 404 wenn leer.
- Kein Job, kein Polling — synchron wie chatplay. „Sending…"-Feedback-JS von
  chatplay übernehmen (`_CHATPLAY_JS`-Muster).

**Verify (B):** Compile → Instanz →
```bash
K=<master-key>
curl -s localhost:4000/v1/audio/speech -H "Authorization: Bearer $K" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice-cpp","input":"Hallo Welt.","voice":"voices/kai-ref.wav",
       "params":{"ref_text":"…"}}' --output /tmp/out.wav
file /tmp/out.wav          # → RIFF WAVE
```
- Header prüfen: `x-gateway-backend` gesetzt, Content-Type `audio/wav`.
- Stats-Tab: Zeile mit endpoint `/v1/audio/speech`, 0 Tokens, plausible Dauer;
  **kein** Response-Blob (has_body leer/klein).
- `stream:true` im Body → funktioniert trotzdem (wird gestrippt).
- Unbekanntes Modell → 503/404 wie bei Chat; Backend down → Failover/Park.
- UI: Voice-Sub-Tab generiert hörbares Audio, Download funktioniert.
- (B4) Alias mit gepinntem voice: Call ohne `voice`-Feld liefert die geklonte
  Stimme; explizites `voice` im Body übersteuert.

### Später (bewusst nicht v1)

- **Referenz-WAV-Upload** über UI/API: kein bestehendes Plumbing auf dem
  Direktpfad (das `images`/`upload_images`-System ist ComfyUI/Bild-spezifisch).
  Braucht eigenes kleines Konzept (multipart oder base64-Feld → wohin auf dem
  Backend? LocalAI erwartet einen Server-Pfad → ggf. Upload-Verzeichnis-
  Konvention klären). Erst angehen, wenn der Pfad-Workflow nervt.
- **Endpoint-Typ-Guardrail** im Routing: nur nötig, falls dieselbe model-id
  je Backend mal chat, mal audio bedient. Heute nicht der Fall.
- **`/v1/audio/transcriptions`** (STT): gleiches Muster, aber multipart-Input —
  eigener Schritt.

---

## Teil C — Voice via ComfyUI + Mapping (Mechanismus fertig)

Nichts zu bauen außer dem, was der User verdrahtet: Generation-Alias mit
Audio-Workflow (SaveAudio) → Jobs liefern `kind:"audio"`, Playground/Job-Detail
rendern `<audio>`. Läuft über den **Media**-Sub-Tab (Alias auswählen), nicht
über den Voice-Sub-Tab — Voice-Sub-Tab = Direktpfad. Wenn später gewünscht,
kann der Voice-Sub-Tab zusätzlich Audio-Generation-Aliase listen (dann dort
entscheiden, nicht jetzt).

---

## Entscheidungen (Defaults, falls der User nichts anderes sagt)

1. Voice-Referenz v1 = **serverseitiger Pfad** (Upload später).
2. **B4 Voice-Aliase: ja**, im selben Zug (bester UX-Hebel, kleines Delta).
3. Voice-Sub-Tab v1 = **nur Direktpfad** (ComfyUI-Voice über Media-Sub-Tab).

## Hinweise für die umsetzende Session (Opus 4.8)

- **Constraints (CLAUDE.md):** kein Modul importiert `main` (admin bekommt
  alles via `bind()` — für B5 KEINE neuen main-Funktionen nötig, `_self_api`
  reicht); `store.py` bleibt dependency-frei; keine neuen pip-Deps;
  `python-multipart`-frei (Voice-Form ist urlencoded, kein File-Upload in v1).
- **Reihenfolge:** A zuerst (schafft den `?sub=`-Rahmen), dann B1+B2 (Proxy),
  B5 (Sub-Tab), B4 (Aliase). Jeder Teil = ein in sich geschlossener Commit,
  danach deploybar.
- **Verifikation:** `venv/bin/python -m py_compile *.py` vor jedem Deploy;
  lokale Instanz nur die eine (Memory: single instance); UI-Renders per curl +
  Login-Cookie (Muster: `POST /ui/login --data-urlencode key=$K` → Cookie-Jar →
  GET Seite → grep auf Marker). Prod-Deploy: `DEPLOY_HOST=root@192.168.8.10
  ./deploy.sh` (Code only, DBs bleiben), danach Health + Live-curl (siehe
  Verify-Blöcke).
- **Live-Test-Ziel:** localai-strix (`192.168.8.38:8080`) bedient
  `omnivoice-cpp`; die Referenz `voices/kai-ref.wav` + Transkript existieren
  dort serverseitig (User-Angabe). Wenn strix 503/lädt: localai-phoenix prüfen
  (`/v1/models`), sonst mit Fehler-Pfaden verifizieren und Audio-Live-Test an
  den User übergeben (Memory: user tests at the UI).
- **Commit-Konvention:** master, Co-Authored-By-Zeile, push auf `github` (Default)
  + `origin` (Forgejo .110).
- Nach Teil A: CLAUDE.md-Absatz zu admin.py um das SUBTABS-Muster ergänzen
  (ein Satz), README-Tab-Liste anpassen falls dort erwähnt.
