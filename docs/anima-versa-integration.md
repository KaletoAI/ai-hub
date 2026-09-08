# Arbeitsanweisung: AI-Hub als Image-Backend in anima-versa einbinden

**Für:** Claude Code im `anima-versa`-Repo
**Ziel:** Den AI-Hub (OpenAI-kompatibler Reverse-Proxy vor ComfyUI) als
Bild-Generierungs-Backend in anima-versa anbinden — über die **OpenAI Images API**.
Das Gateway übersetzt OpenAI-Image-Requests intern auf ComfyUI-Workflows und liefert
fertige Bilder zurück.

Das Gateway spricht den **strikten OpenAI-Image-Standard** (`/v1/images/generations`
+ `/v1/images/edits`) und akzeptiert **zusätzlich** LocalAI-style `ref_images`. Wer
in anima-versa schon den `openai_diffusion`-Provider (für LocalAI) hat, kommt damit
sehr weit — empfohlen ist trotzdem ein eigener, klar benannter Provider-Type, weil
das Gateway ein paar Eigenheiten hat (Auth auf Result-URLs, dynamische Workflow-Params,
gen-alias statt Modellname).

---

## 1. Verbindung

| | |
|---|---|
| Base-URL (prod) | `http://<ai-hub-host>:4000` |
| Auth | HTTP-Header `Authorization: Bearer <API_KEY>` — **auf jedem** Request (auch beim Abholen von Result-URLs) |
| `model` | Der **Generierungs-Alias** des Gateways, **nicht** ein ComfyUI-Checkpoint. Aktuell verfügbar: `Qwen`. Die Liste der erlaubten Aliase hängt am API-Key (User-Allow-List). |

Der `model`-Wert ist ein vom Gateway-Admin angelegter Alias, hinter dem ein konkreter
ComfyUI-Workflow + Mapping + Backend(s) stehen. anima-versa muss davon nichts wissen —
es schickt `model: "<alias>"` und einen Prompt.

### 1.1 Modell-Liste auf Image beschränken (gegen die 400+ Chat-Modelle)

`GET /v1/models` ist der gemeinsame OpenAI-Katalog: ohne Beschränkung listet er **alle**
Chat-/LLM-Modelle aller Backends (400+) **plus** die Image-Generierungs-Aliase
(`owned_by: "ai-hub (image)"`). Zwei Hebel, damit anima-versa nur das Relevante sieht:

1. **Eigenen anima-versa-User mit Allow-List** (empfohlen). Im `/ui` → **Users** einen
   User anlegen, dessen Häkchen **die Image-Aliase namentlich** enthalten (z.B. `Qwen`).
   Ein Grant auf ein ganzes **Backend** wirkt nur für Chat-/LLM-Modelle — Image-Aliase
   müssen einzeln freigegeben werden, sonst sind sie weder in `GET /v1/models` sichtbar
   noch nutzbar (`403`). `GET /v1/models` mit diesem Key ist dann **gefiltert** und
   liefert nur die erlaubten Einträge (verifiziert: `allow=[Qwen]` → genau 1 Eintrag).
   Die Allow-List gated zugleich die **Nutzung** (fremdes Modell → `403`).
2. **`GET /v1/models?type=image`** — liefert ausschließlich die Image-Aliase
   (`?type=chat` = nur Chat). Praktisch zum direkten Testen.

Beides kombinierbar: anima-versa nutzt seinen beschränkten Key, ruft `GET /v1/models`
(oder `?type=image`) und bekommt eine kurze, saubere Alias-Liste statt 400+.

---

## 2. Endpoint A — `POST /v1/images/generations` (Text → Bild)

`Content-Type: application/json`. Das ist der Haupt-Pfad für reine Text2Img-Generierung.

### Request-Body

```jsonc
{
  "model": "Qwen",                 // PFLICHT — gen-alias
  "prompt": "a red apple on a table",   // PFLICHT
  "negative_prompt": "blurry, low quality",  // optional
  "size": "1024x1024",             // optional — "BxH" oder "auto"; unparsebare Werte
                                   // fallen still auf 1024x1024 zurück. Wirkt NUR, wenn
                                   // der Alias width/height im Mapping bindet — sonst
                                   // still ignoriert (wie jeder ungemappte Param, §4).
  "n": 1,                          // wird ANGENOMMEN, aber ignoriert — die Anzahl der
                                   // Bilder bestimmt der Workflow (Batch-Size im Graph).
                                   // data[] hat so viele Einträge, wie der Job liefert.
  "response_format": "b64_json",   // optional — "b64_json" | "url". Default "url"

  // BONUS (LocalAI-kompatibel): Referenzbilder als Liste.
  // Jeweils base64-data-URI ODER nackt-base64 ODER http(s)-URL.
  // Werden positionsweise auf die Bild-Slots des Workflows gemappt.
  "ref_images": ["data:image/png;base64,iVBOR...", "https://.../ref2.png"],

  // ALLES WEITERE wird als dynamischer Workflow-Parameter durchgereicht
  // (siehe §4) — z.B. LoRAs, seed, steps, cfg:
  "seed": 12345,
  "steps": 30,
  "lora_01": "add-detail.safetensors",
  "strength_01": 0.7
}
```

**Empfehlung:** `response_format: "b64_json"` benutzen (Bild kommt inline im JSON,
kein zweiter, auth-pflichtiger Abruf nötig — siehe §5).

### Response (200)

```jsonc
{
  "created": 1782590443,
  "data": [
    { "mime": "image/png", "b64_json": "<base64-PNG>" }   // bei response_format "b64_json"
    // ODER bei "url":
    // { "mime": "image/png", "url": "http://<ai-hub-host>:4000/v1/jobs/<id>/result/0" }
  ]
}
```

### curl

```bash
curl -s -m 240 http://<ai-hub-host>:4000/v1/images/generations \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen","prompt":"a red apple on a table","size":"512x512","n":1,"response_format":"b64_json"}'
```

---

## 3. Endpoint B — `POST /v1/images/edits` (Referenzbild(er) → Bild)

`Content-Type: multipart/form-data`. Für img2img / Bildbearbeitung mit
**hochgeladenen Referenzbildern** (strikt nach OpenAI-Edit-Standard, keine base64-Liste).

### Form-Felder

| Feld | Typ | Bedeutung |
|---|---|---|
| `image` | Datei(en) | **PFLICHT.** 1..16 Bilddateien → **positionsweise** auf die Bild-Slots des Workflows (in **Mapping-Reihenfolge**). Was ein leer gebliebener Slot tut, legt der Admin pro Slot fest: `placeholder` (Default, 8×8-Platzhalter) · `required` (bleibt leer, ComfyUI failt klar) · `disable` (Loader **und** der davon abhängige Zweig werden entfernt — der Job läuft ohne dieses Bild). Mehrfach `image=@...`. |
| `mask` | Datei | optional (OpenAI-Inpaint). Wird **nach** den `image`-Dateien als nächster positionaler Slot angehängt — der Mask-Slot (`LoadImageMask`) ist normal der letzte in der Mapping-Reihenfolge. |
| `model` | Text | **PFLICHT.** gen-alias. |
| `prompt` | Text | optional |
| `negative_prompt` | Text | optional |
| `size` | Text | optional, `BxH`/`auto` |
| `n` | Text | akzeptiert, aber **wirkungslos** — der Workflow bestimmt die Bildanzahl |
| `response_format` | Text | `b64_json` \| `url` |
| *alles andere* | Text | dynamische Scalar-Params (loras, seed, …) — Strings werden auto-gecastet (`"0.8"`→0.8) |

Effektives Limit ist `min(16, Anzahl Bild-Slots des Alias)` — überzählige Dateien
werden **still verworfen** (kein Fehler). Wie viele Slots ein Alias hat und wie sie
heißen, liefert `GET /v1/generations/{alias}/schema` (Liste `images`).

Response identisch zu §2 (`{created, data:[…]}`). Task ist intern `img2img`.
**Reihenfolge:** `image[0]` → erster Bild-Slot, `image[1]`/`mask` → nächster. Welcher
Slot was ist, steuert die Mapping-Reihenfolge im Gateway (per Drag sortierbar). Für
Inpaint also: Referenzbild als erstes `image`, Maske als `mask` (oder zweites `image`).

### curl

```bash
curl -s -m 240 http://<ai-hub-host>:4000/v1/images/edits \
  -H "Authorization: Bearer $KEY" \
  -F model=Qwen -F prompt="make it watercolor" -F response_format=b64_json \
  -F image=@ref1.png -F image=@ref2.png \
  -F lora_01=watercolor.safetensors -F strength_01=0.8
```

---

## 4. Dynamische Workflow-Parameter (LoRAs, seed, steps, … — ohne Presets)

Jeder Request-Key, der **nicht** zum OpenAI-Standard-Set gehört, wird als nativer
Workflow-Parameter weitergereicht und über das (gateway-seitige) Mapping des Alias in
den ComfyUI-Workflow injiziert. Damit lassen sich z.B. LoRAs pro Request setzen —
**vorausgesetzt, der Alias-Workflow exponiert diese Parameter** (das pflegt der
Gateway-Admin in der Gateway-UI ein, nicht anima-versa).

- `generations` (JSON): Extra-Keys = alle außer
  `prompt, model, n, size, response_format, negative_prompt, ref_images, quality,
  style, background, output_format, user, mode, ttl_s, params, stream`.
- `edits` (multipart): Extra-Felder = alle außer
  `model, prompt, negative_prompt, size, n, response_format, image, mask`.

anima-versa muss diese Parameter nur **als zusätzliche Felder im Request mitschicken**
(JSON-Keys bzw. Form-Felder). Welche Namen gültig sind (`lora_01`, `strength_01`, `seed`,
`steps`, `cfg`, …), ergibt sich aus dem konkreten Alias-Workflow — das ist Konfiguration
auf Gateway-Seite, nicht in anima-versa hartzucodieren. Ein anima-versa-UI-Feld
„Extra-Params (JSON)" oder ein generischer Key/Value-Block reicht.

### 4.1 LoRAs — gültige Liste, Kaskade, Routing

- **Gültige LoRAs pro Alias abfragen:** `GET /v1/generations/{alias}/loras` →
  `{"alias": "...", "loras": ["a.safetensors", …]}` (die Vereinigung der auf den
  Alias-Backends installierten LoRAs). Damit baust du in anima-versa einen **korrekten
  LoRA-Picker pro Alias**, statt Namen zu raten.
- **Kaskade (Slot-frei):** Schick deine LoRAs als `lora_1`, `lora_2`, … (+ optional
  `strength_N`). Die Gateway legt sie in die nächsten **freien** Slots des LoRA-Stacks —
  reservierte (gepinnte) Slots werden übersprungen. Du musst also **nicht** wissen,
  welcher physische Slot belegt ist; `lora_1` landet automatisch im ersten freien.
- **LoRA-bewusstes Routing:** Fragt ein Request eine LoRA an, die nur auf bestimmten
  Backends installiert ist, routet die Gateway **automatisch dorthin** (bzw. parkt
  darauf), statt auf ein Backend ohne die LoRA auszuweichen. Eine LoRA, die es nirgends
  gibt, wird ignoriert (dann entscheidet die normale Scheduler-Reihenfolge: unbezahlte
  Backends vor bezahlten, danach das schnellste laut gemessener Generierungszeit). Du
  brauchst also **kein** Backend vorzugeben — die Verfügbarkeit der LoRA steuert die
  Auswahl.
- **Zwei Schreibweisen, ein Pfad je Endpoint:** Auf `/v1/images/generations` und
  `/v1/images/edits` gilt **nur** die flache Form `lora_1`, `lora_2`, … (+ `strength_N`).
  Die im Alias-Schema beworbene Array-Form `loras: [{name, strength}]` versteht **nur**
  der native `POST /v1/generations` — auf den Image-Shims wird sie stillschweigend
  ignoriert. Umgekehrt müssen `lora_N` beim nativen Endpoint **in `params`** stehen
  (top-level werden dort nur width/height/steps/cfg/seed/sampler/scheduler/seconds gelesen).

---

## 5. `url` vs `b64_json` — wichtig wegen Auth

- **`b64_json` (empfohlen):** Bild ist base64-kodiert direkt im Response-JSON. Ein
  Request, kein Folge-Abruf. Robust.
- **`url`:** Liefert `http://<ai-hub-host>:4000/v1/jobs/<id>/result/<n>`. Diese URL ist
  **NICHT öffentlich** — der Abruf verlangt denselben `Authorization: Bearer`-Header
  (Job-Owner-Check). Wenn anima-versa `url` nutzt, muss der Image-Downloader den
  API-Key mitschicken. Anders als bei LocalAI (offene URLs). Im Zweifel `b64_json`.

Die Result-URLs bleiben bis zum TTL des Jobs abrufbar (Default 24 h).

**Video/Audio-Ausgabe:** Manche Aliase erzeugen kein Bild, sondern **Video/Audio**
(z.B. img2video). Beim Images-Shim trägt **jeder** `data[]`-Eintrag ein `mime` (bei
solchen Aliasen z.B. `video/mp4`) — die `url` liefert die Datei mit korrektem
Content-Type. Für
solche Aliase ist der **native `/v1/generations`** sauberer: `results[]` trägt pro
Artefakt `kind` (`image`/`video`/`audio`) + `mime` + `url`. Consumer sollten am
`mime`/`kind` entscheiden, nicht Bild annehmen.

---

## 6. Fehler- & Lastverhalten

Die Endpoints sind **synchron**: der HTTP-Request blockiert, bis das Bild fertig ist
(typisch ~30–90 s pro Bild). Ist das Backend belegt, kommt die **Parkzeit** obendrauf:
bis `async_park_timeout_s` (Default **600 s**) wartet der Job auf einen freien Slot,
erst danach failt er (502). Worst case also Parkzeit + Generierungszeit —
**Client-Timeout ≥ 660 s setzen** (oder den nativen Async-Pfad aus §7 nehmen).

| Code | Bedeutung | anima-versa |
|---|---|---|
| 200 | Bild(er) fertig | normal verarbeiten |
| 400 | `prompt` bzw. `image` fehlt | Request-Fehler, nicht retrien |
| 401 | API-Key ungültig | Config-Fehler |
| 402 | Monats-Kostenlimit des Users erreicht | dem User melden |
| 403 | Modell/Alias für diesen Key nicht erlaubt | Config-Fehler |
| 429 | Tages-Request-Kontingent des Users erschöpft | mit Backoff / am nächsten Tag |
| 502 | Generierung fehlgeschlagen / Park-Timeout (Backend zu lange busy) | optional 1× retrien |
| 503 | **Kein** gesundes Backend für den Alias (down/nicht gemappt) | mit Backoff retrien |

**Fehler-Body:** Das Gateway antwortet auf den Image-Pfaden im FastAPI-Format
`{"detail": "<Meldung>"}` — **nicht** im OpenAI-Format `{"error":{"message":…}}`.
Ein wiederverwendeter OpenAI-Provider muss die Meldung aus `detail` lesen, sonst
bleibt sie leer.

**Busy ≠ Fehler:** Wenn das ComfyUI-Backend gerade ausgelastet ist (Concurrency-Cap),
**queued das Gateway den Job intern und blockiert den Request, bis ein Slot frei wird**
(bis `async_park_timeout_s`, Default 600 s, dann 502). anima-versa braucht **keine eigene
Concurrency-Drosselung** — nur ein großzügiges Timeout. Mehrere parallele Requests
reihen sich auf dem Backend sauber ein.

---

## 7. (Optional) Nativer Async-Pfad für lange Jobs

Wenn anima-versa nicht minutenlang synchron blocken will, gibt es den nativen Job-Pfad:

1. `POST /v1/generations` → `{"model":"Qwen","prompt":"…","mode":"async","params":{"width":1024,"height":1024,"seed":1}}`
   → **202** `{"job_id":"…","status":"queued"}`
2. `GET /v1/jobs/<job_id>` → `{"job_id":"…","status":"queued|running|done|failed",
   "alias":"…","backend":"…","error":null,
   "results":[{"n":0,"kind":"image","mime":"image/png","name":"…","sha256":"…","url":"…"}]}`
   — `results` ist erst bei `status:"done"` gefüllt; laufende Jobs tragen zusätzlich
   `elapsed_s` und (wenn das Backend seinen Schrittzähler meldet) `progress`,
   `progress_step` und `eta_s`.
3. `GET /v1/jobs/<job_id>/result/<n>` → die Bilddatei (mit Bearer-Header).

Das ist optional und **nicht** OpenAI-Standard — nur nehmen, wenn ein nicht-blockierender
Flow gewünscht ist. Für die Standard-Anbindung reichen §2/§3.

---

## 8. Implementierungs-Aufgaben in anima-versa

1. **Provider-Type anlegen** (z.B. `ai_hub` / `comfyui_gateway`). Alternativ den
   bestehenden `openai_diffusion`-Provider wiederverwenden — das Gateway ist auf
   `/v1/images/generations` LocalAI-kompatibel (akzeptiert `ref_images` + Extra-Params).
   Eigener Type ist sauberer wegen Auth-auf-Result-URLs + gen-alias-Semantik.
2. **Config-Felder:** `base_url`, `api_key`, `model` (= gen-alias), Defaults für `size`,
   `n`, `response_format` (Default **`b64_json`** setzen), optional ein generischer
   Extra-Params-Block (für LoRAs/seed/steps).
3. **Request bauen:**
   - Text2Img → `POST /v1/images/generations` (JSON).
   - Mit hochgeladenen Referenzbildern → `POST /v1/images/edits` (multipart) **oder**
     `ref_images`-Liste im generations-JSON. Edits-multipart ist der saubere Weg für
     echte Datei-Uploads.
   - Extra-Params als zusätzliche JSON-Keys bzw. Form-Felder anhängen.
4. **Response parsen:** `data[*].b64_json` → Bytes dekodieren; bei `url` → GET mit
   demselben Bearer-Header.
5. **Timeout** auf ~660 s (oder Async-Pfad §7); Fehler-Codes aus §6 mappen.
6. **Auth:** Bearer-Header **immer** mitschicken — auch beim Result-URL-Abruf.

### Nicht-Ziele / Abgrenzung
- Keine ComfyUI-Workflow-Logik in anima-versa — der Alias kapselt das komplett.
- Keine eigene Job-Queue/Throttle — das Gateway parkt unter Last selbst.
- LoRA-/Param-Namen nicht hartzucodieren — pro Alias konfigurierbar lassen.

---

## 9. Verifikation (gegen prod)

```bash
KEY=<gateway-api-key>
BASE=http://<ai-hub-host>:4000

# 1) Text2Img, inline b64 — erwartet HTTP 200, data[0].b64_json gesetzt
curl -s -m 240 "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen","prompt":"a red apple","size":"512x512","response_format":"b64_json"}' \
  | python3 -c 'import sys,json;d=json.load(sys.stdin);print("ok", len(d["data"]), len(d["data"][0].get("b64_json","")))'

# 2) Edit mit Referenzbild — erwartet HTTP 200
curl -s -m 240 "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $KEY" \
  -F model=Qwen -F prompt="watercolor" -F response_format=b64_json -F image=@ref.png

# 3) Falscher Key → 401 ; unbekannter Alias → 403/503
```

Erwartung: (1) liefert in ~30–90 s ein PNG (b64 ~300–400 KB für 512²). Bei
ausgelastetem Backend blockiert der Call, statt sofort zu failen.
