# Tripo3D Developer API — technische Zusammenfassung

Recherchestand: 2026-09-03. Quellen: `https://developers.tripo3d.ai/en/docs/*` (V3),
`https://docs.tripo3d.ai/*` (V2, VitePress), `https://developers.tripo3d.ai/llms.txt`,
`https://github.com/VAST-AI-Research/tripo-python-sdk`.

---

## 0. WICHTIG ZUERST: V2 vs. V3 — welche API implementieren?

Es gibt **zwei parallele API-Generationen** mit unterschiedlicher Doku-Site:

| | **V2 (alt)** | **V3 (aktuell)** |
|---|---|---|
| Base-URL | `https://api.tripo3d.ai/v2/openapi` | `https://openapi.tripo3d.ai/v3` |
| Doku | `https://docs.tripo3d.ai` | `https://developers.tripo3d.ai/en/docs` |
| Aufrufstil | **EIN** Endpoint `POST /task` + Feld `type` | **pro Fähigkeit ein Endpoint**, kein `type` |
| Task-Query | `GET /v2/openapi/task/{task_id}` | `GET /v3/tasks/{task_id}` |
| Upload | `POST /v2/openapi/upload` → `image_token` | `POST /v3/files` → `file_token` |
| Eingabe-Referenz | `original_model_task_id`, `file: {…}`, `files: […]` | ein einziges Feld `input` (bzw. `inputs`) |
| Modellwahl | `model_version` | `model` |
| Zeitstempel/Credits | `create_time`, `consumed_credit` | `created_at`, `credits_consumed` |

**V2 wird abgeschaltet.** Laut Migrationsankündigung endet die V2-Wartung am
**2026-10-01**, und **ab 2026-11-01 00:00 UTC+8** nehmen alle V2-Endpoints keine
Requests mehr an.
→ **Empfehlung für eine Neuimplementierung: ausschließlich V3.**
UNKLAR: Die offizielle Changelog-Seite (`/en/docs/changelog`) nannte in meinem Abruf
KEIN Sunset-Datum; die Daten oben stammen aus Migrations-/SDK-Ankündigungen
(Websuche), nicht aus einer von mir direkt gelesenen Doku-Seite. Vor dem Bau bitte
einmal auf `/en/docs/migration-v2-to-v3` gegenprüfen.

Der Rest dieses Dokuments beschreibt **V3** als Primärquelle; V2-Äquivalente stehen
in Abschnitt 12.

---

## 1. Grundlagen (V3)

- **Base-URL:** `https://openapi.tripo3d.ai/v3`
- **Auth-Header:** `Authorization: Bearer <API_KEY>`
  (API-Key in der Tripo-Console unter „API Keys" erstellen:
  `https://platform.tripo3d.ai/api-keys` bzw. `https://developers.tripo3d.ai/en/keys`)
- **Content-Type:** `application/json` (Ausnahme: File-Upload = `multipart/form-data`)
- **Alle Generierungs-Endpoints sind ASYNCHRON.** Sie liefern sofort `task_id`
  zurück; das Ergebnis holt man per Polling (`GET /v3/tasks/{task_id}`) oder per
  Webhook.
- **Antwort-Hülle (immer gleich):**
  ```json
  { "code": 0, "data": { ... } }
  ```
  `code: 0` = **Aufruf auf HTTP/Protokollebene erfolgreich**. Ob der *Task*
  erfolgreich war, steht in `data.status` bzw. `data.error_code`. `code != 0`
  = Fehler, dann statt `data` die Felder `message` / `suggestion` / `request_id`
  (siehe Abschnitt 10).

### Empfohlener Ablauf
1. (optional) Bild/Mesh hochladen → `POST /v3/files` → `file_token`
2. Task erzeugen → `POST /v3/generation/...` → `task_id`
3. Pollen → `GET /v3/tasks/{task_id}` alle **2 s**, bis `status == "success"`
4. `output.model_url` **sofort** herunterladen (Signed URL, kurze Gültigkeit!)

---

## 2. File Upload

### 2.1 Normaler Upload — `POST /v3/files`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/files`
- **Header:** `Authorization: Bearer <API_KEY>`
- **Body:** `multipart/form-data`, Feldname **`file`**
- **Akzeptierte Formate / Größen:**
  - Bilder: **JPEG, PNG** (die Doku nennt an anderer Stelle auch **WebP** als
    zulässiges Eingabeformat für Bilder) — **max. 20 MB**
  - 3D-Modelle: **GLB, GLTF, FBX, OBJ, STL** — **max. 150 MB**
  - Bild-Mindestauflösung für Generierung: **256 × 256 px**
- **Beispiel:**
  ```bash
  curl -s \
    -H "Authorization: Bearer YOUR_API_KEY" \
    -F "file=@product-photo.png" \
    https://openapi.tripo3d.ai/v3/files
  ```
- **Response:**
  ```json
  { "code": 0, "data": { "file_token": "file_abc123" } }
  ```
- Der `file_token` wird danach direkt als `input` in jedem Generierungs-/
  Verarbeitungs-Endpoint verwendet.

UNKLAR: spezifische Fehlercodes des Upload-Endpoints (die Doku-Seite listet keine).
UNKLAR: ob WebP hier wirklich zulässig ist — die Upload-Seite nennt nur JPEG/PNG,
die Generierungsseiten nennen PNG/JPEG/WebP.

### 2.2 Große Dateien — `POST /v3/files/upload-credentials`

- Die Doku empfiehlt ab **> 60 MB** den Weg über vorsignierte S3-Credentials
  (Seite „File Upload (Large Files)", `/en/docs/files-presign`).
- Endpoint laut Changelog: **`POST /v3/files/upload-credentials`** — liefert
  temporäre STS-Credentials für einen direkten S3-Upload.
- UNKLAR: Ich konnte die Detailseite nicht laden (sie wird nur client-seitig
  gerendert). Request-Felder, Response-Shape (STS-AK/SK/Session-Token, Bucket,
  Key) und die Art der späteren Referenzierung (`object {bucket, key}` vs. ein
  zurückgegebener Token) sind **NICHT verifiziert**.
  Aus dem V2-Pendant (`POST /v2/openapi/upload/sts`) ist die Shape bekannt:
  ```json
  { "code": 0, "data": {
      "s3_host": "s3.us-west-2.amazonaws.com",
      "resource_bucket": "tripo-data",
      "resource_uri": "<key im bucket>",
      "sts_ak": "...", "sts_sk": "...", "session_token": "..." } }
  ```
  In V2 wird das danach als `file: {"type":"png","object":{"bucket":"tripo-data",
  "key":"<resource_uri>"}}` referenziert. Ob V3 identisch funktioniert: UNKLAR.
- **60 MB ist eine Performance-Empfehlung, keine harte Grenze** — der normale
  `POST /v3/files` akzeptiert Modelle bis 150 MB.

### 2.3 Kann ein Bild inline (base64 / URL) übergeben werden?

- **URL: JA.** Das Feld `input` akzeptiert eine öffentlich erreichbare
  Direkt-URL (z. B. `https://example.com/photo.png`).
- **base64 / Data-URI: NEIN** — in der V3-Doku nicht vorgesehen.
  `input` akzeptiert **genau eine** von drei Varianten: `file_token`, `url` oder
  `task_id`. „Do not pass multiple types simultaneously."
- In V2 gab es zusätzlich die verschachtelte Objektform
  `file: {type, file_token | url | object:{bucket,key}}`; in V3 ist das zu einem
  flachen String `input` zusammengefasst (bei Multiview weiterhin objektförmig,
  siehe 4.).

---

## 3. `image_to_model` — `POST /v3/generation/image-to-model`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/generation/image-to-model`
- **Header:** `Authorization: Bearer <API_KEY>`, `Content-Type: application/json`
- **Zwei Varianten (gleicher Pfad, unterschiedliche `model`-Werte):**
  H-Serie (`/en/docs/generation-image-to-model/standard`) und
  P-Serie (`/en/docs/generation-image-to-model/p`).

### Request-Body

| Feld | Typ | Pflicht | Default | Erlaubte Werte | Bedeutung |
|---|---|---|---|---|---|
| `input` | string | **ja** | — | `file_token` \| public URL \| `task_id` | Bildquelle; genau EINE Form |
| `model` | string | **ja** | — | `v3.1-20260211`, `v3.0-20250812`, `v2.5-20250123` (H-Serie); `P1-20260311`, `P2-20260801` (P-Serie) | Modellversion |
| `enable_image_autofix` | boolean | nein | `false` | | Eingabebild vor Generierung automatisch optimieren |
| `texture` | boolean | nein | `true` | | Texturen erzeugen (`false` spart 10 Credits) |
| `pbr` | boolean | nein | `true` | | PBR-Materialmaps erzeugen |
| `texture_quality` | string | nein | `standard` | `standard`, `detailed`, `extreme` | Texturauflösung; `detailed` +10, `extreme`/8K +20 Credits |
| `texture_alignment` | string | nein | `original_image` | `original_image`, `geometry` | Textur an Bild oder an Geometrie ausrichten |
| `texture_seed` | integer | nein | random | beliebig | reproduzierbare Textur |
| `model_seed` | integer | nein | random | beliebig | reproduzierbare Geometrie |
| `geometry_quality` | string | nein | `standard` | `standard`, `detailed` | nur ab `model >= v3.0`; `detailed` +20 Credits. UNKLAR, ob ältere Modelle das Feld ablehnen oder ignorieren — das Gateway sendet es über `tripo._PASSTHROUGH` bei JEDEM Generierungsrequest, auch für `v2.5-20250123` und die P-Serie (anders als bei `generate_parts` erzwingt `options_of` hier keine Modellregel). Ein 400 wäre ein finaler Job-Fehler. |
| `face_limit` | integer | nein | adaptiv | s. u. | Obergrenze Polygone |
| `quad` | boolean | nein | `false` | | Quad-Mesh statt Tris (+5 Credits); erzwingt in `convert` FBX |
| `smart_low_poly` | boolean | nein | `false` | | „handgebaute" saubere Topologie (+10 Credits) |
| `generate_parts` | boolean | nein | `false` | | segmentierte, editierbare Teile (+20 Credits) |
| `auto_size` | boolean | nein | `false` | | Skalierung auf reale Meter |
| `orientation` | string | nein | `default` | `default`, `align_image` | Rotation am Eingabebild ausrichten |
| `export_orientation` | string | nein | `+x` | `+x`, `-x`, `+y`, `-y` | Vorwärtsachse beim Export |
| `compress` | string | nein | — (meshopt) | `geometry` | Geometrie-/Meshopt-Kompression |
| `export_uv` | boolean | nein | `true` | | UV-Unwrap während der Generierung |
| `style` | string | nein | — | UNKLAR | Stilvorgabe; in der V3-H-Serie-Tabelle **nicht** aufgeführt, im Python-SDK und in V2 vorhanden (`ModelStyle`) |

**Bild-Constraints:** PNG/JPEG/WebP, max. 20 MB, min. 256 × 256 px, Motiv gut
sichtbar und wenig verdeckt.

**`face_limit`-Obergrenzen:**

| Modell | Tris (standard) | Tris (ultra/detailed) | Quads |
|---|---|---|---|
| `v3.1-20260211` | 1.500.000 | 2.000.000 | 150.000 |
| `v3.0-20250812` | 1.000.000 | 2.000.000 | 150.000 |
| `v2.5-20250123` | 500.000 | — | 150.000 |
| `P1-20260311` | UNKLAR (Gateway nimmt 50.000 an) | — | UNKLAR |
| `P2-20260801` | 48–50.000 | — | 48–25.000 |

**Im Gateway** (`tripo.FACE_MAX` / `FACE_MAX_QUAD` / `_FACE_MIN`): 1,5 M / 1 M / 500 k je
H-Modell, 50.000 für **beide** P-Modelle, 2 M für ein unbekanntes Modell — und mit
`quad: true` **pauschal 150.000 für jedes Modell**, also auch für die P-Serie, deren
Quad-Grenze diese Tabelle bei 25.000 sieht. Die Untergrenze ist 100, nicht 48. Ein
Client-`input_face_num` wird in [100, cap] geklemmt (`face_limit_for`), ein
Admin-`face_limit` wird NICHT geklemmt, sondern außerhalb des Bereichs verworfen
(`opt_face_limit` → Tripos adaptiver Default).

Die Spalte *Tris (ultra/detailed)* ist im Gateway nicht umgesetzt: `tripo._face_cap`
klemmt immer gegen die Standard-Obergrenze des Modells und kennt `geometry_quality`
nicht — ein `v3.1`-Alias landet also auch mit `geometry_quality: detailed` bei
1.500.000. Die 2.000.000 verwendet der Code nur als Fallback für ein Modell, das seine
Modelltabelle nicht kennt.

Empfehlungen der Doku: Game-ready 50.000–100.000; Web/Mobile 10.000–50.000;
mit `smart_low_poly` typisch 500–10.000 Quads.

**Inkompatibilitäten:** `generate_parts` lässt sich **nicht** mit `texture`,
`pbr`, `quad` oder `smart_low_poly` kombinieren (erfordert `texture: false`,
`pbr: false`).
Die Doku rät ab, `export_orientation` direkt zu setzen — besser weglassen und am
Ende über `POST /v3/models/convert` orientieren, da sonst die Post-Processing-
Kompatibilität umgangen wird.

### Beispiel-Request
```json
{
  "input": "https://example.com/image.png",
  "model": "v3.1-20260211",
  "texture": true,
  "pbr": true,
  "texture_quality": "detailed"
}
```

### Response (Task-Erzeugung)
```json
{ "code": 0, "data": { "task_id": "task_abc123" } }
```

---

## 4. `multiview_to_model` — `POST /v3/generation/multiview-to-model`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/generation/multiview-to-model`
- Alle Optionsfelder aus Abschnitt 3 gelten identisch (`model`, `texture`,
  `pbr`, `texture_quality`, `texture_alignment`, `geometry_quality`,
  `face_limit`, `quad`, `smart_low_poly`, `generate_parts`, `auto_size`,
  `orientation`, `export_orientation`, `compress`, `model_seed`, `texture_seed`,
  `export_uv`).
  Nur die Bildübergabe unterscheidet sich: **`inputs`** statt `input`.

### `inputs` — drei erlaubte Formen

**Form 1 (empfohlen): View-Key-Objekte**
```json
"inputs": [
  {"front": "https://example.com/front.png"},
  {"back":  "https://example.com/back.png"},
  {"right": "https://example.com/right.png"}
]
```
- Erlaubte Keys: **`front`, `left`, `back`, `right`**
- Werte: URL-String, `file_token`-String oder verschachteltes Objekt
  (`{"url": …}` / `{"file_token": …}` / `{"object": {"bucket": …, "key": …}}`)
- **Reihenfolge egal** — der Server kanonisiert intern auf
  `[front, left, back, right]`
- **`front` ist PFLICHT**; `left`, `back`, `right` sind optional
- **Minimum 2 Bilder** („Do not use less than two images to generate")

**Form 2 (Legacy, positionell):** genau 4 Strings in der Reihenfolge
`[front, left, back, right]`, ausgelassene Views als **leerer String `""`**:
```json
"inputs": ["file_front", "", "file_back", ""]
```

**Form 3 (Task-Referenz):** einelementiges Array mit Task-ID eines
vorangegangenen *Image-to-Multiview*- oder *Edit-Multiview*-Tasks:
```json
"inputs": [{"task_id": "<uuid>"}]
```
Der Quelltask muss `status: "success"` haben.

**Bildanforderungen:** PNG/JPEG/WebP, min. 256 × 256 px, alle Bilder zeigen
dasselbe Objekt bei konsistenter Beleuchtung.

UNKLAR: Ein Feld **`mode`** (wie in der Fragestellung vermutet) existiert in der
V3-Doku **nicht**. In V2 gab es bei manchen Multiview-Varianten `mode: "LEFT"`
bzw. die feste 4er-Reihenfolge; in V3 ist das durch die View-Keys ersetzt.

---

## 5. `text_to_model` — `POST /v3/generation/text-to-model`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/generation/text-to-model`
- Pflichtfelder: **`prompt`** (string) und **`model`**.
- Zusätzlich zu den Optionen aus Abschnitt 3: `negative_prompt` (string),
  `image_seed` (integer).
- Kein `input` (keine Bildquelle).
- Beispiel aus dem Quick Start:
  ```json
  { "prompt": "a cute cat", "model": "tripo-v3.1" }
  ```
  UNKLAR: Der Quick Start nennt hier den Modellwert `"tripo-v3.1"`, die
  Endpoint-Seiten dagegen `"v3.1-20260211"`. Vermutlich ein Alias; vor dem Bau
  einmal praktisch verifizieren.

---

## 6. Task-Status — `GET /v3/tasks/{task_id}`

- **Methode/Pfad:** `GET https://openapi.tripo3d.ai/v3/tasks/{task_id}`
- **Header:** `Authorization: Bearer <API_KEY>`
- **Wichtig:** Ein Task muss mit **demselben API-Key** abgefragt werden, mit dem
  er erzeugt wurde — sonst „not found".

### Status-Werte

| Status | Bedeutung |
|---|---|
| `queued` | wartet in der Warteschlange |
| `running` | in Bearbeitung |
| `success` | erfolgreich abgeschlossen |
| `failed` | fehlgeschlagen |
| `cancelled` | vom Nutzer abgebrochen |

V2 kannte zusätzlich `banned` (Content-Policy-Verstoß), `expired`
(Aufbewahrungsfrist überschritten) und `unknown`.
UNKLAR: Ob V3 diese drei ebenfalls liefern kann — die V3-Seite listet nur die
fünf oben. **Defensiv behandeln: jeden unbekannten Status als terminalen
Fehlschlag werten**, nicht endlos pollen.

### Response

```json
{
  "code": 0,
  "data": {
    "task_id": "task_abc123",
    "type": "text_to_model",
    "status": "success",
    "progress": 100,
    "output": {
      "model_url": "https://cdn.tripo3d.ai/output/model_pbr.glb",
      "rendered_image_url": "https://cdn.tripo3d.ai/output/preview.png"
    },
    "credits_consumed": 100.00,
    "created_at": "2026-04-28T12:00:00Z",
    "completed_at": "2026-04-28T12:01:30Z"
  }
}
```

| Feld | Typ | Bemerkung |
|---|---|---|
| `task_id` | string | wie angefragt |
| `type` | string | Task-Kategorie |
| `status` | string | s. o. |
| `progress` | integer | 0–100 |
| `output` | object | **nur bei `status == "success"`** |
| `credits_consumed` | number | Dezimal, 2 Nachkommastellen; 0 bei Fehlschlag |
| `created_at` / `completed_at` | string | ISO 8601; `completed_at` optional |
| `error_code` | integer | nur bei Fehlschlag |
| `error_message` | string | nur bei Fehlschlag |
| `input` | object | Task-Konfiguration (in V2 dokumentiert) |

**V2-only-Felder** (in V3 nicht mehr dokumentiert): `queuing_num`
(Warteschlangenposition, `-1` wenn nicht in der Queue), `running_left_time`
(geschätzte Restsekunden, `-1` wenn terminal).
UNKLAR: ob V3 diese weiterhin mitliefert.

### `output`-Felder je Task-Typ

| Task | Felder |
|---|---|
| Generierung (text/image/multiview → model) | `model_url`, `rendered_image_url`; V2 zusätzlich `base_model` (untexturiert), `pbr_model` |
| Rig-Check | `riggable` (bool), `rig_type` (string) |
| Auto-Rig | `model_url` |
| Retarget | `model_url` |
| Convert | `model_url` |
| Bildgenerierung | `generated_image_url` |
| Image-to-Multiview | `front_view_url`, `left_view_url`, `back_view_url`, `right_view_url` (V2-Namen) |

UNKLAR: Die V3-Doku zeigt in ihren Beispielen durchgängig nur
`model_url` + `rendered_image_url`. Ob V3 die V2-Äquivalente `base_model` /
`pbr_model` als `base_model_url` / `pbr_model_url` mitliefert, ist **nicht
belegt**. Praktisch: `output` einmal komplett loggen und dann fest verdrahten.

### Signed URLs / Download

- Die Ergebnis-URLs sind **vorsignierte CDN-URLs mit sehr kurzer Gültigkeit:
  laut V2-Doku und V3-Quick-Start ca. 5 Minuten**. Danach ist ein erneutes
  `GET /v3/tasks/{task_id}` nötig, um frische URLs zu bekommen.
- Der **Download selbst braucht KEINEN `Authorization`-Header** — die Signatur
  steckt in der URL. (Die V3-Seite sagt dazu nichts explizit; das ist aus der
  Signed-URL-Natur und der V2-Doku abgeleitet — UNKLAR, aber sehr wahrscheinlich.)
- **Konsequenz für eine Gateway-Implementierung:** direkt nach `success`
  herunterladen, nicht die URL an den Client durchreichen.
  ACHTUNG, im heutigen Gateway nur bei Ein-Task-Jobs erfüllt:
  `CloudTaskAdapter.generate` lädt `state.downloads` erst, wenn `_run` fertig ist —
  bei einem Tripo-Job mit Convert-/Clip-Tasks liegen zwischen dem `success` des
  Haupttasks und seinem Download also alle Folge-Tasks. Falls die
  5-Minuten-Gültigkeit stimmt, muss vor dem Download ein frisches
  `GET /v3/tasks/{task_id}` geholt werden (oder jeder Task sofort nach seinem
  eigenen `success`).

### Polling & weitere Task-Endpoints
- Empfohlenes Poll-Intervall: **2 Sekunden**.
- **Batch-Abfrage:** `POST /v3/tasks/list` (seit V3 neu).
- UNKLAR: ein **Cancel-Endpoint** ist in der V3-Doku nicht auffindbar, obwohl
  der Status `cancelled` existiert. Für ein Gateway heißt das: ein laufender
  Tripo-Task lässt sich vermutlich **nicht** abbrechen (wie bei Meshy) — Credits
  sind dann verbraucht.

---

## 7. Rigging & Animation

### 7.1 Rig-Check — `POST /v3/animations/rig-check`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/animations/rig-check`
- **Kosten: 0 Credits (kostenlos).**
- Die Doku fordert, **vor** jedem Auto-Rig einen Rig-Check zu fahren.

**Request:**

| Feld | Typ | Pflicht | Werte |
|---|---|---|---|
| `input` | string | ja | `task_id` \| `file_token` \| public URL — genau EINE Form |

Format: **nur GLB**, max. **150 MB** — im Widerspruch zur Feldtabelle des Auto-Rigs in
7.2, die für `input` zusätzlich GLTF/FBX/OBJ/STL nennt. Da die Doku den Rig-Check vor
jedem Rig verlangt, ist GLB der gemeinsame Nenner. Das Gateway (`tripo.mesh_ext`)
akzeptiert für `input_mesh_path` deshalb ausschließlich binäres glTF und weist alles
andere ab, bevor Bytes hochgeladen oder 25 Credits ausgegeben werden.

```bash
curl --request POST \
  --url https://openapi.tripo3d.ai/v3/animations/rig-check \
  --header 'Authorization: Bearer <token>' \
  --header 'Content-Type: application/json' \
  --data '{"input": "task_abc123"}'
```

**Response (fertiger Task):**
```json
{
  "code": 0,
  "data": {
    "task_id": "task_def456",
    "status": "success",
    "output": { "riggable": true, "rig_type": "biped" },
    "credits_consumed": 0.00
  }
}
```

`rig_type`-Werte: `biped`, `quadruped`, `hexapod`, `octopod`, `avian`,
`serpentine`, `aquatic`.

### 7.2 Auto-Rig — `POST /v3/animations/rig`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/animations/rig`
- **Kosten: 25 Credits** (Preisliste). Das Doku-Beispiel zeigt `30.00` —
  vermutlich illustrativ bzw. inkl. Zuschlag. UNKLAR: exakter Preis.

**Request:**

| Feld | Typ | Pflicht | Default | Erlaubte Werte |
|---|---|---|---|---|
| `input` | string | **ja** | — | `task_id` \| `file_token` \| public URL (laut Feldtabelle GLB, GLTF, FBX, OBJ, STL; max. 150 MB — der Rig-Check in 7.1 nennt dagegen **nur GLB**, siehe dort) |
| `model` | string | nein | `v1.0-20240301` | `v1.0-20240301`, `v2.5-20260210` |
| `rig_type` | string | nein | `biped` | `biped`, `quadruped`, `hexapod`, `octopod`, `avian`, `serpentine`, `aquatic` |
| `spec` | string | nein | `tripo` | **`tripo`, `mixamo`** |
| `out_format` | string | nein | `glb` | `glb`, `fbx` |

**Modellunterschiede:**
- `v1.0-20240301`: **nur `biped`**, dafür 90+ Animations-Presets im Retarget.
- `v2.5-20260210`: alle `rig_type`-Werte inkl. nicht-humanoider Kreaturen,
  aber deutlich kleinerer Preset-Katalog (s. 7.3).

**`spec: "mixamo"`** liefert **Mixamo-kompatible Bone-Namen** — genau das, was
für Unity/Unreal-Pipelines und für Mixamo-Retargeting gebraucht wird.
`spec: "tripo"` für eigene Pipelines.
UNKLAR: ob `spec: "mixamo"` mit **beiden** `model`-Werten funktioniert — die Doku
listet `spec` ohne Modell-Einschränkung, sagt es aber nicht explizit.

**Response (fertiger Task):**
```json
{
  "code": 0,
  "data": {
    "task_id": "task_def456",
    "status": "success",
    "output": { "model_url": "https://cdn.tripo3d.ai/output/rigged.glb" },
    "credits_consumed": 30.00
  }
}
```

**→ Antwort auf die Kernfrage: JA, ein HOCHGELADENES Mesh kann gerigged werden.**
In **V3** akzeptiert `input` direkt einen `file_token` (aus `POST /v3/files`)
**oder** eine öffentliche URL — ein Tripo-Generierungstask ist **nicht** nötig,
und ein separater „Import"-Schritt entfällt.
In **V2** ging das nicht direkt: dort verlangt `animate_rig` zwingend
`original_model_task_id`; für Fremd-Meshes gibt es deshalb den eigenen Task-Typ
**`import_model`** (siehe 12.4), der eine hochgeladene Datei (bis 150 MB) in eine
`task_id` verwandelt, die dann rigbar/konvertierbar ist. In V3 ist das Pendant
`POST /v3/models/import`, aber praktisch überflüssig.

**Wichtiger Hinweis der Doku:** Remeshing, Mesh-Segmentierung und
Mesh-Completion **zerstören Skelett-Binding und Animationsdaten**. Reihenfolge
also: alle Mesh-Bearbeitungen **zuerst**, Rigging **zuletzt**.
V2 ergänzt: Quellmodelle der Version 1.x werden **nicht** unterstützt (erst
`Turbo-v1.0-20250506` / `v2.0+`).

### 7.3 Retarget — `POST /v3/animations/retarget`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/animations/retarget`
- **Kosten: 10 Credits pro Animation.**

**Request:**

| Feld | Typ | Pflicht | Default | Werte |
|---|---|---|---|---|
| `input` | string | **ja** | — | `task_id` eines **gerigten** Modells (Präfix `task_`) |
| `animation` | string | bedingt | — | ein Preset; exklusiv zu `animations` |
| `animations` | string[] | bedingt | — | mehrere Presets, **max. 5**; exklusiv zu `animation` |
| `out_format` | string | nein | `glb` | `glb`, `fbx` |
| `bake_animation` | boolean | nein | `true` | nur bei GLB wirksam |
| `export_with_geometry` | boolean | nein | `true` | Geometrie mit exportieren (Python-SDK: Default `false`) |
| `animate_in_place` | boolean | nein | `false` | Animation ohne Wurzelversatz („in place") |

Genau eines von `animation` / `animations` muss gesetzt sein.
**Retarget akzeptiert nur Task-IDs von GERIGTEN Modellen** — kein `file_token`.

**Animations-Presets bei Rig `v2.5-20260210`:**
`preset:idle`, `preset:walk`, `preset:run`, `preset:dive`, `preset:climb`,
`preset:jump`, `preset:slash`, `preset:shoot`, `preset:hurt`, `preset:fall`,
`preset:turn`, `preset:quadruped:walk`, `preset:hexapod:walk`,
`preset:octopod:walk`, `preset:serpentine:march`, `preset:aquatic:march`

**Rig `v1.0-20240301`:** 90+ Biped-Presets (u. a. `afraid`, `agree`,
`angry_01`–`angry_03`, `basketball_shot`, `bow`, diverse Kicks, Tänze,
Emotionen, Sportbewegungen).
UNKLAR: die vollständige 90+-Liste habe ich nicht vollständig extrahieren können.

**Beispiele:**
```json
{ "input": "task_abc123", "animation": "preset:walk",
  "out_format": "glb", "bake_animation": true }
```
```json
{ "input": "task_abc123",
  "animations": ["preset:idle", "preset:walk", "preset:run"],
  "out_format": "fbx" }
```

**Response:**
```json
{ "code": 0, "data": { "task_id": "task_def456", "status": "success",
  "output": { "model_url": "https://cdn.tripo3d.ai/output/animated.glb" },
  "credits_consumed": 20.00 } }
```

**Hinweis:** Wenn Rig-Check `riggable: false` liefert, empfiehlt die Doku, das
Modell mit sauberem Prompt neu zu generieren (komplexe Posen vermeiden).

---

## 8. Format-Konvertierung — `POST /v3/models/convert`

- **Methode/Pfad:** `POST https://openapi.tripo3d.ai/v3/models/convert`
- **Kosten: 5 Credits (Basis), 10 Credits mit erweiterten Parametern.**

| Feld | Typ | Pflicht | Default | Werte / Bemerkung |
|---|---|---|---|---|
| `input` | string | **ja** | — | `task_id` \| `file_token` \| public URL (GLB, GLTF, FBX, OBJ, STL; max. 150 MB) |
| `format` | string | **ja** | — | `GLTF`, `FBX`, `USDZ`, `OBJ`, `STL`, `3MF` (Großschreibung!); `GLB` ist in der Feldtabelle **nicht** gelistet — UNKLAR, ob der Endpoint es annimmt |
| `quad` | boolean | nein | `false` | Vierecke statt Dreiecke; **erzwingt FBX** |
| `force_symmetry` | boolean | nein | `false` | nur wirksam bei `quad: true` |
| `face_limit` | integer | nein | — (Original bleibt) | V2-Default war `10000` |
| `flatten_bottom` | boolean | nein | `false` | Unterseite planieren |
| `flatten_bottom_threshold` | float | nein | `0.01` | Tiefe der Planierung |
| `texture_size` | integer | nein | `4096` | Diffuse-Textur in Pixeln (V2 nennt 2048 od. 4096) |
| `texture_format` | string | nein | `JPEG` | `JPEG`, `PNG`, `WEBP`, `BMP`, `DPX`, `HDR`, `OPEN_EXR`, `TARGA`, `TIFF` |
| `bake` | boolean | nein | `true` | erweiterte Materialien in Basistexturen backen |
| `pack_uv` | boolean | nein | `false` | UV-Layout vereinheitlichen |
| `export_vertex_colors` | boolean | nein | `false` | nur für OBJ und GLTF |
| `pivot_to_center_bottom` | boolean | nein | `false` | Pivot auf Mitte-Unten setzen |
| `scale_factor` | float | nein | `1` | Skalierungsfaktor |
| `with_animation` | boolean | nein | `true` | Skelett-/Animationsdaten mitnehmen (Python-SDK: Default `false`) |
| `animate_in_place` | boolean | nein | `false` | Wurzelversatz unterdrücken |
| `part_names` | string[] | nein | — | Namen der zu exportierenden Meshes/Segmente |
| `export_orientation` | string | nein | `+x` | `+x`, `-x`, `+y`, `-y` |
| `fbx_preset` | string | nein | `blender` | `blender`, `3dsmax`, `mixamo`, `bake_scale` |

**Gateway-Hinweis:** `adapters.TripoAdapter._run` schickt jedes Zielformat, das der
Haupttask nicht selbst liefert, als eigenen Convert-Task (`tripo.build_convert`,
`format` in Großbuchstaben). Beim **Rig**-Endpoint ist das native Format das ERSTE
gehakte Lieferformat, ein Rig-Alias mit `deliver formats = fbx, glb` löst deshalb einen
Convert nach `GLB` aus. Solange oben nicht geklärt ist, ob `GLB` zulässig ist: bei
Rig-Aliassen `glb` zuerst haken.

**Beispiel:**
```json
{
  "input": "task_abc123",
  "format": "FBX",
  "texture_size": 2048,
  "texture_format": "PNG",
  "pivot_to_center_bottom": true,
  "fbx_preset": "blender"
}
```

V2-Zusatzeinschränkung: `original_model_task_id` muss von
`Turbo-v1.0-20250506` oder neuer stammen.

---

## 9. Welche Formate liefert `image_to_model` direkt?

- **Direkt aus der Generierung: GLB.** `output.model_url` zeigt in allen
  Doku-Beispielen auf eine `.glb`-Datei; ein `format`/`out_format`-Feld gibt es
  bei den Generierungs-Endpoints **nicht**.
- **FBX/OBJ/STL/USDZ/3MF/GLTF nur über `POST /v3/models/convert`** (Abschnitt 8),
  als eigener Task mit eigenen Credits.
- **Ausnahme Animation:** `POST /v3/animations/rig` und `.../retarget` haben ein
  eigenes `out_format` mit `glb` | `fbx` — ein gerigtes/animiertes Modell kann
  also **direkt als FBX** geliefert werden, ohne Convert-Schritt.
- `quad: true` in der Generierung erzeugt zwar ein Quad-Mesh, aber die
  Quad-Topologie ist praktisch nur über FBX transportierbar — die Convert-Doku
  sagt explizit, `quad` erzwinge FBX.
- UNKLAR: ob die Generierung neben `model_url` (PBR/texturiert) noch eine
  untexturierte `base_model`-Variante liefert (V2 tat das). Siehe Abschnitt 6.

---

## 10. Fehler, Statuscodes, Rate Limits, Concurrency

### 10.1 Fehler-Body (V3)

```json
{
  "code": 1000,
  "message": "Invalid API Key",
  "suggestion": "Check whether the API Key is correct or has been deleted",
  "request_id": "..."
}
```

`code: 0` = **kein Fehler** (Transportebene ok). Bei `code: 0` steht das
eigentliche Ergebnis in `data`; ein *fehlgeschlagener Task* kommt ebenfalls mit
`code: 0`, aber `data.status == "failed"` und `data.error_code` /
`data.error_message`.

### 10.2 Fehlercodes V3

| `code` | HTTP | Message | Behandlung |
|---|---|---|---|
| 1000 | 401 | Invalid API Key | Key prüfen / gelöscht? |
| 1001 | 401 | Unauthorized | `Authorization`-Header fehlt |
| 1007 | 429 | Rate limit exceeded (Request-Rate) | s. 10.4 |
| 2000 | 429 | Rate limit exceeded (Concurrency) | Backoff, `Retry-After` beachten |
| 2002 | 400 | Unsupported request parameter | Feldnamen/Werte prüfen |
| 2003 | 400 | Empty input file | Datei leer/ungültig |
| 2004 | 400 | Unsupported file type | Format prüfen |
| 2008 | 400 | Content policy violation | verbotener Inhalt |
| 2010 | 403 | Insufficient credits | Credits aufladen |
| 2015 | 400 | Version deprecated | neuere API-/Modellversion nutzen |
| 2018 | 400 | Model too complex | Polycount/Komplexität senken |

**Retry-Politik der Doku:** 429 und 500 sind retrybar — exponentielles Backoff
1 s, 2 s, 4 s …, max. 5 Versuche, Deckel bei 32 s. 400/401/403 sind **nicht**
retrybar.

### 10.3 Fehlercodes V2 (zur Referenz)

| HTTP | `code` | Message |
|---|---|---|
| 500 | 1000 | Unknown error on server side |
| 500 | 1001 | Fatal error on server side |
| 401 | 1002 | Authentication failed |
| 400 | 1003 | The request body is malformed |
| 400 | 1004 | One or more of your parameter is invalid |
| 403 | 1005 | You are not allowed to access this resource |
| 429 | 1007 | Rate limit exceeded |
| 429 | 2000 | You have exceeded the limit of generation |
| 404 | 2001 | The task is not found |
| 400 | 2002–2018 | diverse Task-/Input-Validierungsfehler |
| 403 | 2010 | You don't have enough credit |
| 404 | 2019 | The file is not found |

**Achtung Kollision:** In V2 ist `1000` ein Serverfehler (HTTP 500), in V3 ein
ungültiger API-Key (HTTP 401). Codes also nie versionsübergreifend mappen.

### 10.4 Rate Limits (Request-Rate, V3)

- Gilt **pro API-Key**.
- Generierungs-Endpoints (`/v3/generation/*`) haben niedrigere Limits als
  Query-Endpoints (`/v3/tasks/*`).
- Überschreitung → **HTTP 429, `code: 1007`**.
- Response-Header:
  - `X-RateLimit-Limit` — max. Requests im aktuellen Fenster
  - `X-RateLimit-Remaining` — verbleibende Requests
  - `X-RateLimit-Reset` — Unix-Timestamp des Fenster-Resets
- UNKLAR: konkrete Zahlenwerte (Requests/Minute) nennt die Doku nicht.
- V2 nannte für Uploads **10 qps**.

### 10.5 Concurrency-Limits (V3, Default)

Gilt **pro Account**, je Kategorie ein eigener Pool:

| Kategorie | Default |
|---|---|
| 3D Generation — H-Serie | 10 |
| 3D Generation — P-Serie | 5 |
| Image Generation | 1 |
| Animation (rig-check, rig, retarget) | 10 |
| Model Processing (convert, texture, import) | 5 |
| Mesh Operations | 10 |

Überschreitung → **HTTP 429, `code: 2000`**, mit **`Retry-After`**-Header
(Sekunden). Backoff bei 1 s starten, verdoppeln, bei 32 s deckeln;
`Retry-After` und `X-RateLimit-Reset` haben Vorrang vor eigener Heuristik.

V2-Werte zum Vergleich: text/image/multiview_to_model = 10 (P1-20260311: 5),
`refine_model` = 5, alle `animate_*` = 10,
`generate_multiview_image`/`edit_multiview_image` = 1, sonst 10.

### 10.6 402 / Guthaben

- **Kein HTTP 402.** Zu wenig Guthaben ist **HTTP 403 mit `code: 2010`**
  („Insufficient credits" / „You don't have enough credit").
  (V2 identisch: 403/2010.)

---

## 11. Guthaben & Preise

### 11.1 Balance — `GET /v3/account/balance`

```json
{ "code": 0, "data": { "balance": 10000.00, "frozen": 200.00 } }
```

- `balance` (number): verfügbares Guthaben, Dezimal mit 2 Nachkommastellen
- `frozen` (number): für **laufende Tasks reservierte** Credits

Zusätzlich: **`GET /v3/account/usage`** — Verbrauchshistorie pro Task mit
`task_id`, `type`, `credits_consumed`, `created_at`.

UNKLAR: V2 hatte vermutlich `GET /v2/openapi/user/balance` mit derselben Shape;
diese Seite ist in der V2-Navigation **nicht** enthalten und ich konnte sie nicht
direkt verifizieren. Für V3 ist `/v3/account/balance` belegt.

### 11.2 Credits (1 Credit = 0,01 USD; 100 Credits = 1,00 USD)

**Basis-Generierung:**

| Task | ohne Textur | mit Standard-Textur |
|---|---|---|
| Text → 3D | 10 | 20 |
| **Image → 3D** | **20** | **30** |
| Multiview → 3D | 20 | 30 |

Für die **P-Serie** deutlich teurer: V2-Preisliste nennt für P1
Image/Multiview 40 (ohne Textur) / 50 (mit); der Changelog nennt für
**P2-20260801** 100 Basis, 110–130 mit Texturoptionen.

**Aufschläge (stapelbar, additiv auf die Basis):**

| Option | Aufschlag |
|---|---|
| HD-Textur (`texture_quality: detailed`) | +10 |
| 8K-Ultra-Textur (`texture_quality: extreme`) | +20 |
| HD-Geometrie (`geometry_quality: detailed`) | +20 |
| Quad-Mesh (`quad: true`) | +5 |
| Smart Low-poly (`smart_low_poly: true`) | +10 |
| Generate Parts (`generate_parts: true`) | +20 |

**Nachbearbeitung / Animation:**

| Operation | Credits |
|---|---|
| **Rig-Check** | **0 (kostenlos)** |
| **Auto-Rig** | **25** |
| **Retarget** | **10 pro Animation** |
| Format-Convert (Basis) | 5 |
| Format-Convert (erweiterte Parameter) | 10 |
| Texture (standard / HD / 8K) | 10 / 20 / 30 |
| Retopology v2.0 (smart) / v1.0 (basic) | 30 / 10 |
| Mesh-Segmentierung (V2) | 40 |
| Mesh-Completion (V2) | 50 |

**Beispielrechnung für eine typische Gateway-Kette**
(Bild → Modell mit Textur → Rig-Check → Auto-Rig mit Mixamo-Spec, FBX-Ausgabe):
30 + 0 + 25 = **55 Credits ≈ 0,55 USD**. Ein Convert-Schritt entfällt, weil
`animations/rig` `out_format: "fbx"` direkt kann.

---

## 12. V2-Referenz (nur nötig, falls doch V2 gebaut wird)

Base-URL: `https://api.tripo3d.ai/v2/openapi`, Header
`Authorization: Bearer YOUR_TRIPO_API_KEY`.

### 12.1 Ein Endpoint für alles
```bash
curl -X POST https://api.tripo3d.ai/v2/openapi/task \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TRIPO_API_KEY" \
  -d '{ "type": "text_to_model", "prompt": "A dachshund standing on a stool." }'
```
Polling: `GET https://api.tripo3d.ai/v2/openapi/task/{task_id}` (alle 2 s).

UNKLAR: Die V2-Seite „Get your task result" nannte in meinem Abruf den Pfad
`GET /v1/tasks/{task_id}`, was dem Quick Start (`/v2/openapi/task/{id}`)
widerspricht. Ich halte den Quick-Start-Pfad für richtig; das war vermutlich ein
Extraktionsfehler.

### 12.2 Task-Typen (V2)
`text_to_model`, `image_to_model`, `multiview_to_model`, `import_model`,
`refine_model`, `texture_model`, `convert_model`, `stylize_model`,
`mesh_segmentation`, `mesh_completion`, `highpoly_to_lowpoly`,
`animate_prerigcheck`, `animate_rig`, `animate_retarget`,
`text_to_image`, `generate_image`, `generate_multiview_image`,
`edit_multiview_image`.

### 12.3 Bildübergabe (V2)
```json
{ "type": "image_to_model",
  "file": { "type": "png",
            "file_token": "…"       // ODER
            // "url": "https://…"    // JPEG/PNG, max 20 MB
            // "object": {"bucket": "tripo-data", "key": "<resource_uri>"}
          },
  "model_version": "v3.1-20260211" }
```
Bei `multiview_to_model`: **`files`** = Array von 4 solchen Objekten in der
Reihenfolge **[front, left, back, right]**, `front` Pflicht, ausgelassene Views
als leeres Objekt/leerer String; alternativ `original_task_id` (exklusiv zu
`files`). Mindestens 2 Bilder.

Upload:
- `POST /v2/openapi/upload` — multipart, Feld `file`, nur **webp/jpeg/png**,
  max. 20 MB, **keine Modelldateien** → `{"code":0,"data":{"image_token":"…"}}`
- `POST /v2/openapi/upload/sts` — Feld `format` (Bilder: `webp|jpeg|png`;
  Modelle: `glb|obj|fbx|stl`), max. 20 MB → STS-Credentials (s. 2.2)

### 12.4 `import_model` (V2) — Fremd-Mesh in eine Task-ID verwandeln
```json
{ "type": "import_model",
  "file": { "object": { "bucket": "tripo-data", "key": "<resource_uri>" } } }
```
- versionsunabhängig, **max. 150 MB**, STS-Upload empfohlen
- Ergebnis: `task_id`, die dann in `animate_prerigcheck` / `animate_rig` /
  `convert_model` als `original_model_task_id` einsetzbar ist.
- **Das ist in V2 der einzige Weg, ein hochgeladenes Mesh zu riggen.**

### 12.5 V2-Rigging
- `animate_prerigcheck`: `{type, original_model_task_id}` → `riggable`, `rig_type`
- `animate_rig`: `{type, original_model_task_id, out_format(glb|fbx, def. glb),
  model_version(v1.0-20240301|v2.0-20250506|v2.5-20260210), rig_type(def. biped),
  spec(tripo|mixamo, def. tripo)}`
- `animate_retarget`: `{type, original_model_task_id, animation|animations(max 5),
  out_format, bake_animation(def. true, nur glb), export_with_geometry(def. true),
  animate_in_place(def. false)}`
- Quellmodelle müssen `Turbo-v1.0-20250506` oder neuer sein; **1.x wird nicht
  unterstützt**.

### 12.6 V2-Modellversionen `image_to_model`
`v3.1-20260211`, `v3.0-20250812` (H3) · `v2.5-20250123`, `v2.0-20240919` (H2) ·
`Turbo-v1.0-20250506` · `v1.4-20240625` (Legacy) · `P1-20260311`, `P2-20260801`
(P-Serie).
Ergänzende V2-Felder gegenüber V3: `enable_image_autofix`, `export_uv`,
`geometry_quality`, `style` — inhaltlich identisch.

---

## 13. Offene Punkte (Zusammenfassung aller UNKLAR-Marken)

1. **V2-Sunset-Daten** (2026-10-01 / 2026-11-01) stammen aus der Websuche, nicht
   aus einer direkt gelesenen Doku-Seite.
2. **Large-File-Upload** (`POST /v3/files/upload-credentials`, `/en/docs/files-presign`):
   Request/Response-Shape nicht verifiziert.
3. **`output`-Felder in V3**: nur `model_url` + `rendered_image_url` belegt.
   Ob `base_model_url` / `pbr_model_url` existieren: unbekannt.
4. **URL-Gültigkeit in V3**: 5 Minuten stammt aus V2-Doku + Quick Start;
   auf der V3-Task-Seite nicht wiederholt.
5. **Download ohne Auth-Header**: sehr wahrscheinlich (Signed URL), nicht explizit
   dokumentiert.
6. **Cancel-Endpoint**: in V3 nicht auffindbar, obwohl Status `cancelled` existiert.
7. **Status `banned` / `expired` / `unknown`**: in V3 nicht mehr gelistet, aber
   möglicherweise weiterhin geliefert.
8. **`queuing_num` / `running_left_time`**: V2-Felder, V3-Verfügbarkeit unbekannt.
9. **`style`-Feld** bei `image_to_model` in V3: im SDK und in V2 vorhanden, in der
   V3-Feldtabelle nicht gelistet.
10. **`text_to_model`-Modellwert** `"tripo-v3.1"` (Quick Start) vs.
    `"v3.1-20260211"` (Endpoint-Seite).
11. **`spec: "mixamo"` × `model: "v2.5-20260210"`**: Kombinierbarkeit nicht
    explizit bestätigt.
12. **Auto-Rig-Preis**: Preisliste 25 Credits, Doku-Beispiel 30.
13. **Vollständige 90+-Preset-Liste** für Rig `v1.0-20240301` nicht extrahiert.
14. **Konkrete Request-Rate-Zahlen** (req/min) nennt die Doku nicht.
15. **WebP im Upload**: Upload-Seite nennt nur JPEG/PNG, Generierungsseiten
    PNG/JPEG/WebP.
16. **`GET /v2/openapi/user/balance`**: nicht verifiziert (V2-Navigation kennt
    keine Balance-Seite).
