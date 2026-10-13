# Mesh-Generierung über das Gateway — Client-Spezifikation

Stand 2026-09-03. Gilt für die Mesh-Aliase auf dem Prod-Gateway. Maschinenlesbare
Quelle der Wahrheit ist immer `GET /v1/generations/{alias}/schema` — dieses Dokument
erklärt die Semantik dahinter; bei Abweichungen gewinnt das Schema.

---

## 1. API-Mechanik

Alle Aufrufe mit `Authorization: Bearer <key>`.

| Schritt | Aufruf |
|---|---|
| Aliase entdecken | `GET /v1/models?type=image` |
| Parameter eines Alias | `GET /v1/generations/{alias}/schema` (Namen, Typen, Defaults, Bild-Slots `images[]`, Datei-Felder `files[]`) |
| Job starten | `POST /v1/generations` mit `{"model": "<alias>", "mode": "sync"\|"async", "params": {…}, "images": {…}, "files": {…}}` |
| Status/Ergebnis | `GET /v1/jobs/{job_id}` — `sync` blockt und liefert dieselbe Job-View direkt (Status `done`), `async` antwortet `202 {job_id, status: "queued", …}` |
| Artefakt laden | `GET /v1/jobs/{job_id}/result/{n}` (owner-gated, gleicher Bearer-Key) |
| Abbrechen | `POST /v1/jobs/{job_id}/cancel` |

Mesh-Jobs laufen minutenlang — **`mode: "async"` verwenden** und `GET /v1/jobs/{id}`
pollen (`elapsed_s` ist bei `queued`/`running` immer dabei; `progress`, `eta_s` und
`progress_step` nur, wenn das Backend Live-Schritte meldet oder der Alias schon
Laufzeit-Historie hat — sonst fehlen die Felder ganz, ein Client darf sie nicht
voraussetzen). Ist das Backend belegt, wird der Job **geparkt** statt abgelehnt
(Status bleibt `queued`); läuft die Park-Zeit ab, endet der Job mit
`status: "failed"` und `error: "park timeout: backend busy for >…s"` — nicht mit
`503`. Ein `503` kommt nur beim Start und nur dann, wenn für den Alias überhaupt kein
gesundes Backend existiert (ohne `Retry-After`). Im `sync`-Modus antwortet
`POST /v1/generations` auf einen fehlgeschlagenen Job mit HTTP `502` und der Job-View
im Body.

Parameter werden unter ihrem **öffentlichen Namen** (dem Label aus dem Schema)
gesendet — bei den Mesh-Aliasen durchgehend `input_*`. Unbekannte Namen werden
**still ignoriert** (dann greift der Default); nach Client-Updates lohnt ein
Abgleich gegen das Schema. `input_image` ist bei allen `img2mesh`-Aliasen Pflicht
(`on_empty: required`): fehlt es, setzt das Gateway **kein** Platzhalterbild ein — es
entsteht kein stilles Schwarz-Mesh. Der Request wird dabei nicht mit `400` abgelehnt;
der Job wird angelegt und endet mit `status: "failed"` und der Fehlermeldung des
Backends.

**Dateien** reisen nie in `params`, sondern in eigenen Objekten: Bilder unter
`images`, alle übrigen Dateien (Meshes) unter `files` — Schlüssel ist der
öffentliche Parametername, Wert Base64, ein data-URI oder eine http(s)-URL, die
das Gateway selbst abholt. Das Gateway lädt die Datei auf genau das Backend, auf
dem der Job dann läuft; Parken und Failover ändern daran nichts, der Client sieht
nie einen Backend-Pfad. `files` ist bewusst **streng** (anders als `params`):
unbekannter Schlüssel oder unlesbarer Wert → `400`, Datei über **64 MB** → `413`
— eine still verschluckte Datei würde einen technisch erfolgreichen Job mit
falschem Eingang liefern. Die Bytes werden nicht als Job-Input gespeichert; in
der Job-Ansicht steht nur ein Vermerk `<upload:… (N MB)>`.

Welche Schlüssel ein Alias unter `files` annimmt, steht im Schema als eigene Liste
**`files[]`** — neben `params[]` und `images[]`:

```json
"files": [{"name": "input_mesh_path", "required": true, "accept": ["glb"]}]
```

`required: true` heißt: ohne diese Datei wird der Request abgelehnt (so bei
`Meshy-Rig`). Bei den ComfyUI-Aliasen steht `required: false`, weil dasselbe
Eingangs-Mesh alternativ als Backend-Pfad in `params` genannt werden kann — nur mit
Admin-Key (siehe 3.4); `accept` nennt, wenn vorhanden, die zulässigen Container.

### Antwortform (Job-View)

```json
{
  "job_id": "…", "status": "queued|running|done|failed",
  "alias": "…", "backend": "…", "error": null,
  "rig": "tripo",                       // Beispiel: Tripo-Rig (siehe 3.1)
  "rig_spec": "mixamo",                 // Knochennamen dieses Cloud-Rigs
  "warnings": [],
  "results": [
    {"n": 0, "name": "rigged.glb", "mime": "model/gltf-binary", "kind": "file", "sha256": "…", "url": "…/result/0"},
    {"n": 1, "name": "walk.glb",   "mime": "model/gltf-binary", "kind": "file", "sha256": "…", "url": "…/result/1"}
  ]
}
```

(`kind` ist bei Meshes — GLB wie FBX — immer `"file"`, bei Karten `"image"`;
ein Filter auf `kind == "model"` findet nichts.)

---

## 2. Artefakte identifizieren und speichern — die zwei Grundregeln

**Regel 1 — Zuordnung über das Artefakt-Token im Namen**, nie über Position oder
Endung. Basis ist `<name>` = Wert von `input_name`; das Token steht als Suffix
davor der Endung. Match als Substring, z. B. `*_basecolor*`:

| Quelle | Namensmuster | Zähler? |
|---|---|---|
| ComfyUI-Export (Meshes, Texturkarten) | `<name>_00001_.glb`, `<name>_basecolor_00001_.png`, `<name>_metallic_00001_.jpg` | ja — ComfyUI hängt `_00001_` an, **auch ans Haupt-GLB** |
| UniRig (`mesh-rig-unirig`) | `<name>_articulationxl.fbx` | nein |
| Make-It-Animatable (`mesh-mia`) | `…_rigged.glb` | nein |
| Meshy (Cloud) | feste Namen ohne `<name>`: `model.glb`, `preview.png`; beim Rigging `rigged.glb` / `rigged.fbx` und optional `walking.glb` / `running.glb` | nein |
| Tripo (Cloud) | feste Namen ohne `<name>`: `model.glb`, `preview.png`; jedes **weitere** bestellte Format als `model.<fmt>` (`model.fbx`, `model.obj` …). Beim Rigging `rigged.<fmt>` (erstes Format nativ, weitere als `rigged.<fmt>`) und je Animations-Preset ein `<preset>.<fmt>`, z. B. `walk.glb` | nein |

Tokens: `_basecolor`, `_metallic`, `_articulationxl` (FBX), `_rigged` (Humanoid-GLB).
Verlasse dich für das Haupt-Asset nicht auf `<name>`; das Mesh ist zuverlässig
das Result mit `kind: "file"`. Dateinamen stammen aus der jeweiligen Alias-Ausgabe.

**Regel 2 — Format, Endung und MIME kommen ausschließlich aus der Antwort**
(`name` + `mime` je Result), nie aus einer Annahme. Grund: die Umkodierung nach
JPEG (bei entsprechend konfigurierten Output-Cases, siehe unten) ist **pro Datei**, nicht
pro Job: das Gateway kodiert nur die Texturen um, die **keinen echten
Alphakanal** nutzen; eine Basecolor mit Transparenz bleibt PNG, während die
Metallic-Map im selben Job als JPEG (`.jpg`, `image/jpeg`, Qualität 90)
rausgeht. Ein Satz wie „bei diesem Alias sind Texturen JPEG" ist deshalb
grundsätzlich falsch. Name und MIME wechseln dabei immer gemeinsam (gleiche
Quelle) — sie können nicht auseinanderlaufen.

Weitere Zusicherungen des Gateways:

* Textur-Normalisierung und JPEG-Umkodierung richten sich nach den Output-Cases
  des angefragten Alias. Ein separater Rigging-Job übernimmt keine Texturen aus
  dem Mesh-Job. Der Client muss benötigte Karten selbst aufbewahren und binden;
  aus einem Aliasnamen oder einem lokalen Rigging-Ergebnis folgt keine automatische
  V-Korrektur, Paar-Prüfung oder Skin-Validierung.
* Dateien über ~30 MB erzeugen einen Eintrag in `warnings` (Hinweis auf
  Web-Tauglichkeit, kein Fehler) — bei ComfyUI-Auslieferungen.
  **Cloud-Auslieferungen (Meshy, Tripo) prüft das Gateway nicht**;
  dort fehlt `warnings` auch bei großen Dateien.
* `sha256` je Result für Integritätsprüfung.
* **Eingangs-Isolation (Zusicherung):** Gleichzeitig laufende Jobs teilen sich
  keinen Eingangs-Zustand — auch nicht bei verschiedenen Aliasen auf derselben
  Backend-Instanz. Jede hochgeladene Datei (Bild wie Mesh) liegt unter einem
  job-eindeutigen Namen; schlägt ein Upload fehl, scheitert der Job, statt mit
  fremden Bytes weiterzurechnen. Muss serialisiert werden, wird der zweite Job
  wie gehabt **geparkt** (`queued`), nie abgelehnt.
* **`input_images[]` in der Job-View** führt `sha256` und `bytes` je
  Referenzbild — analog zu `results[]`. Damit lässt sich nachweisen, welche
  Bytes tatsächlich in den Job gingen: den lokal berechneten Hash des
  hochgeladenen Bildes dagegen prüfen. (Jobs von vor dem 03.08.2026 haben den
  Wert nicht und liefern `null`.)

---

## 3. Alias-Katalog

### 3.1 Varianten-Schema

Mesh-Erzeugung, Verdichtung und Rigging sind unabhängige Operationen mit
unterschiedlichem **Speicher-Kontrakt**:

| Operation | Alias / Anfrage | Auslieferung | Pflicht zu speichern |
|---|---|---|---|
| Mesh erzeugen | `img2mesh`-Alias | GLB mit eingebetteter Textur und je Workflow getrennte Karten | GLB; benötigte Karten für späteres FBX-Rigging behalten |
| Mesh verdichten | `mesh-shrink` / `mesh-shrink-quad`, eigener Request mit `files.input_mesh_path` | GLB und getrennte Karten | GLB und benötigte Karten |
| Lokal riggen (FBX) | `mesh-rig-unirig`, eigener Request mit `files.input_mesh_path` | FBX, ohne automatisch übernommene Basecolor | FBX **und** passende Basecolor aus dem Mesh-Job; der Client bindet sie neu |
| Lokal riggen (GLB) | `mesh-mia`, eigener Request mit `files.input_mesh_path` | `*_rigged.glb` | GLB; Skin und Textur anhand des konkreten Ergebnisses beurteilen |
| Lokal animieren | `mesh-unimate`, geriggtes GLB über `files.input_mesh_path` plus `prompt` | animiertes GLB pro Wiederholung | GLB mit nativer Animation, Skin und eingebetteten Texturen |
| Bewegung auf Mixamo-Rig | `motion-kimodo`, Mixamo-FBX über `files.input_rig_path` (Pflicht) plus `prompt` | ein FBX pro Variante | FBX mit Animation auf dem gesendeten Rig |
| Cloud-Rigging | `Meshy-Rig` / `Tripo-Rig`, eigener Request mit `files.input_mesh_path` | `rigged.*` und optionale Animationsdateien | benötigte Results unverändert speichern |

Jeder Request führt ausschließlich den gewählten Alias aus. Der Client lädt ein
Mesh-Ergebnis herunter und lädt es für eine weitere Operation erneut hoch.
Parameter, Texturen und sonstige Results werden nicht automatisch zwischen Jobs
übernommen. Jeder Job hat eine eigene ID, Planung, Stornierung und Abrechnung.
`input_no_fingers` wirkt beim lokalen `mesh-mia`-Request, nicht bei der Mesh-Erzeugung.

**Das Job-Feld `rig`** wird von den Cloud-Riggern gesetzt:

| `rig` | Skelett | Auslieferung |
|---|---|---|
| `meshy` | Meshys eigenes Skelett (Biped) | `rigged.glb`, optional `rigged.fbx` und `walking.*` / `running.*` |
| `tripo` | Tripos Auto-Rig; `rig_spec` nennt `mixamo` oder `tripo` | `rigged.<fmt>` und je konfiguriertem Animations-Preset weitere Results |

Das Gateway kennzeichnet Cloud-Rigs nur: kein V-Flip, keine JPEG-Umkodierung,
keine lokale Paar- oder Skin-Prüfung. Dateien so speichern, wie sie kommen.
Clips sind eine Zugabe, kein garantierter Bestandteil; das Rig-Mesh selbst fehlt
nie, dafür scheitert der Job. Ein `tripo`-GLB mit Mixamo-Namen ist kein
Make-It-Animatable-GLB. Lokale Rigger bekommen kein automatisch abgeleitetes
`generic`-/`mixamo`-Tag.

### 3.2 `img2mesh`-Familien

Gemeinsame Parameter (Defaults je Familie in der Tabelle): `input_image`
(Pflicht), `input_name` (string — **immer setzen**, er bestimmt die Dateinamen;
Leerlassen übernimmt Workflow-Reste), `input_face_num` (int),
`input_texture_resolution` (int), `input_remove_background` (bool, Default true —
bei Bildern mit sauberem Alphakanal auf false setzbar), `input_no_fingers` (bool).

| Familie (Aliase) | face_num | texture_resolution | Besonderheit |
|---|---|---|---|
| `Trellis2-High` | 20000 | 1024 | High-Workflow; explizite GLB- und Texturausgabe |
| `Trellis2-Low` | 50000 | **2048, fest** | Pipeline **`1536_cascade`**, für Gesichtsqualität fest gepinnt; `input_texture_resolution` ist kein Request-Feld. Backends: `k12-gpu`, `dx10-02`. |
| `Pixal3D` | 50000 | 2048 | höchste Auflösung |
| `Hunyuan3D` | 40000 | 1024 | ⚠ **`input_face_num` nie über 40000** — größere Werte frieren das Backend ein (kein Fehler, der Job hängt bis zum Timeout). 40000 ist der höchste nachweislich stabile Wert. |
| `Meshy-Object`, `Meshy-Multiview` | Meshy-Default (30000 bei Remesh) | 2048 | Cloud (Meshy.ai, bezahlt pro Task, nur als Fallback oder gezielt). `-Multiview` nimmt `input_image_front` (Pflicht) + optional `input_image_back` / `_left` / `_right`. Zusätzlich `input_texture_prompt` (string) und `input_pose` (`a-pose`/`t-pose`). `input_remove_background`/`input_no_fingers` werden angenommen, wirken nicht. Kein `files`-Upload (`400` — Bilder gehören unter `images`). Liefert `model.glb` (Texturen eingebettet) + `preview.png`. |
| `Meshy-Humanoid`, `Meshy-Humanoid-Multiview` | wie oben | 2048 | Cloud-Mesh in t-pose, ohne Rigging. `-Multiview` nimmt dieselben vier Bild-Slots wie `Meshy-Multiview`. Rigging separat über `files.input_mesh_path` anfragen. |
| `Meshy-Humanoid-Cloud` | wie oben | 2048 | Ebenfalls ein unabhängiger Cloud-Mesh-Job in t-pose. Für ein Cloud-Rig das Ergebnis herunterladen und separat `Meshy-Rig` aufrufen (3.5). |
| `Tripo-Object`, `Tripo-Multiview` | Tripo-Default (adaptiv) | 2048 | Cloud (Tripo3D, bezahlt pro Task, wie Meshy nur als Fallback oder gezielt). `-Multiview` nimmt `input_image_front` (**Pflicht**) **plus mindestens eine weitere** Ansicht (`input_image_back` / `_left` / `_right`) — mit nur einem Bild lehnt das Gateway ab, bevor irgendetwas hochgeladen wird (Tripo verlangt zwei Ansichten). `input_name`, `input_remove_background`, `input_no_fingers` werden angenommen, wirken nicht; `input_texture_prompt` und `input_pose` gibt es bei Tripo nicht (still ignoriert). Kein `files`-Upload (`400` — Bilder gehören unter `images`). Liefert `model.glb` (Textur eingebettet) + `preview.png`; jedes weitere vom Betreiber bestellte Format kommt als `model.<fmt>` dazu. |
| `Tripo-Humanoid` | wie oben, Betreiber-Budget 150000 | 2048 | Unabhängiger Cloud-Mesh-Job. Für ein Rig das Ergebnis herunterladen und separat `Tripo-Rig` mit `files.input_mesh_path` und Rigging-Parametern aufrufen (3.5). |

**`input_face_num` bei den Meshy-Aliasen.** Hier gibt es — anders als bei den
ComfyUI-Familien — keinen vom Gateway gesetzten Default: Bleibt der Parameter weg,
entscheidet Meshys eigener Modell-Default über die Dreieckszahl. Wird er gesendet,
geht er als `target_polycount` (100 … 300000) an Meshy **und schaltet für diese
Anfrage den Remesh-Durchlauf ein** — die Zahl wirkt also nur zusammen mit einer
Neuvernetzung. Texturen, PBR, Topologie, Ausgabeformate und das Vorschaubild sind
Alias-Defaults des Betreibers, keine Client-Parameter. Ein fehlgeschlagener
Meshy-Task ist endgültig (Meshy erstattet die Credits); `/v1/jobs/{id}/cancel`
beendet nur den Gateway-Job — Meshy rechnet den Task trotzdem ab.

**`input_face_num` bei den Tripo-Aliasen.** Auch hier gibt es keinen vom Gateway
gesetzten Default: ohne den Parameter entscheidet Tripos adaptiver Default (bzw. das
Face-Budget, das der Betreiber am Alias hinterlegt hat — dann steht es als `default` im
Schema). Wird er gesendet, geht er als `face_limit` an Tripo und wird auf **100 … die
Obergrenze des Modells** geklemmt, nicht abgelehnt: v3.1 1,5 Mio., v3.0 1 Mio., v2.5
500.000, P-Serie 50.000; mit Quad-Topologie 150.000. Einen Remesh-Schalter wie bei
Meshy gibt es nicht. `input_texture_resolution` wird in Tripos Stufen übersetzt (≤2048
`standard`, ≤4096 `detailed`, darüber `extreme`). Textur, PBR, Geometriequalität,
Quads, Ausgabeformate und das Vorschaubild sind Alias-Defaults des Betreibers. Ein
`failed` Tripo-Task ist endgültig; `/v1/jobs/{id}/cancel` beendet nur den Gateway-Job —
Tripo kennt in V3 keinen Cancel-Endpunkt und rechnet den Task ab.

#### LOD-Stufen (`input_lod_faces`) — zurückgezogen

Der Parameter `input_lod_faces` (kommaseparierte Ziel-Dreieckszahlen, mehrere
reduzierte Fassungen pro Job) existierte bis September 2026 nur auf `Hunyuan3D-Object`.
Der Workflow erzeugt keine zusätzlichen LOD-Fassungen mehr (Stand 2026-09-08);
ein mitgeschickter Wert wird **stillschweigend ignoriert** — der Job liefert nur
das Hauptergebnis, der Parameter erscheint nicht unter `applied`.
Wer reduzierte Fassungen braucht, lädt das Mesh herunter und stellt separate
Requests an `mesh-shrink` / `mesh-shrink-quad` (Abschnitt 1 der Workflow-README)
oder dezimiert clientseitig.

V-Flip und JPEG-Umkodierung (Abschnitt 2) richten sich nach den konkreten
Output-Cases, nicht nach dem Namenssuffix eines Alias.

### 3.3 `Triposplat-Object` — der Sonderfall

Gaussian-Splat-Pipeline, andere Parameter: **kein** `input_face_num` und kein
`input_no_fingers`; stattdessen `input_num_gaussians` (Default 10000) und
`input_texture_resolution` (Default 512, steuert Mesh-Auflösung **und**
Preprocess-Größe). Sonst wie oben (`input_image`, `input_name`,
`input_remove_background`).

Auslieferung: **nur** `<name>_00001_.glb`, und dieses GLB ist technisch anders
gebaut als das aller anderen Aliase: die Farbe steckt **in den Vertices**
(`COLOR_0`), es hat **keine UVs und keine Texturbilder**. Für den Client heißt
das:

* Es gibt **keine** getrennten Basecolor-/Metallic-Karten und keinen Weg, sie
  nachträglich zu erzeugen — `mesh-shrink` kann Triposplat-Meshes **nicht**
  verarbeiten (es braucht UVs, siehe 3.4).
* Es gibt bewusst **keine** `Triposplat-Generic`-Variante — FBX-Rigging ohne
  separate Basecolor ist unmöglich.
* Beim Rendern beachten: Vertex-Farben brauchen einen Shader, der `COLOR_0`
  auswertet (das Material ist `KHR_materials_unlit`); Engines, die nur
  Texturmaterialien erwarten, zeigen das Modell sonst einfarbig.
* Wer texturierte Meshes mit separaten Karten braucht, nimmt eine der anderen
  Familien (`Trellis2-*`, `Pixal3D-*`, `Hunyuan3D-*`).

### 3.4 `mesh-shrink` und `mesh-shrink-quad` — Nachverdichtung

Nehmen ein vorhandenes GLB und reduzieren die Polygonzahl inkl. Neu-Bake der
Texturen. Gleiche Parameter:

| Alias | Verfahren | Ergebnis-Topologie | Status |
|---|---|---|---|
| `mesh-shrink` | Dezimierung (Quadric Edge Collapse) | Tris, formtreu | **verfügbar**, trifft `input_face_num` exakt (gemessen: 5.000 angefordert → 5.000 Dreiecke) |
| `mesh-shrink-quad` | Remesh (Instant Meshes) | Quad-Topologie (z. B. für Sculpting/Subdivision) | verfügbar, aber `input_face_num` wirkt nur als **grober Richtwert** (gemessen: 5.000 angefordert → ~38.000 Dreiecke). Wer eine verlässliche Zielgröße braucht, nimmt `mesh-shrink`. |

**Voraussetzung an das Eingangs-Mesh:** es muss **UVs und eine eingebettete
Textur** mitbringen — die Verkleinerung backt die Textur auf die neue Topologie
um und braucht dafür das texturierte Original. Ergebnisse von `Trellis2-*`,
`Pixal3D-*` und `Hunyuan3D-*` erfüllen das. **Triposplat-GLBs nicht** (nur
Vertex-Farben, siehe 3.3): der Job scheitert dann mit
`node 21 Trellis2RenderMultiViewNvdiffrast: expected np.ndarray (got NoneType)`
— das ist genau dieser Fall, kein vorübergehender Fehler, ein Retry hilft nicht.

Parameter: `input_mesh_path` (Pflicht — das Mesh selbst, siehe unten),
`input_name`, `input_face_num` (Default **5000**), `input_texture_resolution`
(Default 1024), **`input_keep_borders`** (Default `"true"`, als String — siehe
unten), `input_no_fingers` (nur Durchreichung, hier ohne Wirkung).

**`input_keep_borders` entscheidet über Reduktion vs. Formtreue** und ist die
wichtigste Stellschraube dieses Alias:

* `"true"` (Default) — Randkanten bleiben erhalten. Bei kompakten, geschlossenen
  Objekten wird die Zielzahl trotzdem exakt getroffen. Bei Meshes aus vielen
  getrennten Flächen (**Vegetation**: Bäume, Büsche, Gras) verhindert es die
  Reduktion weitgehend — das Ergebnis bleibt groß, aber **unversehrt**.
* `"false"` — erzwingt die Zielzahl. Nur für Meshes verwenden, die eine
  zusammenhängende Oberfläche haben. Bei Vegetation zerfallen die Blattflächen
  zu losen, verdrehten Dreiecks-Splittern; das ist nicht reparierbar, weil die
  Flächen dabei zerstört und nicht nur vereinfacht werden.

**Für Bäume und ähnliche Geometrie gilt darum:** nicht nachträglich verkleinern,
sondern die Polygonzahl gleich bei der Erzeugung über `input_face_num` des
`img2mesh`-Alias begrenzen. Ein Mesh aus tausenden Einzelflächen lässt sich
nicht auf einen Bruchteil reduzieren, ohne diese Flächen zu verlieren.

Das Eingangs-Mesh kommt als **Datei im Request** — unter `files`, nicht in
`params`:

```json
{"model":"mesh-shrink","mode":"async",
 "params":{"input_name":"Held","input_face_num":5000},
 "files":{"input_mesh_path":"data:model/gltf-binary;base64,…"}}
```

Erlaubt sind Base64, ein data-URI (die Endung leitet das Gateway aus dem MIME
ab, sonst `.glb`) und eine http(s)-URL, die das Gateway abholt; Grenze 64 MB
(Abschnitt 1). Damit ist jedes GLB verdichtbar — auch eines, das nie über dieses
Gateway lief: Upload-Ziel, Backend-Wahl und Pfad regelt das Gateway pro Job
selbst. Ein vorheriges Job-Ergebnis wird also normal abgeholt
(`GET /v1/jobs/{id}/result/{n}`) und beim Shrink-Aufruf wieder mitgeschickt.

(Ein direkt in `params` gesetzter `input_mesh_path` bleibt weiterhin ein
Dateipfad auf dem Backend — aber nur mit **Admin-Key** (oder wenn das Mapping den
Eintrag mit `client_path: true` freigibt); ein normaler User-Key bekommt dafür `400`.
Für Clients ist `files` der Weg.)

Auslieferung: `<name>_00001_.glb` + `*_basecolor*.png` + `*_metallic*.png` — hier
**ohne** JPEG-Umkodierung, die Karten bleiben PNG.

**Dreiecke vs. Vertices.** `input_face_num` steuert die **Dreiecke** und wird exakt
getroffen. Die Vertex-Zahl bleibt deutlich darüber (gemessen: 5.000 Dreiecke →
11.129 Vertices): Das Texturbacken trennt das Mesh entlang der UV-Nähte auf, und
jeder Naht-Vertex existiert danach mehrfach. Das ist unvermeidlich, wenn ein Mesh
eine Textur tragen soll — wer Vertices zählt, misst das UV-Layout mit, nicht die
Geometrie.

### 3.5 Rigger direkt (`mesh-rig-unirig`, `mesh-mia`, `Meshy-Rig`, `Tripo-Rig`)

`mesh-rig-unirig` und `mesh-mia` werden als unabhängige Jobs aufgerufen:
Mesh über `files.input_mesh_path` wie in 3.4 hochladen und Rigging-Parameter im
selben Request angeben. Sie liefern **nur** das Rig-Ergebnis: `mesh-rig-unirig`
die nackte FBX (ohne Basecolor — der Client muss die passende Textur aus dem
Mesh-Job behalten und binden), `mesh-mia` das geriggte GLB `*_rigged.glb`
(Parameter `input_mesh_path`, `input_no_fingers`; kein `input_name`).
Achtung Namensfalle: `mesh-mia` führt den **Make-It-Animatable**-Workflow aus —
der gleichnamige FBX-Rigger „MIA" (`MIAAutoRig`, Workflow `mesh-reg-mia`) ist
auf dem Gateway **nicht registriert**.

**`Meshy-Rig`** ist ein Cloud-Rigger: ein `mesh2rig`-Alias
auf dem Meshy-Backend (Endpoint `rigging`, **5 Credits je Lauf**), der jedes GLB riggt —
auch eines aus einer lokalen Pipeline oder von außerhalb des Gateways.

| Punkt | Regel |
|---|---|
| Eingang | `files.input_mesh_path` — **Pflicht**, und ausschließlich ein **binäres glTF** (`.glb`). Das Gateway prüft die Container-Signatur und lehnt alles andere ab, **bevor** der Task angelegt wird (eine umbenannte OBJ kostet also nichts). |
| Parameter | `input_name` (string, benennt den Task bei Meshy) und `input_height_m` (float, Default **1.7** — die angenommene Körpergröße, an der Meshy das Skelett ausrichtet). `input_no_fingers` wird angenommen und ignoriert; die Mesh-Optionen (Textur, Topologie, Polycount …) gibt es hier nicht. |
| Auslieferung | `rigged.glb`, zusätzlich `rigged.fbx`, wenn der Betreiber `fbx` in den Ausgabeformaten führt (mehr als glb/fbx kann Rigging nicht). Ist die Option **animations** gesetzt, kommen `walking.glb` / `running.glb` (je Format) als weitere Results dazu — Zugabe, kein garantierter Bestandteil. |
| Grenzen | Nur **Bipeds** mit klar getrennten Gliedmaßen, Blick in **+Z**, höchstens **300.000** Dreiecke. Ein `failed` mit Meshys Meldung ist endgültig (Meshy erstattet die Credits) — kein blinder Retry. |

```json
{"model": "Meshy-Rig", "mode": "async",
 "params": {"input_name": "Held", "input_height_m": 1.8},
 "files":  {"input_mesh_path": "data:model/gltf-binary;base64,…"}}
```

Anders als bei `mesh-shrink` ist ein Backend-**Pfad** in `params` hier kein Ersatz:
Meshy hat kein Dateisystem, das Mesh reist immer als Datei mit (das Schema führt es
deshalb mit `required: true`).

**`Tripo-Rig`** ist der vierte Rigger und der zweite in der Cloud: ein `mesh2rig`-Alias
auf dem Tripo-Backend (Endpoint `rig`, **25 Credits je Lauf** nach einer kostenlosen
Vorprüfung), der ebenfalls jedes GLB riggt — auch eines aus einer lokalen Pipeline oder
von außerhalb des Gateways. Sein Unterschied zu `Meshy-Rig`: die Knochennamen sind auf
Wunsch **Mixamo-kompatibel** (Betreiber-Option `spec`, Default `mixamo`; das Job-Feld
`rig_spec` sagt, was tatsächlich geliefert wurde).

| Punkt | Regel |
|---|---|
| Eingang | `files.input_mesh_path` — **Pflicht**, und ausschließlich ein **binäres glTF** (`.glb`). Das Gateway prüft die Container-Signatur, bevor die Datei überhaupt hochgeladen wird; eine umbenannte OBJ kostet nichts. |
| Vorprüfung | Vor dem Rig läuft Tripos **Rig-Check** (0 Credits) — sofern der Betreiber ihn am Alias nicht abgeschaltet hat (Option `rig_check`, Default an). Hält er das Mesh für nicht riggbar, endet der Job **vor** den 25 Credits mit einer Meldung, die den erkannten Rig-Typ nennt. |
| Parameter | `input_rig_type` — welcher Typ zulässig ist, hängt vom Rig-Modell des Alias ab: `v1.0-20240301` kann **nur `biped`** (das Schema führt dann auch nur diese Auswahl), `v2.5-20260210` kann `biped`, `quadruped`, `hexapod`, `octopod`, `avian`, `serpentine`, `aquatic`. Ein nicht unterstützter Wert lässt den Job scheitern, **bevor** der Rig-Task angelegt wird (also ohne Credits) — er wird nicht stillschweigend auf `biped` gebogen. `input_name` und `input_no_fingers` werden angenommen und ignoriert; die Mesh-Optionen gibt es hier nicht. |
| Auslieferung | `rigged.<Format>` für das **erste** vom Betreiber geführte Ausgabeformat (glb oder fbx — mehr kann das Rigging nicht), Textur eingebettet. Jedes **weitere** geführte Format ist bei Tripo ein eigener Convert-Task und kommt als `rigged.<fmt>` dazu; schlägt einer fehl, scheitert der Job (die Lieferung schrumpft nie stillschweigend). Sind **Animations-Presets** konfiguriert, kommt je Preset ein Result `<preset>.<fmt>` dazu (`preset:walk` → `walk.glb`) — Zugabe, kein garantierter Bestandteil. |
| Grenzen | Der Rig-Check ist die Grenze: was er ablehnt, wird nicht gerigt. Ein `failed` Task ist endgültig — kein blinder Retry. `/v1/jobs/{id}/cancel` beendet nur den Gateway-Job; Tripo hat in V3 keinen Cancel-Endpunkt und rechnet den Task ab. |

```json
{"model": "Tripo-Rig", "mode": "async",
 "params": {"input_rig_type": "biped"},
 "files":  {"input_mesh_path": "data:model/gltf-binary;base64,…"}}
```

Auch hier ist ein Backend-**Pfad** in `params` kein Ersatz — Tripo hat kein
Dateisystem, das Mesh reist als Datei mit (`required: true` im Schema).

---

### 3.6 Text → Animation auf vorhandenem Rig (`mesh-unimate`)

Lokaler `mesh2anim`-Alias auf VM 402 (`k12-gpu`). Eingang ist ein **bereits geriggtes
GLB**, etwa das Ergebnis von `mesh-mia`; ein ungeriggter Mesh braucht einen separaten
Rigging-Job. Keine Übertragung eines fremden Standardskelett-Clips: UniMate erzeugt
die Bewegung direkt für das eingesandte Rig.

```json
{"model": "mesh-unimate", "mode": "async",
 "prompt": "An object walks forward.",
 "params": {"seed": 42, "input_cfg_scale": 3.0, "input_num_repetitions": 1,
            "input_chain_overlap": 10},
 "files": {"input_mesh_path": "data:model/gltf-binary;base64,…"}}
```

Einzeilig: 60 Frames bei 30 fps (2-s-Fenster; letzte Keyframe-Zeit 1,9667 s).
Mehrere Prompt-Zeilen verketten je eine Bewegung; mit 10 Frames Überlappung ergeben
zwei Zeilen 110 Frames. `input_num_repetitions` erzeugt getrennte GLB-Results.
Grenze des installierten v3-Checkpoints: 71 Gelenke. Keine GGUF-Variante.
Modelllizenz **CC BY-NC 4.0**: keine kommerzielle Nutzung.

Jeder Result ist `kind: file`, MIME `model/gltf-binary`; den vollständigen GLB
speichern und die enthaltene Animation abspielen. Dateiname `UniMate_<counter>_.glb`,
kein `input_name`-Token. Die AI-Hub-Vorschau spielt den Originalclip mit Play/Pause.
Der Betreiber hat für den ersten Einsatz den aktuellen Rig-Mesh als Backend-Pfad
vorbelegt; Clients sollen trotzdem eine eigene Datei senden, statt auf diesen Default
zu vertrauen. Upload schlägt Default.

### 3.7 Text → Bewegung auf Mixamo-Rig (`motion-kimodo`)

Lokaler `text2anim`-Alias auf VM 402 (`k12-gpu`), NVIDIA Kimodo (SOMA-RP-v1). Eingang
ist ein **Mixamo-Skelett als FBX** (`mixamorig:`-Bones, ohne Mesh genügt); es gibt
keinen Server-Default — ohne `files.input_rig_path` antwortet der AI-Hub `400`.

```json
{"model": "motion-kimodo", "mode": "async",
 "prompt": "A person holds both hands together in front of the chest, then spreads the arms wide open.",
 "params": {"seed": 42, "input_duration": 4.0, "input_num_samples": 3},
 "files": {"input_rig_path": "data:application/octet-stream;base64,…"}}
```

Eine Prompt-Zeile = ein Segment (Punkte innerhalb der Zeile trennen nicht);
mehrere Zeilen werden nacheinander abgespielt. `input_duration` = Sekunden pro
Segment (0,5–30, Default 4), 30 fps. `input_num_samples` 1–4 Varianten aus einem
Lauf (mehr → Job `failed` mit Bereichsfehler). `results[]` enthält genau so viele
FBX (`kind: file`, `application/octet-stream`, mit `sha256`), Reihenfolge =
Sample-Index; Armature und Bone-Namen sind die des gesendeten Rigs. Ein Rig ohne
Mixamo-Kernknochen (Hips, Spine, Head, Arme, Oberschenkel) lässt den Job mit Fehler
enden, statt eine unbewegte Datei zu liefern. Laufzeit ≈ 1 min pro Lauf.

## 4. Fehlerbilder

| Symptom | Bedeutung / Reaktion |
|---|---|
| `status: "failed"` + `error` | Workflow-/Validierungsfehler (z. B. „no basecolor PNG", „embedded texture is a 2x2 dummy", per-Node-Fehler des Backends). Nicht blind retrien — Fehlertext auswerten. |
| `503` beim Start | Für den Alias existiert kein gesundes Backend (bzw. keines, das das geforderte `backend`-Pin/LoRA erfüllt) — Alias und `/health` prüfen (die Details dort nur mit Admin-Key), kein blinder Retry. |
| `status: "failed"`, `error` beginnt mit `park timeout:` | Alle Backends waren die ganze Park-Zeit belegt — später neu einreichen. |
| Job hängt lange in `running` | Mesh-Jobs dauern Minuten; `progress` beachten. Hunyuan3D mit `face_num` > 40000: siehe 3.2 — vermeiden. |
| `status: "failed"` bei `Meshy-Rig` | Rigging abgelehnt (kein erkennbarer Biped, zu viele Dreiecke, unbrauchbare Pose): Mesh prüfen (3.5), nicht blind retrien. Fehlendes `files.input_mesh_path` oder eine Datei ohne binäre glTF-Signatur scheitern vor einem Task und kosten keine Credits. `400` beim Start bedeutet unbekannter `files`-Schlüssel oder unlesbarer Wert; ab 64 MB kommt `413`. |
| `status: "failed"` bei `Tripo-Rig` | `rig-check: mesh is not riggable` bedeutet: kostenlose Vorprüfung abgelehnt, keine Rig-Credits verbraucht. Mesh prüfen, nicht retrien. Fehlende Datei, falsche Container-Signatur oder ein zum Rig-Modell unpassender `input_rig_type` scheitern vor dem bezahlten Task (3.5). `400` beim Start bedeutet unbekannter `files`-Schlüssel oder unlesbarer Wert. |
| `404` auf `result/{n}` | Job-TTL abgelaufen (Default **24 h**, `ttl_s` 86400) — Artefakte werden serverseitig aufgeräumt; Ergebnisse zeitnah abholen und selbst speichern. |

---

## 5. Minimalbeispiel

```bash
B=https://<gateway>; K=<key>

# Job starten (async)
JOB=$(curl -s $B/v1/generations -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"Trellis2-Low","mode":"async",
       "params":{"input_name":"Held","input_face_num":20000},
       "images":{"input_image":"data:image/png;base64,…"}}' | jq -r .job_id)

# Pollen bis done/failed
curl -s $B/v1/jobs/$JOB -H "Authorization: Bearer $K" | jq '{status, progress, rig, warnings}'

# Ergebnisse: Token matchen, Name/MIME aus der Antwort übernehmen
curl -s $B/v1/jobs/$JOB -H "Authorization: Bearer $K" \
  | jq -r '.results[] | [.n, .name, .mime] | @tsv'
curl -s -o Held.glb $B/v1/jobs/$JOB/result/0 -H "Authorization: Bearer $K"
```

Dasselbe GLB nachträglich verdichten — das Mesh geht als Datei mit, ein
Backend-Pfad ist nicht nötig:

```bash
# GLB als data-URI einbetten (Base64 ohne Zeilenumbrüche)
MESH="data:model/gltf-binary;base64,$(base64 -w0 Held.glb)"

SHRINK=$(jq -n --arg m "$MESH" \
  '{model:"mesh-shrink",mode:"async",
    params:{input_name:"Held_lo",input_face_num:5000,input_texture_resolution:1024},
    files:{input_mesh_path:$m}}' \
  | curl -s $B/v1/generations -H "Authorization: Bearer $K" \
         -H "Content-Type: application/json" -d @- | jq -r .job_id)

curl -s $B/v1/jobs/$SHRINK -H "Authorization: Bearer $K" | jq '{status, error}'
```
