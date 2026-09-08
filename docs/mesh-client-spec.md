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
pollen (`elapsed_s`/`progress` sind enthalten). Ist das Backend belegt, wird der Job
**geparkt** statt abgelehnt (Status bleibt `queued`); erst nach Ablauf der Park-Zeit
kommt `503` mit `Retry-After`.

Parameter werden unter ihrem **öffentlichen Namen** (dem Label aus dem Schema)
gesendet — bei den Mesh-Aliasen durchgehend `input_*`. Unbekannte Namen werden
**still ignoriert** (dann greift der Default); nach Client-Updates lohnt ein
Abgleich gegen das Schema. `input_image` ist bei allen `img2mesh`-Aliasen Pflicht
(`on_empty: required` — fehlt es, wird der Request abgelehnt, es entsteht kein
stilles Schwarz-Mesh).

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
Eingangs-Mesh alternativ als Backend-Pfad in `params` genannt werden kann (siehe
3.4); `accept` nennt, wenn vorhanden, die zulässigen Container.

### Antwortform (Job-View)

```json
{
  "job_id": "…", "status": "queued|running|done|failed",
  "alias": "…", "backend": "…", "error": null,
  "rig": "generic|mixamo|meshy|tripo",  // bei Rigging-Ketten und bei Meshy-Rig / Tripo-Rig (siehe 3.1)
  "rig_spec": "mixamo",                 // nur bei rig: "tripo" — die Knochennamen (siehe 3.1)
  "warnings": ["x.glb is 41 MB (> 30 MB guideline)"],
  "results": [
    {"n": 0, "name": "Held_articulationxl.fbx",   "mime": "application/octet-stream", "kind": "file",  "sha256": "…", "url": "…/result/0"},
    {"n": 1, "name": "Held_basecolor_00001_.png", "mime": "image/png",                "kind": "image", "sha256": "…", "url": "…/result/1"},
    {"n": 2, "name": "Held_metallic_00001_.jpg",  "mime": "image/jpeg",               "kind": "image", "sha256": "…", "url": "…/result/2"}
  ]
}
```

(Beispiel: eine `-Generic`-Kette. `kind` ist bei Meshes — GLB wie FBX — immer
`"file"`, bei Karten `"image"`; ein Filter auf `kind == "model"` findet nichts.)

---

## 2. Artefakte identifizieren und speichern — die zwei Grundregeln

**Regel 1 — Zuordnung über das Artefakt-Token im Namen**, nie über Position oder
Endung. Basis ist `<name>` = Wert von `input_name`; das Token steht als Suffix
davor der Endung. Match als Substring, z. B. `*_basecolor*`:

| Quelle | Namensmuster | Zähler? |
|---|---|---|
| ComfyUI-Export (Meshes, Texturkarten) | `<name>_00001_.glb`, `<name>_basecolor_00001_.png`, `<name>_metallic_00001_.jpg` | ja — ComfyUI hängt `_00001_` an, **auch ans Haupt-GLB** |
| UniRig (`-Generic`) | `<name>_articulationxl.fbx` | nein |
| Make-It-Animatable (`-Humanoid`) | `…_rigged.glb` — in Ketten mit **technischem** Präfix `gwchain_<jobid>…`, nicht `<name>` | nein |
| Meshy (Cloud) | feste Namen ohne `<name>`: `model.glb`, `preview.png`; beim Rigging `rigged.glb` / `rigged.fbx` und optional `walking.glb` / `running.glb` | nein |
| Tripo (Cloud) | feste Namen ohne `<name>`: `model.glb`, `preview.png`; jedes **weitere** bestellte Format als `model.<fmt>` (`model.fbx`, `model.obj` …). Beim Rigging `rigged.<fmt>` (erstes Format nativ, weitere als `rigged.<fmt>`) und je Animations-Preset ein `<preset>.<fmt>`, z. B. `walk.glb` | nein |

Tokens: `_basecolor`, `_metallic`, `_articulationxl` (FBX), `_rigged` (Humanoid-GLB).
Verlasse dich für das Haupt-Asset nicht auf `<name>` — bei Humanoid-Ketten trägt
das GLB den internen Ketten-Namen; das Mesh ist zuverlässig das Result mit
`kind: "file"`.

**Regel 2 — Format, Endung und MIME kommen ausschließlich aus der Antwort**
(`name` + `mime` je Result), nie aus einer Annahme. Grund: die Umkodierung nach
JPEG (nur bei `-Generic`-Auslieferungen, siehe unten) ist **pro Datei**, nicht
pro Job: das Gateway kodiert nur die Texturen um, die **keinen echten
Alphakanal** nutzen; eine Basecolor mit Transparenz bleibt PNG, während die
Metallic-Map im selben Job als JPEG (`.jpg`, `image/jpeg`, Qualität 90)
rausgeht. Ein Satz wie „bei diesem Alias sind Texturen JPEG" ist deshalb
grundsätzlich falsch. Name und MIME wechseln dabei immer gemeinsam (gleiche
Quelle) — sie können nicht auseinanderlaufen.

Weitere Zusicherungen des Gateways:

* **Nur bei `rig: "generic"`-Auslieferungen** (die `-Generic`-Ketten) normalisiert
  das Gateway die Texturkarten: V-Flip passend zu den FBX-UVs plus die
  JPEG-Umkodierung von oben — der Client speichert sie unverändert und
  kompensiert nichts. Bei `-Object`-, `-Humanoid`- und Shrink-Auslieferungen
  werden die Karten **unverändert wie vom Backend erzeugt** durchgereicht
  (immer PNG); dort gilt die Orientierung des jeweiligen Bakes.
* Dateien über ~30 MB erzeugen einen Eintrag in `warnings` (Hinweis auf
  Web-Tauglichkeit, kein Fehler).
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

Jede `img2mesh`-Familie gibt es in bis zu drei Varianten, die über das
**Speicher-Kontrakt** entscheiden:

| Variante | Pipeline | Auslieferung | Pflicht zu speichern |
|---|---|---|---|
| `-Object` | nur Mesh | `<name>_00001_.glb` (Textur eingebettet) + `*_basecolor*.png` + `*_metallic*.png` (unverändert, siehe Abschnitt 2) | **das GLB**; Karten optional (nur wenn die Engine eigene Maps will) |
| `-Generic` | Mesh → UniRig (Auto-Rig, FBX) | `<name>_articulationxl.fbx` + `*_basecolor*` + `*_metallic*` (V-korrigiert, ggf. JPEG); Job-Feld `rig: "generic"` | **FBX + Basecolor-Bild, untrennbar**: die FBX referenziert ihre Textur nur über einen Temp-Pfad, der beim Client ins Leere zeigt — über den gelieferten Dateinamen neu binden. Ohne das Bild ist das Asset unbrauchbar und nicht wiederherstellbar. Das Gateway garantiert das Paar in der Auslieferung (sonst schlägt der Job fehl) — das Speichern liegt beim Client. |
| `-Humanoid` | Mesh → Make-It-Animatable (mixamo-Rig, GLB) | ein GLB `*_rigged.glb` (technischer Name!) mit mixamorig-Skin und **eingebetteter** Textur (+ `*_metallic*.png` optional); Job-Feld `rig: "mixamo"` | **das GLB — genügt allein.** Das Gateway validiert Skin und Textur hart (bekannter Node-Bug „2×2-Dummy-Textur" → Job `failed` statt kaputtem Asset). |

`input_no_fingers` wird bei allen Familien angenommen, wirkt aber **nur** in der
`-Humanoid`-Rig-Stufe (Hände ohne Einzelfinger riggen).

**Das Job-Feld `rig`** benennt die Art des ausgelieferten Skeletts und damit den
Speicher-Kontrakt. Es gibt vier Werte:

| `rig` | Skelett | Auslieferung |
|---|---|---|
| `generic` | UniRig / ArticulationXL, FBX | FBX + Basecolor-Bild, untrennbar (Zeile oben); Karten V-korrigiert, ggf. JPEG |
| `mixamo` | Make-It-Animatable, mixamorig-Namen, GLB | ein selbsttragendes GLB; Skin und Textur werden hart validiert |
| `meshy` | **Meshys eigenes Skelett** (Cloud-Rigging, Biped) | `rigged.glb` (ggf. zusätzlich `rigged.fbx`) mit eingebetteter Textur; bei eingeschalteten Animationen zusätzlich `walking.*` / `running.*` als eigene Results |
| `tripo` | **Tripos Auto-Rig** (Cloud) — die Knochennamen nennt das Job-Feld **`rig_spec`**: `mixamo` (Mixamo-kompatibel, Default des Betreibers) oder `tripo` (Tripos eigenes Skelett) | `rigged.glb` bzw. `rigged.fbx` mit eingebetteter Textur; je konfiguriertem Animations-Preset ein weiteres Result `<preset>.<fmt>` (`walk.glb`) |

Die beiden Cloud-Werte `meshy` und `tripo` **kennzeichnet** das Gateway nur, es
normalisiert und validiert sie nicht — kein V-Flip, keine JPEG-Umkodierung, keine
Paar-Prüfung. Beide Dienste riggen nach eigenen Konventionen; die Dateien so speichern,
wie sie kommen. Die Clips sind eine Zugabe, kein garantierter Bestandteil: liefert der
Dienst einen Clip nicht (oder läuft dessen Task in die Zeitgrenze), fehlt er
kommentarlos — das Rig-Mesh selbst fehlt nie, dafür scheitert der Job.
Bei `tripo` sagt zusätzlich **`rig_spec`**, nach welcher Konvention die Knochen heißen
(`mixamo` oder `tripo`) — ein Feld, das es bei den anderen Rig-Arten nicht gibt. Ein
`tripo`-GLB mit Mixamo-Namen ist trotzdem **kein** Make-It-Animatable-GLB: die harte
Skin-/Textur-Validierung der `mixamo`-Zeile läuft dafür bewusst nicht.
`rig` setzt das Gateway bei **Ketten**-Ergebnissen (dort nennt es die Rig-Variante
der Kette) **und** bei einem direkten `Meshy-Rig`- bzw. `Tripo-Rig`-Job (3.5) — dort
immer `meshy` bzw. `tripo`. Ein Rigging-Ergebnis ist also am `rig`-Feld erkennbar, egal
auf welchem Weg es entstand.

### 3.2 `img2mesh`-Familien

Gemeinsame Parameter (Defaults je Familie in der Tabelle): `input_image`
(Pflicht), `input_name` (string — **immer setzen**, er bestimmt die Dateinamen;
Leerlassen übernimmt Workflow-Reste), `input_face_num` (int),
`input_texture_resolution` (int), `input_remove_background` (bool, Default true —
bei Bildern mit sauberem Alphakanal auf false setzbar), `input_no_fingers` (bool).

| Familie (Aliase) | face_num | texture_resolution | Besonderheit |
|---|---|---|---|
| `Trellis2-Generic-High`, `Trellis2-Humanoid-High` | 20000 | 1024 | höchste Qualität, längste Laufzeit |
| `Trellis2-Generic-Low`, `Trellis2-Humanoid-Low`, `Trellis2-Object-Low` | 20000 | 1024 | schnellere Pipeline |
| `Pixal3D-Generic`, `Pixal3D-Humanoid`, `Pixal3D-Object` | 50000 | 2048 | höchste Auflösung |
| `Hunyuan3D-Generic`, `Hunyuan3D-Humanoid`, `Hunyuan3D-Object` | 40000 | 1024 | ⚠ **`input_face_num` nie über 40000** — größere Werte frieren das Backend ein (kein Fehler, der Job hängt bis zum Timeout). 40000 ist der höchste nachweislich stabile Wert. |
| `Meshy-Object`, `Meshy-Multiview` | Meshy-Default (30000 bei Remesh) | 2048 | Cloud (Meshy.ai, bezahlt pro Task, nur als Fallback oder gezielt). `-Multiview` nimmt `input_image_front` (Pflicht) + optional `input_image_back` / `_left` / `_right`. Zusätzlich `input_texture_prompt` (string) und `input_pose` (`a-pose`/`t-pose`). `input_remove_background`/`input_no_fingers` werden angenommen, wirken nicht. Kein `files`-Upload (`400` — Bilder gehören unter `images`). Liefert `model.glb` (Texturen eingebettet) + `preview.png`. |
| `Meshy-Humanoid`, `Meshy-Humanoid-Multiview` | wie oben | 2048 | Cloud-Mesh (t-pose) → **lokales** Rigging mit Make-It-Animatable, Kette wie `Trellis2-Humanoid-*`: Auslieferung ein `*_rigged.glb`, Job-Feld `rig: "mixamo"`, Speicher-Kontrakt der `-Humanoid`-Zeile in 3.1. `-Multiview` nimmt dieselben vier Bild-Slots wie `Meshy-Multiview`. |
| `Meshy-Humanoid-Cloud` | wie oben | 2048 | Cloud-Mesh (t-pose) → **Cloud**-Rigging (`Meshy-Rig`, 3.5), das Mesh verlässt Meshy nie. Auslieferung `rigged.glb` (+ optionale Clips), Job-Feld `rig: "meshy"` — nicht normalisiert, nicht validiert (3.1). Ein mitgeschicktes `input_height_m` wird an die Rig-Stufe durchgereicht (Default 1.7 m); zwei Tasks = Mesh-Credits + 5 Credits. |
| `Tripo-Object`, `Tripo-Multiview` | Tripo-Default (adaptiv) | 2048 | Cloud (Tripo3D, bezahlt pro Task, wie Meshy nur als Fallback oder gezielt). `-Multiview` nimmt `input_image_front` (**Pflicht**) **plus mindestens eine weitere** Ansicht (`input_image_back` / `_left` / `_right`) — mit nur einem Bild lehnt das Gateway ab, bevor irgendetwas hochgeladen wird (Tripo verlangt zwei Ansichten). `input_name`, `input_remove_background`, `input_no_fingers` werden angenommen, wirken nicht; `input_texture_prompt` und `input_pose` gibt es bei Tripo nicht (still ignoriert). Kein `files`-Upload (`400` — Bilder gehören unter `images`). Liefert `model.glb` (Textur eingebettet) + `preview.png`; jedes weitere vom Betreiber bestellte Format kommt als `model.<fmt>` dazu. |
| `Tripo-Humanoid` | wie oben, Betreiber-Budget 150000 | 2048 | Cloud-Mesh → **Cloud**-Rigging (`Tripo-Rig`, 3.5), das Mesh verlässt Tripo nie. Auslieferung `rigged.glb` (+ konfigurierte Clips wie `walk.glb`), Job-Felder `rig: "tripo"` und `rig_spec` — nicht normalisiert, nicht validiert (3.1). Ein mitgeschicktes `input_rig_type` wird an die Rig-Stufe durchgereicht; zwei bezahlte Tasks = Mesh-Credits + 25 Credits (die Rig-Prüfung davor ist gratis). |

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
Der Workflow trägt die LOD-Kette nicht mehr (Stand 2026-09-08); ein mitgeschickter
Wert wird **stillschweigend ignoriert** — der Job liefert nur das Hauptergebnis, der
Parameter erscheint nicht unter `applied`. Wer reduzierte Fassungen braucht, hängt
`mesh-shrink` / `mesh-shrink-quad` (Abschnitt 1 der Workflow-README) als Folgestufe an
oder dezimiert clientseitig.

V-Flip und JPEG-Umkodierung (Abschnitt 2) greifen nur bei den
`-Generic`-Aliasen; bei `-Object` und `-Humanoid` kommen die Karten unverändert
als PNG.

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
Dateipfad auf dem Backend — nützlich für Server-Admins mit Zugriff auf dessen
Dateisystem, für Clients ist `files` der Weg.)

Auslieferung: `<name>_00001_.glb` + `*_basecolor*.png` + `*_metallic*.png` — hier
**ohne** JPEG-Umkodierung, die Karten bleiben PNG.

**Dreiecke vs. Vertices.** `input_face_num` steuert die **Dreiecke** und wird exakt
getroffen. Die Vertex-Zahl bleibt deutlich darüber (gemessen: 5.000 Dreiecke →
11.129 Vertices): Das Texturbacken trennt das Mesh entlang der UV-Nähte auf, und
jeder Naht-Vertex existiert danach mehrfach. Das ist unvermeidlich, wenn ein Mesh
eine Textur tragen soll — wer Vertices zählt, misst das UV-Layout mit, nicht die
Geometrie.

### 3.5 Rigger direkt (`mesh-rig-unirig`, `mesh-mia`, `Meshy-Rig`, `Tripo-Rig`)

`mesh-rig-unirig` und `mesh-mia` sind API-sichtbar, primär aber die Stage-2 der `-Generic`/`-Humanoid`-
Ketten. Direktaufruf ist möglich (Mesh über `files.input_mesh_path` wie in 3.4),
liefert aber **nur** das Rig-Ergebnis: `mesh-rig-unirig` die nackte FBX (ohne
Basecolor — das Pflichtpaar aus 3.1 muss der Client dann selbst sicherstellen),
`mesh-mia` das geriggte GLB `*_rigged.glb` (Parameter nur `input_mesh_path`,
`input_no_fingers`; kein `input_name`). Achtung Namensfalle: der Alias
`mesh-mia` führt das **Make-It-Animatable**-Workflow aus (mixamo-Rig → GLB) —
der gleichnamige FBX-Rigger „MIA" (`MIAAutoRig`, Workflow `mesh-reg-mia`) ist
auf dem Gateway **nicht registriert**. Im Zweifel die Ketten-Aliase verwenden —
dort garantiert das Gateway die vollständige Auslieferung.

**`Meshy-Rig`** ist der dritte Rigger und der einzige in der Cloud: ein `mesh2rig`-Alias
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
| Vorprüfung | Vor dem Rig läuft Tripos **Rig-Check** (0 Credits). Hält er das Mesh für nicht riggbar, endet der Job **vor** den 25 Credits mit einer Meldung, die den erkannten Rig-Typ nennt. |
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

## 4. Fehlerbilder

| Symptom | Bedeutung / Reaktion |
|---|---|
| `status: "failed"` + `error` | Workflow-/Validierungsfehler (z. B. „no basecolor PNG", „embedded texture is a 2x2 dummy", per-Node-Fehler des Backends). Nicht blind retrien — Fehlertext auswerten. |
| `503` + `Retry-After` beim Start | Park-Zeit abgelaufen, alle Backends belegt — nach `Retry-After` neu einreichen. |
| Job hängt lange in `running` | Mesh-Jobs dauern Minuten; `progress` beachten. Hunyuan3D mit `face_num` > 40000: siehe 3.2 — vermeiden. |
| `status: "failed"` bei `Meshy-Rig` / `Meshy-Humanoid-Cloud` | Meshys Rigging hat abgelehnt (kein erkennbarer Biped, zu viele Dreiecke, unbrauchbare Pose) — endgültig, Credits werden erstattet. Mesh prüfen (3.5), nicht retrien. `400` schon beim Start heißt: Datei fehlt oder ist kein binäres glTF. |
| `status: "failed"` bei `Tripo-Rig` / `Tripo-Humanoid` | Enthält der Fehlertext **`rig-check: mesh is not riggable`**, hat Tripos kostenlose Vorprüfung abgelehnt (der erkannte Rig-Typ steht dabei) — es wurden keine Rig-Credits verbraucht; Mesh prüfen, nicht retrien. Eine fehlende Datei, eine, die kein binäres glTF ist, und ein `input_rig_type`, das nicht zum Rig-Modell des Alias passt, lassen den Job ebenfalls scheitern, bevor ein bezahlter Task entsteht (3.5); `400` schon beim Start heißt: unbekannter `files`-Schlüssel oder unlesbarer Wert. |
| `404` auf `result/{n}` | Job-TTL abgelaufen (Default **24 h**, `ttl_s` 86400) — Artefakte werden serverseitig aufgeräumt; Ergebnisse zeitnah abholen und selbst speichern. |

---

## 5. Minimalbeispiel

```bash
B=https://<gateway>; K=<key>

# Job starten (async)
JOB=$(curl -s $B/v1/generations -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
  -d '{"model":"Trellis2-Humanoid-Low","mode":"async",
       "params":{"input_name":"Held","input_face_num":20000,"input_no_fingers":true},
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
