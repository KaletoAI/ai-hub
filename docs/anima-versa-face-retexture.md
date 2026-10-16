# Anima-Versa: Gesichtstextur nach img2mesh korrigieren

## Vertrag und Freischaltung

Diese Integration ist **ein unabhängiger zweiter AI-Hub-Job nach erfolgreichem img2mesh und vor jedem Rigging**. Modellname: `GLB-Face-Retexture`, Task: `mesh2mesh`. Bestehende img2mesh-, Rigging- und Legacy-Face-Workflows bleiben unverändert; keine automatische Gateway-Kette voraussetzen.

Auf **dx10-02 und k12** sind Node **v1.0.2** und sichtbarer Workflow installiert. Der AI-Hub-Alias wurde auf dx10-02 mit einem echten GLB-/RGB-Upload und Ergebnisdownload geprüft; auf k12 lief der native ComfyUI-Workflow erfolgreich. Vor Aktivierung in Anima-Versa mit dem normalen Benutzer-Key `GET /v1/generations/GLB-Face-Retexture/schema` abrufen und die Freigabe in der User-Allow-List sicherstellen. Die Anima-Versa-Clientintegration selbst ist nicht Teil dieser Installation.

## Eingangsbedingungen

- Originales, statisches GLB: **ein Mesh, ein indiziertes Dreiecks-Primitive**, eingebettete unkomprimierte opake Textur. Keine externen Assets, Skinning/Rigs, Animation oder Morph-Targets. Ein beliebiges img2mesh-Ergebnis erfüllt diese Bedingungen nicht automatisch; ungeeignete Assets klar ablehnen, nicht still konvertieren.
- **Y-up, +Z nach vorne**. `yaw` wählt die Registrierungsansicht in Grad (−180 bis 180), ohne die Geometrie zu drehen/deformieren. Mit 0 beginnen; nur bei abweichender Frontausrichtung anpassen. Ein erkennbares Gesicht muss im Mesh-Render und im Foto sichtbar sein.
- Genau ein passendes frontales **Original-RGB-Foto**, nicht StableX-Normalmap, Tiefenbild, Maske oder diagnostischer Render. Das Foto aus dem Anima-Versa-Asset behalten und erneut hochladen.
- Jede Korrektur beginnt erneut am **Original-GLB**. Das korrigierte GLB enthält getrennte Körper-/Gesichts-Primitives und ist kein unterstützter Eingang für einen weiteren Projektionslauf.

Der Node projiziert/blendet Textur; er erzeugt kein neues Mesh, erhöht nicht die Face-Anzahl, rekonstruiert keine Gesichtsanatomie und repariert keine Silhouette. Vertexpositionen, UVs, Normalen und Tangenten bleiben erhalten; GLB-Struktur und Materialien ändern sich, daher ist die Datei nicht byte-identisch.

## Konkreter Ablauf für den Entwickler

1. Auf `done` des img2mesh-Jobs warten. Original-Job-ID und Original-Artefakt getrennt speichern.
2. In dessen `results[]` das GLB anhand `mime: "model/gltf-binary"` auswählen (zusätzlich `kind`/Dateityp beachten), **nie pauschal `results[0]`**. Ein OBJ/FBX oder PNG ist kein Ersatz. Bei mehreren GLBs eine explizite Auswahl treffen.
3. GLB über dessen AI-Hub-Result-URL mit `Authorization: Bearer <KEY>` herunterladen. Nur erwartete AI-Hub-URLs mit diesem Header abrufen; keinen Key an fremde Hosts weitergeben. Auch Schema, Polling, Cancel und korrigierter Download verwenden die normale Bearer-Authentifizierung.
4. GLB-Bytes und Original-RGB-Foto als Base64/Data-URI an folgenden nativen Endpoint senden. **Keine Backend-Dateipfade**, keine öffentliche Freigabe privater Result-URLs und keine Änderung ihrer Authentifizierung erforderlich. Das Gateway lädt beide Eingaben job-isoliert zum ausgewählten Backend hoch.

```http
POST /v1/generations
Authorization: Bearer <KEY>
Content-Type: application/json
```

```json
{
  "model": "GLB-Face-Retexture",
  "mode": "async",
  "files": {
    "input_mesh_path": "data:model/gltf-binary;base64,<BASE64_ORIGINAL_GLB>"
  },
  "images": {
    "reference_image": "data:image/png;base64,<BASE64_ORIGINAL_RGB_PHOTO>"
  },
  "params": {
    "strength": 1,
    "feather": 0.18,
    "render_resolution": 768,
    "yaw": 0,
    "light_balance": 1,
    "tone_match": 1,
    "brightness": 1
  }
}
```

Ab Node **v1.1.0** sind `light_balance`, `tone_match` und `brightness` optional (Node-Defaults 0 / 0 / 1 = Verhalten wie v1.0.2, bytegleich). Sie bearbeiten das Foto im linearen Licht vor der Projektion: Seitenlicht-Ausgleich entlang der aus Landmarken bestimmten Gesichtsmittellinie innerhalb der semantischen Hautmaske (Merkmale werden nie gespiegelt), dann Hautton-Abgleich an die eigene Gesichtshaut des Meshes (semantische Gesichtsmaske des Mesh-Renders; ab v1.1.1, v1.1.0 nahm einen Ring um das Gesicht, der auf generierten Meshes vor allem Hals-Schatten und Haaransatz traf), dann Helligkeit. Das Proof meldet aktive Werte unter `photo_adjust`. Anima-Versa erkennt die drei Namen im Schema und schickt dann seine gespeicherten Werte (Default 1 / 1 / 1); ohne sie gleicht es lokal aus.

Bei JPEG-Foto `data:image/jpeg;base64,` verwenden. Die Base64-Platzhalter durch echte Bytes ersetzen. Einzelne Uploads sind auf 64 MB begrenzt; zusätzlich gilt das konfigurierte HTTP-Body-Limit einschließlich Base64-Overhead. Ein direkter URL-Upload ist hier nicht erforderlich: geschützte AI-Hub-URLs nicht als `files`-URL weiterreichen, da der Gateway-URL-Fetch keine Benutzer-Download-Authentifizierung übernimmt und private Adressen verweigert.

5. Antwort `202 {"job_id": "…"}` als **separate Korrektur-Job-ID** speichern. `GET /v1/jobs/{job_id}` etwa alle zwei Sekunden abfragen. `queued`/`running` anzeigen; vorhandene Fortschritts-/ETA-Felder verwenden, aber keinen Prozentwert erfinden, wenn der CPU-Node keine Schrittfortschritte meldet. Bei Transportfehlern zuerst denselben Job weiter abfragen, nicht erneut generieren.
6. Bei `done` wiederum nach GLB-MIME/Typ auswählen und mit Bearer-Header herunterladen. Original und korrigiertes Asset samt jeweiligen Job-IDs, Result-URLs und verfügbaren SHA256 getrennt halten; das Original nicht überschreiben. Die Vergleichs-PNG dient **nur der Diagnose**, nicht als Textur-Sidecar oder Mesh-Eingang. Das korrigierte GLB enthält die benötigten Texturen.
7. Erst mit diesem korrigierten GLB den bestehenden Rigging-Schritt starten. Der Rigger muss beide Körper-/Gesichts-Primitives und ihre Materialien erhalten; nicht nur das erste Primitive laden. Bei `failed`/`cancelled` Fehler anzeigen und keine Rigging-Fortsetzung auslösen. Bei abgelaufenem/gelöschtem Job (HTTP 404 bzw. 410) den Ablauf als nicht mehr abrufbar beenden; lokale Originalkopie behalten. Cancel: `POST /v1/jobs/{job_id}/cancel`. Kein unendliches Polling und kein stiller Rückfall auf das Original als angeblich korrigiertes Ergebnis.

Resultate bleiben nur bis zum Job-TTL verfügbar (Standard 24 Stunden): beide GLBs rechtzeitig in Anima-Versas eigener Asset-Ablage persistieren. Der Gateway speichert Mesh-Uploads nicht dauerhaft als Job-Eingabe.

## Workflow und Admin-Mapping

- Sichtbarer ComfyUI-Workflow: [`../sample_comfyui_workflows/mesh-face-retexture.json`](../sample_comfyui_workflows/mesh-face-retexture.json). In ComfyUI öffnen, eigene GLB-/Foto-Slots füllen, die benannten Parameter an Node 3 setzen. Vorschau 4 zeigt das korrigierte GLB, Vorschau 5 den Vergleich, Note 7 erläutert Grenzen.
- API-Workflow: [`../sample_comfyui_workflows/mesh-face-retexture_api.json`](../sample_comfyui_workflows/mesh-face-retexture_api.json).
- Alias `GLB-Face-Retexture`, Task `mesh2mesh`, Backend-Namen `dx10-02` und `k12-gpu` (Health-IDs `comfyui:dx10-02` und `comfyui:k12-gpu`); andere Aliase nicht bearbeiten.

| Öffentliches Feld | Node | Workflow-Feld | Default |
|---|---|---|---|
| `files.input_mesh_path` | 1 `Load3DAdvanced` | `model_file` | `none` (native Leerauswahl), Original-GLB hochladen |
| `images.reference_image` | 2 `LoadImage` | `image` | leer; Slot `on_empty: required` |
| `params.strength` | 3 `GLBFaceRetexture` | `strength` | 1 |
| `params.feather` | 3 | `feather` | 0.18 |
| `params.render_resolution` | 3 | `render_resolution` | 768 |
| `params.yaw` | 3 | `yaw` | 0 |
| `params.light_balance` | 3 | `light_balance` | 0 |
| `params.tone_match` | 3 | `tone_match` | 0 |
| `params.brightness` | 3 | `brightness` | 1 |

Autoritativer Output: Node **6 `SaveGLB`**, Titel `output_final`, Prefix `3d/GLBFaceRetexture`. Zusätzlich Node **8 `SaveImage`**, Prefix `GLB-Face-Retexture_Comparison`; `output_globs: ["GLB-Face-Retexture_Comparison_*.png"]`. Keine Legacy-Skripte, keine Image-Normalisierung/Texture-Sidecar-Regeln hinzufügen. Native `FILE_3D`-Verbindungen zur Vorschau und SaveGLB erhalten; Loader-/Preview-Viewport-State ist leer, Breite/Höhe 1024, vorhandener Kamera-Link 1→4 bleibt bestehen.

## Installation auf dx10-02 (Operator, nicht Anima-Versa)

Öffentliches Paket: <https://github.com/KaletoAI/ComfyUI-GLB-Face-Retexture>. Als ComfyUI-Benutzer auf dx10-02:

```sh
cd /home/kai/ComfyUI
git clone https://github.com/KaletoAI/ComfyUI-GLB-Face-Retexture custom_nodes/ComfyUI-GLB-Face-Retexture
cd custom_nodes/ComfyUI-GLB-Face-Retexture
/home/kai/comfyui-env/bin/python install.py --python /home/kai/comfyui-env/bin/python
```

Der `--python`-Interpreter muss Python 3.12 sein. Bei bereits vorhandenem Clone nicht erneut klonen: mindestens den veröffentlichten Paketstand **v1.0.1** verwenden (`git pull --ff-only` im unveränderten Clone). `install.py` erstellt den **privaten CPU-Worker `.venv`** und lädt die offiziellen Modelle; **kein `pip install -r requirements-worker.txt` im Haupt-ComfyUI-Environment**. Linux ARM64/aarch64 verwendet **MediaPipe 0.10.18** im privaten Worker; andere unterstützte Plattformen behalten 0.10.21. Keinen Fake-Fallback einsetzen und das bestehende Haupt-Environment nicht ändern.

Nach Installation erst die aktuelle ComfyUI-Queue prüfen. **Nur bei leerer laufender und wartender Queue und freigegebenem Idle-Zeitpunkt** ComfyUI mit dem vorhandenen Service-Verfahren neu starten, danach Browser hart aktualisieren. AI-Hub nicht neu starten. Der Operator prüft anschließend Node-Registrierung und native Schema-Kompatibilität auf dx10-02, öffnet den sichtbaren Workflow im echten Browser und führt einen echten Job einschließlich GLB- und Vergleichsdownload aus, bevor Anima-Versa die Funktion freischaltet.

## Geprüfter Stand

- Node **v1.1.1** (2026-10-16, Tag `v1.1.1`) auf dx10-02 und k12 installiert; je 36 Worker-Tests grün; GitHub-CI Ubuntu/Windows grün. Echter Lauf mit generiertem Gesicht und Trellis-Mesh: Wangenverhältnis 0,71 → 0,94 bei `light_balance` 1, Hautton-Gewinn RGB 1,21/1,04/1,09 bei `tone_match` 1 (v1.1.1).

- Node v1.0.2 auf dx10-02 (Linux ARM64/NVIDIA GB10) und k12 (Linux x86_64); ComfyUI 0.37.0 und Frontend 1.53.6. Je 31 Verhaltenstests im isolierten Python-3.12-Worker bestanden; die ComfyUI-Hauptumgebungen bleiben unverändert.
- Echter asynchroner AI-Hub-Job: originales Journal-GLB und RGB-Referenz über die oben genannten Upload-Felder, GLB und Vergleichs-PNG heruntergeladen.
- Sichtbarer ComfyUI-Workflow per **Run** ausgeführt, auch mit `strength: 0.85`; alle vier Widgets benannt, keine fehlende Node. In der Bibliothek unter **GLB-Face-Retexture** verfügbar.
- Im Test blieben alle 101.600 Dreiecke, Vertexattribute, ursprünglichen Binärdaten und Szenentransformationen erhalten; geschützte Körperbereiche: 0 veränderte Render-Pixel.
- v1.0.2: Proof-Schema 2, echter Workflow-Lauf auf beiden Hosts; Geometrie unverändert und 0 veränderte Pixel in geschützter Körpertextur. Der Alias-Editor zeigt für `input_mesh_path` die native Leerauswahl `none` ohne `stale`.
- Das Test-Ergebnis war 47 MB groß und erzeugte die bestehende AI-Hub-Größenwarnung. Asset-Größe im Client berücksichtigen; keine automatische Kompression oder Vereinfachung behaupten.
