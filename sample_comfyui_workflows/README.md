# Mesh-Workflows — Ein- und Ausgaben

Stand 2026-09-08. Die Tabellen sind aus den `*_api.json` erzeugt, nicht von Hand gepflegt.

**Dateiablage:** `<name>_api.json` = API-Format (das, was das Gateway einreicht, git-versioniert).
`org/<name>.json` = UI-Format zum Bearbeiten in ComfyUI — **`org/` ist gitignored**, Sicherungen
liegen in `org/backups/`.

**Konventionen:**

* Nodes mit Titel `input_*` sind die Bindepunkte des Gateways.
* Der Haupt-Export trägt den Titel **`Output`** und bekommt `filename_prefix` aus `input_name`.
* Dateinamen: `<name>.<ext>` für das Hauptergebnis, `<name>_<artefakt>.<ext>` für alles andere,
  Artefaktnamen klein. `<name>` = Wert von `input_name`.
* ComfyUI hängt an Export-/Save-Nodes einen laufenden Zähler an: aus `<name>_basecolor` wird auf
  der Platte `<name>_basecolor_00001_.png`. Globs deshalb immer mit `*` am Ende.

---

## 1. Sicht LLM-Gateway

### 1.1 Eingaben (Bindepunkte)

Node-IDs sind stabil, solange der Workflow nicht neu exportiert wird. Bindungen immer über
**Label** anlegen, nie über den rohen Feldnamen — `value` kommt in jedem Workflow mehrfach vor
und kollidiert in der Kette (siehe `main.py`, Kommentar zu `s2_params`).

#### img2mesh-trellis2_high

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image` | `58` | `image` | `'0.png'` | ja |
| `input_name` | `47` | `value` | `'Test3'` | ja |
| `input_face_num` | `62` | `value` | `20000` | ja |
| `input_texture_resolution` | `84` | `value` | `1024` | ja |
| `input_remove_background` | `61` | `value` | `True` | ja |
| `input_no_fingers` | `80` | `value` | `False` | **nein** — Pass-through an Stage 2 |

#### img2mesh-trellis2_low

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image` | `58` | `image` | — | ja |
| `input_name` | `47` | `value` | `''` | ja |
| `input_face_num` | `62` | `value` | `20000` | ja |
| `input_texture_resolution` | `102` | `value` | `1024` | ja |
| `input_remove_background` | `61` | `value` | `True` | ja |
| `input_no_fingers` | `66` | `value` | `False` | **nein** — Pass-through an Stage 2 |

#### img2mesh-trellis2_multiview

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image_front` | `94` | `image` | — | ja |
| `input_image_back` | `95` | `image` | — | ja |
| `input_image_left` | `96` | `image` | — | ja |
| `input_image_right` | `97` | `image` | — | ja |
| `input_name` | `89` | `value` | `'kira'` | ja |
| `input_face_num` | `90` | `value` | `50000` | ja |
| `input_steps` | `91` | `value` | `35` | ja |
| `input_remove_background` | `92` | `value` | `True` | ja |
| `input_no_fingers` | `93` | `value` | `False` | **nein** — Pass-through an Stage 2 |

Vier Bild-Slots; leere Seitenansichten werden über `on_empty: disable` samt ihrem
Zweig entfernt (siehe CLAUDE.md, `_prune_branch`).

#### img2mesh-Pixal3D

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image` | `311` | `image` | — | ja |
| `input_name` | `306` | `value` | `''` | ja |
| `input_face_num` | `307` | `value` | `50000` | ja |
| `input_texture_resolution` | `318` | `value` | `2048` | ja |
| `input_remove_background` | `309` | `value` | `True` | ja |
| `input_no_fingers` | `310` | `value` | `False` | **nein** — Pass-through an Stage 2 |

#### img2mesh-triposplat

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image` | `58` | `image` | `'source.png'` | ja |
| `input_name` | `47` | `value` | `''` | ja |
| `input_num_gaussians` | `62` | `value` | `262144` | ja |
| `input_texture_resolution` | `102` | `value` | `512` | ja — steuert Mesh-Auflösung **und** Preprocess-Größe |
| `input_remove_background` | `61` | `value` | `True` | ja (seit 2026-08-03) |

Kein `input_face_num`: `SplatToMesh` kennt kein Face-Ziel, die Dichte kommt über
`input_num_gaussians` und `input_texture_resolution`.

#### img2mesh-hunyuan3d

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_image` | `1` | `image` | `'source.png'` | ja |
| `input_name` | `5` | `value` | `''` | ja |
| `input_face_num` | `3` | `value` | `40000` | ja |
| `input_texture_resolution` | `4` | `value` | `1024` | ja |
| `input_remove_background` | `2` | `value` | `True` | ja |
| `input_no_fingers` | `6` | `value` | `False` | **nein** — Pass-through an Stage 2 |

> `input_face_num` **nicht über 40000** setzen. Bei 100000 friert die VM im UV-Unwrap ein
> (kein OOM-Kill, kein Traceback — der Gast reagiert nicht mehr). 40000 ist der höchste
> nachweislich durchlaufende Wert auf VM 402.

#### mesh-shrink / mesh-shrink-quad

| Bindepunkt | Node | Feld | Default | wirkt im Graphen |
|---|---|---|---|---|
| `input_mesh_path` | `10` | `glb_path` | `'output/3D/Trellis2_00001_.glb'` | ja — **Mesh-Eingang der Kette** |
| `input_name` | `4` | `value` | `''` | ja |
| `input_face_num` | `2` | `value` | `5000` | ja |
| `input_texture_resolution` | `3` | `value` | `1024` | ja |
| `input_keep_borders` | `8` | `value` | `'true'` | ja |
| `input_no_fingers` | `6` | `value` | `False` | **nein** — Pass-through an Stage 2 |

Unterschied: `mesh-shrink` dezimiert (Quadric Edge Collapse), `mesh-shrink-quad` remesht
(QuadriFlow, Quad-Topologie). Gleiche Bindepunkte, austauschbar.

#### mesh-reg-unirig / mesh-reg-mia / mesh-make-it-animatable

| Workflow | `input_mesh_path` | `input_name` | `input_face_num` | `input_no_fingers` |
|---|---|---|---|---|
| `mesh-reg-unirig` | `4` (`file_path`) | `6` | `9` (20000) | — |
| `mesh-reg-mia` | `4` (`file_path`) | `6` | — | `5` |
| `mesh-make-it-animatable` | `9` (`value`) | — | — | `5` |

**Abweichung:** Diese drei haben **keinen** Node mit Titel `Output`. Ihr Ergebnis ist der
Rückgabewert des Rig-Nodes (`fbx_output_path` bzw. `rigged_glb_path`), der über den
Preview-Node in `/history` landet. `mesh-make-it-animatable` hat außerdem kein `input_name`.

### 1.2 Ausgaben

| Workflow | Output-Node | erzeugte Dateien |
|---|---|---|
| `img2mesh-trellis2_high` | `82` | `<name>.glb` · `_whitemesh.glb` (#33) · `_refined.glb` (#36) · `_basecolor.png` (#70) · `_metallic.png` (#71) |
| `img2mesh-trellis2_low` | `100` | `<name>.glb` · `_whitemesh.glb` (#68) · `_basecolor.png` (#92) · `_metallic.png` (#93) |
| `img2mesh-Pixal3D` | `312` | `<name>.glb` · `_whitemesh.glb` (#322) · `_basecolor.png` (#313) · `_metallic.png` (#315) |
| `img2mesh-triposplat` | `107` | `<name>.glb` — **sonst nichts**, siehe unten |
| `img2mesh-hunyuan3d` | `50` | `<name>.glb` · `_whitemesh.glb` (#41) · `_basecolor.png` (#45) · `_metallic.png` (#47) · `_multiview.png` (#43) |
| `mesh-shrink` / `-quad` | `30` | `<name>.glb` · `_basecolor.png` (#33) · `_metallic.png` (#35) |
| `mesh-reg-unirig` | — (#7) | `<name>.fbx` |
| `mesh-reg-mia` | — (#2) | `<name>.fbx` |
| `mesh-make-it-animatable` | — (#7) | `<name>.glb` (mixamo-Skin) |

**triposplat liefert keine getrennten Texturen.** `SplatToMesh` gibt den Core-Typ `MESH` aus,
die gesamte Textur-Werkzeugkette arbeitet auf `TRIMESH`, und es existiert kein Node, der
zwischen beiden übersetzt. Seine Textur steckt eingebettet im GLB. Folge: **triposplat darf
nicht direkt an einen `generic`-Rigger gekettet werden** (siehe 2.2) — entweder `mesh-shrink`
dazwischenschalten, das die PNGs aus dem GLB erzeugt, oder nur den `mixamo`-Weg nutzen.

### 1.3 Ketten

Konfiguration des Nachfolgers: `{alias, export_node, mesh_param, relay?, keep_from_mesh?, rig?}`.

* `export_node` — die Node-ID aus 1.2, deren Datei an Stage 2 übergeben wird.
* `mesh_param` — Label des Mesh-Eingangs bei Stage 2, hier durchgehend `input_mesh_path`.
* `keep_from_mesh` — Globs für Stage-1-Dateien, die in die Auslieferung übernommen werden.
  Für `generic`-Rigger **zwingend** `*_basecolor*.png` (Begründung in 2.2).
* `rig` — `generic` (FBX-Ergebnis) oder `mixamo` (GLB-Ergebnis); steuert Validierung und
  Textur-Normalisierung.

Übergabe: bei geteiltem Dateisystem als absoluter Pfad, sonst per Upload ins `input/` von
Stage 2. Relative Pfade (`output/3D/…glb`) funktionieren, weil ComfyUI mit
`WorkingDirectory=/home/kai/ComfyUI` läuft.

**Genau zwei Stufen.** Das Gateway kennt `1/2` → `2/2`, ein Nachfolger ohne Rekursion.
`img2mesh → mesh-shrink` und `img2mesh → rig` gehen, `img2mesh → shrink → rig` in **einem**
Job nicht.

Stage-1-Parameter werden per **Label** an Stage 2 weitergereicht — daher der unverdrahtete
`input_no_fingers` in den img2mesh-Workflows: er ist der Anker, damit der Parameter an Stage 1
annehmbar ist, wirken tut er erst beim Rigger.

---

## 2. Sicht Abnehmer

### 2.1 Ohne Rigger (nur Mesh-Workflow)

| Datei | Inhalt | speichern? |
|---|---|---|
| `<name>.glb` | fertiges Mesh **mit eingebetteter Textur** | **Pflicht** — das ist das Asset |
| `<name>_basecolor.png` | Albedo-Karte, separat | optional — nur wenn die Engine eigene Maps will |
| `<name>_metallic.png` | Metallic/Roughness, separat | optional, wie oben |
| `<name>_whitemesh.glb` | unbemaltes Mesh vor der Texturierung | optional — Kollision, LOD-Quelle, Rigging-Eingang |
| `<name>_refined.glb` | Zwischenstufe (nur trellis2_high) | nein — Diagnose |
| `<name>_multiview.png` | die 6 gerenderten Ansichten (nur hunyuan3d) | nein — Diagnose |

Das GLB ist selbsttragend: Geometrie, UVs und Texturen stecken darin. Wer nur rendert,
braucht ausschließlich diese eine Datei.

### 2.2 Mit `generic`-Rigger (UniRig, MIA → FBX)

| Datei | Inhalt | speichern? |
|---|---|---|
| `<name>.fbx` | geriggtes Mesh mit Skelett | **Pflicht** |
| `<name>_basecolor.png` | Albedo — aus der Mesh-Stufe übernommen | **Pflicht, untrennbar vom FBX** |
| `<name>_metallic.png` | Metallic/Roughness | optional |
| `<name>.glb` (Stage 1) | untexturiertes Vorstadium | optional — Referenz |

**Die FBX enthält die Textur nicht.** Sie referenziert sie nur über einen Temp-Pfad, der beim
Abnehmer ins Leere zeigt; der Client bindet über den ausgelieferten Dateinamen neu. Wird das
PNG nicht mitgespeichert, ist das Asset unbrauchbar und nicht wiederherstellbar. Das Gateway
erzwingt das Paar (`validate_delivery` bricht ohne PNG hart ab), aber nur die Auslieferung —
**das Speichern liegt beim Abnehmer.**

Die Texturen kommen **V-gespiegelt** aus dem Bake und werden vom Gateway einmal zentral
gedreht (`normalize_delivery`), optional nach JPEG umkodiert. Der Abnehmer speichert sie
unverändert und kompensiert nichts.

### 2.3 Mit `mixamo`-Rigger (Make-It-Animatable → GLB)

| Datei | Inhalt | speichern? |
|---|---|---|
| `<name>.glb` | geriggtes Mesh, mixamorig-Skin, **eingebettete** Textur | **Pflicht — genügt allein** |

Hier gilt das Gegenteil von 2.2: eine Datei, alles drin. Das Gateway prüft Skin und Textur
(eine 2×2-Dummy-Textur ist ein bekannter Node-Bug und schlägt fehl).

### 2.4 Kurzfassung fürs Speichern

* **Immer:** die Datei des `Output`-Nodes (`<name>.glb`) bzw. das Rigger-Ergebnis.
* **Zusätzlich Pflicht bei FBX-Rigging:** `<name>_basecolor.png`.
* **Nie nötig:** `_multiview`, `_refined`.
* **Nach Bedarf:** `_whitemesh` (Kollision/LOD), `_metallic` (PBR-Rendering).
* Dateien über ~30 MB erzeugen eine Warnung im Job-Meta — Hinweis auf Web-Tauglichkeit,
  kein Fehler.
