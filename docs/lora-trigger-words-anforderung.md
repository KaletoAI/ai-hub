# Anforderung: LoRA-Trigger-Wörter im AI-Hub

**Für:** Claude Code im `ai-hub`-Repo (Design und Umsetzung dort)
**Von:** quartermaster-Session, 2026-10-02
**Status:** Anforderung. Das Design (Spec unter `docs/superpowers/specs/`) entsteht im ai-hub-Projekt.

## Ziel

Der AI-Hub soll zu jeder LoRA die Trigger-Wörter kennen, sie dauerhaft speichern
und API-Clients abfragen lassen. Ein Client, der eine LoRA per Request wählt
(`loras: [...]` oder `lora_N`), soll vom AI-Hub erfahren können, welche Wörter in den
Prompt gehören, damit die LoRA greift. Heute muss der Client das selbst wissen.

Quelle der Trigger-Wörter ist Civitai, zugeordnet über den **SHA256 der LoRA-Datei**
(nicht über den Dateinamen: Die Dateien auf dem Share sind umbenannt).

## Ist-Stand (Bezugspunkte im Code)

- LoRA-Namen kommen aus `/object_info` der ComfyUI-Backends (`adapters._comfy_loras`),
  nur im Speicher (`main.backend_loras`), ohne Hash und ohne Metadaten.
- Client-Sicht: `GET /v1/generations/{alias}/loras` liefert nur sortierte Dateinamen;
  `GET /v1/generations/{alias}/schema` hat einen `loras`-Block.
- Konsole: Input & Routing → LoRAs (`_routing_loras_body`): LoRA → Backends.
- Der Share wird komplett gelistet (`LanSource`), `models/loras/...` inkl. Größe.
  SHA256 rechnet der AI-Hub nur bei LAN-Transfer oder Check & save
  (`modelsrc_sha`-Cache). Die meisten LoRAs haben dort noch keinen Hash.
- Es gibt noch keine Civitai-Anbindung.

## Datenlage (Stichprobe 2026-10-02, Share auf K12)

- 135 LoRA-Dateien unter `loras/`, alle gehasht.
- 121 per Hash auf Civitai gefunden, davon 71 mit nicht-leerer `trainedWords`-Liste.
  14 nicht auf Civitai (eigene oder anderswo bezogene Dateien), eine davon auf HF.
- Bis zu 14 Einträge pro LoRA, 152 insgesamt.
- Zwei Civitai-Versionen sind je mehreren Dateien zugeordnet (z. B. High/Low-Hälften).
- **Wichtig, Qualität der Rohdaten:** `trainedWords` ist oft kein sauberes Trigger-Token.
  Vorkommende Formen:
  - echtes Trigger-Token (`m4pdr4w`, `RPGmap`)
  - lange kommagetrennte Tag-Ketten (Prompt-Beispiele in einem einzigen Eintrag)
  - ganze Sätze und Edit-Anweisungen (`Convert the character in the image to ...`)
  - Hinweise ohne Trigger-Charakter (`Follow the prompt template I left in the workflow.`)
  - Schmutz: führende/abschließende Leerzeichen, doppelte Leerzeichen, `,,`
- Basismodell (`baseModel`) ist fast immer gesetzt (ZImageTurbo, Flux.2 Klein 9B,
  Wan 2.2 I2V, Krea 2, Qwen …) und für Clients nützlich.

Seed-Daten (Pfad, Größe, SHA256, Civitai-IDs, trainedWords, baseModel) für alle 135
Dateien liegen auf dev-claude unter
`/home/dev/projekte/quartermaster/tmp/model-sources-2026-10-02.json` (Schlüssel `loras`).
Nur als optionaler Startbestand bzw. Testdaten. Nicht ins Repo übernehmen
(private Dateinamen, das Repo ist öffentlich).

## Funktionale Anforderungen

**F1 Zuordnung über den Inhalt.** Metadaten hängen am SHA256 der Datei, nicht am Namen.
Eine Datei, die umbenannt oder in einen Unterordner verschoben wird, behält ihre
Trigger-Wörter. Ändert sich der Inhalt (Größe oder Hash anders), gelten die alten
Metadaten nicht mehr.

**F2 Ermittlung.** Für jede LoRA auf dem Model-Share ermittelt der AI-Hub den SHA256
und fragt Civitai per Hash ab (`GET https://civitai.com/api/v1/model-versions/by-hash/<sha256>`;
funktioniert **ohne** API-Token, am 2026-10-02 für alle 135 Dateien geprüft).
Gespeichert werden mindestens: `trainedWords` (Rohliste, Reihenfolge erhalten),
Civitai `modelId`, `versionId`, Modellname, Versionsname, `baseModel`, Link zur
Modellseite, Abrufzeitpunkt. „Nicht auf Civitai“ (404) wird ebenfalls gespeichert,
damit nicht bei jedem Lauf neu gefragt wird.

**F3 Aktualität.** Neue LoRAs auf dem Share werden ohne Zutun erfasst (Hash und
Civitai-Abfrage im Hintergrund). Geänderte Dateien werden neu gehasht und neu
abgefragt. Zusätzlich gibt es eine manuelle Auffrischung (alle oder einzelne,
auch für frühere 404-Einträge).

**F4 Kuratierte Trigger-Wörter.** Weil die Rohdaten oft Tag-Ketten oder Anweisungen
sind, gibt es neben der Civitai-Rohliste eine **kuratierte Liste** pro LoRA, die der
Operator in der Konsole pflegt. Sie
- überschreibt für Clients die Rohliste, sobald sie gesetzt ist,
- übersteht jede Civitai-Auffrischung,
- ist auch für LoRAs ohne Civitai-Eintrag setzbar,
- darf leer sein (= „bewusst keine Trigger“, nicht „unbekannt“).
Die Rohliste bleibt daneben sichtbar und abrufbar. Grundlegende Bereinigung
(Trimmen, doppelte Leerzeichen, leere Einträge) darf automatisch erfolgen;
inhaltliches Zerlegen von Tag-Ketten nicht.

**F5 Client-Abfrage.** Clients können die Trigger-Wörter abfragen:
- je Alias: zu jeder LoRA, die für den Alias gültig ist (heutige Menge von
  `/v1/generations/{alias}/loras`), die Metadaten aus F2/F4;
- je LoRA-Name einzeln.
Die Antwort unterscheidet erkennbar: kuratiert / nur Civitai / nicht auf Civitai /
noch nicht ermittelt / Datei nicht auf dem Share (LoRA nur lokal auf einem Backend,
kein Hash möglich).
**Abwärtskompatibel:** Bestehende Clients von `/v1/generations/{alias}/loras` (heute eine
Liste von Strings) dürfen nicht brechen. Auth und Allow-List wie bei den bestehenden
Generation-Discovery-Routen. Das Schema (`/schema`) verweist auf die neue Abfrage.

**F6 Konsole.** Im LoRAs-Tab pro LoRA sichtbar: Trigger-Wörter (kuratiert und roh),
Quelle, Basismodell, Link zu Civitai, Stand der Ermittlung. Dort auch Pflege (F4)
und Auffrischen (F3).

**F7 Paare.** High/Low-Paare (`lora_groups`/`lora_counterpart`) sind zwei Dateien mit
je eigenem Hash. Ein Client, der eine Hälfte wählt, muss die Trigger-Wörter des Paars
ohne Sonderwissen bekommen.

## Nicht-funktionale Anforderungen

- **N1** Der Request-Pfad wartet nie auf Hashing oder Civitai. Fehlende Daten heißen
  „noch nicht ermittelt“, die Generierung läuft unverändert.
- **N2** Civitai-Ausfall, Rate-Limit (429) oder Timeout beeinträchtigen nichts außer der
  Ermittlung selbst. Zurückhaltende Abfragerate, Wiederholung später.
- **N3** Hashing belastet den Share nicht spürbar und läuft hinter Transfer-Hashes
  (Größenordnung: 135 LoRAs = 54 GB).
- **N4** Daten sind persistent (überstehen Restart und Deploy).
- **N5** Nach außen geht nur der Hash, nie Datei oder Dateiname. Kein Civitai-Token nötig.
  Falls später doch einer gebraucht wird: wie `hf_token` verschlüsselt speichern.
- **N6** Ohne konfigurierten Model-Share (`modelsrc_host`) bleibt das Feature inaktiv
  (Status „nicht ermittelbar“), sonst keine Seiteneffekte.

## Nicht Teil dieser Anforderung (offene Entscheidung)

- **Automatisches Einsetzen** der Trigger-Wörter in den Prompt durch den AI-Hub, wenn ein
  Request eine LoRA wählt. Erst einmal nur Auskunft; der Client baut den Prompt.
  Falls später gewünscht: abschaltbar pro Alias oder Request, und ein gepinntes
  Workflow-Feld bleibt maßgeblich (CLAUDE.md-Regel zu gepinnten `(node, field)`).
- Andere Quellen als Civitai (z. B. HF-Modellkarten).
- Metadaten für andere Modelltypen (Checkpoints, Text-Encoder …).

## Abnahmekriterien

1. Nach dem ersten Lauf hat jede LoRA auf dem Share einen Status. Für die Stichprobe
   oben: 121 mit Civitai-Daten, 14 „nicht auf Civitai“.
2. Eine neu auf den Share gelegte LoRA erscheint ohne manuellen Schritt mit Status und,
   falls auf Civitai, mit Trigger-Wörtern.
3. Umbenennen einer LoRA-Datei verliert weder Civitai-Daten noch kuratierte Liste.
4. Eine kuratierte Liste übersteht eine manuelle Auffrischung aller LoRAs.
5. Ein bestehender Client von `/v1/generations/{alias}/loras` funktioniert unverändert.
6. Civitai nicht erreichbar: Generierungen laufen normal, Konsole zeigt den Fehlerstand.
7. Ein Client kann für einen Alias in **einem** Aufruf alle gültigen LoRAs samt
   Trigger-Wörtern und Status holen.
