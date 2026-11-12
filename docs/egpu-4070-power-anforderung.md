# Anforderung: RTX 4070 (USB4-eGPU an K12) sicher aus- und einschalten

**Für:** Claude Code im `ai-hub`-Repo (Design und Umsetzung dort)
**Von:** quartermaster-Session, 2026-11-12
**Status:** Anforderung. Wie der AI-Hub das umsetzt (Konsole, Ablauf, Speicherung), entscheidet das ai-hub-Projekt.

## Ziel

Im AI-Hub soll es für das Backend `rtx-4070` (ComfyUI in CT 462 auf K12) eine Funktion
**„4070 ausschalten“** geben, die die Karte so vorbereitet, dass Kai sie danach gefahrlos vom
Strom nehmen kann, und eine Gegenfunktion **„4070 ist wieder an“**, die sie nach dem
Einschalten wieder in Betrieb nimmt.

Das physische Ein- und Ausschalten macht weiterhin Kai am Gerät. Der AI-Hub sagt nur, **wann**
es gefahrlos ist.

## Warum das nötig ist

Die 4070 hängt per USB4/Thunderbolt am K12. Wird sie im laufenden Betrieb einfach
ausgeschaltet, ist das für den Kernel ein Surprise-Removal. Mit dem NVIDIA-Treiber 595 bleibt
der PCIe-Hotplug-Thread dann im Treiber hängen (belegt einen CPU-Kern dauerhaft) und hält die
globale PCI-Rescan-Sperre. Folge: Die wieder eingeschaltete Karte wird nicht mehr erkannt,
`nvidia-smi` meldet `Unable to determine the device handle ... Unknown Error`, und **nur ein
Reboot von K12** hilft. Das trifft auch CT 460 (RTX 3090, Produktiv-ComfyUI) und CT 451.

Beobachtet 2026-11-12: Ausschalten 00:01, Xid 79 + Xid 154, Wiedereinschalten 15:06 ohne
PCI-Erkennung.

Wird die Karte dagegen **vor** dem Ausschalten per Software abgemeldet (PCI-Remove, solange sie
noch antwortet), läuft das Entfernen sauber durch.

## Host-Seite (fertig, quartermaster)

Skript `/usr/local/sbin/egpu-4070` auf K12 (Quelle: quartermaster `hosts/pveK12/egpu-4070`).

| Aufruf | Wirkung |
|---|---|
| `egpu-4070 status --json` | read-only Zustand |
| `egpu-4070 off --stop-ct --json` | stoppt CT 462 (nur wenn dessen ComfyUI-Queue leer ist), wartet bis kein Prozess die Karte offen hat, PCI-Remove, prüft das Verschwinden |
| `egpu-4070 on --start-ct --json` | wartet auf die Erkennung (stößt bei Bedarf einen Rescan an), wartet auf den NVIDIA-Treiber, legt `/dev/nvidiaN` an, passt die Geräteeinträge von CT 462 an (`ct-gpu-devmap.sh`), startet CT 462 |

Ohne `--stop-ct` bricht `off` ab, wenn CT 462 läuft. Ohne `--start-ct` bleibt CT 462 nach `on` aus.

**Laufzeiten:** `off` bis ca. 2 min (CT-Shutdown bis 60 s, Warten auf Nutzer bis 10 s, Remove
bis 30 s). `on` bis ca. 3 min. Also asynchron ausführen, nicht im Request.

**Exit-Codes und `result`:**

| Exit | `result` | Bedeutung | Was der AI-Hub zeigt |
|---|---|---|---|
| 0 | `ok` | Aktion erfolgreich | bei `off`: „Jetzt ausschalten“; bei `on`: Backend wieder freigeben |
| 0 | `absent` | Karte nicht am Bus (aus oder schon abgemeldet) | gefahrlos ausschaltbar bzw. „ist aus“ |
| 0 | `ready` | nur `status`: Karte bereit | normaler Betrieb |
| 1 | `busy` | nicht möglich (CT-Queue nicht leer, Prozess auf der GPU, zweiter Lauf parallel). Karte unverändert; CT 462 kann bei `--stop-ct` schon gestoppt sein, siehe `ct.status` | Grund aus `message`, Kai entscheidet |
| 1 | `absent` | nur `on`: Dock nicht am Thunderbolt-Bus | „4070 eingeschaltet und Kabel dran?“ |
| 2 | `hung` | Hotplug hängt, **K12-Reboot nötig** | deutlich warnen, **nicht ausschalten**, nichts automatisch tun |
| 3 | – | Aufruffehler (kein JSON) | Bug im AI-Hub |
| 4 | `error`/`degraded` | sonstiger Fehler | `message` anzeigen |

**JSON (eine Zeile auf stdout, Diagnose auf stderr):**

```json
{"action": "off", "result": "ok", "exit_code": 0,
 "message": "4070 abgemeldet, kann jetzt ausgeschaltet werden",
 "gpu": {"pci": "0000:0a:00.0", "present": false, "driver": null,
         "nvidia_registered": false, "nvidia_ok": false,
         "device_node": null, "users": []},
 "thunderbolt_authorized": true, "hotplug_hung": false,
 "ct": {"id": 462, "status": "stopped"},
 "safe_to_power_off": true}
```

Maßgeblich für die Freigabe „jetzt ausschalten“ ist **nur** `safe_to_power_off == true`.
`users` sind Host-PIDs als `pid:comm`.

## Funktionale Anforderungen

**F1 Ausschalten.**
1. Backend `rtx-4070` drainen: keine neuen Jobs, laufende fertig werden lassen. Dazu gehört
   auch die eigene Hub-Queue (`jobs.db` queued/running für dieses Backend), nicht nur die
   ComfyUI-Queue.
2. Danach `egpu-4070 off --stop-ct --json` auf K12.
3. Bei `safe_to_power_off`: deutliche Meldung „4070 kann jetzt ausgeschaltet werden“.
   Backend bleibt deaktiviert.
4. Bei `busy`: Grund anzeigen, Backend-Freigabe auf den Stand vor Schritt 1 zurück.
5. Bei `hung`: Warnung „nicht ausschalten, K12-Reboot nötig“. Backend bleibt deaktiviert.

**F2 Wieder einschalten.** Kai schaltet die Karte ein und löst „4070 ist wieder an“ aus.
Der AI-Hub ruft `egpu-4070 on --start-ct --json`, wartet bei `ok` auf die normale
ComfyUI-Health von CT 462 und gibt das Backend erst dann frei.

**F3 Status.** Der Zustand (`status --json`) ist dort sichtbar, wo die Funktion ausgelöst
wird. Nicht in einem engen Takt pollen (der Aufruf startet `nvidia-smi`), auf Anforderung oder
selten reicht. K12 ist ein On-Demand-Host: Ist er per SSH nicht erreichbar, ist das „K12 aus“,
kein Fehler.

**F4 Kein Fault-Rauschen.** Ein CT 462, den dieser Ablauf gestoppt hat, und eine bewusst
ausgeschaltete 4070 sind kein Ausfall des Backends.

**F5 Nur diese Funktion.** Der Zugang zu K12 darf nichts anderes können als dieses Skript
(siehe unten). Kein Admin-Text im SSH-Kommando, nur die festen Aufrufe aus der Tabelle.

## Zugang AI-Hub → K12

Heute hat CT 110 keinen Schlüssel auf K12. Vorschlag:

1. Der AI-Hub erzeugt einen eigenen Schlüssel nur für diesen Zweck und gibt den Public Key an
   quartermaster bzw. Kai.
2. quartermaster trägt ihn auf K12 in `/root/.ssh/authorized_keys` **mit Zwangskommando** ein:

   ```
   command="/usr/local/sbin/egpu-4070 --ssh-forced",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-pty ssh-ed25519 AAAA... ai-hub-egpu
   ```

3. Der AI-Hub ruft dann `ssh -i <key> root@<K12> egpu-4070 off --stop-ct --json` usw. auf. Das
   Skript nimmt im `--ssh-forced`-Modus nur `status|off|on` und `--json|--stop-ct|--start-ct`
   an, alles andere endet mit Exit 3 (geprüft: `status; reboot` wird abgelehnt).

## Grenzen und offene Punkte

- `status` und die Abbruchpfade (`hung`, Aufrufprüfung) sind auf K12 getestet. **Der saubere
  `off`/`on`-Durchlauf ist noch nicht getestet**, weil K12 zum Zeitpunkt der Erstellung schon im
  Hängezustand war. Erster echter Test nach dem nächsten K12-Reboot (quartermaster).
- Physisches Schalten bleibt manuell. Falls die 4070 später an einer schaltbaren Steckdose
  hängt, kann F1/F2 das ergänzen; das ist nicht Teil dieser Anforderung.
- Das Backend `rtx-4070` ist seit 2026-10-27 wegen GPU-Fehlern offline (quartermaster kb/01).
  Die Funktion hier ändert daran nichts, sie verhindert nur den Hotplug-Hänger.
