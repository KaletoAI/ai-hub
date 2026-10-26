# docs/

| Ort | Inhalt | Pflege |
|---|---|---|
| `architecture.md` | Moduldesign, Routing-/Auth-Invarianten und Regressionstest-Referenz | lebend |
| `mesh-client-spec.md` | Client-Vertrag der Mesh-Aliase (Parameter, Lieferung, Rigging) | lebend |
| `host-coordination-plan.md` | GPU-Host-Koordination: Flags, VRAM-Free, Scheduler-Affinität | lebend |
| `tripo-api-v3-notes.md` | Tripo3D API V3 — technische Zusammenfassung | lebend |
| `anima-versa-integration.md` | AI-Hub als Image-Backend in anima-versa | lebend |
| `install-test-2026-09.md` | Protokoll der Frischinstallation auf einer leeren VM (2026-09-08, Debian 13) | Protokoll |
| `archive/` | erledigte Pläne/Reviews mit Index | eingefroren |
| `superpowers/specs/`, `superpowers/plans/` | Design-Specs und Umsetzungspläne je Feature (datiert) | eingefroren nach Umsetzung |

Die Quelle der Wahrheit für Betrieb und Konfiguration ist `../README.md`; für
Mechanismen und ihre Invarianten `architecture.md`. Der Abschnitt
[Client-OAuth](../README.md#keep-the-claude-code-subscription-login-client-oauth)
beschreibt Claude-Code-Subscription-Passthrough, getrennte Gateway-Authentifizierung
und die Grenzen der automatischen Modellauswahl. `../CLAUDE.md` enthält die
kompakte Architekturkarte und Repository-Regeln.
