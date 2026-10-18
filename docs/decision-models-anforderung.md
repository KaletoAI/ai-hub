# Anforderung: Entscheidungsmodelle (System One) über den AI-Hub

**Für:** Claude Code im `ai-hub`-Repo (Design und Umsetzung dort)
**Von:** anima-verse-Session, 2026-10-19
**Status:** Anforderung. Wie der AI-Hub das umsetzt (Routen, Konsole, Speicherung), entscheidet das ai-hub-Projekt.

## Ziel

Anima-Versa soll das Entscheidungsmodell **`typesafe/jev-1.14`** von OpenRouter über den AI-Hub
nutzen können, mit dem normalen AI-Hub-Benutzer-Key. Anima-Versa soll keinen eigenen
OpenRouter-Key halten. Kosten, Statistik und Allow-List laufen wie bei den übrigen Modellen über den
AI-Hub.

Ein Entscheidungsmodell erzeugt keinen Text. Es bekommt einen Zustand und typisierte Fragen
(`choice` / `noul` / `score`) und liefert pro Frage eine Wahrscheinlichkeitsverteilung über die
vorgegebenen Antworten. Es ist also kein Chat-Modell, und `/v1/chat/completions` passt dafür nicht.

## Ist-Stand

- Der AI-Hub listet `openrouter/typesafe/jev-router` unter `/v1/models`. Das ist ein Chat-Router und
  KEIN Entscheidungsmodell. Die Entscheidungsmodelle von OpenRouter fehlen in der Liste.
- `POST /v1/systemone`, `/v1/decisions` und `/api/alpha/decisions` am AI-Hub antworten mit 404.
- Die lokalen Aliase `laya` / `openjev-4b` stehen in `/v1/models`, sind über den AI-Hub aber ebenfalls
  nicht im System-One-Format erreichbar. Anima-Versa spricht sie heute direkt über llama-swap an
  (`http://<host>:8080/upstream/<modell>`).

## Schnittstelle, die Anima-Versa spricht (fest)

Anima-Versa hängt an eine konfigurierbare Basis-URL immer `/v1/systemone` an. Die Basis-URL wählt der
AI-Hub, Anima-Versa trägt sie nur ein.

```http
POST <basis-url>/v1/systemone
Authorization: Bearer <AI-Hub-Key>
Content-Type: application/json
```

```json
{
  "model": "<frei konfigurierbarer Modellname, unverändert durchgereicht>",
  "state": {"text": "smiles and waves at a friend"},
  "questions": {
    "entry": {
      "type": "choice",
      "instructions": "Which facial expression fits the text best?",
      "criteria": {
        "neutral": "neutral: no particular emotion",
        "positive": "positive: happy, friendly, warm",
        "sad": "sad: unhappy, downcast",
        "none": "none of these fits the text"
      }
    }
  }
}
```

- `state` ist ein Objekt mit Textfeldern. Je nach Entscheidungsstelle sind das ein kurzer Text oder bis
  etwa 3.000 Zeichen Kontext.
- `questions` enthält eine oder mehrere Fragen. Heute schickt Anima-Versa nur `choice` mit 2 bis 255
  Optionen. `noul` und `score` sind im Format vorgesehen.

Erwartete Antwort (`200`):

```json
{
  "answers": {
    "entry": {
      "choice": "positive",
      "probabilities": {"neutral": 0.03, "positive": 0.95, "sad": 0.0, "none": 0.02}
    }
  }
}
```

Was Anima-Versa daran prüft, sonst verwirft es die Antwort:
- `choice` ist einer der gesendeten Schlüssel.
- `probabilities` enthält **genau** die gesendeten Schlüssel, jeder Wert liegt in [0, 1], die Summe
  ist 1 ± 0,02.
- Bei `noul` muss `noul` die Wahrscheinlichkeit für „ja" sein (0..1). Bei `score` gilt
  `score` + `probabilities` mit den Schlüsseln `"0"` … `"n-1"`.
- Weitere Felder (`confidence`, `model`, `usage`, …) sind erlaubt. Die Konfidenz berechnet Anima-Versa
  selbst aus `probabilities`.

Normale Zustände: Jeder Nicht-200-Status, ein Timeout oder unpassendes JSON führt in Anima-Versa zum
bisherigen Weg ohne Modell. Das ist kein Fehlerfall für den Spielablauf. Anima-Versa wiederholt nicht.
Nach drei Fehlern in Folge sperrt es den Endpunkt 60 s. Eine klare 4xx-/5xx-Antwort ist besser als ein
langes Hängen.

## Erwartungen

**E1 Modell.** `typesafe/jev-1.14` ist über die obige Schnittstelle erreichbar. Der Modellname bleibt
**auf diese Version festgenagelt**, nicht `~typesafe/jev-latest`: Die Schwellen in Anima-Versa gelten
für genau diesen Build. Wie der Name in Anima-Versa lautet (z. B. `typesafe/jev-1.14` oder ein
AI-Hub-Alias), entscheidet der AI-Hub. Anima-Versa trägt ihn ein.

**E2 Upstream.** OpenRouter bietet zwei Wege an. Welchen der AI-Hub nutzt, entscheidet er:
- Decisions-API `POST https://openrouter.ai/api/alpha/decisions` mit der vollen ID
  (`typesafe/jev-1.14`). Die Antwort trägt pro Frage zusätzlich `type`. Am 2026-10-18 hat die
  anima-verse-Session diesen Weg mit 1.284 Anfragen geprüft, ohne Fehler.
- System-One-Pfad `POST https://openrouter.ai/api/v1/systemone` mit der Modell-ID ohne
  Anbieter-Präfix. Laut OpenRouter-Doku gilt das für Modelle mit TypeSafe-System-One-Vertrag.
  **Nicht selbst geprüft.**

**E3 Latenz.** Anima-Versa wartet je nach Stelle 2 bis 6 s. Direkt bei OpenRouter lag die Latenz
bei p50 ≈ 300 ms und p90 ≈ 350 ms. Der Aufschlag durch den AI-Hub sollte klein bleiben. Kein
Streaming, keine Warteschlange hinter langen Chat-Jobs.

**E4 Inhalt.** Der Zustand kann Inhalte für Erwachsene enthalten (Spielwelt, vom Nutzer
konfiguriert). Er muss unverändert durchgereicht werden, ohne Moderation oder Umschreiben. Protokolliert
wird wie bei Chat-Aufrufen, nicht mehr.

**E5 Kosten und Statistik.** Aufrufe erscheinen in der AI-Hub-Statistik mit Kosten (OpenRouter liefert
`usage.cost` in USD). Gemessen: etwa 0,00004 USD pro Aufruf, Volumen einige hundert Aufrufe pro Tag.

**E6 Freigabe.** Das Modell ist für den Anima-Versa-Benutzer-Key freigeschaltet (Allow-List). Ein
Aufruf mit dem obigen Beispiel liefert eine gültige Antwort.

## Wünschenswert, kein Muss

- Dieselbe Schnittstelle für die lokalen Entscheidungsmodelle (`laya`, `openjev-4b` auf den
  llama-swap-Hosts), damit Anima-Versa alle Entscheidungsmodelle über einen Key und eine Statistik
  bezieht.
- Weitere OpenRouter-Entscheidungsmodelle nach demselben Muster. Der Katalog steht unter
  `GET https://openrouter.ai/api/v1/models?output_modalities=decisions`. Ein Kandidat ist
  `perplexity/pplx-decider-v1.2`: In derselben Messung war er fast so gut, er ist Apache 2.0 und es gibt
  ihn auch als 4B-Variante `pplx-decider-flash-v1.2` (≈ 9 GiB BF16, CUDA). Er ließe sich also später
  lokal betreiben.

## Referenzwissen (kein Vertrag)

Anima-Versa-Messung vom 2026-10-18 mit 642 echten Fällen aus dem Schattenbetrieb, direkt gegen
OpenRouter. Die Trefferquoten sind handbewertet:

| Stelle | jev-1.14 | pplx-decider-v1.2 | bisheriger Weg (Embedding) | openjev-4b / Laya (lokal) |
|---|---|---|---|---|
| Pose-Text → Körperhaltung | 98 % | 90 % | 73 % | ~42 % |
| Stimmungstext → Gesichtsausdruck | 91 % | 93 % | 77 % | 79 % / 52 % |

Abnahme auf Anima-Versa-Seite: In `/admin/settings → Decision models` einen Endpunkt mit der
AI-Hub-Basis-URL, dem Modellnamen aus E1 und dem AI-Hub-Key eintragen. Der dortige **Test**-Knopf
schickt eine Probe-Frage (30-s-Timeout) und muss eine gültige Antwort melden.
