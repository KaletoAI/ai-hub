# Chat Completions → Messages: Claude hinter `/v1/chat/completions`

Status: **entworfen, nicht gebaut.** Spike am 2026-09-18 ergab, dass der Anlassfall
(ein OpenAI-sprechendes Bench-Tool soll Sonnet messen) ohne Codeänderung lösbar ist.
Dieses Dokument hält das Design fest, falls der Bedarf mit einem echten
Anthropic-API-Key wiederkommt.

## Anlass und warum nicht gebaut

`anthropic_bridge.py` übersetzt heute nur eine Richtung: Messages → Chat, damit ein
`type: openai`-Backend `/v1/messages` bedienen kann. Die Gegenrichtung fehlt, und
`main.serves_path()` sperrt ein `type: anthropic`-Backend bewusst auf `/v1/messages`
— eine Lizenzgrenze, kein technisches Hindernis.

Gemessen am 2026-09-18 gegen prod (.10), jede Zeile ein echter Call:

- Das Abo-Backend ist für Benchmarks unbrauchbar: identischer Request **mit**
  „You are Claude Code, Anthropic's official CLI for Claude." im `system` → 200,
  **ohne** ihn → `429 rate_limit_error`, ohne `retry-after`. Eine Übersetzung, die
  den Marker selbst einsetzte, verfälschte die Messung und umginge genau die Regel,
  die `serves_path()` schützt.
- Der bestehende Weg deckt alles ab: `openrouter/anthropic/claude-sonnet-5` an
  `/v1/chat/completions` liefert `usage.prompt_tokens/completion_tokens`,
  `finish_reason: "length"` bei Abschnitt, Streaming mit finalem Usage-Chunk und
  `[DONE]` bei `stream_options.include_usage`, `delta.reasoning` über
  `reasoning_effort`, akzeptiert `temperature: 0` und ignoriert `chat_template_kwargs`.
- `"provider": {"only": ["anthropic"]}` wird vom OpenAI-Adapter durchgereicht und
  erzwingt Anthropic statt des Default-Providers „Claude Platform on AWS" — nötig,
  sobald jemand Geschwindigkeit misst.

Das Feature lohnt sich daher erst mit einem **echten Anthropic-API-Key** als Backend:
dann ohne OpenRouter-Hop und mit Anthropics eigenen Latenzen.

## Design

**Ansatz: Spiegelung im `AnthropicAdapter`.** `OpenAIAdapter._dispatch_messages()`
(`adapters.py:941`) macht heute exakt die Gegenrichtung — übersetzen, über den
normalen Pfad dispatchen, zurückübersetzen. `AnthropicAdapter._dispatch_chat()` wird
sein Zwilling; der eigentliche Call geht weiterhin verbatim über `/v1/messages`,
sodass In-Flight-Zählung, Stats, Failover und `retry-after` unverändert auf dem
bestehenden Weg liegen. Die reinen Funktionen kommen nach `anthropic_bridge.py`
(das Modul heißt bereits „Messages ↔ Chat"), nicht in ein zweites Modul: die
Mapping-Tabelle `stop_reason ↔ finish_reason` wird sonst dupliziert.

### Lizenzgrenze

Neues Backend-Feld `serve_chat_api` (Checkbox im `#anthopts`-Pane, default aus).
`main.serves_path()` gibt für ein Anthropic-Backend dann **nur** `/v1/chat/completions`
frei — nicht `/v1/completions`, `/v1/embeddings`, `/v1/audio`. Die Regel bleibt
dokumentiert und wird pro Backend zur bewussten Entscheidung.
`main.anthropic_only_candidates()` muss das Flag mitlesen, sonst erklärt der 404
weiterhin „reachable through POST /v1/messages only", obwohl der Pfad offen ist.
`tests/test_backend_form_tabs.py` leitet die Feldliste per AST aus `backend_save` ab —
das neue Feld muss also in beiden stehen.

### Request `chat_to_messages_body()`

Whitelist, keine Blacklist: unbekannte Felder (`chat_template_kwargs`,
`presence_penalty`, `logit_bias`, `n`) fallen still heraus statt ein 400 auszulösen.

- `system`-Rollen → `system`, Rest als `messages`.
- **`max_tokens` ist Pflicht** und fehlt bei Chat-Clients oft → Fallback aus dem
  Backend-Feld `max_tokens_default` (Vorschlag 4096). Die einzige Stelle, an der die
  Übersetzung etwas erfinden muss.
- `stop` → `stop_sequences`.
- **Sampling wird NICHT übersetzt.** Gemessen: `claude-sonnet-5` antwortet auf
  `temperature` mit `400 … "temperature is deprecated for this model"`, ganz ohne
  Thinking; mit Thinking AN verlangt Anthropic zusätzlich `temperature == 1` und
  `top_p >= 0.95`. Ein durchgereichtes Sampling erzeugt je nach Modell und
  Thinking-Zustand drei verschiedene 400er. Der Verzicht gehört in einen
  Response-Header (`x-gateway-dropped: temperature`), damit er nicht still ist.
- **`reasoning: off|on`** liegt als `req.reasoning` an (`main._normalize_reasoning()`,
  `main.py:1550`): `on` → `thinking: {type: enabled, budget_tokens: …}`, sonst kein
  `thinking`. Budget geklemmt auf `1024 <= budget < max_tokens`. Das ist die einzige
  Stelle, an der dieser Pfad etwas anwendet statt verbatim zu senden — vertretbar,
  weil ein Chat-Client gar kein `thinking` senden kann, das zu respektieren wäre.
- **`tools` im Request → 400**, nicht stilles Verwerfen: ein Client, der Werkzeuge
  schickt und keine bekommt, erhält eine schweigend falsche Antwort. Dieselbe Policy,
  die das Modul in der Gegenrichtung für Dokumente/PDFs fährt.

### Antwort `messages_to_chat_response()`

`text`-Blöcke → `message.content`, `thinking` → `message.reasoning_content`.
`stop_reason` → `finish_reason`: `max_tokens`→`length`, `end_turn`/`stop_sequence`→
`stop`, `tool_use`→`tool_calls`, `refusal`→`content_filter`.
`usage` → `prompt_tokens` (inkl. `cache_read_input_tokens` und
`cache_creation_input_tokens`, so wie `AnthropicAdapter._usage_of` zählt) /
`completion_tokens` / `total_tokens`, dazu `prompt_tokens_details.cached_tokens`.

### Stream `messages_to_chat_stream()`

`text_delta` → `delta.content`, `thinking_delta` → `delta.reasoning_content`,
`message_delta` → `finish_reason`, `message_stop` → `[DONE]`. Der abschließende
Usage-Chunk nur bei `stream_options.include_usage` — dieselbe Regel, die
`_StreamNormalizer` für alle anderen Backends fährt. Ein `error`-Event beendet den
Stream als Fehler, nicht als sauberes `[DONE]`: ein abgeschnittener Lauf darf nicht
wie ein fertiger aussehen. Den Quell-Iterator im `finally` schließen (`aclose`), wie
im Gegenstück — sonst hält ein Client-Abbruch den In-Flight-Slot.

### Fehler

429 und `retry-after` laufen bereits durch (`adapters._ratelimit_headers`,
`adapters.py:808`, gepinnt von `tests/test_ratelimit_headers.py`). Zusätzlich wird
der Anthropic-Fehlerkörper in OpenAI-Form umgeschrieben (`{"error": {"message", …}}`)
— Spiegel von `adapters._anthropic_error()` —, damit ein Client nur ein Fehlerformat
kennen muss.

### Nicht gebaut

Tools, Bilder, `n > 1`, logprobs, andere Endpunkte.

## Tests (`tests/test_chat_to_anthropic.py`)

Alles hier fällt still falsch aus, nicht laut — deshalb gepinnt:
`finish_reason`-Mapping (ein verpasstes `length` lässt einen abgeschnittenen Lauf wie
einen vollständigen aussehen), Usage-Mapping inkl. Cache-Anteile, der
`max_tokens`-Fallback, dass Sampling nicht durchgereicht wird, `reasoning on|off` →
`thinking` mit Budget unter `max_tokens`, `tools` → 400, und der Stream: Text- und
`reasoning_content`-Deltas, Usage-Chunk nur bei `include_usage`, Abbruch bei
`error`-Event. Dazu `serves_path()` mit und ohne Flag.
