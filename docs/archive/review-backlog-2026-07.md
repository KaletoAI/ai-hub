# Review-Backlog (xhigh-Review vom 2026-07-15, Bereich `0d80ee3..HEAD`)

Rest-Findings des Multi-Agent-Reviews: 56 Kandidaten wurden verifiziert, die 15
schwersten Root-Causes sind bereits gefixt (Commit `c4b73ad`), 2 wurden widerlegt.
Übrig bleiben die 15 Einträge unten (3 Symlink-Duplikate zusammengefasst) —
überwiegend Robustheit/Effizienz/Duplikation, nichts davon blockiert den Betrieb.

## Arbeitsregeln

- **Zeilennummern sind Stand des Review-Baums und damit STALE.** Seit dem Review
  wurden `_run_chain` komplett umgebaut und die Artefakt-Delivery restrukturiert
  (Commits `c4b73ad`, `ed19146`, `b6ff280`). Jedes Finding ZUERST gegen HEAD
  verifizieren; erledigte/hinfällige direkt hier in der Datei als `~~erledigt~~`
  mit Ein-Zeilen-Begründung markieren statt sie zu fixen.
- Konventionen aus CLAUDE.md gelten: kein Test-Framework — verifizieren per
  `venv/bin/python -m py_compile *.py` + App-Import/Route-Check (und für reine
  Funktionen gern ein Wegwerf-Skript unter /tmp); `jobs`/`store`/`stats`/
  `reasoning`/`responses_bridge` bleiben dependency-frei; `adapters` ist die
  Protokoll-Naht (kein ComfyUI-HTTP in `main.py` NEU einführen).
- **Nicht deployen, nicht die laufende Instanz anfassen** — nur committen; der
  User deployt selbst.
- Reihenfolge: Abschnitt A vor B vor C.

## A — Robustheit (können Jobs/Features real brechen) — ✓ ERLEDIGT

1. ~~**adapters.py:679 — GLB-Textur-Erkennung kennt nur PNG + Baseline-JPEG (erste 4 KB).**~~
   ✓ `_glb_info` zählt jetzt `embedded_images` separat von lesbaren Dims (Präsenz ≠
   Dims); `validate_delivery` prüft Präsenz darüber, Scan-Fenster 4 KB→64 KB,
   `_image_dims` kann zusätzlich WebP (VP8/VP8L/VP8X). Unbekannt (KTX2) → present-but-unknown,
   kein „no embedded texture" mehr.

2. ~~**adapters.py:1488 — 2×2-Dummy-Check ohne Opt-out im Flat-Modus.**~~
   ✓ Per-Alias-Opt-out `dummy_check` (NormalizedRequest-Feld, default on; Output-Checkbox
   „texture check"); flat-Mode `_check_glb_not_dummy` nur noch wenn `req.dummy_check`.

3. ~~**admin.py:2762/2772/2773 — abspath/realpath-Mismatch in `static_asset`.**~~
   ✓ `_STATIC_DIR = os.path.realpath(...)` — beide Seiten kanonisch.

4. ~~**previewanim.py:86 — `add_idle` nimmt Buffer 0 = BIN-Chunk an.**~~
   ✓ Guard: `_parse` liefert `had_bin`; `add_idle` gibt unverändert zurück, wenn kein
   BIN-Chunk ODER buffer[0] eine `uri` trägt.

5. ~~**previewanim.py:90 — Bone-Lookup verfehlt kolonlose Mixamo-Namen.**~~
   ✓ Neuer `_bone_key` strippt `mixamorig`-Präfix in allen Schreibweisen (`:`/`_`/keins).

6. ~~**adapters.py:1670 — nichtdeterministische Delivery-Reihenfolge bei Glob-Siblings.**~~
   ✓ Set-Literal → geordnete Liste: reported file zuerst, Siblings nach `sorted(glob_exts)`.

7. ~~**admin.py:2788 — `?anim=idle` blockiert den Event-Loop.**~~
   ✓ `jobs.result_path` + File-Read + `add_idle` laufen über `asyncio.to_thread`.

## B — Effizienz — ✓ ERLEDIGT

8. ~~**adapters.py:1533 — `/queue`-Poll auf jedem Tick.**~~
   ✓ `/queue`-Probe gedrosselt auf ~3× pro Grace-Fenster (`queue_every = max(poll_interval,
   grace/3)`); ein übersprungener Tick lässt `still = None` (unknown) → Gone-Timer trägt
   unverändert weiter.

9. ~~**main.py:1961 — path-Relay lädt das Mesh komplett, braucht aber nur Existenz.**~~
   ✓ `need_bytes = relay == "upload"`; path-Modus prüft Existenz per 1-Byte-Range-GET
   (200/206), Bytes nur beim upload-Relay geholt.

## C — Duplikation / Altitude (Cleanup) — ✓ ERLEDIGT

10. ~~**adapters.py:1298 — `upload_input` dupliziert `_upload_image`**~~
    ✓ Beide POSTs delegieren an einen gemeinsamen `_post_upload` (roher `/upload/image`-POST);
    Success/Subfolder + Fehlerpolitik bleiben je Aufrufer.
11. ~~**adapters.py:691 — 2×2-Dummy-Prädikat doppelt**~~
    ✓ Gemeinsames `_is_dummy(dims)`; Single-Use-Wrapper `_glb_texture_dims` entfernt
    (`_check_glb_not_dummy` nutzt `_glb_info` direkt).
12. ~~**previewanim.py:34 — GLB-Chunk-Walk re-implementiert**~~
    ✓ Gemeinsamer Leaf `adapters.glb_chunks` (pure, kein `main`); `_glb_info` und
    `previewanim._parse` (lazy import) nutzen ihn.
13. ~~**main.py:1931 — ComfyUI-Protokoll leckt in `main._run_chain`**~~
    ✓ Als ComfyUIAdapter-Methoden gekapselt (`export_pin`, `pinned_output_name`,
    `fetch_output`, Geschwister von `upload_input`); `main` ruft nur noch adapter-agnostisch.
14. ~~**main.py:1928 — `_gen_inputs_params`/`_apply_seconds` doppelt berechnet**~~
    ✓ Endpoint berechnet einmal, reicht `inputs`/`params` in `_run_chain` durch (fps/mapping
    sind alias-weit → identisch); per-Attempt `_apply_seconds` bleibt als Failover-No-op.
15. ~~**main.py:1734 — `view['workflow']` dup; admin.py — Download-Karten**~~
    ✓ Redundantes `view['workflow']` entfernt (Consumer nutzen `alias`); vier Download-Karten
    → ein `_dl_card()`-Helper (`_media_tag` file + `_job_thumbs` GLB/FBX/other).

## Abschluss

Alle 15 Findings verifiziert (keiner hinfällig) und gefixt. `py_compile` + App-Import +
Helfer-Smoke-Tests grün. Drei Commits (A/B/C). Nicht deployt — der User deployt selbst.
