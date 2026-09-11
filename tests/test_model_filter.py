"""Per-backend model allow/deny globs — the filter that must never work silently.

Why this fails SILENTLY: a whitelist is a typo away from matching nothing, and a
backend whose discovered model set has been narrowed to zero stays UP, keeps
answering /v1/models with an empty list and reports no error anywhere. Every
symptom then points somewhere else — an alias with no candidates reads as "all
backends busy", a missing model id as a backend that never listed it. So both
halves are pinned here: the pure rule (allow narrows, deny wins over allow, empty
means unfiltered, matching is case-sensitive) and the fact that refresh_backend
applies it to the SET the rest of the gateway reads while recording kept/total for
the console to show.
"""
import asyncio
import os
import sys
import tempfile
import unittest

_here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_prev = os.getcwd()
_tmp = tempfile.TemporaryDirectory()
with open(os.path.join(_tmp.name, "config.yaml"), "w") as _f:
    _f.write('api_key: ""\nbackends: []\n')
os.chdir(_tmp.name)
sys.path.insert(0, _here)
try:
    import main
    import adapters
finally:
    os.chdir(_prev)
    _tmp.cleanup()
    del _tmp

import logging

logging.getLogger("store").setLevel(logging.WARNING)


# ── parsing: a stray comma must not become a pattern ──────────────────────────

class ParseModelFilter(unittest.TestCase):
    def test_comma_separated_string(self):
        self.assertEqual(adapters.parse_model_filter("gpt-*, claude-*"), ["gpt-*", "claude-*"])

    def test_whitespace_and_empty_entries_dropped(self):
        self.assertEqual(adapters.parse_model_filter("  a ,, , b  ,"), ["a", "b"])

    def test_already_a_list(self):
        self.assertEqual(adapters.parse_model_filter([" a ", "", "b"]), ["a", "b"])

    def test_none_and_empty_string_are_no_filter(self):
        self.assertEqual(adapters.parse_model_filter(None), [])
        self.assertEqual(adapters.parse_model_filter(""), [])
        self.assertEqual(adapters.parse_model_filter("   "), [])
        self.assertEqual(adapters.parse_model_filter([]), [])


# ── the rule: allow narrows, deny wins, empty = unchanged ─────────────────────

MODELS = {"gpt-4", "gpt-4-embed", "gpt-5", "claude-opus", "llama-70b", "GPT-4"}


class FilterModels(unittest.TestCase):
    def test_no_filter_returns_the_set_unchanged(self):
        self.assertEqual(adapters.filter_models(MODELS, {"name": "b"}), MODELS)
        self.assertEqual(adapters.filter_models(MODELS, {"models_allow": "", "models_deny": ""}), MODELS)

    def test_allow_only(self):
        out = adapters.filter_models(MODELS, {"models_allow": "gpt-*"})
        self.assertEqual(out, {"gpt-4", "gpt-4-embed", "gpt-5"})

    def test_deny_only(self):
        out = adapters.filter_models(MODELS, {"models_deny": "gpt-*, GPT-*"})
        self.assertEqual(out, {"claude-opus", "llama-70b"})

    def test_deny_wins_over_allow(self):
        # The reason for the ordering: "all of gpt-* except the embedders" in two lines.
        out = adapters.filter_models(MODELS, {"models_allow": "gpt-*", "models_deny": "gpt-*-embed"})
        self.assertEqual(out, {"gpt-4", "gpt-5"})

    def test_exact_ids_work_like_patterns(self):
        out = adapters.filter_models(MODELS, {"models_allow": "gpt-4, claude-opus"})
        self.assertEqual(out, {"gpt-4", "claude-opus"})

    def test_case_sensitive(self):
        # fnmatchcase, as in model_context_for — "gpt-*" never reaches "GPT-4".
        self.assertEqual(adapters.filter_models(MODELS, {"models_allow": "gpt-*"}),
                         {"gpt-4", "gpt-4-embed", "gpt-5"})
        self.assertEqual(adapters.filter_models(MODELS, {"models_allow": "GPT-*"}), {"GPT-4"})

    def test_allow_matching_nothing_yields_the_empty_set(self):
        # NOT a silent fall back to unfiltered: an allow-list that matches nothing means
        # nothing is served, and the console's kept/total is what makes that visible.
        self.assertEqual(adapters.filter_models(MODELS, {"models_allow": "gtp-*"}), set())

    def test_a_list_from_yaml_is_accepted(self):
        out = adapters.filter_models(MODELS, {"models_allow": ["gpt-4", "gpt-5"]})
        self.assertEqual(out, {"gpt-4", "gpt-5"})

    def test_empty_input_set(self):
        self.assertEqual(adapters.filter_models(set(), {"models_allow": "gpt-*"}), set())


# ── discovery: the filtered set is the one the gateway stores ─────────────────

class _StubAdapter:
    """Answers discovery with a fixed model set — no network, no protocol."""

    def __init__(self, models):
        self.models = set(models)

    async def discover(self, client):
        return adapters.Capabilities(models=set(self.models), pricing={})


class RefreshBackendApplies(unittest.TestCase):
    def setUp(self):
        self._saved = {k: getattr(main, k) for k in
                       ("backends", "backend_adapters", "backend_models", "backend_pricing",
                        "backend_loras", "backend_context", "backend_healthy", "backend_error",
                        "backend_model_counts")}
        main.backend_models, main.backend_pricing, main.backend_loras = {}, {}, {}
        main.backend_context, main.backend_healthy, main.backend_error = {}, {}, {}
        main.backend_model_counts = {}

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(main, k, v)
        main.rebuild_route_index()

    def _run(self, backend, models):
        bid = main.backend_id(backend)
        main.backends = [backend]
        main.backend_adapters = {bid: _StubAdapter(models)}
        main.rebuild_route_index()
        asyncio.run(main.refresh_backend(backend, None))
        return bid

    def test_filtered_set_lands_in_backend_models_and_counts(self):
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True,
             "models_allow": "gpt-*", "models_deny": "gpt-*-embed"}
        bid = self._run(b, {"gpt-4", "gpt-4-embed", "claude-opus"})
        self.assertEqual(main.backend_models[bid], {"gpt-4"})
        self.assertEqual(main.backend_model_counts[bid], (1, 3))
        self.assertTrue(main.backend_healthy[bid])          # healthy AND narrowed

    def test_typo_whitelist_is_visible_not_silent(self):
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True,
             "models_allow": "gtp-*"}
        bid = self._run(b, {"gpt-4", "gpt-5"})
        self.assertEqual(main.backend_models[bid], set())
        self.assertEqual(main.backend_model_counts[bid], (0, 2))
        self.assertEqual(main._model_filter_info(b), {"models_filtered": {"kept": 0, "total": 2}})

    def test_no_filter_reports_no_key(self):
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True}
        bid = self._run(b, {"gpt-4", "gpt-5"})
        self.assertEqual(main.backend_models[bid], {"gpt-4", "gpt-5"})
        self.assertEqual(main._model_filter_info(b), {})    # absent = no filter configured

    def test_summary_carries_the_globs_for_the_editor(self):
        # The backend editor pre-fills from the store, and for a CONFIG-defined backend
        # from this summary instead. A field the summary omits renders blank, and the
        # next Save writes that blank back — the filter would disappear from a form the
        # operator opened to change something else. A YAML list must arrive as the comma
        # string the text input round-trips.
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True,
             "models_allow": ["gpt-*", " claude-* "], "models_deny": "*-embed"}
        self._run(b, {"gpt-4", "gpt-4-embed"})
        row = next(r for r in main.gateway_info()["backends"] if r["name"] == "dx")
        self.assertEqual(row["models_allow"], "gpt-*, claude-*")
        self.assertEqual(row["models_deny"], "*-embed")

    def test_never_polled_backend_reports_nothing(self):
        # A backend that is simply UNREACHABLE has no measurement, and deriving one
        # from its empty model set reads as "kept 0 of 0" — which the console renders
        # as "filter matches nothing", blaming the whitelist for a dead host. Measured
        # 2026-09-09 on a fresh instance: a down backend with a filter showed exactly
        # that badge beside "⇥ unreachable". Absent key = say nothing.
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True,
             "models_allow": "gpt-*"}
        self.assertEqual(main._model_filter_info(b), {})

    def test_counts_survive_the_backend_going_down(self):
        # Measured numbers stay reportable once the backend stops answering — they
        # were true when they were taken, and the down badge says the rest.
        b = {"name": "dx", "type": "openai", "url": "http://dx", "enabled": True,
             "models_allow": "gpt-*"}
        bid = self._run(b, {"gpt-4", "claude-opus"})
        main.backend_healthy[bid] = False
        self.assertEqual(main._model_filter_info(b), {"models_filtered": {"kept": 1, "total": 2}})

    def test_applies_to_non_openai_types_too(self):
        # The filter sits in refresh_backend, not in extract_models — so a comfyui
        # checkpoint list is narrowed by exactly the same two knobs.
        b = {"name": "gpu", "type": "comfyui", "url": "http://gpu", "enabled": True,
             "models_deny": "*.safetensors"}
        bid = self._run(b, {"flux.safetensors", "sd15.ckpt"})
        self.assertEqual(main.backend_models[bid], {"sd15.ckpt"})
        self.assertEqual(main.backend_model_counts[bid], (1, 2))


class AddModelExtras(unittest.TestCase):
    """The pure rule for `models_extra` — the knob that ADDS."""

    def test_no_extras_returns_the_set_unchanged(self):
        ms = {"a", "b"}
        self.assertIs(adapters.add_model_extras(ms, {}), ms)
        self.assertIs(adapters.add_model_extras(ms, {"models_extra": ""}), ms)
        self.assertIs(adapters.add_model_extras(ms, {"models_extra": " , "}), ms)

    def test_ids_are_added(self):
        self.assertEqual(adapters.add_model_extras({"a"}, {"models_extra": "b, c"}),
                         {"a", "b", "c"})

    def test_a_list_from_yaml_is_accepted(self):
        self.assertEqual(adapters.add_model_extras({"a"}, {"models_extra": ["b"]}), {"a", "b"})

    def test_an_id_discovery_already_found_changes_nothing(self):
        self.assertEqual(adapters.add_model_extras({"a", "b"}, {"models_extra": "b"}), {"a", "b"})

    def test_the_input_set_is_not_mutated(self):
        ms = {"a"}
        adapters.add_model_extras(ms, {"models_extra": "b"})
        self.assertEqual(ms, {"a"})


class RefreshBackendAddsExtras(RefreshBackendApplies):
    """`models_extra` through refresh_backend — where it has to be ROUTABLE.

    Why this needs pinning: routing checks `real in backend_models[bid]` in three
    places, so a model the backend serves but does not list is unreachable by every
    name — and no whitelist can bring it back, because a filter only subtracts. That
    is the whole point of the knob, and it is also how it breaks: added in the wrong
    ORDER it is filtered straight back out, and counted into `kept/total` it makes the
    console's filter badge report a number no poll measured.
    """

    def test_an_unlisted_model_becomes_routable(self):
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_extra": "whisper-v3-turbo"}
        bid = self._run(b, {"qwen3:4b"})
        self.assertEqual(main.backend_models[bid], {"qwen3:4b", "whisper-v3-turbo"})
        ready, _busy = main.resolve_routes("whisper-v3-turbo")
        self.assertEqual([(bk["name"], real) for bk, real in ready], [("npu", "whisper-v3-turbo")])

    def test_extras_survive_a_whitelist_that_excludes_them(self):
        # The order is the contract: filter first, add second. Reversed, the operator
        # would have to repeat every extra id in models_allow to keep it — two knobs
        # kept in sync for nothing, and silently losing the model when they drift.
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_allow": "qwen3*", "models_extra": "Whisper-V3-Turbo-NPU2"}
        bid = self._run(b, {"qwen3:4b", "gemma3:1b"})
        self.assertEqual(main.backend_models[bid], {"qwen3:4b", "Whisper-V3-Turbo-NPU2"})

    def test_extras_are_not_counted_into_the_filter_verdict(self):
        # kept/total describes what the FILTER did to what discovery MEASURED. Counting
        # an addition into it would report a total no poll ever saw, and could even hide
        # a whitelist matching nothing behind a non-zero `kept`.
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_allow": "qwen3*", "models_extra": "whisper"}
        bid = self._run(b, {"qwen3:4b", "gemma3:1b"})
        self.assertEqual(main.backend_model_counts[bid], (1, 2))
        self.assertEqual(main._model_filter_info(b),
                         {"models_filtered": {"kept": 1, "total": 2}, "models_added": ["whisper"]})

    def test_extras_alone_report_only_themselves(self):
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_extra": "whisper, embed"}
        self._run(b, {"qwen3:4b"})
        self.assertEqual(main._model_filter_info(b), {"models_added": ["whisper", "embed"]})

    def test_a_never_polled_backend_publishes_no_extras(self):
        # An unreachable backend must not offer a model: refresh_backend adds the ids
        # only on a successful poll, so the report has to follow the same rule or the
        # console would list models every call fails on.
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_extra": "whisper"}
        self.assertEqual(main._model_filter_info(b), {})
        self.assertEqual(main.backend_models.get(main.backend_id(b)), None)

    def test_summary_carries_the_extras_for_the_editor(self):
        # Same reason as the globs: the editor pre-fills a config-defined backend from
        # this summary, and a field it omits comes back blank on the next Save.
        b = {"name": "npu", "type": "openai", "url": "http://npu", "enabled": True,
             "models_extra": ["whisper", " embed "]}
        self._run(b, {"qwen3:4b"})
        row = next(r for r in main.gateway_info()["backends"] if r["name"] == "npu")
        self.assertEqual(row["models_extra"], "whisper, embed")      # config string, for the form
        self.assertEqual(row["models_added"], ["whisper", "embed"])  # verdict, for the badge

    def test_applies_to_non_openai_types_too(self):
        b = {"name": "gpu", "type": "comfyui", "url": "http://gpu", "enabled": True,
             "models_extra": "hidden.safetensors"}
        bid = self._run(b, {"sd15.ckpt"})
        self.assertEqual(main.backend_models[bid], {"sd15.ckpt", "hidden.safetensors"})


if __name__ == "__main__":
    unittest.main()
