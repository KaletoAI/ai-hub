"""Tests for the Anthropic Messages ↔ Chat Completions bridge.

stdlib only, no server needed:  venv/bin/python -m unittest tests.test_anthropic_bridge -v

The bridge is what a non-Anthropic backend (OpenRouter, LocalAI, …) sees when Claude
Code talks to the gateway, so these cases mirror real Claude Code traffic: system
blocks, tool-use round-trips, tool_result blocks, images, and the full SSE sequence.
"""
import asyncio
import json
import unittest

import anthropic_bridge as ab


class MessagesToChat(unittest.TestCase):
    """Request direction: Anthropic Messages body → Chat Completions body."""

    def test_system_string_becomes_leading_system_message(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "system": "be brief",
                                    "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(chat["messages"][0], {"role": "system", "content": "be brief"})

    def test_system_blocks_are_joined(self):
        """Claude Code sends system as a list of text blocks (with cache_control)."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [],
                                    "system": [{"type": "text", "text": "part one"},
                                               {"type": "text", "text": "part two",
                                                "cache_control": {"type": "ephemeral"}}]})
        self.assertEqual(chat["messages"][0]["content"], "part one\n\npart two")

    def test_plain_string_message_passes_through(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100,
                                    "messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(chat["messages"], [{"role": "user", "content": "hello"}])

    def test_text_blocks_are_flattened_to_a_string(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]}]})
        self.assertEqual(chat["messages"][0]["content"], "a\n\nb")

    def test_base64_image_becomes_a_data_uri_image_part(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": "QUJD"}}]}]})
        parts = chat["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "look"})
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD")

    def test_url_image_keeps_its_url(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]}]})
        self.assertEqual(chat["messages"][0]["content"][0]["image_url"]["url"], "https://x/y.png")

    def test_tool_use_block_becomes_a_tool_call(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "assistant", "content": [
                {"type": "text", "text": "let me look"},
                {"type": "tool_use", "id": "toolu_1", "name": "Read",
                 "input": {"file_path": "/tmp/x"}}]}]})
        msg = chat["messages"][0]
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "let me look")
        self.assertEqual(msg["tool_calls"][0]["id"], "toolu_1")
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "Read")
        self.assertEqual(json.loads(msg["tool_calls"][0]["function"]["arguments"]),
                         {"file_path": "/tmp/x"})

    def test_tool_result_becomes_its_own_tool_message(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "file contents"}]}]}]})
        self.assertEqual(chat["messages"], [{"role": "tool", "tool_call_id": "toolu_1",
                                             "content": "file contents"}])

    def test_tool_result_and_text_in_one_user_turn_keep_their_order(self):
        """Claude Code packs the tool result and the next instruction into one turn."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
                {"type": "text", "text": "now continue"}]}]})
        self.assertEqual([m["role"] for m in chat["messages"]], ["tool", "user"])
        self.assertEqual(chat["messages"][1]["content"], "now continue")

    def test_error_tool_result_is_marked_for_the_model(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                 "content": "file not found"}]}]})
        self.assertIn("file not found", chat["messages"][0]["content"])
        self.assertIn("rror", chat["messages"][0]["content"])   # flagged as an error

    def test_tools_are_translated_to_function_schemas(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [], "tools": [
            {"name": "Read", "description": "read a file",
             "input_schema": {"type": "object", "properties": {"p": {"type": "string"}}}}]})
        self.assertEqual(chat["tools"][0], {
            "type": "function",
            "function": {"name": "Read", "description": "read a file",
                         "parameters": {"type": "object", "properties": {"p": {"type": "string"}}}}})

    def test_server_side_tools_are_dropped(self):
        """A backend that only speaks OpenAI functions cannot run Anthropic's
        server-side tools — forwarding them verbatim would 400 the whole request."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [], "tools": [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "Read", "input_schema": {"type": "object"}}]})
        self.assertEqual([t["function"]["name"] for t in chat["tools"]], ["Read"])

    def test_tool_choice_variants(self):
        mk = lambda tc: ab.messages_to_chat({"model": "m", "max_tokens": 1, "messages": [],
                                             "tool_choice": tc})["tool_choice"]
        self.assertEqual(mk({"type": "auto"}), "auto")
        self.assertEqual(mk({"type": "any"}), "required")
        self.assertEqual(mk({"type": "none"}), "none")
        self.assertEqual(mk({"type": "tool", "name": "Read"}),
                         {"type": "function", "function": {"name": "Read"}})

    def test_sampling_fields_map_to_their_chat_names(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 512, "messages": [],
                                    "temperature": 0.2, "top_p": 0.9, "top_k": 40,
                                    "stop_sequences": ["END"], "stream": True,
                                    "metadata": {"user_id": "u1"}})
        self.assertEqual(chat["max_tokens"], 512)
        self.assertEqual(chat["temperature"], 0.2)
        self.assertEqual(chat["top_p"], 0.9)
        self.assertEqual(chat["top_k"], 40)
        self.assertEqual(chat["stop"], ["END"])
        self.assertTrue(chat["stream"])
        self.assertEqual(chat["user"], "u1")

    def test_thinking_blocks_from_history_are_dropped(self):
        """Signed thinking blocks are Anthropic-only; an OpenAI backend has no field
        for them and they carry no instruction the model needs."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "answer"}]}]})
        self.assertEqual(chat["messages"][0]["content"], "answer")

    def test_cache_control_is_not_forwarded(self):
        body = {"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "x",
                                          "cache_control": {"type": "ephemeral"}}]}]}
        self.assertNotIn("cache_control", json.dumps(ab.messages_to_chat(body)))

    def test_cache_control_survives_when_the_backend_understands_it(self):
        """OpenRouter forwards cache_control to Anthropic/Gemini models. Dropping it
        for those backends means paying full price for the whole context every turn,
        so a backend can opt in (`prompt_cache: true`) and keep the breakpoints."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100,
                                    "system": [{"type": "text", "text": "preamble",
                                                "cache_control": {"type": "ephemeral"}}],
                                    "messages": [{"role": "user", "content": [
                                        {"type": "text", "text": "hi",
                                         "cache_control": {"type": "ephemeral"}}]}]},
                                   keep_cache_control=True)
        self.assertEqual(chat["messages"][0]["content"],
                         [{"type": "text", "text": "preamble",
                           "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(chat["messages"][1]["content"],
                         [{"type": "text", "text": "hi",
                           "cache_control": {"type": "ephemeral"}}])

    def test_kept_cache_control_does_not_reshape_uncached_turns(self):
        """Only the turns that actually carry a breakpoint become part lists; the
        rest stay plain strings, so nothing changes for backends by accident."""
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
            {"role": "user", "content": [{"type": "text", "text": "plain"}]}]},
            keep_cache_control=True)
        self.assertEqual(chat["messages"][0]["content"], "plain")

    def test_document_block_raises_instead_of_silently_dropping_it(self):
        """Dropping a PDF the user asked about would answer about nothing at all —
        the one case where silence produces a wrong result."""
        with self.assertRaises(ab.UnsupportedContent):
            ab.messages_to_chat({"model": "m", "max_tokens": 100, "messages": [
                {"role": "user", "content": [
                    {"type": "document", "source": {"type": "base64", "media_type": "application/pdf",
                                                    "data": "JVBER"}}]}]})

    def test_gateway_private_keys_are_not_forwarded(self):
        chat = ab.messages_to_chat({"model": "m", "max_tokens": 1, "messages": [],
                                    "_reasoning": "on"})
        self.assertNotIn("_reasoning", chat)


class ChatToMessages(unittest.TestCase):
    """Response direction: Chat Completions response → Anthropic Messages object."""

    def chat(self, message, finish_reason="stop", usage=None):
        return {"id": "chatcmpl-1", "model": "some/model", "created": 1,
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": usage or {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}}

    def test_text_answer_becomes_a_text_block(self):
        msg = ab.chat_to_messages(self.chat({"role": "assistant", "content": "hello"}), "alias-x")
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], [{"type": "text", "text": "hello"}])
        self.assertTrue(msg["id"].startswith("msg_"))

    def test_model_is_the_alias_the_client_asked_for(self):
        """Claude Code checks the model it gets back; the alias is what it knows."""
        msg = ab.chat_to_messages(self.chat({"content": "x"}), "alias-x")
        self.assertEqual(msg["model"], "alias-x")

    def test_usage_is_renamed_to_anthropic_fields(self):
        msg = ab.chat_to_messages(self.chat({"content": "x"}), "m")
        self.assertEqual(msg["usage"], {"input_tokens": 11, "output_tokens": 3})

    def test_stop_reasons_are_mapped(self):
        mk = lambda fr: ab.chat_to_messages(self.chat({"content": "x"}, fr), "m")["stop_reason"]
        self.assertEqual(mk("stop"), "end_turn")
        self.assertEqual(mk("length"), "max_tokens")
        self.assertEqual(mk("tool_calls"), "tool_use")
        self.assertEqual(mk("function_call"), "tool_use")

    def test_tool_calls_become_tool_use_blocks_with_parsed_input(self):
        msg = ab.chat_to_messages(self.chat({
            "content": "checking", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "Read", "arguments": '{"file_path": "/tmp/x"}'}}]},
            "tool_calls"), "m")
        self.assertEqual(msg["content"][0], {"type": "text", "text": "checking"})
        self.assertEqual(msg["content"][1], {"type": "tool_use", "id": "call_1", "name": "Read",
                                             "input": {"file_path": "/tmp/x"}})
        self.assertEqual(msg["stop_reason"], "tool_use")

    def test_unparsable_tool_arguments_yield_an_empty_input(self):
        """A truncated argument stream must not take the whole response down."""
        with self.assertLogs("anthropic_bridge", level="WARNING"):
            msg = ab.chat_to_messages(self.chat({
                "content": None, "tool_calls": [
                    {"id": "c1", "function": {"name": "Read", "arguments": '{"file_pa'}}]}), "m")
        self.assertEqual(msg["content"][0]["input"], {})

    def test_reasoning_text_becomes_a_leading_thinking_block(self):
        msg = ab.chat_to_messages(self.chat({"content": "answer", "reasoning": "let me think"}), "m")
        self.assertEqual(msg["content"][0]["type"], "thinking")
        self.assertEqual(msg["content"][0]["thinking"], "let me think")
        self.assertEqual(msg["content"][1]["type"], "text")

    def test_empty_content_yields_no_blocks(self):
        msg = ab.chat_to_messages(self.chat({"content": ""}), "m")
        self.assertEqual(msg["content"], [])


class MessageShell(unittest.TestCase):
    def test_shell_has_the_fields_every_client_reads(self):
        shell = ab.message_shell("msg_1", "m")
        self.assertEqual(shell["type"], "message")
        self.assertEqual(shell["role"], "assistant")
        self.assertEqual(shell["content"], [])
        self.assertIsNone(shell["stop_reason"])
        self.assertEqual(shell["usage"], {"input_tokens": 0, "output_tokens": 0})


class FakeStream:
    """Stands in for the adapter's StreamingResponse: yields chat SSE bytes."""

    def __init__(self, chunks):
        self.chunks = chunks

    @property
    def body_iterator(self):
        async def gen():
            for c in self.chunks:
                yield c.encode() if isinstance(c, str) else c
        return gen()


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def delta_chunk(**delta):
    return sse({"id": "c1", "model": "real/model", "choices": [{"index": 0, "delta": delta}]})


def collect(gen):
    """Drain an async generator into a list of (event_type, payload) pairs."""
    async def run():
        out = []
        async for piece in gen:
            for block in piece.split("\n\n"):
                line = [ln for ln in block.split("\n") if ln.startswith("data:")]
                if line:
                    out.append(json.loads(line[0][5:].strip()))
        return out
    return asyncio.run(run())


class MessagesStream(unittest.TestCase):
    """Chat SSE → Anthropic SSE. Claude Code drives entirely off these events."""

    def text_stream(self):
        return FakeStream([
            delta_chunk(role="assistant"),
            delta_chunk(content="Hel"),
            delta_chunk(content="lo"),
            sse({"id": "c1", "model": "real/model", "choices": [{"index": 0, "delta": {},
                                                                 "finish_reason": "stop"}]}),
            sse({"id": "c1", "model": "real/model", "choices": [],
                 "usage": {"prompt_tokens": 7, "completion_tokens": 2}}),
            "data: [DONE]\n\n",
        ])

    def test_event_sequence_for_a_text_answer(self):
        events = collect(ab.messages_stream(self.text_stream(), "alias-x"))
        self.assertEqual([e["type"] for e in events],
                         ["message_start", "content_block_start", "content_block_delta",
                          "content_block_delta", "content_block_stop", "message_delta",
                          "message_stop"])

    def test_deltas_carry_the_text(self):
        events = collect(ab.messages_stream(self.text_stream(), "alias-x"))
        texts = [e["delta"]["text"] for e in events if e["type"] == "content_block_delta"]
        self.assertEqual(texts, ["Hel", "lo"])

    def test_message_start_announces_the_requested_model(self):
        events = collect(ab.messages_stream(self.text_stream(), "alias-x"))
        self.assertEqual(events[0]["message"]["model"], "alias-x")
        self.assertEqual(events[0]["message"]["role"], "assistant")

    def test_usage_lands_where_anthropic_puts_it(self):
        """A chat backend reports usage only in its LAST chunk, but message_start
        goes out first — so it carries the caller's estimate and message_delta
        carries the real numbers once they arrive."""
        events = collect(ab.messages_stream(self.text_stream(), "alias-x", input_tokens=5))
        self.assertEqual(events[0]["message"]["usage"]["input_tokens"], 5)
        delta = [e for e in events if e["type"] == "message_delta"][0]
        self.assertEqual(delta["usage"], {"input_tokens": 7, "output_tokens": 2})
        self.assertEqual(delta["delta"]["stop_reason"], "end_turn")

    def test_tool_call_streams_as_tool_use_with_json_deltas(self):
        stream = FakeStream([
            delta_chunk(content="looking"),
            delta_chunk(tool_calls=[{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "Read", "arguments": ""}}]),
            delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"file_'}}]),
            delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": 'path": "/x"}'}}]),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
            "data: [DONE]\n\n",
        ])
        events = collect(ab.messages_stream(stream, "m"))
        starts = [e for e in events if e["type"] == "content_block_start"]
        self.assertEqual(starts[0]["content_block"]["type"], "text")
        self.assertEqual(starts[1]["content_block"],
                         {"type": "tool_use", "id": "call_1", "name": "Read", "input": {}})
        self.assertEqual(starts[1]["index"], 1)
        json_deltas = [e["delta"]["partial_json"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"]["type"] == "input_json_delta"]
        self.assertEqual("".join(json_deltas), '{"file_path": "/x"}')
        self.assertEqual([e for e in events if e["type"] == "message_delta"][0]
                         ["delta"]["stop_reason"], "tool_use")

    def test_every_opened_block_is_closed(self):
        stream = FakeStream([
            delta_chunk(content="hi"),
            delta_chunk(tool_calls=[{"index": 0, "id": "c1", "function": {"name": "R",
                                                                         "arguments": "{}"}}]),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ])
        events = collect(ab.messages_stream(stream, "m"))
        self.assertEqual(len([e for e in events if e["type"] == "content_block_start"]),
                         len([e for e in events if e["type"] == "content_block_stop"]))

    def test_reasoning_deltas_stream_as_thinking_blocks(self):
        stream = FakeStream([
            delta_chunk(reasoning="think"),
            delta_chunk(content="answer"),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        ])
        events = collect(ab.messages_stream(stream, "m"))
        starts = [e["content_block"]["type"] for e in events if e["type"] == "content_block_start"]
        self.assertEqual(starts, ["thinking", "text"])
        tdelta = [e for e in events if e["type"] == "content_block_delta"][0]
        self.assertEqual(tdelta["delta"], {"type": "thinking_delta", "thinking": "think"})

    def test_tool_args_never_land_on_a_closed_block(self):
        """A backend may interleave text between tool-call fragments. Anthropic
        blocks are strictly sequential, so a delta after content_block_stop would
        desync the client's block bookkeeping."""
        stream = FakeStream([
            delta_chunk(tool_calls=[{"index": 0, "id": "c1", "type": "function",
                                     "function": {"name": "Read", "arguments": '{"a"'}}]),
            delta_chunk(content="thinking out loud"),
            delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": ':1}'}}]),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ])
        events = collect(ab.messages_stream(stream, "m"))
        closed = set()
        for e in events:
            if e["type"] == "content_block_stop":
                closed.add(e["index"])
            if e["type"] == "content_block_delta":
                self.assertNotIn(e["index"], closed,
                                 f"delta on closed block {e['index']}: {e}")
        json_deltas = [e["delta"]["partial_json"] for e in events
                       if e["type"] == "content_block_delta"
                       and e["delta"]["type"] == "input_json_delta"]
        self.assertEqual("".join(json_deltas), '{"a":1}')

    def test_tool_calls_without_an_index_field_stay_separate(self):
        """Not every OpenAI-compatible server numbers streamed tool calls; keying
        on a missing index would merge two calls' arguments into one."""
        stream = FakeStream([
            delta_chunk(tool_calls=[{"id": "c1", "type": "function",
                                     "function": {"name": "Read", "arguments": '{"a":1}'}}]),
            delta_chunk(tool_calls=[{"id": "c2", "type": "function",
                                     "function": {"name": "Write", "arguments": '{"b":2}'}}]),
            sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ])
        events = collect(ab.messages_stream(stream, "m"))
        tools = [e["content_block"] for e in events if e["type"] == "content_block_start"
                 and e["content_block"]["type"] == "tool_use"]
        self.assertEqual([t["id"] for t in tools], ["c1", "c2"])
        per_block = {}
        for e in events:
            if e["type"] == "content_block_delta" and e["delta"]["type"] == "input_json_delta":
                per_block.setdefault(e["index"], "")
                per_block[e["index"]] += e["delta"]["partial_json"]
        self.assertEqual(sorted(per_block.values()), ['{"a":1}', '{"b":2}'])

    def test_a_broken_upstream_stream_reports_an_error_event(self):
        """Silently finishing with stop_reason=end_turn would present a truncated
        answer as a complete one."""
        class Exploding(FakeStream):
            @property
            def body_iterator(self):
                async def gen():
                    yield delta_chunk(content="partial").encode()
                    raise ConnectionError("upstream died")
                return gen()

        with self.assertLogs("anthropic_bridge", level="WARNING"):
            events = collect(ab.messages_stream(Exploding([]), "m"))
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["error"]["type"], "api_error")
        self.assertNotIn("message_stop", [e["type"] for e in events])

    def test_a_stream_that_never_produced_content_still_closes_cleanly(self):
        events = collect(ab.messages_stream(FakeStream(["data: [DONE]\n\n"]), "m"))
        self.assertEqual([e["type"] for e in events],
                         ["message_start", "message_delta", "message_stop"])


class CountTokens(unittest.TestCase):
    def test_estimate_counts_system_and_message_text(self):
        n = ab.estimate_input_tokens({"system": "a" * 40, "messages": [
            {"role": "user", "content": "b" * 40}]})
        self.assertGreater(n, 10)
        self.assertLess(n, 60)

    def test_estimate_is_never_zero_for_a_real_request(self):
        self.assertGreaterEqual(ab.estimate_input_tokens(
            {"messages": [{"role": "user", "content": "hi"}]}), 1)


if __name__ == "__main__":
    unittest.main()
