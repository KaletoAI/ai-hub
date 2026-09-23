"""The Responses ↔ Chat bridge — what N8N/LangChain.js and every Responses client see.

Why this fails SILENTLY: the bridge always produces a well-formed Responses object or
event stream, whatever it lost on the way. A streamed tool call whose `delta.tool_calls`
is ignored ends in a perfectly valid `response.completed` with an empty message — the
agent simply "decides" not to call its tool. Two parallel `function_call` items turned
into two assistant messages each carrying one call are rejected by strict chat servers
only (vLLM/OpenAI answer 400, llama.cpp quietly renders a different prompt). And a
backend stream that dies mid-answer used to close with `status: completed` around the
truncated text — the client stores half an answer as the whole one.

stdlib only:  venv/bin/python -m unittest tests.test_responses_bridge -v
"""
import asyncio
import json
import unittest

import responses_bridge as rb


class FakeStream:
    """Stands in for the adapter's StreamingResponse: yields chat SSE bytes."""

    def __init__(self, chunks, explode=None):
        self.chunks, self.explode = chunks, explode
        self.closed = False

    @property
    def body_iterator(self):
        outer = self

        async def gen():
            try:
                for c in outer.chunks:
                    yield c.encode() if isinstance(c, str) else c
                if outer.explode is not None:
                    raise outer.explode
            finally:
                outer.closed = True
        return gen()


def sse(obj):
    return f"data: {json.dumps(obj)}\n\n"


def chunk(finish=None, **delta):
    ch = {"index": 0, "delta": delta}
    if finish:
        ch["finish_reason"] = finish
    return sse({"id": "c1", "model": "real/model", "choices": [ch]})


def tc(index, id=None, name=None, args=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    out = {"index": index, "function": fn}
    if id is not None:
        out["id"], out["type"] = id, "function"
    return out


def collect(gen):
    async def run():
        out = []
        async for piece in gen:
            for block in piece.split("\n\n"):
                line = [ln for ln in block.split("\n") if ln.startswith("data:")]
                if line:
                    out.append(json.loads(line[0][5:].strip()))
        return out
    return asyncio.run(run())


USAGE = sse({"id": "c1", "model": "real/model", "choices": [],
             "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13}})
DONE = "data: [DONE]\n\n"


class StreamedToolCalls(unittest.TestCase):
    """K2: `delta.tool_calls` must become function_call output items."""

    def fragmented(self):
        # Two parallel calls, arguments split across chunks and interleaved — the shape
        # vLLM/llama.cpp/OpenRouter actually stream.
        return FakeStream([
            chunk(role="assistant"),
            chunk(tool_calls=[tc(0, id="call_a", name="get_weather", args="")]),
            chunk(tool_calls=[tc(0, args='{"ci')]),
            chunk(tool_calls=[tc(1, id="call_b", name="get_time", args='{"tz"')]),
            chunk(tool_calls=[tc(0, args='ty":"Berlin"}')]),
            chunk(tool_calls=[tc(1, args=':"CET"}')]),
            chunk(finish="tool_calls"),
            USAGE, DONE,
        ])

    def test_each_call_is_announced_streamed_and_finished(self):
        events = collect(rb.responses_stream(self.fragmented(), {"model": "m"}, "m"))
        added = [e for e in events if e["type"] == "response.output_item.added"
                 and e["item"]["type"] == "function_call"]
        self.assertEqual([a["item"]["name"] for a in added], ["get_weather", "get_time"])
        self.assertEqual([a["item"]["call_id"] for a in added], ["call_a", "call_b"])
        by_item = {}
        for e in events:
            if e["type"] == "response.function_call_arguments.delta":
                by_item[e["item_id"]] = by_item.get(e["item_id"], "") + e["delta"]
        done = {e["item_id"]: e["arguments"] for e in events
                if e["type"] == "response.function_call_arguments.done"}
        self.assertEqual(sorted(done.values()), ['{"city":"Berlin"}', '{"tz":"CET"}'])
        self.assertEqual(by_item, done)                  # deltas add up to the final text
        item_done = [e["item"] for e in events if e["type"] == "response.output_item.done"
                     and e["item"]["type"] == "function_call"]
        self.assertEqual([i["status"] for i in item_done], ["completed", "completed"])

    def test_completed_response_lists_the_calls_in_output_index_order(self):
        events = collect(rb.responses_stream(self.fragmented(), {"model": "m"}, "m"))
        final = events[-1]
        self.assertEqual(final["type"], "response.completed")
        calls = [o for o in final["response"]["output"] if o["type"] == "function_call"]
        self.assertEqual([(c["call_id"], c["name"], json.loads(c["arguments"])) for c in calls],
                         [("call_a", "get_weather", {"city": "Berlin"}),
                          ("call_b", "get_time", {"tz": "CET"})])
        # every item's output_index names its position in the final output array
        positions = {e["item"]["id"]: e["output_index"] for e in events
                     if e["type"] == "response.output_item.added"}
        for i, item in enumerate(final["response"]["output"]):
            self.assertEqual(positions[item["id"]], i)

    def test_usage_still_reaches_completed(self):
        events = collect(rb.responses_stream(self.fragmented(), {"model": "m"}, "m"))
        self.assertEqual(events[-1]["response"]["usage"]["output_tokens"], 4)

    def test_calls_without_index_stay_separate(self):
        stream = FakeStream([
            chunk(tool_calls=[{"id": "x1", "function": {"name": "a", "arguments": '{"p":1}'}}]),
            chunk(tool_calls=[{"id": "x2", "function": {"name": "b", "arguments": '{"q":'}}]),
            chunk(tool_calls=[{"function": {"arguments": "2}"}}]),
            chunk(finish="tool_calls"), DONE,
        ])
        final = collect(rb.responses_stream(stream, {"model": "m"}, "m"))[-1]
        calls = [(o["call_id"], o["arguments"]) for o in final["response"]["output"]
                 if o["type"] == "function_call"]
        self.assertEqual(calls, [("x1", '{"p":1}'), ("x2", '{"q":2}')])

    def test_text_and_reasoning_still_stream_as_before(self):
        stream = FakeStream([chunk(reasoning="hmm"), chunk(content="Hi"), chunk(finish="stop"), DONE])
        events = collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        self.assertEqual(final["output_text"], "Hi")
        self.assertEqual([o["type"] for o in final["output"]], ["message", "reasoning"])


class AbortedStream(unittest.TestCase):
    """K7: a stream that dies must end as `response.failed`, never `completed`."""

    def test_an_exception_mid_stream_fails_the_response(self):
        stream = FakeStream([chunk(content="half an ans")], explode=ConnectionError("peer reset"))
        with self.assertLogs("responses_bridge", level="WARNING"):
            events = collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        types = [e["type"] for e in events]
        self.assertNotIn("response.completed", types)
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertEqual(events[-1]["response"]["status"], "failed")
        self.assertIn("peer reset", events[-1]["response"]["error"]["message"])

    def test_a_stream_that_just_stops_is_not_complete(self):
        # No finish_reason, no [DONE]: the connection ended early without an exception.
        stream = FakeStream([chunk(content="half")])
        with self.assertLogs("responses_bridge", level="WARNING"):
            events = collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_an_in_band_error_event_fails_the_response(self):
        stream = FakeStream([chunk(content="x"),
                             sse({"error": {"message": "CUDA out of memory"}}), DONE])
        with self.assertLogs("responses_bridge", level="WARNING"):
            events = collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        self.assertEqual(events[-1]["type"], "response.failed")
        self.assertIn("CUDA out of memory", events[-1]["response"]["error"]["message"])

    def test_the_upstream_iterator_is_closed(self):
        stream = FakeStream([chunk(content="x")], explode=ConnectionError("gone"))
        with self.assertLogs("responses_bridge", level="WARNING"):
            collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        self.assertTrue(stream.closed)

    def test_a_clean_end_without_done_marker_but_with_finish_reason_completes(self):
        stream = FakeStream([chunk(content="ok"), chunk(finish="stop")])
        events = collect(rb.responses_stream(stream, {"model": "m"}, "m"))
        self.assertEqual(events[-1]["type"], "response.completed")


class ParallelFunctionCallsInInput(unittest.TestCase):
    """K6: consecutive function_call items are ONE assistant turn."""

    def test_consecutive_calls_share_one_assistant_message(self):
        chat = rb.responses_to_chat({"model": "m", "input": [
            {"type": "message", "role": "user", "content": "weather and time?"},
            {"type": "function_call", "call_id": "a", "name": "w", "arguments": "{}"},
            {"type": "function_call", "call_id": "b", "name": "t", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "sunny"},
            {"type": "function_call_output", "call_id": "b", "output": "noon"},
        ]})
        roles = [m["role"] for m in chat["messages"]]
        self.assertEqual(roles, ["user", "assistant", "tool", "tool"])
        self.assertEqual([t["id"] for t in chat["messages"][1]["tool_calls"]], ["a", "b"])

    def test_calls_follow_the_assistant_text_of_the_same_turn(self):
        chat = rb.responses_to_chat({"model": "m", "input": [
            {"type": "message", "role": "user", "content": "q"},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "let me check"}]},
            {"type": "function_call", "call_id": "a", "name": "w", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "ok"},
        ]})
        self.assertEqual([m["role"] for m in chat["messages"]], ["user", "assistant", "tool"])
        self.assertEqual(chat["messages"][1]["content"], "let me check")
        self.assertEqual(chat["messages"][1]["tool_calls"][0]["id"], "a")

    def test_a_reasoning_item_between_calls_does_not_split_the_turn(self):
        chat = rb.responses_to_chat({"model": "m", "input": [
            {"type": "function_call", "call_id": "a", "name": "w", "arguments": "{}"},
            {"type": "reasoning", "summary": []},
            {"type": "function_call", "call_id": "b", "name": "t", "arguments": "{}"},
        ]})
        self.assertEqual(len(chat["messages"]), 1)
        self.assertEqual(len(chat["messages"][0]["tool_calls"]), 2)

    def test_separate_turns_stay_separate(self):
        chat = rb.responses_to_chat({"model": "m", "input": [
            {"type": "function_call", "call_id": "a", "name": "w", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "a", "output": "x"},
            {"type": "function_call", "call_id": "b", "name": "t", "arguments": "{}"},
        ]})
        self.assertEqual([m["role"] for m in chat["messages"]], ["assistant", "tool", "assistant"])


if __name__ == "__main__":
    unittest.main()
