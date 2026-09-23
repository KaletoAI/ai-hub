"""Responses API ↔ Chat Completions translation layer.

Pure functions only — no imports from `main`/`adapters`, no gateway state — so the
bridge stays hot-reload- and test-friendly (same rule as reasoning.py). `main.py`
owns the /v1/responses endpoints, dispatch/parking, and the background mode; this
module owns every body/stream translation those endpoints need:

- request:  `responses_to_chat()`   Responses request body → Chat Completions body
- response: `chat_to_responses()`   Chat Completions response → full Responses object
- shell:    `response_shell()`      the ONE Responses-object skeleton (completed
            bodies, stream events, and background/queued states all build on it)
- stream:   `responses_stream()`    backend chat SSE → Responses API SSE events
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _oid(prefix: str) -> str:
    """OpenAI-style object id: <prefix>_<24 hex chars> (msg_/fc_/resp_…)."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _content_parts_to_text(content: Any) -> Any:
    """Flatten a Responses-style content array to a chat-completions content value."""
    if not isinstance(content, list):
        return content
    text_pieces: list[str] = []
    image_parts: list[dict] = []
    for part in content:
        ptype = part.get("type")
        if ptype in ("input_text", "output_text", "text"):
            text_pieces.append(part.get("text", ""))
        elif ptype == "input_image":
            url = part.get("image_url") or part.get("url")
            if url:
                image_parts.append({"type": "image_url", "image_url": {"url": url}})
    if not image_parts:
        return "".join(text_pieces)
    parts: list[dict] = []
    if text_pieces:
        parts.append({"type": "text", "text": "".join(text_pieces)})
    parts.extend(image_parts)
    return parts


def responses_to_chat(body: dict) -> dict:
    """Translate an OpenAI Responses API request body to Chat Completions."""
    passthrough = {
        "model", "temperature", "top_p", "stop", "seed", "user", "metadata",
        "presence_penalty", "frequency_penalty", "logit_bias",
        "parallel_tool_calls", "response_format",
    }
    chat: dict = {k: v for k, v in body.items() if k in passthrough}

    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    # stream: silently downgrade — translating SSE event streams isn't supported yet
    chat["stream"] = False

    # Tools: Responses uses flat {type, name, description, parameters};
    #        Chat uses nested {type, function: {name, description, parameters}}.
    if tools := body.get("tools"):
        chat_tools = []
        for t in tools:
            if t.get("type") != "function":
                continue  # skip built-in tools (web_search, code_interpreter, …)
            fn = {k: t[k] for k in ("name", "description", "parameters", "strict") if k in t}
            chat_tools.append({"type": "function", "function": fn})
        if chat_tools:
            chat["tools"] = chat_tools
    if "tool_choice" in body:
        chat["tool_choice"] = body["tool_choice"]

    messages: list[dict] = []
    if instructions := body.get("instructions"):
        messages.append({"role": "system", "content": instructions})

    inp = body.get("input")
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            itype = item.get("type", "message")
            if itype == "message":
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                content = _content_parts_to_text(item.get("content", ""))
                messages.append({"role": role, "content": content})
            elif itype == "function_call":
                call = {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                }
                # Parallel calls are ONE assistant turn in chat: consecutive
                # function_call items (and the assistant text right before them) join
                # the same message. One message per call — assistant, assistant, tool,
                # tool — is rejected by strict servers (vLLM/OpenAI 400) and renders a
                # different prompt on lenient ones.
                prev = messages[-1] if messages else None
                if prev is not None and prev.get("role") == "assistant":
                    prev.setdefault("tool_calls", []).append(call)
                    if prev.get("content") == "":
                        prev["content"] = None
                else:
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            elif itype == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id"),
                    "content": item.get("output", ""),
                })
    chat["messages"] = messages
    return chat


def output_text_of(output: Optional[list]) -> str:
    """Concatenated `output_text` parts of a Responses `output` array."""
    return "".join(p.get("text", "") for o in (output or []) if o.get("type") == "message"
                   for p in (o.get("content") or []) if p.get("type") == "output_text")


def response_shell(rid: str, status: str, model: Optional[str], created: int,
                   output: Optional[list] = None, usage: Optional[dict] = None,
                   error: Optional[dict] = None, background: bool = False,
                   **extra) -> dict:
    """The one OpenAI Responses-object skeleton. Completed bodies, stream-event
    snapshots, and background queued/failed states all build on this; `extra`
    lays additional top-level fields (e.g. `_finish_reason`) over the base."""
    out = output or []
    shell = {
        "id": rid, "object": "response", "created_at": created, "status": status,
        "error": error, "incomplete_details": None, "model": model,
        "output": out, "output_text": output_text_of(out),
        "usage": usage, "metadata": {}, "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [], "temperature": 1.0, "top_p": 1.0,
    }
    if background:
        shell["background"] = True
    shell.update(extra)
    return shell


def _reasoning_item(text: str) -> dict:
    """A Responses-API reasoning output item carrying the model's thinking text as a
    summary part — how thinking-model output (delta/message `reasoning`) is surfaced
    instead of being dropped (some models stream EVERYTHING there; see the
    thinking-models note in the repo docs)."""
    return {"type": "reasoning", "id": _oid("rs"),
            "summary": [{"type": "summary_text", "text": text}], "status": "completed"}


def chat_to_responses(chat_resp: dict) -> dict:
    """Translate a Chat Completions response body to a Responses API body. A
    `message.reasoning`/`reasoning_content` field becomes a reasoning output item
    (listed first, like OpenAI); `output_text` stays content-only."""
    choice = (chat_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}

    output: list[dict] = []
    think = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(think, str) and think:
        output.append(_reasoning_item(think))

    text_content = message.get("content")
    if text_content:
        output.append({
            "type": "message",
            "id": _oid("msg"),
            "role": message.get("role", "assistant"),
            "status": "completed",
            "content": [{"type": "output_text", "text": text_content, "annotations": []}],
        })

    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        output.append({
            "type": "function_call",
            "id": _oid("fc"),
            "call_id": tc.get("id"),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", ""),
            "status": "completed",
        })

    usage_in = chat_resp.get("usage") or {}
    usage = {
        "input_tokens": usage_in.get("prompt_tokens", 0),
        "output_tokens": usage_in.get("completion_tokens", 0),
        "total_tokens": usage_in.get("total_tokens", 0),
    }

    return response_shell(
        chat_resp.get("id") or _oid("resp"), "completed",
        chat_resp.get("model"), chat_resp.get("created", int(time.time())),
        output=output, usage=usage, _finish_reason=choice.get("finish_reason"),
    )


class _UpstreamStreamError(Exception):
    """An error the backend reported INSIDE its SSE stream (`data: {"error": …}`)."""


def _tool_slot(tc: dict, last_key):
    """Which collected tool call a chat `delta.tool_calls` fragment belongs to — the
    same rule as anthropic_bridge.messages_stream: the index when the backend numbers
    its calls, else a new id starts a new call and an id-less fragment continues the
    last one (keying everything on a missing index would merge two calls)."""
    if tc.get("index") is not None:
        return ("i", tc["index"])
    if tc.get("id"):
        return ("id", tc["id"])
    return last_key or ("i", 0)


async def responses_stream(chat_resp, raw_body: dict, alias: str):
    """A3: translate a backend chat-completion SSE stream into Responses API SSE
    events. Consumes the adapter StreamingResponse's body_iterator (so in-flight
    accounting + stats still fire in the adapter when it drains) and closes it
    explicitly on every exit — a client disconnect lands in THIS generator, and the
    adapter's (which holds the slot and the upstream connection) must not wait for
    garbage collection.

    Thinking-model chunks (`delta.reasoning`/`reasoning_content`) are forwarded as
    reasoning-summary events on a reasoning item (output_index 1 — the message item
    stays at 0, its events having already been announced) instead of being dropped;
    clients that don't know reasoning events simply ignore them.

    Tool calls (`delta.tool_calls`) become `function_call` output items. Like the
    Messages bridge, their fragments are COLLECTED and each call is emitted complete
    once the stream ends (`output_item.added` → `function_call_arguments.delta` →
    `.done` → `output_item.done`, after the message and reasoning items): a chat
    backend may interleave two calls' fragments, number them inconsistently or omit
    the index, and a client can only run a tool once its arguments parse anyway.

    A stream that dies — an exception from upstream, an in-band `error` payload, or an
    end with neither `finish_reason` nor `[DONE]` — ends in `response.failed`, never
    `response.completed`: completing around a truncated text makes the client store
    half an answer as the whole one."""
    resp_id = _oid("resp")
    item_id = _oid("msg")
    rs_id = _oid("rs")
    created = int(time.time())
    model = raw_body.get("model") or alias
    seq = 0

    def ev(etype: str, payload: dict) -> str:
        nonlocal seq
        body = {"type": etype, "sequence_number": seq, **payload}
        seq += 1
        return f"event: {etype}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"

    yield ev("response.created",
             {"response": response_shell(resp_id, "in_progress", model, created)})
    yield ev("response.in_progress",
             {"response": response_shell(resp_id, "in_progress", model, created)})
    yield ev("response.output_item.added", {"output_index": 0, "item": {
        "id": item_id, "type": "message", "status": "in_progress", "role": "assistant", "content": []}})
    yield ev("response.content_part.added", {"item_id": item_id, "output_index": 0,
             "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})

    full, think, buf, usage = "", "", "", None
    tools: list[dict] = []                    # collected tool calls, in arrival order
    slots: dict = {}                          # chat slot key → index into `tools`
    last_key = None
    finished = False                          # finish_reason or [DONE] seen
    failure: Optional[str] = None
    source = chat_resp.body_iterator
    try:
        async for chunk in source:
            buf += chunk.decode("utf-8", "ignore") if isinstance(chunk, (bytes, bytearray)) else chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    finished = True
                    continue
                if not data:
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                err = obj.get("error")
                if err and not obj.get("choices"):
                    msg = err.get("message") if isinstance(err, dict) else err
                    raise _UpstreamStreamError(str(msg or err))
                if obj.get("model"):
                    model = obj["model"]
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = (obj.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    finished = True
                d = choice.get("delta") or {}
                rdelta = d.get("reasoning") or d.get("reasoning_content")
                if isinstance(rdelta, str) and rdelta:
                    if not think:                       # first thinking token → announce the item
                        yield ev("response.output_item.added", {"output_index": 1, "item": {
                            "id": rs_id, "type": "reasoning", "status": "in_progress", "summary": []}})
                    think += rdelta
                    yield ev("response.reasoning_summary_text.delta", {"item_id": rs_id,
                             "output_index": 1, "summary_index": 0, "delta": rdelta})
                delta = d.get("content")
                if delta:
                    full += delta
                    yield ev("response.output_text.delta", {"item_id": item_id,
                             "output_index": 0, "content_index": 0, "delta": delta})
                for tc in d.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function") or {}
                    key = last_key = _tool_slot(tc, last_key)
                    if key not in slots:
                        slots[key] = len(tools)
                        tools.append({"id": tc.get("id") or "", "name": fn.get("name") or "",
                                      "args": ""})
                    t = tools[slots[key]]
                    if tc.get("id") and not t["id"]:
                        t["id"] = tc["id"]
                    if fn.get("name") and not t["name"]:
                        t["name"] = fn["name"]
                    t["args"] += fn.get("arguments") or ""
    except _UpstreamStreamError as e:
        failure = f"upstream error: {e}"
    except Exception as e:
        failure = f"upstream stream failed: {str(e) or type(e).__name__}"
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:                  # already closed / never started
                pass
    if failure is None and not finished:
        failure = "upstream stream ended before the answer was complete"

    u = None
    if usage:
        u = {"input_tokens": usage.get("prompt_tokens", 0),
             "output_tokens": usage.get("completion_tokens", 0),
             "total_tokens": usage.get("total_tokens", 0)}
    if failure is not None:
        logger.warning(f"responses SSE translate aborted: {failure}")
        yield ev("response.failed", {"response": response_shell(
            resp_id, "failed", model, created, usage=u,
            error={"code": "server_error", "message": failure})})
        return

    part = {"type": "output_text", "text": full, "annotations": []}
    yield ev("response.output_text.done", {"item_id": item_id, "output_index": 0,
             "content_index": 0, "text": full})
    yield ev("response.content_part.done", {"item_id": item_id, "output_index": 0,
             "content_index": 0, "part": part})
    final_item = {"id": item_id, "type": "message", "status": "completed",
                  "role": "assistant", "content": [part]}
    yield ev("response.output_item.done", {"output_index": 0, "item": final_item})
    output = [final_item]
    if think:
        rs_item = dict(_reasoning_item(think), id=rs_id)
        yield ev("response.reasoning_summary_text.done", {"item_id": rs_id,
                 "output_index": 1, "summary_index": 0, "text": think})
        yield ev("response.output_item.done", {"output_index": 1, "item": rs_item})
        output.append(rs_item)
    for t in tools:                            # complete function_call items, in arrival order
        idx = len(output)
        item = {"type": "function_call", "id": _oid("fc"), "call_id": t["id"] or _oid("call"),
                "name": t["name"], "arguments": "", "status": "in_progress"}
        yield ev("response.output_item.added", {"output_index": idx, "item": item})
        if t["args"]:
            yield ev("response.function_call_arguments.delta", {"item_id": item["id"],
                     "output_index": idx, "delta": t["args"]})
        yield ev("response.function_call_arguments.done", {"item_id": item["id"],
                 "output_index": idx, "name": t["name"], "arguments": t["args"]})
        done_item = dict(item, arguments=t["args"], status="completed")
        yield ev("response.output_item.done", {"output_index": idx, "item": done_item})
        output.append(done_item)
    yield ev("response.completed",
             {"response": response_shell(resp_id, "completed", model, created,
                                         output=output, usage=u)})
