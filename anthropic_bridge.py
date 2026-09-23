"""Anthropic Messages ↔ Chat Completions translation layer.

Pure functions only — no imports from `main`/`adapters`, no gateway state — so the
bridge stays hot-reload- and test-friendly (same rule as `reasoning.py` and
`responses_bridge.py`). `main.py` owns the `/v1/messages` endpoints and dispatch;
`adapters.py` calls this module when a NON-Anthropic backend serves a Messages
request. An Anthropic backend never touches it: there the request is forwarded
verbatim, because `cache_control`, thinking signatures and fine-grained tool
streaming must survive exactly as Claude Code sent them.

- request:  `messages_to_chat()`    Messages body → Chat Completions body
- response: `chat_to_messages()`    Chat Completions response → Messages object
- shell:    `message_shell()`       the ONE Messages-object skeleton
- stream:   `messages_stream()`     backend chat SSE → Messages API SSE events
- tokens:   `estimate_input_tokens()`  count_tokens for backends without the endpoint

Translation policy (Claude Code sends more than an OpenAI backend can take):
drop what is inert (cache_control, thinking blocks from history, server-side
tools), raise `UnsupportedContent` where dropping would silently produce an
answer about the wrong thing (documents/PDFs), and MOVE what chat can carry
elsewhere (images inside a tool_result → a user message right after the turn's
tool messages, since the chat `tool` role takes text only).
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)


class UnsupportedContent(Exception):
    """A content block this backend cannot be given a faithful translation of.
    The endpoint turns it into a 400 — the alternative (dropping it) would answer
    a question about content the model never saw."""


def _mid() -> str:
    """Anthropic-style message id."""
    return f"msg_{uuid.uuid4().hex[:24]}"


def _blocks_text(content: Any) -> str:
    """Text of a content value that may be a plain string or a block list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n\n".join(b.get("text", "") for b in content
                       if isinstance(b, dict) and b.get("type") == "text")


def _image_part(block: dict) -> dict:
    """Anthropic image block → OpenAI image_url part (base64 → data URI)."""
    src = block.get("source") or {}
    if src.get("type") == "base64":
        url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
    else:
        url = src.get("url", "")
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_result_text(block: dict) -> str:
    """Text a tool_result carries, flagged when the tool reported an error (the
    chat `tool` role has no is_error field — the model must still see it failed).
    A result that is ONLY images gets a pointer to where they went (see
    `_tool_result_images`) — an empty tool message is refused by strict servers."""
    text = _blocks_text(block.get("content"))
    if not text and _tool_result_images(block):
        text = "(image result — attached in the next message)"
    return f"Error: {text}" if block.get("is_error") else text


def _tool_result_images(block: dict) -> list[dict]:
    """Image blocks inside a tool_result, as chat image parts. The chat `tool` role
    takes text only, so these cannot stay where Anthropic put them; the caller moves
    them into a user message right after the turn's tool messages. Dropping them — the
    old behaviour — answered a question about a screenshot the model never saw
    (Claude Code's Read tool returns image files exactly this way)."""
    content = block.get("content")
    if not isinstance(content, list):
        return []
    return [_image_part(b) for b in content if isinstance(b, dict) and b.get("type") == "image"]


_STOP_REASON = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
                "function_call": "tool_use", "content_filter": "end_turn"}


def _stop_reason(finish_reason: Optional[str], has_tool_calls: bool) -> str:
    """Anthropic stop_reason for a chat finish_reason. A message that carries tool
    calls is `tool_use` whatever the backend said: several servers (Ollama, some
    vLLM/LocalAI builds) report `stop` there, and `end_turn` makes Claude Code end the
    turn instead of running the tool. `max_tokens` stays — a call cut off mid-arguments
    must not be run."""
    reason = _STOP_REASON.get(finish_reason or "", "end_turn")
    if has_tool_calls and reason == "end_turn":
        return "tool_use"
    return reason


def message_shell(mid: str, model: Optional[str], content: Optional[list] = None,
                  stop_reason: Optional[str] = None, usage: Optional[dict] = None) -> dict:
    """The one Anthropic Messages-object skeleton. Completed bodies and the
    stream's `message_start` snapshot both build on this."""
    return {
        "id": mid, "type": "message", "role": "assistant", "model": model,
        "content": content if content is not None else [],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": usage or {"input_tokens": 0, "output_tokens": 0},
    }


def _tool_use_block(tc: dict) -> dict:
    """Chat tool_call → Anthropic tool_use block. Unparsable arguments (a
    truncated stream) yield an empty input rather than failing the response."""
    fn = tc.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        logger.warning(f"anthropic bridge: tool call '{fn.get('name')}' had unparsable arguments")
        args = {}
    return {"type": "tool_use", "id": tc.get("id", ""), "name": fn.get("name", ""),
            "input": args if isinstance(args, dict) else {}}


def chat_to_messages(chat_resp: dict, model: Optional[str]) -> dict:
    """Translate a Chat Completions response body into a Messages object. `model`
    is the alias the client asked for — Claude Code matches it against what it
    sent, so the backend's real model id must not leak into it."""
    choice = (chat_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict] = []

    think = message.get("reasoning") or message.get("reasoning_content")
    if isinstance(think, str) and think:
        content.append({"type": "thinking", "thinking": think})
    text = message.get("content")
    if isinstance(text, list):                       # some backends answer in parts
        text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
    if text:
        content.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        content.append(_tool_use_block(tc))

    usage_in = chat_resp.get("usage") or {}
    return message_shell(
        _mid(), model, content=content,
        stop_reason=_stop_reason(choice.get("finish_reason"), bool(message.get("tool_calls"))),
        usage={"input_tokens": int(usage_in.get("prompt_tokens") or 0),
               "output_tokens": int(usage_in.get("completion_tokens") or 0)},
    )


def estimate_input_tokens(body: dict) -> int:
    """Rough prompt-token count (~chars/4) over a Messages body — what
    `/v1/messages/count_tokens` answers for a backend that has no such endpoint,
    and the input figure `message_start` carries before the backend reports real
    usage. Never 0 for a non-empty request: a 0 reads as "nothing sent"."""
    chars = len(_blocks_text(body.get("system") or ""))
    for m in body.get("messages") or []:
        content = m.get("content")
        if isinstance(content, str):
            chars += len(content)
            continue
        for block in content or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                chars += len(block.get("text", ""))
            elif block.get("type") == "tool_use":
                chars += len(json.dumps(block.get("input") or {}))
            elif block.get("type") == "tool_result":
                chars += len(_blocks_text(block.get("content")))
    for t in body.get("tools") or []:
        chars += len(json.dumps(t.get("input_schema") or {})) + len(t.get("description") or "")
    return max(1, chars // 4) if chars else 0


async def messages_stream(chat_resp, model: Optional[str], input_tokens: int = 0):
    """Translate a backend chat-completion SSE stream into Anthropic Messages SSE.

    Consumes the adapter StreamingResponse's body_iterator (so the adapter's
    in-flight accounting and stats still fire when it drains). Emits exactly the
    event sequence Claude Code drives off: message_start → per content block
    start/delta/stop → message_delta (stop_reason + real usage) → message_stop.

    Blocks are opened lazily and strictly sequentially — a text or thinking block
    is closed before the next one opens, and whatever is open when the stream ends
    is closed — so a client's block bookkeeping never desyncs.

    Tool calls are the exception: their fragments are COLLECTED and emitted as
    complete blocks at the end of the stream. A chat backend may interleave text
    between two fragments of the same call, number its calls inconsistently, or
    omit the index entirely — while Anthropic blocks are strictly sequential, so a
    late fragment would have to land on an already-closed block. Nothing is lost:
    a client can only run a tool once its arguments parse, which is after the last
    fragment either way. (The verbatim Anthropic path keeps true fine-grained tool
    streaming — this only concerns translated backends.)"""
    mid = _mid()
    state = {"index": -1, "kind": None}       # currently OPEN block
    stop_reason, usage = None, None
    tools: list[dict] = []                    # collected tool calls, in arrival order
    slots: dict = {}                          # chat slot key → index into `tools`
    last_slot = {"key": None}

    def ev(etype: str, payload: dict) -> str:
        return f"event: {etype}\ndata: {json.dumps({'type': etype, **payload}, ensure_ascii=False)}\n\n"

    def close_open() -> str:
        if state["kind"] is None:
            return ""
        out = ev("content_block_stop", {"index": state["index"]})
        state["kind"] = None
        return out

    def open_block(kind: str, block: dict) -> str:
        state["index"] += 1
        state["kind"] = kind
        return ev("content_block_start", {"index": state["index"], "content_block": block})

    yield ev("message_start", {"message": message_shell(
        mid, model, usage={"input_tokens": input_tokens, "output_tokens": 0})})

    # Hold the iterator in a local so it can be closed explicitly: when the CLIENT
    # disconnects, GeneratorExit lands in THIS generator, and the inner adapter
    # generator — which holds the in-flight slot and the upstream connection — would
    # otherwise only be finalized whenever the event loop gets around to collecting
    # it. The project's rule is explicit release on every path.
    source = chat_resp.body_iterator
    try:
        buf = ""
        async for chunk in source:
            buf += chunk.decode("utf-8", "ignore") if isinstance(chunk, (bytes, bytearray)) else chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = (obj.get("choices") or [{}])[0]
                if choice.get("finish_reason"):
                    stop_reason = choice["finish_reason"]
                d = choice.get("delta") or {}

                think = d.get("reasoning") or d.get("reasoning_content")
                if isinstance(think, str) and think:
                    if state["kind"] != "thinking":
                        yield close_open()
                        yield open_block("thinking", {"type": "thinking", "thinking": ""})
                    yield ev("content_block_delta", {"index": state["index"],
                             "delta": {"type": "thinking_delta", "thinking": think}})

                text = d.get("content")
                if isinstance(text, str) and text:
                    if state["kind"] != "text":
                        yield close_open()
                        yield open_block("text", {"type": "text", "text": ""})
                    yield ev("content_block_delta", {"index": state["index"],
                             "delta": {"type": "text_delta", "text": text}})

                for tc in d.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    # Slot identity: the index when the backend numbers its calls, else
                    # a new id starts a new call and an id-less fragment continues the
                    # last one. Keying everything on a missing index would merge two
                    # calls' arguments into one.
                    if tc.get("index") is not None:
                        key = ("i", tc["index"])
                    elif tc.get("id"):
                        key = ("id", tc["id"])
                    else:
                        key = last_slot["key"] or ("i", 0)
                    last_slot["key"] = key
                    if key not in slots:
                        slots[key] = len(tools)
                        tools.append({"id": tc.get("id", ""), "name": fn.get("name", ""),
                                      "args": ""})
                    t = tools[slots[key]]
                    if tc.get("id") and not t["id"]:
                        t["id"] = tc["id"]
                    if fn.get("name") and not t["name"]:
                        t["name"] = fn["name"]
                    t["args"] += fn.get("arguments") or ""
    except Exception as e:
        # A stream that died mid-answer must NOT be dressed up as a finished one:
        # closing with stop_reason=end_turn would present a truncated answer as
        # complete. Anthropic has an `error` event for exactly this.
        logger.warning(f"anthropic SSE translate aborted: {e}")
        yield close_open()
        yield ev("error", {"error": {"type": "api_error",
                                     "message": f"upstream stream failed: {e}"}})
        return
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:                  # already closed / never started
                pass

    yield close_open()
    for t in tools:                            # complete tool blocks, in arrival order
        yield open_block("tool", {"type": "tool_use", "id": t["id"], "name": t["name"],
                                  "input": {}})
        if t["args"]:
            yield ev("content_block_delta", {"index": state["index"],
                     "delta": {"type": "input_json_delta", "partial_json": t["args"]}})
        yield close_open()
    final_usage = {"input_tokens": int((usage or {}).get("prompt_tokens") or input_tokens),
                   "output_tokens": int((usage or {}).get("completion_tokens") or 0)}
    yield ev("message_delta", {"delta": {"stop_reason": _stop_reason(stop_reason, bool(tools)),
                                         "stop_sequence": None}, "usage": final_usage})
    yield ev("message_stop", {})


def _text_parts(blocks: list, keep_cache_control: bool):
    """Text blocks → a chat `content` value. Normally that is one flattened string;
    with `keep_cache_control` and at least one breakpoint present it stays a part
    LIST so the breakpoints ride along (OpenRouter passes them to Anthropic/Gemini
    models — dropping them means paying full price for the whole context every
    turn). Turns without a breakpoint keep the plain-string shape, so opting in
    never reshapes more of the request than it has to.

    Note the one visible difference: in part-list form the blocks stay separate
    instead of being joined with a blank line, so a turn's rendered prompt can
    differ marginally between `prompt_cache` on and off. It is consistent per
    backend, which is what caching needs."""
    texts = [b.get("text", "") for b in blocks]
    if not keep_cache_control or not any(b.get("cache_control") for b in blocks):
        return "\n\n".join(texts)
    out = []
    for b in blocks:
        part = {"type": "text", "text": b.get("text", "")}
        if b.get("cache_control"):
            part["cache_control"] = b["cache_control"]
        out.append(part)
    return out


def messages_to_chat(body: dict, keep_cache_control: bool = False) -> dict:
    """Translate an Anthropic Messages request body to Chat Completions.

    Raises UnsupportedContent for blocks that cannot be dropped without changing
    the answer. Gateway-private `_`-prefixed keys are never forwarded.

    `keep_cache_control` (backend opt-in) carries prompt-cache breakpoints into the
    chat body instead of dropping them — see `_text_parts`."""
    chat: dict = {}
    for src, dst in (("model", "model"), ("max_tokens", "max_tokens"), ("temperature", "temperature"),
                     ("top_p", "top_p"), ("top_k", "top_k"), ("stream", "stream")):
        if src in body:
            chat[dst] = body[src]
    if body.get("stop_sequences"):
        chat["stop"] = body["stop_sequences"]
    user_id = (body.get("metadata") or {}).get("user_id")
    if user_id:
        chat["user"] = user_id

    if tools := body.get("tools"):
        chat_tools = []
        for t in tools:
            # Server-side tools (web_search, code_execution, computer use) are typed;
            # a plain function tool has an input_schema and no `type`. Only the latter
            # has an OpenAI equivalent — forwarding the rest would 400 the request.
            if t.get("type") and t.get("type") != "custom":
                continue
            fn = {"name": t.get("name", "")}
            if t.get("description"):
                fn["description"] = t["description"]
            fn["parameters"] = t.get("input_schema") or {"type": "object"}
            chat_tools.append({"type": "function", "function": fn})
        if chat_tools:
            chat["tools"] = chat_tools
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        kind = tc.get("type")
        if kind == "tool" and tc.get("name"):
            chat["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        elif kind in ("auto", "none"):
            chat["tool_choice"] = kind
        elif kind == "any":
            chat["tool_choice"] = "required"

    messages: list[dict] = []
    if system := body.get("system"):
        if isinstance(system, str):
            content = system
        else:
            content = _text_parts([b for b in system if isinstance(b, dict)
                                   and b.get("type") == "text"], keep_cache_control)
        if content:
            messages.append({"role": "system", "content": content})

    for m in body.get("messages") or []:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        text_blocks: list[dict] = []
        images: list[dict] = []
        tool_images: list[dict] = []          # images out of this turn's tool results
        tool_calls: list[dict] = []
        for block in content or []:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_blocks.append(block)
            elif btype == "image":
                images.append(_image_part(block))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""), "type": "function",
                    "function": {"name": block.get("name", ""),
                                 "arguments": json.dumps(block.get("input") or {},
                                                         ensure_ascii=False)},
                })
            elif btype == "tool_result":
                # A tool result is its own chat message and must precede whatever the
                # user wrote in the same turn — flush it right here to keep the order.
                messages.append({"role": "tool", "tool_call_id": block.get("tool_use_id", ""),
                                 "content": _tool_result_text(block)})
                if imgs := _tool_result_images(block):
                    tool_images.append({"type": "text", "text": "Image(s) returned by tool call "
                                        f"{block.get('tool_use_id', '')}:"})
                    tool_images.extend(imgs)
            elif btype in ("thinking", "redacted_thinking"):
                continue                      # Anthropic-only, inert for an OpenAI backend
            elif btype in ("document", "search_result"):
                raise UnsupportedContent(
                    f"content block '{btype}' has no chat-completions equivalent — "
                    "route this model to an Anthropic backend")
            else:
                logger.warning(f"anthropic bridge: dropping unknown content block '{btype}'")
        text_content = _text_parts(text_blocks, keep_cache_control)
        if images or tool_images:
            # Tool-result images lead: they answer the calls, the user's own text and
            # images follow as in the original turn.
            parts: list[dict] = list(tool_images)
            if isinstance(text_content, list):
                parts.extend(text_content)
            elif text_content:
                parts.append({"type": "text", "text": text_content})
            parts.extend(images)
            messages.append({"role": role, "content": parts})
        elif text_blocks or tool_calls:
            msg: dict = {"role": role, "content": text_content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
    chat["messages"] = messages
    return chat
