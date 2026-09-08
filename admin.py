"""Generation / management UI — a tabbed console mounted on the main app at /ui.

Plain server-rendered (no JS / no build / no extra deps). The nav shell (`TABS` +
`_page`) is the extension point. A small component layer (`_field` / `_btn` /
`_select`) keeps every form consistent — one label width, one input height, one
button size. POST bodies are parsed by hand (`parse_qs` / minimal multipart) to
avoid a `python-multipart` dependency.

Deferred (Phase 3 hardening): admin auth / multi-user, HTMX live validation.
"""
from __future__ import annotations

import asyncio
import base64
import html
import ipaddress
import json
import logging
import os
import re
import socket
import struct
import time
from typing import Callable, Optional
from urllib.parse import parse_qs, quote, urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

import adapters
import cloudtask
import jobs
import scheduler
import reasoning
import stats
import store

logger = logging.getLogger(__name__)


# Each cloud kind (Meshy, Tripo) is one FIXED endpoint — the form pre-fills it and
# backend_save accepts a blank url for those types. Derived from the modules, never
# a literal here: a new kind must not need a second edit in the console.
def _cloud_urls() -> dict:
    return {k: m.URL for k, m in adapters.CLOUD_MODULES.items()}


def _cloud_url_for(new_type: str, url: str) -> str:
    """The url a backend of type `new_type` is SAVED with. One fixed cloud endpoint per
    kind — nothing to type. Fill it when the field is blank, and REPLACE another cloud
    kind's fixed URL: switching an existing meshy backend to tripo would otherwise store
    api.meshy.ai on a Tripo backend, which surfaces only as an auth error at discovery,
    pointing at the wrong thing. A URL the operator typed himself (a self-hosted proxy)
    is left alone, and a non-cloud type is never touched. The form's JS does the same
    live; this is the authority, since the field is editable."""
    if new_type not in adapters.CLOUD_TYPES:
        return url
    curls = _cloud_urls()
    if not url or any(k != new_type and url == u for k, u in curls.items()):
        return curls.get(new_type, "")
    return url

_MODEL_EXTS = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".onnx")
_LOADER_HINTS = ("loader", "checkpoint", "unet", "clip", "vae", "lora", "gguf", "controlnet")

TABS = [
    ("dashboard", "Dashboard"), ("server", "Server"), ("backends", "Backends"),
    ("routing", "Input & Routing"), ("mapping", "Mapping"),
    ("reasoning", "Reasoning"),
    ("playground", "Playground"),
    ("jobs", "Jobs & Calls"),
    ("statistic", "Statistic"), ("users", "Users"),
]
DEFAULT_TAB = "dashboard"

# Sub-tabs: a top-level tab can group child views, picked via `?sub=<key>` on the
# parent route (first child = default). General pattern — future tab groupings
# register here; the parent page dispatches on `sub` and passes
# _page(..., subnav=_subnav(parent, sub)) so the bar renders under the header.
SUBTABS = {"playground": [("chat", "Chat"), ("media", "Media"), ("voice", "Voice")],
           "mapping": [("chat", "Chat"), ("media", "Media")],
           "jobs": [("llm", "LLM Calls"), ("media", "Media Jobs"), ("voice", "Voice Calls")],
           "routing": [("input", "Input"), ("chat", "Chat aliases"), ("llm", "LLM models"),
                       ("gen", "Media aliases"), ("image", "Image models"), ("loras", "LoRAs")]}


def _subnav(parent: str, active_sub: str) -> str:
    subs = SUBTABS.get(parent) or []
    links = "".join(f'<a class="{"on" if k == active_sub else ""}" '
                    f'href="/ui/{parent}?sub={k}">{_esc(lbl)}</a>' for k, lbl in subs)
    return f'<nav class="subnav">{links}</nav>'


# The bar renders OUTSIDE <main>, directly under the header (via _page(subnav=…)) —
# like the top tabs it never scrolls, and the body layout stays untouched.

# Request fields (the params that vary per generation) are NOT a fixed list — each
# alias defines its own by promoting Available fields to request fields in Mapping.
# The model stays a per-alias fixed choice (Pinned values), never a request field.


def _num(s: str):
    """Coerce a submitted string to int/float when it parses cleanly, else leave it
    a string (so free-text params like a sampler name survive untouched)."""
    s = s.strip()
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


# Injected main.py callables — ONE registry: the module-level defaults below ARE the
# catalog of bindable names; bind(**overrides) rebinds them by keyword (`foo=` sets
# `_foo`). No triple bookkeeping (signature + global stmt + assignments) to keep in sync.
_comfy_backends: Callable[[], list] = lambda: []
_gen_backends: Callable[[], list] = lambda: []      # every generation backend (ComfyUI, Meshy, Tripo)
_gateway_info: Callable[[], dict] = lambda: {}
# LAN scan (Backends tab → Scan network): start one, read the snapshot. Bound by main.
_scan_start: Callable[[], bool] = lambda: False
_scan_status: Callable[[], dict] = lambda: {"running": False, "findings": [], "cidrs": [],
                                            "ports": [], "hosts_total": 0, "hosts_done": 0,
                                            "truncated": False, "error": None, "no_range": False}
_gen_speed_info: Callable[[], dict] = lambda: {"speed": {}, "quarantine": {}}
_job_progress: Callable[[str], Optional[dict]] = lambda job_id: None   # live ws progress, None = none
_cancel_generation = None
_drain_backend = None
_cancel_drain = None
_restart_comfy = None
_set_backend_enabled = None
_apply_backends: Callable[[], None] = lambda: None
_llm_backends: Callable[[], list] = lambda: []
_config_chat_aliases: Callable[[], dict] = lambda: {}
_apply_chat_aliases: Callable[[], None] = lambda: None
_playground_key: Callable[[str], Optional[str]] = lambda name: None
_routing_snapshot: Callable[[], dict] = lambda: {"aliases": [], "models": [], "conflicts": []}
_server_info: Callable[[], dict] = lambda: {"effective": {}, "runtime": {}}
_apply_server_settings: Callable[[], None] = lambda: None
_apply_users: Callable[[], None] = lambda: None
_resolve_admin: Callable = lambda key: None
_ui_locked: Callable[[], bool] = lambda: False
_dashboard_snapshot: Callable[[], dict] = lambda: {}
_apply_reasoning: Callable[[], None] = lambda: None
_llm_backend_names: Callable[[], list] = lambda: []
# (alias_or_model, backend) → real model id; None = alias not mapped on that backend.
_resolve_for_backend: Callable[[str, str], Optional[str]] = lambda m, b: m
# Live reasoning probe: (backend, model, adapter, param, requested, prompt) → result dict.
_probe_reasoning: Callable = None
# Voice reference library (gateway blobs + scp ship to the TTS backend host).
_voice_lib_save: Callable = None          # async (name, data, ref_text) → status dict
_voice_lib_delete: Callable = None        # (name) → None
_voice_lib_ship: Callable = None          # async (name) → (ok, msg)
_voice_ship_config: Callable[[], tuple] = lambda: ([], "")   # → (hosts, dir)
_apply_voice_library: Callable[[], None] = lambda: None
_apply_hosts: Callable[[], None] = lambda: None           # refresh main's hosts_meta cache
# ComfyUI backend name → sorted installed LoRA filenames (discovery, verbatim).
_backend_loras: Callable[[], dict] = lambda: {}


def bind(**overrides) -> None:
    """Inject main.py's callables: bind(foo=fn) sets the module global `_foo`. Every
    keyword must match a `_`-prefixed default defined above — a typo raises instead
    of silently binding into the void."""
    for name, value in overrides.items():
        g = f"_{name}"
        if g not in globals():
            raise KeyError(f"admin.bind: unknown callback '{name}'")
        globals()[g] = value


# ── Shell + components ──────────────────────────────────────────────────────────

def _esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def _nav(active: str) -> str:
    links = "".join(f'<a class="{"on" if k == active else ""}" href="/ui/{k}">{_esc(label)}</a>'
                    for k, label in TABS)
    logout = ('<a href="/ui/logout" style="margin-left:auto;color:#8b97a4">Logout</a>'
              if _ui_locked() else "")
    return f'<header><span class="brand">AI-Hub</span><nav>{links}{logout}</nav></header>'


_CSS = """
*{box-sizing:border-box}
html,body{height:100%}
body{font:14px/1.6 system-ui,-apple-system,sans-serif;margin:0;background:#0f1115;color:#d7dbe0;overflow:hidden;display:flex;flex-direction:column}
a{color:#6cb0ef}
*{scrollbar-width:thin;scrollbar-color:#39414e transparent}
::-webkit-scrollbar{width:11px;height:11px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2d3440;border-radius:7px;border:2px solid #0f1115}
::-webkit-scrollbar-thumb:hover{background:#3d4654}
::-webkit-scrollbar-corner{background:transparent}
header{display:flex;align-items:center;background:#171a21;border-bottom:1px solid #272b33;padding:0 20px;flex:none}
.brand{font-weight:700;padding:14px 16px 14px 0;color:#e7ebf0}
nav{display:flex;flex-wrap:wrap}
nav a{color:#9aa7b4;padding:14px 14px;text-decoration:none;border-bottom:2px solid transparent;font-size:13px}
nav a:hover{color:#dce4ec;background:#1b1f27}
.subnav{display:flex;flex-wrap:wrap;gap:2px;padding:0 20px;background:#12151b;border-bottom:1px solid #272b33;flex:none}
.subnav a{color:#9aa7b4;padding:8px 12px;text-decoration:none;border-bottom:2px solid transparent;font-size:13px}
.subnav a:hover{color:#dce4ec;background:#1b1f27}
.subnav a.on{color:#fff;border-bottom-color:#3b82f6}
nav a.on{color:#fff;border-bottom-color:#3b82f6}
main{flex:1;min-height:0;overflow-y:auto;padding:18px 26px}
h2{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#7e8b99;margin:28px 0 12px;padding-bottom:7px;border-bottom:1px solid #242a33}
h2:first-child{margin-top:4px}
p.hint{color:#9aa7b4;margin:0 0 14px;max-width:62ch;line-height:1.5}
.field{display:flex;align-items:center;gap:14px;margin:10px 0}
.field>label{flex:0 0 140px;text-align:left;color:#9aa7b4;font-size:13px}
.field>.control{flex:1;min-width:0;max-width:480px;display:flex;gap:8px;align-items:center}
.field>.control.wide{max-width:none}
.control.short input,.control.short select{max-width:150px}
input,select,textarea{width:100%;background:#0c0e12;color:#dce4ec;border:1px solid #313a46;border-radius:7px;padding:0 10px;height:36px;font:inherit;outline:none}
input:focus,select:focus,textarea:focus{border-color:#3b82f6}
textarea{height:auto;min-height:150px;padding:8px 10px;font-family:ui-monospace,monospace;font-size:12px}
input[type=file]{padding:7px 10px;height:auto}
input[type=checkbox]{width:auto;height:auto;margin:0 6px 0 0;vertical-align:middle;accent-color:#2563eb}
.ckbox{display:inline-flex;align-items:center;margin-right:18px;color:#cdd5de;font-size:13px;cursor:pointer;white-space:nowrap}
.ckbox code{margin:0}
.btn{height:36px;padding:0 18px;background:#2563eb;color:#fff;border:0;border-radius:7px;font:inherit;font-weight:500;cursor:pointer;display:inline-flex;align-items:center;justify-content:center;text-decoration:none;white-space:nowrap}
.btn:hover{background:#1d4ed8}
.btn.secondary{background:#2a313c}.btn.secondary:hover{background:#343c49}
.btn.danger{background:#b4433f}.btn.danger:hover{background:#9e3a36}
.btn.sm{height:28px;padding:0 11px;font-size:12px;border-radius:6px}
.btn.icon{padding:0;width:32px;min-width:32px;font-size:15px}
.btn.sm.icon{width:28px;min-width:28px;font-size:14px}
.actions{display:flex;gap:10px;margin-top:18px;padding-left:0}
table{border-collapse:collapse;width:100%;margin:6px 0;font-size:13px}
th,td{border-bottom:1px solid #242a33;padding:8px 10px;text-align:left;vertical-align:middle;overflow-wrap:anywhere}
th{color:#7e8b99;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em;border-bottom-color:#313a46}
td .btn{margin-right:4px}
td input,td select{height:30px}
.muted{color:#6b7682}.ok{color:#5cb87f}.bad{color:#e06c6c}
code{background:#1b1f27;padding:2px 6px;border-radius:4px;font-size:12px}
.stub{color:#7e8b99;border:1px dashed #313a46;border-radius:8px;padding:22px;margin-top:8px;line-height:1.8}
img.result,video.result{max-width:512px;border:1px solid #313a46;border-radius:8px;margin:8px 0}
audio.result{width:512px;max-width:100%;margin:8px 0}
pre.err{white-space:pre-wrap;word-break:break-word;background:#1a1113;border:1px solid #5a2a2a;color:#f0b6b6;border-radius:8px;padding:12px 14px;margin:8px 0;max-height:340px;overflow:auto;font:12px/1.5 ui-monospace,monospace;user-select:text}
.cols{display:flex;gap:24px;align-items:flex-start}
.col{flex:1;min-width:0}
.cols{gap:0;align-items:stretch;height:100%}
.cols>.col{flex:1 1 0;min-width:0;height:100%;overflow-y:auto;padding:0 16px 18px 0}
.cols>.col+.col{border-left:1px solid #2a313c;margin-left:22px;padding-left:22px}
/* Mapping image editor: give the editor (col 2) more room for the request-fields
   table, taken from the Available fields column (col 3). */
.cols.map3>.col:nth-child(2){flex:1.9 1 0}
.cols.map3>.col:nth-child(3){flex:0.84 1 0}
.bar{display:flex;align-items:center;justify-content:space-between;gap:14px;margin:0 0 8px;padding:8px 0 8px;position:sticky;top:0;z-index:10;background:#0f1115}
.bar h2{margin:0;border:0;padding:0}
td.acts{white-space:nowrap;text-align:right;width:1%}
td.acts .btn{margin:0 0 0 6px}
tr.sel td{background:#19222e}
.item{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 4px;border-bottom:1px solid #242a33}
.item.sel{background:#19222e}
.item-main{min-width:0}
.item-title{font-weight:600;font-size:13px}
.item-sub{color:#6b7682;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item-acts{white-space:nowrap;flex:none}
.item-acts .btn{margin:0 0 0 6px}
.badge{font-size:11px;font-weight:500;padding:1px 7px;border-radius:10px;margin-left:7px;white-space:nowrap}
.badge.ok{background:#16361f;color:#5cb87f}.badge.bad{background:#3a1b1b;color:#e06c6c}.badge.muted{background:#23262d;color:#7e8b99}
.badge.warn{background:#3a2f12;color:#d8b35a}
.badge.llm{background:#13303a;color:#5fb8c8}.badge.img{background:#2a1d3a;color:#bb8ce6}
tr.grp td{background:#13161c;font-weight:600}
.grouphdr{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:#8b97a4;font-weight:600;margin:16px 0 6px;padding-bottom:4px;border-bottom:1px solid #242a33}
.grouphdr:first-child{margin-top:4px}
.formbar{display:flex;gap:8px;align-items:center;margin:0 0 14px;padding:8px 0 12px;border-bottom:1px solid #242a33;position:sticky;top:0;z-index:10;background:#0f1115}
.formbar h2{margin:0;border:0;padding:0;flex:1}
.formwrap{max-width:560px}
.avail tr td:last-child{text-align:right;white-space:nowrap}
.avail .node{color:#9aa7b4}.avail .node code{background:#13202f}
table.pins td:nth-child(2){width:62%}
table.pins td:last-child{width:1%;white-space:nowrap;text-align:right}
table.reqf tr[draggable]{cursor:grab}
table.reqf tr.dragging{opacity:.45}
table.reqf tr[draggable]:hover{background:#13202f}
table.reqf th:nth-child(3),table.reqf td:nth-child(3){width:72px}   /* node — just an id */
table.reqf th:nth-child(4),table.reqf td:nth-child(4){width:120px}  /* field */
.grip{color:#5a6675;cursor:grab;user-select:none;margin-right:4px}
.tag{font-size:10px;background:#1d3a52;color:#9fd0ff;border-radius:3px;padding:1px 5px;margin-left:4px;vertical-align:middle}
textarea{height:auto;min-height:60px;padding:8px 10px;line-height:1.5}
.chatout{white-space:pre-wrap;word-break:break-word;background:#0c0e12;border:1px solid #242a33;border-radius:8px;padding:12px 14px;user-select:text;font-size:13px}
.ok-banner{background:#16361f;color:#5cb87f;border:1px solid #1f5232;border-radius:8px;padding:8px 12px;margin:8px 0}
.ok-banner.fade{animation:okfade 2.2s ease forwards}
@keyframes okfade{0%,65%{opacity:1}100%{opacity:0;visibility:hidden}}
/* Save confirmation that lives INSIDE the sticky form bar: a banner stacked above the
   form would push the whole editor down by its height, so the restored scroll position
   lands on shifted content — the very jump the scroll restore exists to prevent. */
.ok-chip{color:#5cb87f;font-size:12px;white-space:nowrap}
.acctbl{max-height:360px;overflow-y:auto;border:1px solid #242a33;border-radius:8px}
.acctbl table{margin:0}
.acctbl td:first-child,.acctbl th:first-child{width:1%;text-align:center}
.acctbl thead th{position:sticky;top:0;background:#13161c;z-index:1}
.cards{display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 6px}
.card{background:#13161c;border:1px solid #242a33;border-radius:10px;padding:12px 18px;min-width:130px}
.card .cnum{font-size:22px;font-weight:600;color:#e8edf2}
.card .clbl{font-size:12px;color:#8b97a4;margin-top:2px}
table.recent{font-size:12px}
table.sortable th{cursor:pointer;user-select:none}
table.sortable th:hover{color:#dfe6ee}
table.sortable th .sind{margin-left:4px;color:#5fb8c8;font-size:10px}
details.optblock{border:1px solid #242a33;border-radius:8px;padding:6px 10px;margin:0 0 12px}
details.optblock>summary{cursor:pointer;user-select:none;font-size:12px;color:#8b97a4;padding:2px 0}
details.optblock>summary:hover{color:#cdd6e0}
details.optblock[open]>summary{margin-bottom:8px;border-bottom:1px solid #1c2129;padding-bottom:6px}
"""


# Preserve each scrolling pane's position across the 303-redirect reloads that every
# inline edit action triggers — otherwise the column jumps back to the top on each
# change. Stored in sessionStorage (survives the reload, not a fresh visit) under ONE
# KEY PER PANE, because the panes differ in what makes a position stale:
#   · the master list (first `.col`) is the SAME list under every `?edit=…` → keyed on
#     the path alone, so picking another entry leaves the list where it was;
#   · every other pane is the selection's detail → keyed on the query too, so alias A's
#     editor position never lands on alias B's editor.
# `saved=1` is stripped from that query: it is a transient banner flag the Save redirect
# appends, and keying on it made every Save read as a fresh URL — the whole point of
# Save being the ONE action that must not move the pane you were working in.
# `<main>` is tracked as a pane too, and stays tracked under the live morph. It is the
# page's scroll container, not the window: `body{overflow:hidden}` + `main{overflow-y:
# auto}` mean `<main>.scrollTop` IS the page scroll. What this block preserves is that
# position across REAL navigation — F5, a nav link back to a long list (Media Jobs, LLM
# Calls, the LoRA list), a form POST's 303 — none of which the morph covers. It does
# NOT exist for the live update any more: the morph patches `<main>`'s children and
# never replaces the container, so the position is simply never lost there and the
# restore, which runs once at load, is inert from the first tick onward. The browser's
# own scroll restoration is no substitute: it applies to history navigation (Back /
# Forward), not to F5 or to re-clicking the same nav link.
_SCROLL_JS = ("<script>(function(){"
              "var q=location.search.replace(/([?&])saved=[^&]*&?/,'$1').replace(/[?&]$/,'');"
              "var b='scr:'+location.pathname;"
              "function t(){var o=[],m=document.querySelector('main');"
              "if(m)o.push([m,b+q+'|main']);"
              "[].slice.call(document.querySelectorAll('.col')).forEach(function(e,j){"
              "o.push([e,j===0?b+'|master':b+q+'|c'+j]);});return o;}"
              "try{t().forEach(function(p){var v=sessionStorage.getItem(p[1]);"
              "if(v!=null)p[0].scrollTop=+v;});}catch(e){}"
              "var d=false;function save(){if(d)return;d=true;requestAnimationFrame(function(){d=false;"
              "try{t().forEach(function(p){sessionStorage.setItem(p[1],p[0].scrollTop);});}catch(e){}});}"
              "t().forEach(function(p){p[0].addEventListener('scroll',save);});"
              "window.addEventListener('beforeunload',save);"
              "})();</script>")


# Click-a-header to sort any `table.sortable` (numeric-aware: a cell that is a plain
# number sorts numerically, otherwise lexically). The choice persists per table in
# sessionStorage and is re-applied on load — so it survives the dashboard's 4s
# auto-refresh. A gwLiveHooks entry re-applies it after every live morph too:
# the server renders rows in insertion order and the morph re-imposes that order
# on the live DOM, so without the hook a clicked sort is undone on every tick.
# The hook re-reads sessionStorage rather than closing over the state — that
# store is the source of truth and stays it. For the same reason no call site
# captures the storage KEY STRING: the morph reuses an unkeyed table node for a
# different logical table when a dashboard panel appears or disappears, which
# rewrites its data-sk while the header cells keep their old handlers. Every
# read and write therefore derives 'sort:'+key(tbl,i) at the moment it acts, so a
# data-sk table's writer and reader always agree on the node's LIVE data-sk.
# That guarantee stops at data-sk: the `i` fallback (path+index) IS captured —
# wire()'s click handler closes over the load-time index while the post-morph hook
# passes a freshly recomputed one — so a table without data-sk that changes position
# can read under one key and write under another. Latent only: every current
# `table.sortable` carries a data-sk, which is why the fallback is a fallback.
# Tables with `data-sk` get a stable key; others key off path+index.
# The hook also calls `wire()` on every sortable table, not just `applySort()`: a
# table the morph INSERTS mid-session (a dashboard panel appearing) never went
# through the load-time loop and would show a restored sort while silently ignoring
# clicks until the next real reload — a regression the morph introduces, since before
# it every refresh was a reload that re-bound everything. `wire()` is additive and
# marks the node with the JS property `__gwWired` (a property, not an attribute:
# syncAttrs would otherwise strip or re-add it on every tick, and it must vanish with
# the node). That marker only short-circuits the BINDING; the hook still re-applies
# the sort to already-wired tables, which is the whole point — the reconciler puts
# the server's insertion order back on every morph.
# Grouped tables (a `tr.grp` header row followed by its member rows — the routing
# views) sort as BLOCKS, so a group never gets torn apart: the group row supplies the
# key for column 0 (it is the alias name), later columns key off the first member row.
_SORT_JS = ("<script>(function(){"
            "function num(td){var t=(td.textContent||'').trim().replace(/[$,\\s]/g,'');"
            "return /^-?\\d+(\\.\\d+)?$/.test(t)?parseFloat(t):null;}"
            "function ind(th,a){var s=th.querySelector('.sind');"
            "if(!s){s=document.createElement('span');s.className='sind';th.appendChild(s);}"
            "s.textContent=a||'';}"
            "function blocks(tbl,hdr){var out=[],cur=null;"
            "[].slice.call(tbl.rows).forEach(function(r){if(r===hdr||r.querySelector('th'))return;"
            "if(/(^|\\s)grp(\\s|$)/.test(r.className)){cur={rows:[r],grp:r};out.push(cur);}"
            "else if(cur){cur.rows.push(r);}else{out.push({rows:[r],grp:null});}});return out;}"
            "function cellOf(b,idx){if(b.grp&&idx===0)return b.grp.cells[0];"
            "for(var i=0;i<b.rows.length;i++){var r=b.rows[i];"
            "if(r!==b.grp&&r.cells[idx])return r.cells[idx];}"
            "return b.grp?b.grp.cells[0]:null;}"
            "function sortIt(tbl,idx,dir){var hdr=tbl.rows[0];var bs=blocks(tbl,hdr);"
            "bs.sort(function(a,b){var x=cellOf(a,idx),y=cellOf(b,idx);if(!x||!y)return 0;"
            "var nx=num(x),ny=num(y),r;if(nx!==null&&ny!==null)r=nx-ny;"
            "else r=(x.textContent||'').trim().toLowerCase().localeCompare((y.textContent||'').trim().toLowerCase());"
            "return dir<0?-r:r;});"
            "var tb=tbl.tBodies[0]||tbl;"
            "bs.forEach(function(b){b.rows.forEach(function(r){tb.appendChild(r);});});"
            "var hs=hdr.cells;for(var i=0;i<hs.length;i++)ind(hs[i],i===idx?(dir<0?'\\u25bc':'\\u25b2'):'');}"
            "function key(tbl,i){return tbl.getAttribute('data-sk')||(location.pathname+'#'+i);}"
            "function applySort(tbl,i){var hdr=tbl.rows[0];if(!hdr)return;var s={};"
            "try{s=JSON.parse(sessionStorage.getItem('sort:'+key(tbl,i))||'{}');}catch(e){}"
            "if(s.idx!=null)sortIt(tbl,s.idx,s.dir||1);}"
            "function wire(tbl,i){var hdr=tbl.rows[0];if(!hdr)return;"
            "if(tbl.__gwWired)return;tbl.__gwWired=true;"
            "[].forEach.call(hdr.cells,function(th,idx){th.addEventListener('click',function(){"
            "var k='sort:'+key(tbl,i),c={};"
            "try{c=JSON.parse(sessionStorage.getItem(k)||'{}');}catch(e){}"
            "var dir=(c.idx===idx&&c.dir>0)?-1:1;sortIt(tbl,idx,dir);"
            "try{sessionStorage.setItem(k,JSON.stringify({idx:idx,dir:dir}));}catch(e){}});});"
            "applySort(tbl,i);}"
            "[].slice.call(document.querySelectorAll('table.sortable')).forEach(function(tbl,i){"
            "wire(tbl,i);});"
            "window.gwLiveHooks=window.gwLiveHooks||[];"
            "window.gwLiveHooks.push(function(){"
            "[].slice.call(document.querySelectorAll('table.sortable')).forEach(function(tbl,i){"
            "wire(tbl,i);applySort(tbl,i);});});"
            "})();</script>")


# One auto-update mechanism for the whole console. `_page(refresh=N)` marks <main>
# with data-live=N; this poller re-fetches the SAME url and morphs the response's
# <main> into the live one instead of reloading the page. Nodes are matched by id,
# data-k or data-sk (falling back to position+tag), so a table that gains a row keeps
# every other row's identity — which is what preserves scroll, sort order, a focused
# filter input, an open form, playing media and the model-viewer's camera.
# `data-sk` counts as a key because it is the stable, document-unique NAME of a
# sortable table (dash-backends, dash-jobs, …). The dashboard renders three of its
# four tables conditionally; without that key the morph would keep using table node N
# for a DIFFERENT logical table when a panel appears or disappears — syncAttrs would
# rewrite the data-sk while everything attached to the node (handlers, the sort
# indicator, its wired marker) stayed behind from the table it used to be.
# Five things are deliberately never touched: <script> (a re-inserted _JOB_TICK
# would double its setInterval), [data-live-skip] subtrees, form controls that are
# focused or dirty, media whose src is unchanged, and <details open> (user state the
# server knows nothing about). A response without data-live stops the poller — the
# same signal the meta tag's absence used to carry.
# The <script> rule cuts BOTH ways, and the second way bites: `adopt()` also drops a
# script it would otherwise INSERT, so markup that only appears in a LATER state of a
# live page arrives without its script and silently never initialises (a <model-viewer>
# that never upgrades, a viewer div that stays black), and data-live is usually gone in
# that same response, so the poller stops and it cannot self-heal. Hence the invariant
# every live page owes: it must ALREADY contain every <script> any later state of it
# can render — hoist them (see job_detail_page, _playground_body) and, where the script
# has to act on nodes that arrive later, register the action in window.gwLiveHooks.
_LIVE_JS = ("<script>(function(){"
            "var main=document.querySelector('main');"
            "if(!main)return;"
            "window.gwLiveHooks=window.gwLiveHooks||[];"
            "var base=parseInt(main.getAttribute('data-live')||'0',10)*1000;"
            "if(!(base>0))return;"
            "var MEDIA={IMG:1,VIDEO:1,AUDIO:1,IFRAME:1,SOURCE:1,'MODEL-VIEWER':1};"
            "var FORM={INPUT:1,TEXTAREA:1,SELECT:1};"
            "var seq=0;"
            "function keyOf(n){if(n.nodeType!==1)return null;"
            "return n.id||n.getAttribute('data-k')||n.getAttribute('data-sk')||null;}"
            "function dirty(e){if(!FORM[e.tagName])return false;"
            "if(document.activeElement===e)return true;"
            "if(e.tagName==='SELECT'){var sel=e.querySelector('option[selected]');"
            "var def=sel?sel.value:(e.options[0]?e.options[0].value:'');"
            "return e.value!==def;}"
            "if(e.type==='checkbox'||e.type==='radio')return e.checked!==e.defaultChecked;"
            "return e.value!==e.defaultValue;}"
            "function frozen(e){return e.tagName==='SCRIPT'||e.hasAttribute('data-live-skip')||dirty(e);}"
            "function adopt(n){var c=document.importNode(n,true);"
            "if(c.nodeType===1){var s=c.querySelectorAll?c.querySelectorAll('script'):[];"
            "for(var i=0;i<s.length;i++)s[i].parentNode.removeChild(s[i]);"
            "if(c.tagName==='SCRIPT')return document.createComment('gw-script-skipped');}"
            "return c;}"
            "function syncAttrs(o,n){"
            "if(MEDIA[o.tagName]&&o.getAttribute('src')===n.getAttribute('src'))return;"
            "var i,a;"
            "for(i=n.attributes.length-1;i>=0;i--){a=n.attributes[i];"
            "if(o.tagName==='DETAILS'&&a.name==='open')continue;"
            "if(o.getAttribute(a.name)!==a.value)o.setAttribute(a.name,a.value);}"
            "for(i=o.attributes.length-1;i>=0;i--){a=o.attributes[i];"
            "if(o.tagName==='DETAILS'&&a.name==='open')continue;"
            "if(!n.hasAttribute(a.name))o.removeAttribute(a.name);}"
            "if(o.tagName==='INPUT'){"
            "if(o.type==='checkbox'||o.type==='radio')o.checked=n.hasAttribute('checked');"
            "else o.value=n.getAttribute('value')||'';}"
            "else if(o.tagName==='TEXTAREA'){o.value=n.textContent;}}"
            "function same(o,n){if(o.nodeType!==n.nodeType)return false;"
            "if(o.nodeType!==1)return true;"
            "if(o.tagName!==n.tagName)return false;"
            "var ko=keyOf(o),kn=keyOf(n);"
            "if(ko||kn)return ko===kn;"
            "return true;}"
            "function morph(o,n){"
            "if(o.nodeType===3||o.nodeType===8){if(o.data!==n.data)o.data=n.data;return;}"
            "if(o.nodeType!==1)return;"
            "if(frozen(o))return;"
            "syncAttrs(o,n);"
            "if(MEDIA[o.tagName])return;"
            "reconcile(o,n);}"
            "function reconcile(o,n){var pool={},unkeyed=[],c,k,i;"
            "for(c=o.firstChild;c;c=c.nextSibling){k=keyOf(c);"
            "if(k)pool['#'+k]=c;else unkeyed.push(c);}"
            "var out=[],ui=0,m;"
            "for(c=n.firstChild;c;c=c.nextSibling){k=keyOf(c);m=null;"
            "if(k){m=pool['#'+k]||null;"
            "if(m&&m.tagName!==c.tagName)m=null;"
            "else if(m)pool['#'+k]=null;}"
            "else{while(ui<unkeyed.length&&!same(unkeyed[ui],c))ui++;"
            "if(ui<unkeyed.length){m=unkeyed[ui];ui++;}}"
            "if(m){morph(m,c);out.push(m);}"
            "else out.push(adopt(c));}"
            "var stamp=++seq,nx;"
            "for(i=0;i<out.length;i++)out[i].__gwLive=stamp;"
            "c=o.firstChild;"
            "while(c){nx=c.nextSibling;"
            "if(c.__gwLive!==stamp)o.removeChild(c);"
            "c=nx;}"
            "var cur=o.firstChild;"
            "for(i=0;i<out.length;i++){if(cur===out[i])cur=cur.nextSibling;"
            "else o.insertBefore(out[i],cur);}}"
            "var wait=base,timer=null,catchUp=false;"
            "function schedule(ms){if(timer)clearTimeout(timer);timer=setTimeout(tick,ms);}"
            "function stop(){if(timer)clearTimeout(timer);timer=null;}"
            "function tick(){if(document.hidden){catchUp=true;schedule(wait);return;}"
            "fetch(location.href,{cache:'no-store',credentials:'same-origin'})"
            ".then(function(r){"
            "if(r.redirected&&new URL(r.url).pathname!==location.pathname){"
            "stop();location.href=r.url;return null;}"
            "if(!r.ok){wait=Math.min(wait*2,30000);schedule(wait);return null;}"
            "return r.text();})"
            ".then(function(html){"
            "if(html===null||html===undefined)return;"
            "var doc=new DOMParser().parseFromString(html,'text/html');"
            "var nm=doc.querySelector('main');"
            "if(!nm){stop();return;}"
            "try{morph(main,nm);}"
            "catch(e){main.replaceChildren.apply(main,[].slice.call(nm.childNodes).map(adopt));"
            "if(nm.hasAttribute('data-live'))main.setAttribute('data-live',nm.getAttribute('data-live'));"
            "else main.removeAttribute('data-live');}"
            "for(var i=0;i<window.gwLiveHooks.length;i++){"
            "try{window.gwLiveHooks[i]();}catch(e){}}"
            "var next=parseInt(main.getAttribute('data-live')||'0',10)*1000;"
            "if(!(next>0)){stop();return;}"
            "wait=base=next;"
            "schedule(wait);})"
            ".catch(function(){wait=Math.min(wait*2,30000);schedule(wait);});}"
            "document.addEventListener('visibilitychange',function(){"
            "if(!document.hidden&&catchUp&&timer){catchUp=false;schedule(0);}});"
            "schedule(wait);"
            "})();</script>")


def _page(title: str, body: str, active: str = "", refresh: Optional[int] = None,
          nologin: bool = False, subnav: str = "") -> str:
    # `refresh` no longer reloads the page. It marks <main> as live and _LIVE_JS
    # polls this same URL and MORPHS the new <main> into the old one — the container
    # is never replaced, so scroll position, sort order, a half-typed filter, an
    # open form, playing video and the model-viewer's camera all survive an update.
    # A response without data-live stops the poller, which is exactly what dropping
    # the meta tag used to mean.
    live = f' data-live="{int(refresh)}"' if refresh else ""
    head = "" if nologin else _nav(active)        # login page renders without the nav
    # subnav (see SUBTABS) renders as a second header row — outside <main>, so it
    # never scrolls and sits flush under the tabs.
    return (f'<!doctype html><html><head><meta charset="utf-8"><title>{_esc(title)} · AI-Hub</title>'
            f"<style>{_CSS}</style></head><body>{head}{subnav}<main{live}>{body}</main>"
            f"{_SCROLL_JS}{_SORT_JS}{_LIVE_JS}</body></html>")


def _field(label: str, control: str, short: bool = False, wide: bool = False) -> str:
    cls = "control short" if short else ("control wide" if wide else "control")
    return f'<div class="field"><label>{_esc(label)}</label><div class="{cls}">{control}</div></div>'


def _btn(label: str, href: str = "", kind: str = "", sm: bool = False, submit: bool = False,
         confirm: str = "", title: str = "", icon: bool = False) -> str:
    cls = "btn" + (f" {kind}" if kind else "") + (" sm" if sm else "") + (" icon" if icon else "")
    t = f' title="{_esc(title)}"' if title else ""
    onclick = f' onclick="return confirm(\'{_esc(confirm)}\')"' if confirm else ""
    if submit:
        return f'<button type="submit" class="{cls}"{t}>{_esc(label)}</button>'
    return f'<a class="{cls}" href="{_esc(href)}"{onclick}{t}>{_esc(label)}</a>'


def _icon_acts(*specs) -> str:
    """Compact icon action buttons. Each spec: (glyph, href, kind, title[, confirm])."""
    out = ""
    for s in specs:
        glyph, href, kind, ttl = s[0], s[1], s[2], s[3]
        confirm = s[4] if len(s) > 4 else ""
        out += _btn(glyph, href, kind, sm=True, icon=True, title=ttl, confirm=confirm)
    return out


def _select(name: str, options: list, selected=None) -> str:
    opts = ""
    for o in options:
        val, lbl = (o if isinstance(o, tuple) else (o, o))
        opts += f'<option value="{_esc(val)}"{" selected" if str(val) == str(selected) else ""}>{_esc(lbl)}</option>'
    return f'<select name="{_esc(name)}">{opts}</select>'


_TASK_OPTIONS = ("text2img", "img2img", "text2vid", "img2vid", "img2mesh", "mesh2mesh",
                 "mesh2rig", "text2audio")
# JS that shows the Video section (fps/frames) only for a video task, driven by the
# task dropdown; matched loosely so img2vid / img2video / text2video all count.
_TASK_VIDEO_JS = ("var v=document.getElementById('gw-video');"
                  "if(v){var f=v.querySelector('input[name=fps]');"
                  "v.style.display=(/vid/i.test(this.value)||(f&&f.value))?'':'none';}")


def _task_select(current: str = "text2img", onchange: str = "") -> str:
    """The `task` dropdown. Routing is convention-free (task is a label/hint), but the
    editor keys field visibility on it (video → fps/frames). Preserves an unlisted
    stored value so a custom task is never silently dropped."""
    cur = (current or "text2img").strip()
    opts = list(_TASK_OPTIONS) + ([cur] if cur and cur not in _TASK_OPTIONS else [])
    body = "".join(f'<option value="{_esc(o)}"{" selected" if o == cur else ""}>{_esc(o)}</option>'
                   for o in opts)
    oc = f' onchange="{onchange}"' if onchange else ""
    return f'<select name="task"{oc}>{body}</select>'


def _inp(name: str, value="", placeholder: str = "", typ: str = "text", step: str = "") -> str:
    # `step` only matters for type=number: without step="any" a browser rejects
    # decimals like 0.85 (the implicit step is 1).
    st = f' step="{_esc(step)}"' if step else ""
    return (f'<input type="{typ}" name="{_esc(name)}" value="{_esc(value)}" '
            f'placeholder="{_esc(placeholder)}"{st}>')


def _textarea(name: str, value="", rows: int = 3, placeholder: str = "") -> str:
    return (f'<textarea name="{_esc(name)}" rows="{rows}" placeholder="{_esc(placeholder)}">'
            f'{_esc(value)}</textarea>')


def _checkbox(name: str, checked: bool, label: str, title: str = "") -> str:
    ck = " checked" if checked else ""
    t = f' title="{_esc(title)}"' if title else ""
    return (f'<label class="ckbox"{t}><input type="checkbox" name="{_esc(name)}" value="1"{ck}> '
            f'{_esc(label)}</label>')


def _item(title: str, sub: str, acts: str, sel: bool = False) -> str:
    """A compact two-line list item: bold title (may contain a badge) + a muted,
    single-line truncated sub line, with right-aligned actions."""
    return (f'<div class="item{" sel" if sel else ""}"><div class="item-main">'
            f'<div class="item-title">{title}</div>'
            f'<div class="item-sub" title="{_esc(sub)}">{_esc(sub)}</div></div>'
            f'<div class="item-acts">{acts}</div></div>')


def _badge(text: str, kind: str = "muted", title: str = "") -> str:
    t = f' title="{_esc(title)}"' if title else ""
    return f'<span class="badge {kind}"{t}>{_esc(text)}</span>'


# What a failed discovery poll looked like → the badge that says what to DO about it.
# A rejected credential and an unplugged host both used to read "down", which sends
# you debugging the network when the fix is one field in this very form.
_DOWN_BADGE = {
    "auth": ("🔑 token invalid", "Backend rejected the credential — check the api key "
                                 "field (Anthropic tokens start with sk-ant-oat01- or sk-ant-api03-)"),
    "not_found": ("✖ endpoint not found", "The URL answered 404 — check it: it must be the base "
                                          "URL without /v1, which the gateway appends itself"),
    "rate_limit": ("⏱ rate limited", "The backend is rate-limiting discovery (429) — usually "
                                     "temporary; the next poll retries"),
    "upstream": ("⚠ upstream error", "The backend answered with a server error (5xx)"),
    "unreachable": ("⇥ unreachable", "No TCP connection — host down, wrong port, or firewalled"),
    "timeout": ("⏱ timeout", "The backend accepted the connection but did not answer in time"),
    "stuck": ("⚠ executor stuck", "Answers HTTP but is not draining its queue"),
}


def _down_badge(err: Optional[dict]) -> str:
    """Status chip for an unhealthy backend, naming the actual cause."""
    if not err:
        return _badge("down", "bad")
    label, hint = _DOWN_BADGE.get(err.get("kind") or "", ("down", ""))
    status = err.get("status")
    detail = err.get("detail") or ""
    since = err.get("since")
    parts = [hint] if hint else []
    if status:
        parts.append(f"HTTP {status}")
    if since:
        parts.append(f"for {_age(int(since))}")
    if detail:
        parts.append(detail)
    return _badge(label, "bad", " · ".join(p for p in parts if p))


_MODELVIEWER_SRC = "/ui/static/model-viewer.min.js"   # bundled locally (no CDN); served by static_asset


def _dl_card(src: str, label: str = "download", *, dl: str = " download",
             mime: str = "", compact: bool = False, cls: str = "") -> str:
    """A download anchor for a non-inline job artifact. `compact` = the small card shown
    under a 3D viewer (GLB/FBX); default = a standalone card. `mime` adds a type hint;
    `dl` is the download attribute (a suggested filename, or bare `download`)."""
    c = f' class="{cls}"' if cls else ""
    hint = f" <span class='muted'>({_esc(mime)})</span>" if mime else ""
    pad = "margin-top:6px;padding:8px 12px" if compact else "padding:18px 22px"
    return (f'<a{c} href="{src}" target="_blank"{dl} style="display:inline-block;'
            f'{pad};{_BOX_STYLE};text-decoration:none">⬇ {_esc(label)}{hint}</a>')


def _media_tag(src: str, mime: str = "", kind: str = "", cls: str = "",
               style: str = "", autoplay: bool = False) -> str:
    """Right media element for a generation artifact: <video> for video, <audio>
    for audio, <model-viewer> for glTF/GLB, else <img>. The serving route sets the
    real content-type; mime/kind here only pick the tag (unknown → <img>). `src`
    must already be escaped."""
    m, k = (mime or "").lower(), (kind or "").lower()
    c = f' class="{cls}"' if cls else ""
    s = f' style="{style}"' if style else ""
    if k == "video" or m.startswith("video/"):
        ap = " autoplay" if autoplay else ""
        return f'<video{c}{s} src="{src}" controls loop muted playsinline preload="metadata"{ap}></video>'
    if k == "audio" or m.startswith("audio/"):
        return f'<audio{c}{s} src="{src}" controls preload="metadata"></audio>'
    if m in ("model/gltf-binary", "model/gltf+json"):
        # <model-viewer> from the locally-bundled ES module (module URLs load once
        # even if the tag repeats). Needs an explicit box or it collapses.
        box = style or "width:720px;max-width:100%;height:640px"
        box += ";background:#0c0e12;border:1px solid #313a46;border-radius:10px"
        head = f'<script type="module" src="{_MODELVIEWER_SRC}"></script>'
        if autoplay and m == "model/gltf-binary":
            # inspection view: play an injected idle so bad skin weights show (a spike
            # shoots out, the crotch webs). A toggle pauses back to bind pose. No
            # auto-rotate here — you want to watch the deformation, not the camera.
            sep = "&" if "?" in src else "?"
            mvid = "mv%d" % (abs(hash(src)) % 1000000)
            return (head + f'<model-viewer id="{mvid}"{c} style="{box}" src="{src}{sep}anim=idle" '
                    f'camera-controls autoplay animation-name="idle" shadow-intensity="1" '
                    f'interaction-prompt="none" ar-status="not-presenting"></model-viewer>'
                    f'<label class="ckbox" style="margin-top:6px;display:block"><input type="checkbox" '
                    f'checked onchange="var m=document.getElementById(\'{mvid}\');'
                    f'this.checked?m.play():m.pause();"> idle animation '
                    f'<span class="muted">— reveals bad skin weights (spikes / crotch ring)</span></label>')
        return (head + f'<model-viewer{c} style="{box}" src="{src}" camera-controls auto-rotate '
                f'shadow-intensity="1" interaction-prompt="none" '
                f'ar-status="not-presenting"></model-viewer>')
    if k == "file":                                   # non-previewable artifact (fbx/obj/…) → download
        return _dl_card(src, cls=cls)
    return f'<img{c}{s} src="{src}">'


def _glb_stats(path: str) -> Optional[dict]:
    """Cheap mesh stats from a GLB's JSON chunk (header-only read, never the whole
    file): vertex/triangle counts, vertex-color/texture presence, bounding-box dims.
    Any parse problem returns None — stats are decoration, never an error."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            magic, _ver, _total = struct.unpack("<4sII", fh.read(12))
            if magic != b"glTF":
                return None
            clen, ctype = struct.unpack("<I4s", fh.read(8))
            if ctype != b"JSON" or clen > 64 * 1024 * 1024:
                return None
            doc = json.loads(fh.read(clen))
        acc = doc.get("accessors", [])

        def _count(i):
            return acc[i].get("count", 0) if isinstance(i, int) and 0 <= i < len(acc) else 0

        verts = tris = 0
        colors = False
        lo: list = [None] * 3
        hi: list = [None] * 3
        for m in doc.get("meshes", []):
            for p in m.get("primitives", []):
                attrs = p.get("attributes", {})
                pa = attrs.get("POSITION")
                n = _count(pa)
                verts += n
                colors = colors or any(k.startswith("COLOR_") for k in attrs)
                if p.get("mode", 4) == 4:                       # triangles only
                    tris += (_count(p["indices"]) if isinstance(p.get("indices"), int) else n) // 3
                a = acc[pa] if isinstance(pa, int) and 0 <= pa < len(acc) else {}
                amin, amax = a.get("min"), a.get("max")
                if isinstance(amin, list) and isinstance(amax, list) and len(amin) == 3 == len(amax):
                    for i in range(3):
                        lo[i] = amin[i] if lo[i] is None else min(lo[i], amin[i])
                        hi[i] = amax[i] if hi[i] is None else max(hi[i], amax[i])
        dims = tuple(hi[i] - lo[i] for i in range(3)) if lo[0] is not None else None
        return {"size": size, "vertices": verts, "triangles": tris,
                "colors": colors, "textures": len(doc.get("images", [])), "dims": dims}
    except Exception:
        return None


def _glb_stats_html(path: Optional[str]) -> str:
    """One muted line under a GLB viewer: vertices · triangles · colors · size."""
    st = _glb_stats(path) if path else None
    if not st:
        return ""
    parts = [f"{st['vertices']:,} vertices", f"{st['triangles']:,} triangles"]
    if st["textures"]:
        parts.append(f"{st['textures']} texture{'s' if st['textures'] != 1 else ''}")
    if st["colors"]:
        parts.append("vertex colors")
    if not st["textures"] and not st["colors"]:
        parts.append("no color data")
    if st["dims"]:
        parts.append("bbox " + " × ".join(f"{d:.2f}" for d in st["dims"]))
    parts.append(f"{st['size'] / 1048576:.1f} MB")
    return f"<p class='muted' style='margin:4px 0'>{_esc(' · '.join(parts))}</p>"


def _type_badge(t: str) -> str:
    """Color-coded chip for a backend's protocol type, so the two kinds stand out at
    a glance in mixed lists (LLM and image-generation backends share these tables)."""
    t = (t or "openai").lower()
    if t == "comfyui":
        return _badge("🖼 comfyui", "img", "image-generation backend (ComfyUI)")
    if t in adapters.CLOUD_TYPES:
        mod = adapters.cloud_module(t)
        return _badge(f"☁ {t}", "img", f"{mod.VENDOR} cloud mesh generation (paid, per task)")
    if t == "anthropic":
        # plain glyph on purpose: the enclosed-A (🅐) renders as a blank box in the
        # console's system font stack
        return _badge("✳ anthropic", "warn", "Anthropic backend — reachable via /v1/messages only")
    return _badge(f"💬 {t}", "llm", "LLM backend (OpenAI-compatible)")


def _qp(request: Request, key: str, default: str = "") -> str:
    return request.query_params.get(key, default)


async def root_redirect():
    return RedirectResponse(f"/ui/{DEFAULT_TAB}", status_code=307)


async def ui_root():
    return RedirectResponse(f"/ui/{DEFAULT_TAB}", status_code=307)


def _inactive():
    return HTMLResponse(_page("UI", '<p class="hint">The generation store is not enabled. '
        'Set <code>image_models</code> or <code>jobs.enabled: true</code> in config.</p>'))


# ── Admin login / session guard (Phase 3 hardening) ──────────────────────────────
# /ui requires an admin login once an admin credential exists (master api_key or an
# admin user). Bootstrap-open until then. Session = a store-signed (encrypt-then-MAC)
# cookie carrying the user name + expiry; tamper-proof, no server-side session store.

_SESSION_COOKIE = "gw_session"
_SESSION_TTL = 12 * 3600


def _make_session(name: str) -> str:
    return store.encrypt_secret(json.dumps({"u": name, "exp": int(time.time()) + _SESSION_TTL}))


def _session_user(request: Request) -> Optional[str]:
    tok = request.cookies.get(_SESSION_COOKIE)
    if not tok:
        return None
    try:
        d = json.loads(store.decrypt_secret(tok) or "{}")
        return d["u"] if d.get("exp", 0) > int(time.time()) else None
    except Exception:
        return None


async def _ui_guard(request: Request, call_next):
    """Block /ui (except the login routes) without a valid admin session — but only
    once the gateway is locked (an admin credential exists)."""
    p = request.url.path
    if p.startswith("/ui") and not p.startswith("/ui/login") and _ui_locked():
        if not _session_user(request):
            return RedirectResponse(f"/ui/login?next={quote(p)}", status_code=303)
    return await call_next(request)


def _login_page(error: str = "", nxt: str = "/ui") -> str:
    err = f"<p class='bad'>{_esc(error)}</p>" if error else ""
    body = (f"<div style='max-width:360px;margin:8vh auto;text-align:left'>"
            f"<h2>AI-Hub login</h2><p class='hint'>Enter an <b>admin API key</b> to access the console.</p>"
            f"{err}<form action='/ui/login' method='post'>"
            f"<input type='hidden' name='next' value='{_esc(nxt)}'>"
            f"{_field('admin key', _inp('key', '', placeholder='Bearer token', typ='password'))}"
            f"<div class='field'><label></label><div class='control'>{_btn('Sign in', submit=True)}</div></div>"
            f"</form></div>")
    return body


async def login_page(request: Request):
    if not _ui_locked():
        return RedirectResponse("/ui", status_code=303)
    nxt = request.query_params.get("next", "/ui")
    return HTMLResponse(_page("Login", _login_page(nxt=nxt), active="", nologin=True))


async def login_post(request: Request):
    f = await _form(request)
    key = (f.get("key", "") or "").strip()
    nxt = f.get("next", "/ui") or "/ui"
    admin = _resolve_admin(key)
    if not admin:
        return HTMLResponse(_page("Login", _login_page("Invalid admin key.", nxt), active="", nologin=True),
                            status_code=401)
    resp = RedirectResponse(nxt if nxt.startswith("/ui") else "/ui", status_code=303)
    resp.set_cookie(_SESSION_COOKIE, _make_session(admin["name"]),
                    max_age=_SESSION_TTL, httponly=True, samesite="lax", path="/")
    logger.info(f"ui: admin '{admin['name']}' logged in")
    return resp


async def logout(request: Request):
    resp = RedirectResponse("/ui/login", status_code=303)
    resp.delete_cookie(_SESSION_COOKIE, path="/")
    return resp


# ── POST body parsing (no python-multipart) ─────────────────────────────────────

async def _form(request: Request) -> dict:
    raw = (await request.body()).decode("utf-8", "replace")
    return {k: v[-1] for k, v in parse_qs(raw, keep_blank_values=True).items()}


async def _form_multi(request: Request) -> dict:
    """Like _form(), but keeps EVERY value per key ({k: [v, …]}) — for handlers with
    checkbox lists (reasoning backends, user model grants) that _form would collapse."""
    raw = (await request.body()).decode("utf-8", "replace")
    return parse_qs(raw, keep_blank_values=True)


async def _multipart(request: Request) -> dict:
    ctype = request.headers.get("content-type", "")
    m = re.search(r"boundary=([^;]+)", ctype)
    if not m:
        return await _form(request)
    body = await request.body()
    delim = b"--" + m.group(1).strip().strip('"').encode()
    out: dict = {}
    for part in body.split(delim):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        head_s = head.decode("utf-8", "replace")
        nm = re.search(r'name="([^"]*)"', head_s)
        if nm:
            fn = re.search(r'filename="([^"]*)"', head_s)
            out[nm.group(1)] = content if fn else content.decode("utf-8", "replace")
            if fn:
                # The upload's own name, under a companion key (callers read fields by
                # name, and the prefix loops all filter by their own prefix). A mesh
                # upload needs it: the extension is what tells the API the file type —
                # the bytes of a .fbx and a .glb are both "some binary".
                out[nm.group(1) + "__filename"] = os.path.basename(fn.group(1).replace("\\", "/"))
    return out


# ── Discovery helpers (object_info → model dropdowns) ────────────────────────────

def _backend_url(name: str):
    for b in _comfy_backends():
        if b["name"] == name:
            return b["url"].rstrip("/")
    return None


def _is_model_field(options, current) -> bool:
    if not isinstance(options, list):                # numeric specs are dicts, not combos
        return False
    sample = [str(o) for o in (options[:8] if options else [])] + [str(current or "")]
    return any(s.lower().endswith(_MODEL_EXTS) for s in sample)


_OI_CACHE: dict = {}              # (backend, class) → (monotonic_ts, fields|None)
_OI_TTL = 300.0                   # node defs change rarely; 5 min keeps edits snappy
_OI_TTL_ERR = 60.0                # a FAILED fetch (backend down / class not installed) expires
                                  # fast: cached as None it silently turns every model dropdown in
                                  # the editor into a text box, so a RETURNING backend must not
                                  # stay invisible for the full TTL. Cost: one probe per minute.


async def _fetch_oi_class(client, url: str, cls: str):
    """Fetch + parse one class's input spec from /object_info/{cls}."""
    try:
        r = await client.get(f"{url}/object_info/{cls}")
        if r.status_code != 200:
            return cls, None
        spec = r.json()[cls]["input"]
        fields = {}
        for section in ("required", "optional"):
            for fn, fspec in (spec.get(section) or {}).items():
                if not (isinstance(fspec, list) and fspec):
                    continue
                if isinstance(fspec[0], list):                      # combo (old form) → options list
                    fields[fn] = fspec[0]
                elif fspec[0] == "COMBO" and len(fspec) > 1 and isinstance(fspec[1], dict) \
                        and isinstance(fspec[1].get("options"), list):
                    fields[fn] = list(fspec[1]["options"])          # combo (new form: ["COMBO", {options}])
                elif fspec[0] in ("FLOAT", "INT") and len(fspec) > 1 and isinstance(fspec[1], dict):
                    c = fspec[1]                                    # numeric → discovery constraints
                    fields[fn] = {"_num": fspec[0], "default": c.get("default"),
                                  "min": c.get("min"), "max": c.get("max"), "step": c.get("step")}
        return cls, fields
    except Exception:
        return cls, None


async def _object_info(backend_name: str, wf: dict, mapping: Optional[dict] = None) -> dict:
    """Per-class /object_info for the workflow's loader nodes PLUS every mapped node
    (a request field may be a combo/number on any node — e.g. UniRigAutoRig's
    skeleton_template — and those need widget metadata too). Fetches uncached classes
    in parallel and caches them with a short TTL, so re-opening an alias is instant."""
    url = _backend_url(backend_name)
    if not url:
        return {}
    classes = {n.get("class_type", "") for n in wf.values()
               if any(h in (n.get("class_type", "") or "").lower() for h in _LOADER_HINTS)}
    for m in (mapping or {}).values():                # mapped nodes' combos/numbers → widgets
        cls = (wf.get((m or {}).get("node")) or {}).get("class_type", "")
        if cls:
            classes.add(cls)
    classes.discard("")
    now = time.monotonic()
    out: dict = {}
    missing = []
    for cls in classes:
        hit = _OI_CACHE.get((backend_name, cls))
        if hit and now - hit[0] < (_OI_TTL if hit[1] is not None else _OI_TTL_ERR):
            if hit[1] is not None:
                out[cls] = hit[1]
        else:
            missing.append(cls)
    if missing:
        async with httpx.AsyncClient(timeout=8.0) as c:
            results = await asyncio.gather(*[_fetch_oi_class(c, url, cls) for cls in missing])
        for cls, fields in results:
            _OI_CACHE[(backend_name, cls)] = (now, fields)
            if fields is not None:
                out[cls] = fields
    return out


async def _editor_object_info(cands: list, wf: dict, mapping: Optional[dict]) -> tuple:
    """The node defs the mapping editor renders its widgets from, with a **fallback
    across the alias's own backends**.

    Every widget in the editor — the model dropdowns under Pinned values, the numeric
    bounds, the `▾ n` hints in Available fields — comes from `/object_info`, and the
    editor used to ask ONLY the first candidate. A first backend that is merely DOWN
    therefore degraded every one of those to a free-text box, silently: no error, just
    a text field where a model list belongs (measured 2026-09-02 on `Qwen BL` — its
    primary `dx10-02` was unreachable, so the two model pins, UNETLoader/LoaderGGUF,
    lost their dropdowns while the alias's healthy `k12-gpu` had the very same classes).
    The extra-backend tabs already borrow the primary's defs when they cannot answer
    (`oi_bn or oi`); this is the same courtesy in the other direction.

    All candidates are probed CONCURRENTLY, so a dead backend costs one timeout for the
    page, not one per backend, and the results warm `_OI_CACHE` for `_pinned_block`'s
    per-backend tabs. Returns `(oi, borrowed_from)` — `borrowed_from` names the backend
    whose defs stood in, and is None when the primary answered (the normal case)."""
    names = [str(c.get("backend") or "") for c in cands]
    ois = await asyncio.gather(*[_object_info(n, wf, mapping) for n in names])
    for i, (name, oi) in enumerate(zip(names, ois)):
        if oi:
            return oi, (None if i == 0 else name)
    return {}, None


def _detect_model_bindings(wf: dict, oi: dict) -> list:
    out = []
    for nid, n in wf.items():
        opts_by_field = oi.get(n.get("class_type", ""), {})
        for fn, val in (n.get("inputs") or {}).items():
            opts = opts_by_field.get(fn)
            if opts is not None and _is_model_field(opts, val):
                out.append({"node": nid, "field": fn, "value": val})
    return out


def _num_input(name: str, value, spec: dict) -> str:
    """<input type=number> with discovery default/min/max/step (e.g. LoRA strength,
    steps, cfg). Falls back to the field's default when no value is set."""
    cur = value if value not in (None, "") else spec.get("default")
    attrs = "".join(f' {k}="{_esc(spec[k])}"' for k in ("min", "max", "step") if spec.get(k) is not None)
    return f'<input type="number" name="{_esc(name)}" value="{_esc("" if cur is None else cur)}"{attrs}>'


def _value_control(name: str, node: str, field: str, value, wf: dict, oi: dict) -> str:
    """Render the right input widget for a pinned node field: model dropdown, bounded
    number, true/false, image (placeholder/upload), or a plain text box."""
    cls = wf.get(node, {}).get("class_type", "")
    file_val = (wf.get(node, {}).get("inputs") or {}).get(field)
    cur = value if value is not None else file_val
    opts = oi.get(cls, {}).get(field)
    if isinstance(opts, dict) and opts.get("_num"):       # FLOAT/INT with discovery bounds
        return _num_input(name, cur, opts)
    if opts and _is_model_field(opts, cur):
        o = list(opts)
        stale = cur not in o and o
        if cur not in o:
            o = [cur] + o
        flag = ' <span class="bad">(stale)</span>' if stale else ""
        return _select(name, o, cur) + flag
    if adapters.is_img_loader_class(cls) and field == "image":
        o = [(adapters.UPLOAD_SENTINEL, "playground upload (8×8 if empty)"),
             (adapters.PLACEHOLDER_SENTINEL, "8×8 placeholder (always)")]
        if cur and cur not in (adapters.PLACEHOLDER_SENTINEL, adapters.UPLOAD_SENTINEL):
            o.append((cur, cur))
        return _select(name, o, cur)
    if isinstance(file_val, bool):
        return _select(name, ["true", "false"], str(cur).lower())
    return _inp(name, "" if cur is None else cur)


def _boolean_fields(wf: dict) -> list:
    return [{"node": nid, "field": fn, "value": v, "class": n.get("class_type", "")}
            for nid, n in wf.items()
            for fn, v in (n.get("inputs") or {}).items() if isinstance(v, bool)]


def _image_fields(wf: dict) -> list:
    return [{"node": nid, "field": "image", "value": (n.get("inputs") or {}).get("image", ""),
             "class": n.get("class_type", ""), "title": n.get("_meta", {}).get("title", "")}
            for nid, n in wf.items() if adapters.is_img_loader_class(n.get("class_type"))]


# ── Tab: Backends ───────────────────────────────────────────────────────────────

def _backend_form(b: Optional[dict], hosts: list) -> str:
    g = lambda k, d="": str((b or {}).get(k) if (b or {}).get(k) is not None else d)
    gb = lambda k: bool((b or {}).get(k))
    title = "Edit Backend" if b else "Add Backend"
    orig = f'<input type="hidden" name="orig" value="{_esc(_bid(b))}">' if b else ""
    hlist = "".join(f'<option value="{_esc(h)}">' for h in hosts)
    # The cloud option block is rendered for EVERY cloud kind at once (one hint each,
    # only the current type's visible) — the type select toggles them client-side, so
    # switching type must not need a round trip.
    cur_type = g("type", "openai")
    cmod = adapters.cloud_module(cur_type) if cur_type in adapters.CLOUD_TYPES else None
    num = lambda x: str(int(x)) if float(x) == int(x) else str(x)   # 5.0 → "5" (a placeholder)
    host_inp = (f'<input name="host" value="{_esc(g("host"))}" list="hostlist" '
                f'placeholder="auto: URL host/IP" autocomplete="off">'
                f'<datalist id="hostlist">{hlist}</datalist>')
    return (f'<form action="/ui/backends/save" method="post">{orig}'
            f'<div class="formbar"><h2>{title}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/backends", "secondary")}</div>'
            + _field("name", _inp("name", g("name"), placeholder="evo-comfy"))
            + _field("type", _type_select(g("type", "openai")))
            + "<p class='hint' style='margin:-4px 0 10px'><b>openai</b> = every OpenAI-compatible server "
              "(llama.cpp / llama-swap / vLLM / LocalAI / cloud) — including <b>TTS/voice</b> and whisper "
              "models, which are discovered and routed like any other model. <b>comfyui</b> = workflow-based "
              "media generation. <b>meshy</b> = Meshy.ai cloud mesh generation (image / multi-image → 3D), "
              "billed per task in credits — always <b>paid</b>. <b>tripo</b> = Tripo3D cloud mesh "
              "generation + Mixamo-spec rigging (image / multi-image → 3D), billed per task — always "
              "<b>paid</b>. <b>anthropic</b> = api.anthropic.com for "
              "Claude Code, reachable through <code>/v1/messages</code> only.</p>"
            + _field("url", _inp("url", g("url"), placeholder="http://host:8080"))
            + _field("host", host_inp)
            + "<p class='hint' style='margin:-4px 0 10px'>The physical box this backend runs on — backends "
              "on one host share its GPU/VRAM (basis for host policies). Blank = derived from the URL "
              "host/IP, which groups correctly for most setups.</p>"
            # A cloud backend (Meshy, Tripo) bills per task, so `paid` is not a choice
            # there: shown checked + disabled (a disabled box is NOT submitted —
            # backend_save forces it too).
            + _field("cost tier", ('<label class="ckbox"><input type="checkbox" name="paid" value="1" '
                                   'checked disabled> paid — always, a cloud backend bills per task</label>')
                     if cur_type in adapters.CLOUD_TYPES else
                     _checkbox("paid", gb("paid"), "paid — used only when no unpaid backend is free"))
            + "<p class='hint' style='margin:-4px 0 10px'><b>paid</b>: this backend bills per request "
              "(a cloud API). The scheduler sends a request to the fastest free <b>unpaid</b> backend "
              "and reaches for a paid one only when no unpaid backend is free.</p>"
            + _field("max_concurrent", _inp("max_concurrent", g("max_concurrent"), placeholder="optional, e.g. 1", typ="number"))
            + _field("api key", _inp("api_key", g("api_key"), placeholder="optional — cloud backends"))
            # ComfyUI-only options — hidden for openai (none of these apply to an LLM backend)
            + f'<div id="comfyopts" style="{"" if g("type", "openai") == "comfyui" else "display:none"}">'
            + '<div class="grouphdr">ComfyUI</div>'
            + _field("comfy output dir", _inp("comfy_output_dir", g("comfy_output_dir"),
                     placeholder="e.g. /home/kai/ComfyUI/output"))
            + "<p class='hint' style='margin:-4px 0 10px'>Absolute path to this ComfyUI's <b>output</b> "
              "directory — only needed for <b>workflow chains</b> (a successor stage reads the previous "
              "stage's mesh by full path from here). Leave blank otherwise.</p>"
            + _field("comfy input dir", _inp("comfy_input_dir", g("comfy_input_dir"),
                     placeholder="blank = derived: output dir's …/input sibling"))
            + "<p class='hint' style='margin:-4px 0 10px'>Absolute path to this ComfyUI's <b>input</b> "
              "directory — used by chain <b>upload</b> hand-offs: the relayed mesh lands here and the "
              "successor gets its full path. Blank = derived from the output dir "
              "(<code>…/output</code> → <code>…/input</code>).</p>"
            + _field("comfy watchdog",
                     _checkbox("auto_restart", gb("auto_restart"), "auto_restart",
                               "restart the ComfyUI service automatically when the executor is stuck"))
            + _field("restart cooldown s", _inp("restart_cooldown_s", g("restart_cooldown_s"),
                     placeholder="600", typ="number"))
            + _field("stuck after s", _inp("stuck_after_s", g("stuck_after_s"),
                     placeholder="90", typ="number"))
            + _field("self retries", _inp("self_retries", g("self_retries"),
                     placeholder="0", typ="number"))
            + _field("max wait s", _inp("max_wait", g("max_wait"),
                     placeholder="600", typ="number"))
            + _field("poll interval s", _inp("poll_interval", g("poll_interval"),
                     placeholder="1", typ="number", step="0.1"))
            + "<p class='hint' style='margin:-4px 0 10px'><b>max wait s</b> caps ONE generation: "
              "how long the gateway polls <code>/history</code> for a submitted prompt before it "
              "gives up (it then sends <code>/interrupt</code> to free the GPU). ComfyUI itself has "
              "no such limit — this is the gateway's. Raise it for slow workflows (video, mesh); the "
              "cap is spent per candidate backend, so with two candidates a client can wait twice "
              "this long. <b>poll interval s</b> is the gap between those polls. Blank = 600 / 1.</p>"
            + "<p class='hint' style='margin:-4px 0 10px'>Executor watchdog (comfyui only): the "
              "backend goes <b>down</b> when prompts wait while nothing runs for <b>stuck after s</b> "
              "seconds. <b>auto_restart</b> then reboots the service via the ComfyUI-Manager "
              "extension (requires it installed + a systemd unit with <code>Restart=always</code>), "
              "at most once per <b>restart cooldown s</b>. <b>self retries</b>: a connection-type "
              "fault mid-job retries the <b>same</b> backend this many times (after waiting for "
              "<code>/system_stats</code>) before failing over — for hosts with sporadic driver "
              "faults. Blank/0 = fail over immediately; content errors are never retried.</p>"
            + "</div>"
            # Cloud-only options (Meshy, Tripo) — a cloud task API: no dirs, no watchdog,
            # no self-retry. The fields are named cloud_* because #comfyopts already
            # renders max_wait / poll_interval, and one form may carry each name only once.
            # One hint per kind, all rendered, only the selected type's shown: the type
            # select reveals the right one without a round trip.
            + f'<div id="cloudopts" style="{"" if cmod else "display:none"}">'
            + '<div class="grouphdr">Cloud task API</div>'
            + _field("max wait s", _inp("cloud_max_wait", g("max_wait"), typ="number",
                     placeholder=num(cmod.MAX_WAIT_DEFAULT) if cmod else "900"))
            + _field("poll interval s", _inp("cloud_poll_interval", g("poll_interval"),
                     typ="number", step="0.5",
                     placeholder=num(cmod.POLL_INTERVAL_DEFAULT) if cmod else "5"))
            + "".join(f"<p class='hint' data-cloud-hint=\"{k}\" style=\"margin:-4px 0 10px"
                      f'{"" if k == cur_type else ";display:none"}">{m.BACKEND_HINT}</p>'
                      for k, m in adapters.CLOUD_MODULES.items())
            + "</div>"
            # LLM-only options — hidden for comfyui (none of these apply to ComfyUI)
            + f'<div id="llmopts" style="{"" if g("type", "openai") == "openai" else "display:none"}">'
            + '<div class="grouphdr">LLM</div>'
            + _field("discovery filters",
                     _checkbox("chat_only", gb("chat_only"), "chat_only",
                               "keep only models with type==chat (skip image/video/embedding)")
                     + _checkbox("serverless_only", gb("serverless_only"), "serverless_only",
                                 "keep only priced models (skip dedicated-only; OpenRouter :free)"))
            + "<p class='hint'>Filters for cloud LLM catalogs (Together / OpenRouter); "
              "backends whose models carry no type/pricing are unaffected.</p>"
            + _field("list models under bare id",
                     _checkbox("local", gb("local"), "local"))
            + "<p class='hint'><b>local</b>: also list each of this backend's models under its plain "
              "id (without the <code>backend/</code> prefix). Several local backends sharing a model id "
              "then collapse into one entry the scheduler routes across (fastest free unpaid backend "
              "first) with failover — an implicit "
              "cross-backend alias.</p>"
            + _field("prompt cache passthrough",
                     _checkbox("prompt_cache", gb("prompt_cache"), "prompt_cache"))
            + "<p class='hint'><b>prompt_cache</b>: keep Claude Code's cache breakpoints when this "
              "backend serves <code>/v1/messages</code> (translated). Turn it on for <b>OpenRouter</b>, "
              "which forwards them to Anthropic/Gemini models — without them the full context is billed "
              "again every turn. Off by default: the breakpoints turn a message into a content-part list, "
              "which a strict server may reject. Irrelevant for local models (no token billing) and for "
              "OpenAI models (they cache automatically).</p>"
            + _field("context windows",
                     _textarea("model_context", g("model_context"), rows=3,
                               placeholder="glm-*=32768\nqwen3.8-flash-next-ple4=131072"))
            + "<p class='hint'><b>context windows</b>: one <code>model-glob=tokens</code> per line, "
              "published as <code>context_length</code> in <code>/v1/models</code> — the number a "
              "client (Oh My Pi, Hermes, OpenCode) sizes its prompts by; without one it assumes "
              "128k and a 33k prompt to a 32k model 400s. A rule beats what discovery learned "
              "(OpenRouter/Together <code>context_length</code>, vLLM <code>max_model_len</code>, "
              "llama-server <code>n_ctx</code> — read through llama-swap for <b>loaded</b> models "
              "only, and remembered). Write a rule for a llama-swap model that is rarely loaded, or "
              "when the server's value is not what clients should use.</p>"
            + f'<details class="optblock"{" open" if (b or {}).get("sampling_defaults") else ""}>'
            + "<summary>Sampling defaults <span class='muted'>— used when the caller sends none"
              "</span></summary>"
            + _sampling_inputs((b or {}).get("sampling_defaults"))
            + "<p class='hint'><b>sampling defaults</b>: values filled into every chat request to "
              "this backend, for keys the caller did <b>not</b> send (an explicit client value — and an "
              "alias default — always wins). For backends whose server samples with bare defaults: vLLM "
              "without a truncation sampler (<code>top_p=1</code>, <code>min_p=0</code>) degenerates into "
              "token salad at temperature ≈ 1. Re-derived per backend, so a failover uses the new "
              "backend's values. Applies to chat/completions/responses only.</p>"
            + "</details>"
            + "</div>"
            # Anthropic-only options — the licence warning sits AT the credential field,
            # not in a footnote, because that is where the decision is made.
            + f'<div id="anthopts" style="{"" if g("type", "openai") == "anthropic" else "display:none"}">'
            + '<div class="grouphdr">Anthropic</div>'
            + "<p class='bad' style='margin:0 0 10px'><b>Licence boundary.</b> A Claude "
              "<b>subscription</b> token (<code>claude setup-token</code>) is licensed for your own use of "
              "Claude Code — <b>not</b> for re-serving Claude as an API to other clients or people. This "
              "backend is therefore reachable through <code>/v1/messages</code> only: it never appears in "
              "<code>/v1/chat/completions</code>, <code>/v1/responses</code> or the Playground. Keep it that "
              "way, and don't hand its gateway key to third parties. With a paid <b>API key</b> "
              "(console.anthropic.com) the same restriction applies here — the gateway does not translate "
              "Claude into the OpenAI endpoints.</p>"
            + _field("auth mode", _select("auth_mode", [
                ("subscription", "subscription — claude setup-token (OAuth)"),
                ("api_key", "api key — console.anthropic.com")],
                g("auth_mode", "subscription")))
            + "<p class='hint' style='margin:-4px 0 10px'>Determines how the credential in <b>api key</b> "
              "above is sent: <b>subscription</b> → <code>Authorization: Bearer</code> plus the OAuth beta "
              "header; <b>api key</b> → <code>x-api-key</code>.</p>"
            + _field("models", _inp("models", ", ".join((b or {}).get("models") or [])
                                    if isinstance((b or {}).get("models"), list)
                                    else g("models"),
                                    placeholder="claude-sonnet-5, claude-opus-5"))
            + "<p class='hint' style='margin:-4px 0 10px'>Comma-separated fallback model list. Discovery "
              "asks <code>GET /v1/models</code> first; a subscription token is not guaranteed to be allowed "
              "there, and this list keeps the backend usable when it isn't. Point a chat alias at one of "
              "these ids, then run Claude Code with "
              "<code>ANTHROPIC_BASE_URL=&lt;gateway&gt;</code>.</p>"
            + "</div></form>")


def _sampling_text(d) -> str:
    """Stored sampling defaults → the editor's textarea value (JSON, blank if unset)."""
    return json.dumps(d, ensure_ascii=False) if isinstance(d, dict) and d else ""


def _type_select(current: str) -> str:
    """Backend type select that shows/hides the type-specific option blocks on change
    (LLM / ComfyUI / cloud / Anthropic — the form renders all, only one is ever visible).
    A cloud type also reveals its own backend hint and forces `paid` (it bills per task);
    the disabled box is not submitted, so `backend_save` sets it server-side too.

    The URL field is filled with the chosen kind's fixed endpoint when it is blank OR
    still holds ANOTHER cloud kind's fixed URL — switching meshy → tripo would otherwise
    store a Tripo backend pointing at api.meshy.ai, which only surfaces as an auth error
    at discovery, pointing at the wrong thing. Anything else the operator typed (a
    self-hosted proxy) is never overwritten. `backend_save` applies the same rule.

    Every cloud kind comes from adapters.CLOUD_MODULES, so a new one appears here without
    touching this handler. ES5 only (var/function, no arrows): an inline attribute is
    never transpiled, and test_admin_live pins the console's JS to ES5."""
    opts = "".join(f'<option value="{t}"{" selected" if t == current else ""}>{t}</option>'
                   for t in ("comfyui", "meshy", "tripo", "openai", "anthropic"))
    # single-quoted: this JS sits inside a double-quoted HTML attribute, and JSON's
    # double quotes would close it early.
    urls = json.dumps(_cloud_urls()).replace('"', "'")
    return ('<select name="type" onchange="var t=this.value,cloudUrls=' + urls + ","
            "l=document.getElementById('llmopts'),c=document.getElementById('comfyopts'),"
            "m=document.getElementById('cloudopts'),a=document.getElementById('anthopts'),"
            "u=document.querySelector('input[name=url]'),p=document.querySelector('input[name=paid]');"
            "if(l)l.style.display=t==='openai'?'':'none';"
            "if(c)c.style.display=t==='comfyui'?'':'none';"
            "if(m)m.style.display=cloudUrls[t]?'':'none';"
            "if(a)a.style.display=t==='anthropic'?'':'none';"
            "Array.prototype.forEach.call(document.querySelectorAll('[data-cloud-hint]'),"
            "function(h){h.style.display=h.getAttribute('data-cloud-hint')===t?'':'none'});"
            "if(cloudUrls[t]){if(u){var ow=!u.value;"
            "for(var k in cloudUrls){if(k!==t&&u.value===cloudUrls[k])ow=true}"
            "if(ow)u.value=cloudUrls[t]}"
            "if(p){if(p.dataset.was===undefined)p.dataset.was=p.checked?'1':'';"
            "p.checked=true;p.disabled=true}}"
            "else if(p){p.disabled=false;if(p.dataset.was!==undefined){"
            "p.checked=p.dataset.was==='1';delete p.dataset.was}}\">" + opts + "</select>")


def _bid(b: dict) -> str:
    """Stable backend id = type:name (so LLM + ComfyUI may share a name)."""
    return f'{b.get("type", "openai")}:{b["name"]}'


def _parse_bid(s: str) -> tuple:
    """'type:name' → (name, type). Bare value (legacy) → (value, 'openai')."""
    t, sep, n = s.partition(":")
    return (n, t) if sep else (s, "openai")


async def backends_page(request: Request):
    qp = request.query_params
    edit_id = qp.get("edit", "")
    binfo = _gateway_info().get("backends", [])
    # editable from either source: store (full dict incl. api_key) or the live summary (config)
    editing = None
    if edit_id:
        en, et = _parse_bid(edit_id)
        editing = (store.get_backend(en, et)
                   or next((b for b in binfo if _bid(b) == edit_id), None))
    def render(b):
        bid = _bid(b)
        draining, inflight = b.get("draining"), b.get("inflight", 0)
        if draining:
            badge = _badge(f"⏳ draining · {inflight} in-flight", "warn")
        elif not b["enabled"]:
            badge = _badge("⏻ offline", "warn", "taken offline — use ⏼ bring-online to re-enable")
        elif b["healthy"]:
            badge = _badge("healthy", "ok")
        else:
            badge = _down_badge(b.get("error"))
        if b.get("exec_stuck"):
            badge = _badge("⚠ executor stuck", "bad",
                           "ComfyUI answers HTTP but its executor is not draining the queue "
                           "— restart the service (⟳) or check the box/GPU")
        acts_list = [("✎", f"/ui/backends?edit={quote(bid)}", "secondary", "Edit")]
        if draining:
            acts_list.append(("↺", f"/ui/backends/undrain?id={quote(bid)}", "secondary",
                              "Cancel drain — put back in rotation"))
        elif b["enabled"]:
            acts_list.append(("⏻", f"/ui/backends/drain?id={quote(bid)}", "secondary",
                              "Take offline when idle (drain: stop new requests, finish in-flight)",
                              f"Take {b['name']} offline once idle? New requests stop now; "
                              "in-flight requests finish first."))
        else:
            acts_list.append(("⏼", f"/ui/backends/enable?id={quote(bid)}", "secondary",
                              "Bring online (enable)"))
        if b.get("type") == "comfyui" and b["enabled"]:
            acts_list.append(("⟳", f"/ui/backends/restart?id={quote(bid)}", "secondary",
                              "Restart the ComfyUI service (ComfyUI-Manager reboot)",
                              f"Restart ComfyUI on {b['name']}? Pending prompts there are lost."))
        if b.get("source", "config") == "ui":
            acts_list.append(("✕", f"/ui/backends/delete?id={quote(bid)}", "danger",
                              "Delete", f"Remove backend {b['name']} ({b['type']})?"))
        acts = _icon_acts(*acts_list)
        src = "" if b.get("source") == "ui" else " · config"
        flags = "".join(f" · {fl}" for fl in ("chat_only", "serverless_only", "local", "paid") if b.get(fl))
        host = f" · host {b['host']}" if b.get("host") else ""
        rst = f" · restart: {b['last_restart_result']}" if b.get("last_restart_result") else ""
        # Two rates, never merged: "conn" says the backend keeps falling over, "exec"
        # says it is up and burning every job handed to it. A single number hid the
        # second completely — an execution failure used to count as a clean attempt.
        fr = ""
        if b.get("fail_rate") is not None:
            fr = f" · fail_rate conn {b['fail_rate']:.2f} ({b['gen_fails']}/{b['gen_attempts']})"
            if b.get("exec_fail_rate") is not None:
                fr += f" · exec {b['exec_fail_rate']:.2f} ({b['exec_fails']}/{b['gen_attempts']})"
        # Quarantine DOES change routing, so it must be impossible to miss: a backend
        # sitting idle while jobs run elsewhere has to say why, right here.
        qn = ""
        for q in (b.get("quarantined") or []):
            qn += (f" · <b>⚠ quarantined for {_esc(q['alias'])}</b> "
                   f"({max(1, q['for_s'] // 60)} min left, {q['fails']} execution failures)")
        smp = (f" · sampling {_sampling_text(b['sampling_defaults'])}"
               if b.get("sampling_defaults") else "")
        # Cloud credit balance WITH its age: the number is a snapshot from the last
        # successful discovery, and a stale one is worth spotting before a job fails.
        cr = ""
        if b.get("credits") is not None:
            cr = f" · credits {b['credits']}"
            if b.get("credits_at"):
                cr += f" ({_age(b['credits_at'])} ago)"
        sub = f"{b['url']}{host} · {b['models']} models{flags}{smp}{fr}{qn}{cr}{rst}{src}"
        return _item(f"{_esc(b['name'])}{_type_badge(b['type'])}{badge}", sub, acts, sel=(bid == edit_id))

    # group by kind: LLM (openai-compatible) vs Media (every generation type — ComfyUI,
    # Meshy, Tripo, …), alphabetical within each
    binfo = sorted(binfo, key=lambda b: b["name"].lower())
    llm = [b for b in binfo if b.get("type", "openai") not in adapters.GEN_TYPES]
    img = [b for b in binfo if b.get("type") in adapters.GEN_TYPES]
    items = ""
    for label, group in (("LLM", llm), ("Media", img)):
        if group:
            items += f'<div class="grouphdr">{label}</div>' + "".join(render(b) for b in group)
    items = items or "<p class='muted'>No backends.</p>"
    list_html = (f'<div class="bar"><h2>Backends</h2>{_btn("+ New", "/ui/backends?new=1")}</div>'
                 f"<p class='hint'>Edit a backend to manage it here (editing a config one creates an "
                 f"editable copy that overrides it).</p>{items}"
                 + _hosts_panel(binfo, qp.get("host", "")))
    hosts = sorted({b["host"] for b in binfo if b.get("host")})
    edit_host = qp.get("host", "")
    if editing or qp.get("new"):
        detail = _backend_form(editing, hosts)
    elif edit_host:
        detail = _host_form(edit_host, shared=_host_is_shared([b for b in binfo if b.get("host") == edit_host]))
    else:
        detail = ("<h2>Details</h2><p class='hint'>Select a backend's <b>Edit</b>, "
                  "or <b>+ New</b> to add one.</p>")
    body = (f'<div class="cols"><div class="col">{list_html}</div>'
            f'<div class="col">{detail}</div></div>')
    draining_now = any(b.get("draining") for b in binfo)      # watch the count drain → offline
    return HTMLResponse(_page("Backends", body, "backends", refresh=4 if draining_now else None))


def _hosts_panel(binfo: list, sel_host: str) -> str:
    """Physical-box grouping under the backend list: one row per host that has a
    ComfyUI backend, with its member backends. Membership is edited on the backend
    (its `host` field or the URL IP); this panel edits the per-host extras — a label
    and the GPU policy flags (docs/host-coordination-plan.md).

    Listed: every box with a ComfyUI backend. The LLM-vs-media policies only concern
    a SHARED box (an LLM and a ComfyUI backend on one GPU), but the free-before-job
    policy applies to a dedicated one too — its VRAM is contended by the next alias
    and by nodes that run outside ComfyUI's own process. A host that still carries
    stored meta stays listed regardless, so an old setting never becomes uneditable."""
    by_host: dict = {}
    for b in binfo:
        if b.get("host"):
            by_host.setdefault(b["host"], []).append(b)
    meta = store.get_hosts() if store.is_active() else {}

    def has_comfy(members: list) -> bool:
        return any(b.get("type") == "comfyui" for b in members)
    listed = [h for h, members in by_host.items() if has_comfy(members) or meta.get(h)]
    if not listed:
        return ""
    rows = ""
    for h in sorted(listed):
        hm = meta.get(h) or {}
        label = hm.get("label", "")
        members = " · ".join(f"{b['name']} ({b['type']})" for b in by_host[h])
        shared = _host_is_shared(by_host[h])
        tag = _badge("shared", "warn", "an LLM and a ComfyUI backend share this box (and its GPU/VRAM)") if shared else ""
        # badges mark EXPLICIT non-default values only — the defaults come from the
        # same table the request path reads (scheduler.HOST_FLAGS)
        def non_default(key: str) -> bool:
            return key in hm and bool(hm[key]) != scheduler.host_flag_default(key, shared)
        if non_default("avoid_llm_during_media"):
            tag += " " + _badge("llm-avoid off", "warn",
                                "chat routing does NOT step aside while this host generates media")
        if non_default("comfy_free_before_job"):
            tag += " " + _badge("free-before off", "warn",
                                "VRAM is NOT freed when this box's next media job loads a "
                                "different model set — non-default setting")
        if non_default("comfy_free_after_job"):
            tag += " " + _badge(f"free-after {'on' if hm['comfy_free_after_job'] else 'off'}", "warn",
                                "ComfyUI /free after each media job — non-default setting")
        if non_default("llm_unload_before_media"):
            tag += " " + _badge("unload-llm", "warn",
                                "this host's LLMs are unloaded before each media job")
        acts = _icon_acts(("✎", f"/ui/backends?host={quote(h)}", "secondary", "Edit host"))
        title = f"{_esc(h)}{(' — ' + _esc(label)) if label else ''} {tag}"
        rows += _item(title, members, acts, sel=(h == sel_host))
    return ('<div class="grouphdr" style="margin-top:18px">Hosts · GPU policy</div>'
            "<p class='hint' style='margin:2px 0 6px'>Every box with a ComfyUI backend: who may use "
            "its VRAM and when it is freed. The LLM policies matter only where an LLM shares the "
            f"same GPU (badge <b>shared</b>).</p>{rows}")


def _host_is_shared(members: list) -> bool:
    """A box where an LLM and a ComfyUI backend contend for one GPU. Derived from the
    LIVE backend list wherever it is needed (form, panel, save) — never round-tripped
    through the form: a backend added or removed between render and Save would
    otherwise flip which value counts as the default (review 2026-09-08)."""
    types = {b.get("type", "openai") for b in members}
    return "comfyui" in types and len(types) > 1


def _host_form(host: str, shared: bool) -> str:
    meta = (store.get_hosts() if store.is_active() else {}).get(host) or {}
    avoid = scheduler.host_flag(meta, "avoid_llm_during_media", shared)
    pre = scheduler.host_flag(meta, "comfy_free_before_job", shared)
    free = scheduler.host_flag(meta, "comfy_free_after_job", shared)
    unload = scheduler.host_flag(meta, "llm_unload_before_media", shared)
    return (f'<form action="/ui/backends/host-save" method="post">'
            f'<input type="hidden" name="host" value="{_esc(host)}">'
            f'<div class="formbar"><h2>Host {_esc(host)}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/backends", "secondary")}</div>'
            + _field("label", _inp("label", meta.get("label", ""), placeholder="e.g. K12 box"))
            + "<p class='hint'>Display label for this physical box. Which backends belong to it is "
              "set on each backend (its <b>host</b> field; blank = URL host/IP).</p>"
            + _field("routing", _checkbox("avoid_llm", avoid, "avoid LLM routing during media jobs",
                                          "while this host's ComfyUI is generating, its LLM backends are "
                                          "tried LAST (never skipped) — a llama-swap model load would abort "
                                          "on the VRAM the generation holds"))
            # The three VRAM policies stack VERTICALLY: `.control` is a flex ROW, so a
            # <br> between checkboxes is ignored and they line up side by side, where
            # three long labels read as one sentence. A column wrapper is the fix that
            # stays local to this form.
            + _field("VRAM", '<div style="display:flex;flex-direction:column;gap:6px;'
                             'align-items:flex-start">'
                     + _checkbox("comfy_free_pre", pre,
                                       "free ComfyUI VRAM before a job that changes the model set",
                                       "POST /free once a job is claimed and the weights its workflow "
                                       "loads (its loader nodes, after pins/mapping/LoRAs) are NOT what "
                                       "this box holds — a gateway restart counts as unknown, ComfyUI "
                                       "keeps its cache across one. A job loading the same weights "
                                       "(the same alias, or another one on the same model) keeps them. "
                                       "Leave on: a node running in its own process (rigging, "
                                       "Make-It-Animatable) cannot share ComfyUI's cache and OOMs on it.")
                     + _checkbox("comfy_free", free, "free ComfyUI VRAM after each media job",
                                       "POST /free when a job ends, even if no media job follows — only "
                                       "needed where an LLM shares this GPU, since a llama-swap load "
                                       "aborts on VRAM ComfyUI still holds. Skipped when the queued job "
                                       "this box will take next wants the alias it just ran.")
                     + _checkbox("llm_unload", unload, "unload LLMs before media jobs",
                                          "GET /unload on this host's LLM backends before a generation "
                                          "starts (llama-swap). Rarely needed — the swap TTL usually "
                                          "clears the model first.")
                     + "</div>")
            + f"<p class='hint'>Defaults: routing consideration ON · free-before-job ON · "
              f"free-after-job {'ON (shared box)' if shared else 'OFF (dedicated box)'} · "
              "unload-before OFF. Free-before-job is the one that applies to a dedicated box too; "
              "the rest only matters when an LLM shares this GPU.</p>"
            + "</form>")


async def host_save(request: Request):
    f = await _form(request)
    host = (f.get("host", "") or "").strip()
    if host and store.is_active():
        label = (f.get("label", "") or "").strip()
        # `shared` decides which value is the default (and therefore stored or dropped)
        # — from the LIVE backend list, never the form (see _host_is_shared)
        binfo = _gateway_info().get("backends", [])
        shared = _host_is_shared([b for b in binfo if b.get("host") == host])
        cur = dict(store.get_hosts().get(host) or {})
        if label:
            cur["label"] = label
        else:
            cur.pop("label", None)
        # flags store only the NON-default value (defaults: scheduler.HOST_FLAGS — the
        # table the request path reads) — an untouched host keeps adapting to them.
        for form_key, store_key in (("avoid_llm", "avoid_llm_during_media"),
                                    ("comfy_free_pre", "comfy_free_before_job"),
                                    ("comfy_free", "comfy_free_after_job"),
                                    ("llm_unload", "llm_unload_before_media")):
            val = bool(f.get(form_key))
            if val == scheduler.host_flag_default(store_key, shared):
                cur.pop(store_key, None)
            else:
                cur[store_key] = val
        store.set_host(host, cur or None)      # empty meta → drop the entry
        _apply_hosts()                         # refresh main's request-path cache
        logger.info(f"ui: host '{host}' saved (label={'y' if label else 'n'}, "
                    f"avoid_llm={'on' if f.get('avoid_llm') else 'OFF'}, "
                    f"comfy_free_pre={'on' if f.get('comfy_free_pre') else 'OFF'}, "
                    f"comfy_free={'on' if f.get('comfy_free') else 'off'}, "
                    f"llm_unload={'on' if f.get('llm_unload') else 'off'})")
    return RedirectResponse("/ui/backends", status_code=303)


# Keys a sampling default must never set: they drive routing, streaming, the
# reasoning hand-off and the stats body — a default here would corrupt dispatch.
_SAMPLING_BLOCKED = ("model", "messages", "stream", "stream_options")


# The samplers that get their own named input (label, placeholder). Everything
# else — logit_bias, stop, typical_p, a backend's private knob — goes into the
# free-form "more" JSON box, so no backend sampler is out of reach.
_SAMPLING_FIELDS = (
    ("temperature", "0.85"),
    ("top_p", "0.9"),
    ("top_k", "40"),
    ("min_p", "0.05"),
    ("repetition_penalty", "1.05"),
    ("presence_penalty", "0"),
    ("frequency_penalty", "0"),
)


def _sampling_num(raw: str):
    """A single sampler input → number (or None if blank). Accepts a German
    decimal comma. Returns (value, error)."""
    s = (raw or "").strip().replace(",", ".")
    if not s:
        return None, ""
    try:
        v = json.loads(s)
    except Exception:
        return None, f"'{s}' is not a number"
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return None, f"'{s}' is not a number"
    return v, ""


def _parse_sampling_form(f: dict, prefix: str = "smp_") -> tuple:
    """Read the sampling defaults out of a submitted form: one named input per
    common sampler plus a free-form JSON box for the rest.
    Returns (values, error) — a non-empty error means reject the save."""
    out: dict = {}
    for key, _ph in _SAMPLING_FIELDS:
        v, err = _sampling_num(f.get(prefix + key, ""))
        if err:
            return {}, f"sampling defaults · {key}: {err}"
        if v is not None:
            out[key] = v
    raw = (f.get(prefix + "more", "") or "").strip()
    if raw:
        try:
            d = json.loads(raw)
        except Exception as e:
            return {}, f"sampling defaults · more: invalid JSON ({e})"
        if not isinstance(d, dict):
            return {}, 'sampling defaults · more: must be a JSON object, e.g. {"typical_p": 0.95}'
        bad = [k for k in d if k in _SAMPLING_BLOCKED or k.startswith("_")]
        if bad:
            return {}, f"sampling defaults · more: these keys are not allowed: {', '.join(sorted(bad))}"
        dup = [k for k in d if k in dict(_SAMPLING_FIELDS)]
        if dup:
            return {}, (f"sampling defaults · more: {', '.join(sorted(dup))} "
                        "has its own field above — set it there, not in the JSON box")
        out.update(d)
    return out, ""


def _sampling_inputs(cur, prefix: str = "smp_") -> str:
    """Render the sampler inputs, pre-filled from a stored dict. Unknown keys
    (anything without its own field) land in the 'more' JSON box."""
    d = cur if isinstance(cur, dict) else {}
    known = dict(_SAMPLING_FIELDS)
    rows = "".join(
        _field(key.replace("_", " "),
               _inp(prefix + key, d.get(key, ""), placeholder=ph, typ="number", step="any"),
               short=True)
        for key, ph in _SAMPLING_FIELDS)
    rest = {k: v for k, v in d.items() if k not in known}
    return rows + _field("more (JSON)",
                         _textarea(prefix + "more",
                                   json.dumps(rest, ensure_ascii=False) if rest else "", 2,
                                   '{"typical_p": 0.95, "stop": ["###"]}'))


async def backend_save(request: Request):
    f = await _form(request)
    name = (f.get("name", "") or "").strip()
    url = (f.get("url", "") or "").strip().rstrip("/")
    new_type = (f.get("type", "openai") or "openai").strip()
    url = _cloud_url_for(new_type, url)      # pure rule, tested in test_cloud_editor.py
    if not name or not url:
        return HTMLResponse(_page("Backends", '<p class="bad">name and url are required</p>'
            f'<div class="actions">{_btn("← Back", "/ui/backends", "secondary")}</div>', "backends"))
    orig = (f.get("orig", "") or "").strip()
    oname, otype = _parse_bid(orig) if orig else (name, new_type)
    # start from the existing store backend (by old identity) so fields we don't render
    # (e.g. enabled) survive an edit; merge the form values over it.
    b = dict(store.get_backend(oname, otype) or store.get_backend(name, new_type) or {})
    b.update({"name": name, "type": new_type, "url": url,
              "paid": bool(f.get("paid"))})       # unchecked box = absent from the form = False
    mc = (f.get("max_concurrent", "") or "").strip()
    if mc.isdigit():
        b["max_concurrent"] = int(mc)
    else:
        b.pop("max_concurrent", None)
    host = (f.get("host", "") or "").strip()
    if host:
        b["host"] = host
    else:
        b.pop("host", None)                    # blank = derive from the URL host/IP
    cod = (f.get("comfy_output_dir", "") or "").strip().rstrip("/")
    if cod:
        b["comfy_output_dir"] = cod
    else:
        b.pop("comfy_output_dir", None)
    cid = (f.get("comfy_input_dir", "") or "").strip().rstrip("/")
    if cid:
        b["comfy_input_dir"] = cid
    else:
        b.pop("comfy_input_dir", None)         # blank = derive from the output dir
    if (f.get("api_key", "") or "").strip():
        b["api_key"] = f["api_key"].strip()
    mctx = (f.get("model_context", "") or "").strip()
    if mctx:
        b["model_context"] = mctx               # `glob=tokens` lines; parsed on every read
    else:
        b.pop("model_context", None)
    # boolean flags: checkbox present → True, absent → drop the key (= False)
    for flag in ("chat_only", "serverless_only", "local", "auto_restart",
                 "prompt_cache"):
        if f.get(flag):
            b[flag] = True
        else:
            b.pop(flag, None)
    for nkey in ("restart_cooldown_s", "stuck_after_s", "self_retries", "max_wait"):
        v = (f.get(nkey, "") or "").strip()
        if v.isdigit() and int(v) > 0:
            b[nkey] = int(v)
        else:
            b.pop(nkey, None)                  # blank = defaults (600 / 90 / no self-retry / 600)
    pi = (f.get("poll_interval", "") or "").strip()   # float — sub-second polling is legitimate
    try:
        pi_val = float(pi)
    except ValueError:
        pi_val = 0.0
    if pi_val > 0:
        b["poll_interval"] = pi_val
    else:
        b.pop("poll_interval", None)           # blank/0/garbage = the 1.0 s default
    # A cloud backend (Meshy, Tripo): a cloud task API — bills per task, so `paid` is not
    # the operator's choice (the form's box is disabled and therefore NOT submitted; it is
    # forced here). Its max_wait/poll_interval arrive under cloud_* names because
    # #comfyopts already renders those two names, and none of the ComfyUI-only keys apply.
    if new_type in adapters.CLOUD_TYPES:
        b["paid"] = True                       # bills per task — never an unpaid candidate
        for src, dst, cast in (("cloud_max_wait", "max_wait", int),
                               ("cloud_poll_interval", "poll_interval", float)):
            v = (f.get(src, "") or "").strip()
            try:
                val = cast(float(v))
            except ValueError:
                val = 0
            if val > 0:
                b[dst] = val
            else:
                b.pop(dst, None)               # blank = the kind's own defaults
        for k in ("comfy_output_dir", "comfy_input_dir", "auto_restart", "restart_cooldown_s",
                  "stuck_after_s", "self_retries"):
            b.pop(k, None)
    # Anthropic: how the credential is sent, plus the fallback model list used when
    # a subscription token isn't allowed on GET /v1/models.
    if new_type == "anthropic":
        b["auth_mode"] = "api_key" if (f.get("auth_mode") or "") == "api_key" else "subscription"
        models = [m.strip() for m in (f.get("models", "") or "").split(",") if m.strip()]
        if models:
            b["models"] = models
        else:
            b.pop("models", None)
    else:
        b.pop("auth_mode", None)
        b.pop("models", None)
    sd, sd_err = _parse_sampling_form(f)
    if sd_err:
        return HTMLResponse(_page("Backends", f'<p class="bad">{_esc(sd_err)}</p>'
            f'<div class="actions">{_btn("← Back", "/ui/backends", "secondary")}</div>', "backends"))
    if sd:
        b["sampling_defaults"] = sd
    else:
        b.pop("sampling_defaults", None)        # blank = forward the client body untouched
    renamed = 0
    if orig and (oname != name or otype != new_type):
        store.delete_backend(oname, otype)      # identity changed (rename / type change)
        if oname != name:
            renamed = store.rename_backend_references(oname, name)   # re-point aliases
    store.upsert_backend(b)
    _apply_backends()
    if renamed:
        _apply_chat_aliases()              # chat aliases changed → rebind the router
    logger.info(f"ui: saved backend '{name}' ({b['type']} {url})"
                + (f" — re-pointed {renamed} alias(es) from '{oname}'" if renamed else ""))
    return RedirectResponse("/ui/backends", status_code=303)


# What a delete clears, and what it deliberately does not — the wording the confirm
# and result pages share, so both explain the same rule with the same words.
_REF_CLEARED = (
    ("gen", "media aliases", "the candidate goes; the alias and its workflow stay"),
    ("chat", "chat aliases", "the backend's entry goes"),
    ("chat_empty", "chat aliases left mapping nothing",
     "the alias stays, but routes nowhere until you give it a backend"),
    ("rule", "reasoning rules", "the backend drops out of the rule's list"),
    ("user", "user grants", "the whole-backend grant goes from the allow-list"),
)
_REF_KEPT = (
    ("gen_last", "media aliases whose only backend this is",
     "kept: the workflow lives in that candidate — removing it would leave the alias "
     "standing without its workflow"),
    ("rule_last", "reasoning rules whose only backend this is",
     "kept: an empty backend list means EVERY backend, so removing it would silently "
     "widen the rule"),
    ("user_alias", "allow-list entries that name an alias too",
     "kept: the entry grants that alias, not this backend"),
)


def _refs_rows(found: dict, spec) -> str:
    return "".join(
        f"<tr><td>{len(names)}</td><td>{_esc(label)}</td>"
        f"<td><code>{_esc(', '.join(names))}</code></td>"
        f"<td class='muted'>{_esc(why)}</td></tr>"
        for key, label, why in spec if (names := found.get(key) or []))


def _refs_table(found: dict, spec, title: str, hint: str = "") -> str:
    rows = _refs_rows(found, spec)
    if not rows:
        return ""
    return (f"<h3>{title}</h3>" + (f"<p class='hint'>{hint}</p>" if hint else "")
            + f"<table><tr><th></th><th>what</th><th>which</th><th></th></tr>{rows}</table>")


async def backend_del(request: Request):
    """Delete a backend — and clear its name out of the store with it.

    A delete used to remove the backend row alone, so every alias that named it kept
    a candidate pointing at nothing: the backend stayed visible in Media aliases long
    after it was gone (reported 2026-08-27 for evo-x2-gpu, 25 stale references).
    `rename_backend_references` had done this for renames since forever; this is the
    missing counterpart. When something does point at the backend, the click lands on
    a confirm page first — that is the check that was missing — and REFERENCES are all
    that ever get cleared: no alias, rule or user is deleted here."""
    bid = (request.query_params.get("id", "") or request.query_params.get("name", "") or "").strip()
    if not bid:
        return RedirectResponse("/ui/backends", status_code=303)
    name, typ = _parse_bid(bid)
    found = store.backend_references(name) if store.is_active() else {}
    hits = sum(len(v) for v in found.values())

    if hits and request.query_params.get("confirm") != "1":
        body = (f"<h2>Delete backend <code>{_esc(name)}</code>?</h2>"
                "<p class='hint'>It is still named in the store. Deleting removes those "
                "references as well — nothing else: every alias, rule and user stays, they "
                "just stop pointing at this backend.</p>"
                + _refs_table(found, _REF_CLEARED, "Will be cleared")
                + _refs_table(found, _REF_KEPT, "Will be left alone",
                              "Removing these would destroy or widen something, so they stay "
                              "and keep naming a backend that no longer exists — handle them "
                              "yourself if that matters.")
                + '<div class="actions">'
                + _btn("Delete and clean up", f"/ui/backends/delete?id={quote(bid)}&confirm=1", "danger")
                + _btn("Cancel", "/ui/backends", "secondary") + "</div>")
        return HTMLResponse(_page(f"Delete {name}", body, "backends"))

    if hits:
        store.delete_backend_references(name)
    store.delete_backend(name, typ)
    _apply_backends()
    if found.get("chat") or found.get("chat_empty"):
        _apply_chat_aliases()
    if found.get("rule"):
        _apply_reasoning()
    if found.get("user"):
        _apply_users()
    logger.info(f"ui: deleted backend '{name}' ({typ})"
                + (f" — cleared {hits} reference(s)" if hits else ""))
    if not hits:
        return RedirectResponse("/ui/backends", status_code=303)
    body = (f"<h2>Backend <code>{_esc(name)}</code> deleted</h2>"
            + _refs_table(found, _REF_CLEARED, "Cleared")
            + _refs_table(found, _REF_KEPT, "Left alone — still naming a backend that is gone")
            + f'<div class="actions">{_btn("← Backends", "/ui/backends", "secondary")}</div>')
    return HTMLResponse(_page(f"Deleted {name}", body, "backends"))


async def backend_restart(request: Request):
    """Fire-and-forget ComfyUI service restart (ComfyUI-Manager reboot)."""
    bid = (request.query_params.get("id", "") or "").strip()
    if _restart_comfy and bid:
        _restart_comfy(bid)
    return RedirectResponse("/ui/backends", status_code=303)


async def backend_drain(request: Request):
    """Graceful offline: stop new requests now, disable once in-flight drains to 0."""
    bid = (request.query_params.get("id", "") or "").strip()
    if _drain_backend and bid:
        _drain_backend(bid)
    return RedirectResponse("/ui/backends", status_code=303)


async def backend_undrain(request: Request):
    bid = (request.query_params.get("id", "") or "").strip()
    if _cancel_drain and bid:
        _cancel_drain(bid)
    return RedirectResponse("/ui/backends", status_code=303)


async def backend_enable(request: Request):
    """Bring a disabled backend back online (counterpart to drain)."""
    bid = (request.query_params.get("id", "") or "").strip()
    if _set_backend_enabled and bid:
        _set_backend_enabled(bid, True)
    return RedirectResponse("/ui/backends", status_code=303)


# ── Tab: Input ──────────────────────────────────────────────────────────────────

async def input_page(request: Request):
    """Legacy URL — Input now lives under /ui/routing?sub=input."""
    return RedirectResponse("/ui/routing?sub=input", status_code=307)


def _input_body() -> str:
    info = _gateway_info()
    gen = sorted(store.list_aliases().keys()) if store.is_active() else []
    llm = _llm_backends()

    def chips(items):
        inner = " ".join(f"<code>{_esc(i)}</code>" for i in items) or '<span class="muted">none</span>'
        return f'<div style="flex:1;min-width:0;white-space:nowrap;overflow-x:auto">{inner}</div>'
    # Pass-through: every discovered model is callable WITHOUT an alias — bare (routed
    # by the scheduler across the backends that expose it) or as backend/model. Grouped
    # per backend so the backend/model form is obvious.
    # model → hosting backends (a bare id routes across all of them; backend/model pins
    # one). Chat models only — image models (flux.* on localai etc.)
    # are filtered out here. The backends column shows just the host names (chips), the
    # model name lives in the first column, so backend/model is easy to read.
    model_hosts: dict = {}
    for b in llm:
        for m in b.get("models", []):
            if not _is_image_model(m):
                model_hosts.setdefault(m, []).append(b["name"])
    if model_hosts:
        mrows = ""
        for m in sorted(model_hosts):
            hosts = " ".join(f'<span class="badge muted">{_esc(bn)}</span>'
                             for bn in sorted(model_hosts[m]))
            mrows += f'<tr><td><code>{_esc(m)}</code></td><td>{hosts}</td></tr>'
        models_tbl = (f'<table class="sortable" data-sk="input-models"><tr><th>model id</th>'
                      f'<th>on backends — call bare or <code>backend/id</code></th>'
                      f'</tr>{mrows}</table>')
    else:
        models_tbl = '<p class="muted">none discovered</p>'
    return (f"<h2>Input — what clients can call</h2>"
            f"<p class='hint'>Anything below can be the request <code>model</code>. Aliases are "
            f"shortcuts; every discovered model is <b>also callable without an alias</b> — bare "
            f"(routed across its backends by the scheduler, with failover) or pinned as "
            f"<code>backend/model</code>.</p>"
            + _field("Chat aliases", chips(info.get("virtual_models", [])), wide=True)
            + _field("Generation models", chips(gen), wide=True)
            + _field("Endpoints", chips(info.get("endpoints", [])), wide=True)
            + "<h2>Chat models · bare or backend/model</h2>"
            + models_tbl)


# ── Tab: Routing (models + chat routing overview) ───────────────────────────────

def _route_status(r: dict) -> str:
    """Status badges for one alias→backend route from routing_snapshot()."""
    if r.get("routable"):
        out = _badge("routable", "ok")
    elif not r.get("enabled"):
        out = _badge("disabled")
    elif not r.get("healthy"):
        out = _down_badge(r.get("error"))
    elif not r.get("present"):
        out = _badge("model absent", "warn")
    else:
        out = _badge("—")
    if r.get("busy"):
        out += _badge("busy", "warn")
    return out


def _host_chip(h: dict) -> str:
    """A backend chip in the models table, coloured by health/busy. Shows measured
    throughput (tok/s) once known — the signal the scheduler ranks ready backends on."""
    if not h.get("healthy"):
        kind, suffix = "bad", " down"
    elif h.get("busy"):
        kind, suffix = "warn", " busy"
    else:
        kind, suffix = "ok", ""
    tps = h.get("tps") or 0
    spd = f" · {tps:g} tok/s" if tps else ""
    ctx = h.get("ctx") or 0
    win = f" · ctx {_ctx_text(ctx)}" if ctx else ""
    return f'<span class="badge {kind}">{_esc(h["backend"])}{spd}{win}{suffix}</span>'


def _ctx_text(n: int) -> str:
    """32768 → 32k, 1048576 → 1M, 202752 → 198k — the way model cards write it."""
    if n >= 1024 * 1024 and n % (1024 * 1024) == 0:
        return f"{n // (1024 * 1024)}M"
    if n >= 1024:
        k = n / 1024
        return f"{int(k)}k" if k == int(k) else f"{k:.0f}k"
    return str(n)


def _models_table(models: list, sk: str = "routing-image-models") -> str:
    """Bare model id → hosting backends. There is nothing to choose per model any
    more: every request takes the fastest free unpaid backend of the set."""
    rows = ""
    for m in models:
        chips = " ".join(_host_chip(h) for h in m["hosts"]) or '<span class="muted">none</span>'
        sh = _badge("shadowed by alias", "warn") if m.get("shadowed_by_alias") else ""
        rows += f'<tr><td><code>{_esc(m["model"])}</code>{sh}</td><td>{chips}</td></tr>'
    head = "<th>model</th><th>backends</th>"
    return (f'<table class="sortable" data-sk="{sk}"><tr>{head}</tr>{rows}</table>'
            if rows else "<p class='muted'>none discovered yet</p>")


def _img_status(bm: Optional[dict]) -> str:
    if not bm:
        return _badge("unknown")
    if not bm.get("enabled"):
        return _badge("disabled")
    if not bm.get("healthy"):
        return _down_badge(bm.get("error"))
    return _badge("healthy", "ok")


def _routing_chat_body(snap: dict) -> str:
    arows = ""
    for a in snap.get("aliases", []):
        arows += f'<tr class="grp"><td colspan="3">{_esc(a["alias"])}</td></tr>'
        if not a["routes"]:
            arows += '<tr><td colspan="3" class="muted">no mapped backends</td></tr>'
        for r in a["routes"]:
            arows += (f'<tr><td>{_esc(r["backend"])}</td><td><code>{_esc(r["model"])}</code></td>'
                      f'<td>{_route_status(r)}</td></tr>')
    html = ("<h2>Chat aliases → routes</h2>" + (
        '<table class="sortable" data-sk="routing-chat"><tr><th>alias / backend</th><th>model</th>'
        f'<th>status</th></tr>{arows}</table>'
        if arows else "<p class='muted'>No chat aliases configured.</p>"))
    conf = snap.get("conflicts", [])
    if conf:
        crows = ""
        for c in conf:
            covered = ", ".join(c["covered"]) or "—"
            shadowed = (f'<span class="bad">{_esc(", ".join(c["shadowed"]))}</span> {_badge("unreachable", "bad")}'
                        if c["shadowed"] else '<span class="muted">—</span>')
            crows += f'<tr><td><code>{_esc(c["name"])}</code></td><td>{_esc(covered)}</td><td>{shadowed}</td></tr>'
        html += ("<h2>Alias / model collisions</h2>"
                 "<p class='hint'>An alias named like a real model shadows it. <b>shadowed</b> "
                 "backends host that exact model id but the alias doesn't map them → unreachable "
                 "by that name.</p>"
                 '<table class="sortable" data-sk="routing-conflicts">'
                 f'<tr><th>alias</th><th>covered</th><th>shadowed</th></tr>{crows}</table>')
    return html


def _routing_gen_body(bmeta: dict, sel: Optional[str] = None) -> str:
    """Media aliases → the backends that serve them, in two views behind one picker.

    Unfiltered it stays alias-first: a group per alias with its candidates — that
    answers "where does THIS alias run". Pick a backend and it
    flips to backend-first: one flat row per alias on that backend, carrying what is
    per-BACKEND about it (pinned values, bypassed nodes) beside the shared mapping.
    Keeping the grouped shape while filtered would print a group header above a
    single row — twice the height for none of the answer.

    The picker is built from the aliases, not from the live backend list, so an alias
    still pointing at a renamed or deleted backend stays visible instead of silently
    dropping out of the very overview you would use to find it."""
    gen_aliases = store.list_aliases() if store.is_active() else {}
    per_backend: dict = {}
    for alias, cands in gen_aliases.items():
        for c in cands:
            per_backend.setdefault((c.get("backend") or "").strip() or "—", []).append((alias, c, cands))

    opts = "<option value=''>all backends</option>" + "".join(
        f"<option value='{_esc(b)}'{' selected' if b == sel else ''}>{_esc(b)} ({len(v)})</option>"
        for b, v in sorted(per_backend.items()))
    picker = (f"<div style='margin:6px 0 10px'><select style=\"width:auto;{_BOX_STYLE}\" "
              f"onchange=\"location.href='/ui/routing?sub=gen'+"
              f"(this.value?('&amp;backend='+encodeURIComponent(this.value)):'')\">{opts}</select></div>")

    if sel:
        bm = bmeta.get(sel)
        entries = sorted(per_backend.get(sel, []), key=lambda e: e[0].lower())
        rows = ""
        for alias, c, cands in entries:
            k = adapters.cloud_kind(c)      # a cloud alias has an endpoint where a workflow has a mapping
            mapped = (f"{k} · {adapters.cloud_module(k).endpoint_of(c)}" if k
                      else ", ".join((c.get("mapping") or {}).keys()) or "auto")
            pins = len([b for b in (c.get("fixed") or []) if b.get("node")])
            byp = len(c.get("bypass") or [])
            # pins + bypass are the per-backend half of a workflow — the reason this
            # view exists at all; the mapping beside them is shared across candidates.
            local = " · ".join(x for x in ((f"{pins} pinned" if pins else ""),
                                           (f"{byp} bypassed" if byp else "")) if x) or "—"
            others = ", ".join(sorted(x.get("backend", "") for x in cands
                                      if (x.get("backend") or "").strip() != sel)) or "—"
            rows += (f'<tr><td><a href="/ui/mapping?edit={quote(alias)}"><code>{_esc(alias)}</code></a></td>'
                     f'<td>{_esc(c.get("task", ""))}</td>'
                     f'<td class="muted">{_esc(mapped)}</td><td>{_esc(local)}</td>'
                     f'<td class="muted">{_esc(others)}</td></tr>')
        head = (f"<h2>Media aliases on <b>{_esc(sel)}</b> "
                f"<span class='muted' style='font-weight:normal'>· {len(entries)}</span></h2>"
                f"<p class='hint'>{_img_status(bm)} — <b>mapping</b> is shared by all of an "
                f"alias's backends, <b>pinned</b> and <b>bypassed</b> are this backend's own.</p>")
        return (head + picker
                + ('<table class="sortable" data-sk="routing-gen-on"><tr><th>alias</th><th>task</th>'
                   f'<th>mapping</th><th>this backend</th><th>also on</th></tr>{rows}</table>'
                   if rows else "<p class='muted'>No media alias names this backend.</p>"))

    grows = ""
    for alias, cands in sorted(gen_aliases.items()):
        grows += f'<tr class="grp"><td colspan="3">{_esc(alias)}</td></tr>'
        for c in cands:
            bn = c.get("backend", "")
            bm = bmeta.get(bn)
            grows += (f'<tr><td>{_esc(bn)}</td><td>{_esc(c.get("task", ""))}</td>'
                      f'<td>{_img_status(bm)}</td></tr>')
    return ("<h2>Media Generation aliases → backends</h2>"
            "<p class='hint'>A job goes to the fastest free unpaid backend of this set, with "
            "failover (see Mapping). Pick a backend to see everything mapped onto it instead.</p>"
            + picker
            + ('<table class="sortable" data-sk="routing-gen"><tr><th>alias / backend</th><th>task</th>'
               f'<th>status</th></tr>{grows}</table>'
               if grows else "<p class='muted'>No generation aliases configured.</p>"))


def _routing_loras_body(bmeta: dict) -> str:
    """LoRA → hosting ComfyUI backends (from discovery) — searchable, so a client
    string can be checked byte-for-byte against what the backends really expose."""
    per_backend = _backend_loras()
    hosts: dict = {}
    for bn, loras in per_backend.items():
        for name in loras:
            hosts.setdefault(name, []).append(bn)
    if not hosts:
        return ("<h2>LoRAs → backends</h2><p class='muted'>No LoRAs discovered — no ComfyUI "
                "backend up, or none installed.</p>")
    rows = ""
    for name in sorted(hosts):
        chips = " ".join(
            f'<span class="badge {"ok" if (bmeta.get(bn) or {}).get("healthy") else "bad"}">{_esc(bn)}</span>'
            for bn in sorted(hosts[name]))
        rows += f'<tr><td><code>{_esc(name)}</code></td><td>{chips}</td></tr>'
    search = (f"<input id='sf' autocomplete='off' oninput='sfRun()' placeholder='filter LoRAs…' "
              f"style=\"min-width:260px;max-width:420px;{_BOX_STYLE}\">")
    counts = " · ".join(f"{_esc(bn)}: {len(v)}" for bn, v in sorted(per_backend.items()))
    return ("<h2>LoRAs → backends</h2>"
            "<p class='hint'>Installed LoRAs per ComfyUI backend (from discovery, verbatim incl. "
            "subfolder prefixes — requests must match these strings exactly). "
            f"<span class='muted'>{counts}</span></p>"
            f"<div style='margin:6px 0 10px'>{search}</div>"
            "<table class='filterable sortable' data-sk='routing-loras'>"
            f"<tr><th>lora</th><th>on backends</th></tr>{rows}</table>"
            + _FILTER_JS)


async def routing_page(request: Request):
    """Parent tab Input & Routing: what clients can call + how it resolves —
    sub-tabs Input | Chat aliases | LLM models | Media aliases | Image models |
    LoRAs (?sub=, first child = default)."""
    sub = request.query_params.get("sub") or SUBTABS["routing"][0][0]
    info = _gateway_info()
    bmeta = {b["name"]: b for b in info.get("backends", []) if b.get("type") in adapters.GEN_TYPES}
    if sub == "chat":
        title, body = "Chat aliases", _routing_chat_body(_routing_snapshot())
    elif sub == "llm":
        snap = _routing_snapshot()
        on_llm = [m for m in snap.get("models", [])
                  if any(h.get("type") not in adapters.GEN_TYPES for h in m["hosts"])]
        title, body = "LLM models", (
            "<h2>LLM models → backends</h2>"
            "<p class='hint'>A bare model id goes to the fastest free <b>unpaid</b> backend of "
            "this set (measured tok/s, shown on each chip; unmeasured backends are probed first), "
            "failing over on error. <b>shadowed by alias</b> = a chat alias of the same name "
            "intercepts the bare id.</p>"
            + _models_table([m for m in on_llm if not _is_image_model(m["model"])],
                            sk="routing-llm-models"))
    elif sub == "gen":
        title, body = "Media aliases", _routing_gen_body(
            bmeta, (request.query_params.get("backend") or "").strip() or None)
    elif sub == "image":
        snap = _routing_snapshot()
        img_models = [m for m in snap.get("models", [])
                      if any(h.get("type") in adapters.GEN_TYPES for h in m["hosts"])]
        img_on_llm = [m for m in snap.get("models", [])
                      if any(h.get("type") not in adapters.GEN_TYPES for h in m["hosts"])
                      and _is_image_model(m["model"])]
        title, body = "Image models", (
            "<h2>Image models → backends</h2>"
            "<p class='hint'>ComfyUI checkpoints/models, plus image models served by LLM "
            "backends (e.g. flux.* on localai — matched by name, no type metadata).</p>"
            + _models_table(img_models + img_on_llm))
    elif sub == "loras":
        title, body = "LoRAs", _routing_loras_body(bmeta)
    else:
        sub, title, body = "input", "Input", _input_body()
    return HTMLResponse(_page(title, body, "routing", subnav=_subnav("routing", sub)))


# ── Tab: Chat (LLM alias management) ────────────────────────────────────────────
# A chat alias maps to either one model id (same on every backend → stored as a
# string) or a per-backend table {backend: model}. config aliases are the base; UI
# entries (this store) merge over them — exactly the router's shapes. Older entries
# may still carry the {model, priority} shape; the priority is parsed and ignored.

def _chat_summary(value) -> str:
    """One-line routing summary for the list sub-line."""
    if isinstance(value, str):
        return f"all backends → {value}"
    if isinstance(value, dict):
        parts = []
        for bn, entry in value.items():
            if isinstance(entry, dict):
                parts.append(f"{bn}→{entry.get('model')}")
            else:
                parts.append(f"{bn}→{entry}")
        return " · ".join(parts) or "—"
    return "—"


def _datalist(dlid: str, models: list) -> str:
    return f'<datalist id="{_esc(dlid)}">' + "".join(f'<option value="{_esc(m)}">' for m in models) + "</datalist>"


def _dl_input(name: str, value, dlid: str, placeholder: str = "model id") -> str:
    return (f'<input type="text" name="{_esc(name)}" value="{_esc(value)}" list="{_esc(dlid)}" '
            f'placeholder="{_esc(placeholder)}">')


def _chat_value_for(alias: str) -> dict:
    """An alias's per-backend mapping for editing: the store entry if present, else
    the config value materialised to per-backend form (a config string expands to
    every LLM backend that currently exposes that model)."""
    v = store.get_chat_alias(alias)
    if v is None:
        v = _config_chat_aliases().get(alias)
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, str):
        return {b["name"]: v for b in _llm_backends() if v in b.get("models", [])}
    return {}


def _chat_new_form() -> str:
    """Minimal create form — like registering a generation alias. More backends are
    assigned in the editor afterwards."""
    llm = _llm_backends()
    all_models = sorted({m for b in llm for m in b.get("models", [])})
    bopts = [b["name"] for b in llm] or [("", "(no LLM backends)")]
    return ('<form action="/ui/chat/create" method="post">'
            f'<div class="formbar"><h2>New Chat Alias</h2>{_btn("Create", submit=True)}'
            f'{_btn("Cancel", "/ui/mapping", "secondary")}</div>'
            + _field("alias name", _inp("alias", placeholder="fast"), short=True)
            + _field("backend", _select("backend", bopts), short=True)
            + _field("model", _dl_input("model", "", "cm_all"), short=True)
            + "<p class='hint'>Pick a first backend + model. Assign more backends after "
              "creating.</p>"
            + "</form>" + _datalist("cm_all", all_models))


def _chat_editor(alias: str) -> str:
    """Editor for an existing alias — same logic as the Mapping editor: a list of
    assigned backends, each with the model to use there; add via dropdown, remove via
    ✕. Models save on Save; add/remove are immediate."""
    llm = _llm_backends()
    meta = {b["name"]: b for b in llm}
    assigned = _chat_value_for(alias)
    rows, dls = "", ""
    for i, (bn, entry) in enumerate(assigned.items()):
        model = entry.get("model", "") if isinstance(entry, dict) else entry
        b, dlid = meta.get(bn, {}), f"cm_{i}"
        off = "" if b.get("enabled", True) else " <span class='muted'>(disabled)</span>"
        head = f"{_esc(bn)}{off}"
        rm = (_btn("✕", f"/ui/chat/bdel?alias={_esc(alias)}&backend={_esc(bn)}", "danger",
                   sm=True, icon=True, title="Remove this backend")
              if len(assigned) > 1 else "<span class='muted' title='alias needs ≥1 backend'>—</span>")
        rows += (f"<tr><td>{head}</td><td>{_dl_input('model__' + bn, model, dlid)}</td>"
                 f"<td class='acts'>{rm}</td></tr>")
        dls += _datalist(dlid, b.get("models", []))
    rows = rows or "<tr><td colspan=3 class='muted'>no backends — add one below</td></tr>"
    add_opts = [b["name"] for b in llm if b["name"] not in assigned]
    add_sel = ""
    if add_opts:
        opts = "".join(f"<option>{_esc(o)}</option>" for o in add_opts)
        add_sel = ('<select class="addsel" onchange="if(this.value)location.href=\'/ui/chat/badd?alias='
                   f"{_esc(alias)}&amp;backend='+encodeURIComponent(this.value)\">"
                   f'<option value="">+ Add backend…</option>{opts}</select>')
    cur_park = store.get_alias_park().get(alias)
    park_field = (_field("park seconds",
                         _inp("park_s", "" if cur_park is None else cur_park,
                              placeholder="blank = global default · 0 = off", typ="number"), short=True)
                  + "<p class='hint' style='margin:-4px 0 10px'>When all backends are busy, a call waits in "
                    "the queue up to this long for a free slot, then gets a 503. Blank = the global default "
                    "(Server tab); <b>0</b> disables parking for this alias.</p>")
    cur_rsn = store.get_alias_reasoning().get(alias) or "auto"
    rsn_opts = "".join(f'<option value="{v}"{" selected" if cur_rsn == v else ""}>{v}</option>'
                       for v in ("auto", "on", "off"))
    rsn_field = (_field("reasoning", f'<select name="reasoning">{rsn_opts}</select>', short=True)
                 + "<p class='hint' style='margin:-4px 0 10px'>Default thinking mode for this alias, applied "
                   "via the <a href='/ui/reasoning'>Reasoning</a> rules (model×backend decide the mechanism). "
                   "<b>auto</b> = model default; an explicit client <code>reasoning</code> field always wins. "
                   "Lets e.g. <code>tool</code> (off) and <code>tool-thinking</code> (auto/on) share one "
                   "backend+model.</p>")
    # Voice defaults are TTS-only and irrelevant for the vast majority of chat aliases,
    # so they collapse out of the way — folded open only when this alias has them set.
    # (They live here, not with the media aliases: /v1/audio/speech routes through a
    # CHAT alias on an openai-type backend, not through the generation store.)
    cur_voice = store.get_alias_voice().get(alias) or {}
    voice_field = (
        f'<details class="optblock"{" open" if (cur_voice.get("voice") or cur_voice.get("ref_text")) else ""}>'
        "<summary>Voice defaults <span class='muted'>— TTS aliases only</span></summary>"
        + _field("voice default", _inp("voice_ref", cur_voice.get("voice", ""),
                 placeholder="backend-side reference, e.g. voices/kai-ref.wav"))
        + _field("voice ref text", _inp("voice_ref_text", cur_voice.get("ref_text", ""),
                 placeholder="exact transcript of the reference recording"))
        + "<p class='hint' style='margin:-4px 0 10px'>TTS defaults for <code>/v1/audio/speech</code> "
          "via this alias: filled in when the client sends no <code>voice</code>/<code>ref_text</code> "
          "(explicit client fields always win). Reference recordings are managed in "
          "<a href='/ui/playground?sub=voice'>Playground → Voice</a>.</p>"
        + "</details>")
    # Sampling defaults are set on few aliases, so they collapse like the voice block —
    # folded open only when this alias carries them.
    cur_smp = store.get_alias_sampling().get(alias)
    smp_field = (
        f'<details class="optblock"{" open" if cur_smp else ""}>'
        "<summary>Sampling defaults <span class='muted'>— client &gt; alias &gt; backend</span></summary>"
        + _sampling_inputs(cur_smp)
        + "<p class='hint' style='margin:-4px 0 10px'>Filled into requests on this alias, "
          "for keys the client did <b>not</b> send. Precedence: client &gt; alias &gt; the serving "
          "backend's own <a href='/ui/backends'>sampling defaults</a> (an alias value overrides the "
          "backend's for that key; the backend's other keys still apply). "
          "Chat/completions/responses only.</p>"
        + "</details>")
    return ('<form action="/ui/chat/save" method="post">'
            f'<input type="hidden" name="orig" value="{_esc(alias)}">'
            f'<div class="formbar"><h2>Edit Chat Alias</h2>{_btn("Save", submit=True)}'
            f'{_btn("Cancel", "/ui/mapping", "secondary")}</div>'
            + _field("alias name", _inp("alias", alias, placeholder="fast"), short=True)
            + park_field + rsn_field + voice_field + smp_field
            + "<h2>Backends</h2>"
            + "<p class='hint'>Assign backends to this alias and pick the model on each. A call "
              "takes the fastest free unpaid one of them, with failover.</p>"
            + f"<table class='pins'><tr><th>backend</th><th>model</th><th></th></tr>{rows}</table>"
            + add_sel
            + "</form>" + dls)


async def chat_create(request: Request):
    f = await _form(request)
    alias = (f.get("alias", "") or "").strip()
    backend = (f.get("backend", "") or "").strip()
    model = (f.get("model", "") or "").strip()
    if not alias or not backend:
        return RedirectResponse("/ui/mapping?cnew=1", status_code=303)
    store.upsert_chat_alias(alias, {backend: model})
    _apply_chat_aliases()
    logger.info(f"ui: chat alias '{alias}' created → {backend}/{model or '(no model)'}")
    return RedirectResponse(f"/ui/mapping?cedit={alias}", status_code=303)


async def chat_save(request: Request):
    f = await _form(request)
    alias = (f.get("alias", "") or "").strip()
    orig = (f.get("orig", "") or "").strip()
    value = {}                                # one entry per rendered backend row
    for key in f:
        if not key.startswith("model__"):
            continue
        bn = key[len("model__"):]
        value[bn] = (f.get(key) or "").strip()
    park_s = (f.get("park_s", "") or "").strip()
    rsn = (f.get("reasoning", "") or "").strip()
    smp, smp_err = _parse_sampling_form(f)
    if smp_err:
        return HTMLResponse(_page("Chat aliases", f'<p class="bad">{_esc(smp_err)}</p>'
            f'<div class="actions">{_btn("← Back", "/ui/mapping", "secondary")}</div>', "routing"))
    if not alias or not value:
        return RedirectResponse(f"/ui/mapping?cedit={orig}" if orig else "/ui/mapping", status_code=303)
    if orig and orig != alias and store.get_chat_alias(orig) is not None:
        store.delete_chat_alias(orig)         # renamed a store entry → move it
        store.set_alias_park(orig, None)      # drop the old name's overrides
        store.set_alias_reasoning(orig, None)
        store.set_alias_voice(orig, None)
        store.set_alias_sampling(orig, None)
    store.upsert_chat_alias(alias, value)
    store.set_alias_park(alias, park_s if park_s != "" else None)   # blank → global default
    store.set_alias_reasoning(alias, rsn)                           # 'auto'/blank clears
    store.set_alias_voice(alias, {"voice": f.get("voice_ref", ""),  # blank fields clear
                                  "ref_text": f.get("voice_ref_text", "")})
    store.set_alias_sampling(alias, smp)                            # blank clears
    _apply_chat_aliases()
    logger.info(f"ui: chat alias '{alias}' = {value} (park_s={park_s or 'default'}, "
                f"reasoning={rsn or 'auto'}"
                + (f", sampling={smp}" if smp else "") + ")")
    return RedirectResponse("/ui/mapping", status_code=303)


async def chat_badd(request: Request):
    """Assign a backend to an alias (immediate, like the Mapping editor's + Add)."""
    alias, backend = _qp(request, "alias"), _qp(request, "backend")
    valid = {b["name"] for b in _llm_backends()}
    cur = _chat_value_for(alias)
    if backend in valid and backend not in cur:
        cur[backend] = ""                     # model filled in the editor, then Save
        store.upsert_chat_alias(alias, cur)
        _apply_chat_aliases()
    return RedirectResponse(f"/ui/mapping?cedit={alias}", status_code=303)


async def chat_bdel(request: Request):
    """Unassign a backend from an alias (an alias keeps at least one)."""
    alias, backend = _qp(request, "alias"), _qp(request, "backend")
    cur = _chat_value_for(alias)
    if backend in cur and len(cur) > 1:
        del cur[backend]
        store.upsert_chat_alias(alias, cur)
        _apply_chat_aliases()
    return RedirectResponse(f"/ui/mapping?cedit={alias}", status_code=303)


async def chat_del(request: Request):
    alias = (request.query_params.get("alias", "") or "").strip()
    if alias:
        store.delete_chat_alias(alias)
        store.set_alias_park(alias, None)     # drop the alias's overrides with it
        store.set_alias_reasoning(alias, None)
        store.set_alias_voice(alias, None)
        store.set_alias_sampling(alias, None)
        _apply_chat_aliases()
        logger.info(f"ui: chat alias '{alias}' deleted")
    return RedirectResponse("/ui/mapping", status_code=303)


# ── Tab: Chat Playground ────────────────────────────────────────────────────────

# Heuristic to spot image-generation models served by an OpenAI-compatible backend
# (e.g. localai exposes flux.* with no `type` field). Name-based, since these backends
# carry no metadata to classify by — extend the hint list as needed.
_IMG_MODEL_HINTS = ("flux", "stable-diffusion", "sdxl", "sd3", "sd-", "dall-e", "dalle",
                    "pixart", "kandinsky", "playground-v", "kolors", "wan2", "hidream")


def _is_image_model(mid: str) -> bool:
    m = (mid or "").lower()
    return any(h in m for h in _IMG_MODEL_HINTS)


def _anthropic_backends() -> set:
    return {b["name"] for b in _llm_backends() if b.get("type") == "anthropic"}


def _anthropic_only(alias: str, anthro: set) -> bool:
    """True when only Anthropic backends can serve this alias. Such an alias is
    unreachable outside /v1/messages (see main.serves_path) — the playground must
    not offer it, because a subscription credential is licensed for Claude Code,
    not for a chat console.

    Both alias shapes count: a dict alias names its backends outright, a string
    alias applies to every backend but is only served where the model exists."""
    if not anthro:
        return False
    v = _config_chat_aliases().get(alias)
    if v is None and store.is_active():
        v = store.get_chat_alias(alias)
    if isinstance(v, dict):
        return bool(v) and set(v.keys()) <= anthro
    if isinstance(v, str) and v:
        serving = {b["name"] for b in _llm_backends() if v in (b.get("models") or [])}
        return bool(serving) and serving <= anthro
    return False


def _chat_models() -> list:
    """Everything callable as a chat `model`: aliases first, then bare model ids
    (scheduler-routed) and backend/model forms — feeds the model datalist. Anthropic
    backends and their alias-only entries are left out: they answer /v1/messages
    only, so offering them here would just produce a 503 (and inviting a
    subscription backend into a chat console is exactly what it is not for)."""
    anthro = _anthropic_backends()
    aliases = [a for a in _gateway_info().get("virtual_models", [])
               if not _anthropic_only(a, anthro)]
    llm = [b for b in _llm_backends() if b.get("type") != "anthropic"]
    bare = sorted({m for b in llm for m in b.get("models", [])})
    prefixed = sorted(f"{b['name']}/{m}" for b in llm for m in b.get("models", []))
    out, seen = [], set()
    for x in aliases + bare + prefixed:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


async def _self_api(request: Request, method: str, path: str, **kw) -> httpx.Response:
    """Call the gateway's OWN API as a real client — the playground contract: they
    exist to TEST the API, so they go through it (auth, routing, parking, reasoning,
    stats, quotas) and bypass nothing. Auth = the logged-in admin's own key (correct
    attribution in LLM Calls), else the master key; anonymous in bootstrap-open mode,
    where the x-source header attributes the call as 'playground'."""
    key = _playground_key(_session_user(request) or "admin")
    headers = {"x-source": "playground"}
    if key:
        headers["authorization"] = f"Bearer {key}"
    url = str(request.base_url).rstrip("/") + path       # our own listener, whatever host/port serves /ui
    async with httpx.AsyncClient(timeout=600.0) as c:    # sync calls may park before running
        return await c.request(method, url, headers=headers, **kw)


# On submit, show immediate feedback: replace the (possibly stale) result with a
# "sending…" spinner and disable Send. The full-page POST then re-renders with the
# real reply — so a slow call, or a second Send, no longer leaves the old result up.
_CHATPLAY_JS = ("<script>function cpSending(f){"
                "var r=document.getElementById('cpresult');"
                "if(r)r.innerHTML=\"<h2>Response</h2><p class='muted'>\\u23f3 <b>Sending\\u2026</b> · routing + "
                "waiting for the backend</p>\";"
                "var b=f.querySelector('button[type=submit]');if(b){b.disabled=true;b.textContent='Sending\\u2026';}"
                "return true;}</script>")


def _chatplay_form(vals: dict) -> str:
    v = lambda k: vals.get(k, "")
    llm = _llm_backends()
    bk_models = {b["name"]: sorted(b.get("models", [])) for b in llm}   # bare model ids per backend
    all_models = _chat_models()                                         # aliases + bare + backend/model
    cur_bk = v("backend")
    init_models = bk_models[cur_bk] if (cur_bk in bk_models) else all_models   # pre-filter if a backend is picked
    # Backend picker (manual, so it can filter the model datalist on change without a reload).
    opts = f'<option value=""{"" if cur_bk else " selected"}>— all backends (scheduler picks) —</option>'
    for n in bk_models:
        opts += f'<option{" selected" if n == cur_bk else ""}>{_esc(n)}</option>'
    bk_select = f'<select name="backend" onchange="cpFilterModels(this.value)">{opts}</select>'
    filt_js = ("<script>var CP_BK=%s,CP_ALL=%s;function cpFilterModels(bk){"
               "var dl=document.getElementById('cpmodels');if(!dl)return;"
               "var ms=(bk&&CP_BK[bk])?CP_BK[bk]:CP_ALL;"
               "dl.innerHTML=ms.map(function(m){var o=document.createElement('option');o.value=m;"
               "return o.outerHTML;}).join('');}</script>") % (json.dumps(bk_models), json.dumps(all_models))
    return ('<form action="/ui/chatplay/send" method="post" onsubmit="return cpSending(this)">'
            f'<div class="formbar"><h2>Chat Playground</h2>{_btn("Send", submit=True)}</div>'
            + _field("backend", bk_select, short=True)
            + _field("model", _dl_input("model", v("model"), "cpmodels", "alias or model id"), short=True)
            + _field("system", _textarea("system", v("system"), 2, "optional system prompt"))
            + _field("message", _textarea("user", v("user"), 6, "your message"))
            + _field("max tokens", _inp("max_tokens", v("max_tokens"), typ="number"), short=True)
            + _field("temperature", _inp("temperature", v("temperature"), typ="number"), short=True)
            + _field("reasoning", "<select name='reasoning'>" + "".join(
                f'<option value="{x}"{" selected" if (v("reasoning") or "auto") == x else ""}>{x}</option>'
                for x in ("auto", "on", "off")) + "</select>", short=True)
            + "<p class='hint'>Non-streaming. Routed by the scheduler with failover, exactly like the API. "
              "Pick a <b>backend</b> to pin the call and filter the model list; empty = all backends. "
              "<b>reasoning</b> sends the API's thinking switch (applied via the "
              "<a href='/ui/reasoning'>Reasoning</a> rules; auto = model default — an alias's own "
              "default applies only when calling the alias, not a backend/model form).</p>"
            + "</form>" + _datalist("cpmodels", init_models) + filt_js + _CHATPLAY_JS)


def _chatplay_body(vals: dict, result_html: str) -> str:
    return (f'<div class="cols"><div class="col">{_chatplay_form(vals)}</div>'
            f'<div class="col" id="cpresult">{result_html}</div></div>')


def _chat_result_html(res: dict) -> str:
    data, status = res.get("response") or {}, res.get("status")
    rsn = f" · reasoning <b>{_esc(res['reasoning'])}</b>" if res.get("reasoning") else ""
    meta = (f"backend <b>{_esc(res.get('backend'))}</b> · model {_esc(res.get('model'))} · "
            f"HTTP {status}{rsn}")
    if status != 200:
        return (f"<h2>Response</h2><p class='bad'>{meta}</p>"
                f"<pre class='err'>{_esc(json.dumps(data, indent=2)[:4000])}</pre>")
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        content = json.dumps(data, indent=2)[:4000]
    usage = data.get("usage") or {}
    utxt = (f" · tokens {usage.get('prompt_tokens', '?')}+{usage.get('completion_tokens', '?')}"
            if usage else "")
    return (f"<h2>Response</h2><p class='muted'>{meta}{utxt}</p>"
            f"<div class='chatout'>{_esc(content)}</div>")


_CHATPLAY_KEYS = ("backend", "model", "system", "user", "max_tokens", "temperature", "reasoning")


async def chatplay_page(request: Request):
    """Legacy URL — the Chat Playground now lives under /ui/playground?sub=chat."""
    q = dict(request.query_params)
    q["sub"] = "chat"
    return RedirectResponse(f"/ui/playground?{urlencode(q)}", status_code=307)


async def chatplay_send(request: Request):
    f = await _form(request)
    vals = {k: (f.get(k, "") or "") for k in _CHATPLAY_KEYS}
    model, user = vals["model"].strip(), vals["user"].strip()
    if not model or not user:
        result = "<h2>Response</h2><p class='bad'>model and message are required</p>"
        return HTMLResponse(_page("Chat Playground", _chatplay_body(vals, result), "playground",
                                  subnav=_subnav("playground", "chat")))
    messages = []
    if vals["system"].strip():
        messages.append({"role": "system", "content": vals["system"]})
    messages.append({"role": "user", "content": vals["user"]})
    params = {}
    if vals["max_tokens"].strip():
        try:
            params["max_tokens"] = int(vals["max_tokens"])
        except ValueError:
            pass
    if vals["temperature"].strip():
        try:
            params["temperature"] = float(vals["temperature"])
        except ValueError:
            pass
    backend = vals["backend"].strip()
    # A picked backend pins the request to it via the '<backend>/<model>' convention
    # (same as the API); empty = route across all backends via the scheduler.
    send_model = f"{backend}/{model}" if (backend and not model.startswith(backend + "/")) else model
    # A REAL API call through the gateway's own /v1 endpoint — dispatch, parking,
    # reasoning, stats and quotas all apply, exactly like any external client.
    body = {"model": send_model, "messages": messages, "stream": False, **params}
    if vals["reasoning"].strip() in ("on", "off"):     # auto → field omitted (API default)
        body["reasoning"] = vals["reasoning"].strip()
    try:
        r = await _self_api(request, "POST", "/v1/chat/completions", json=body)
        try:
            data = r.json()
        except Exception:
            data = {"raw": r.text[:4000]}
        result = _chat_result_html({
            "status": r.status_code,
            "backend": r.headers.get("x-gateway-backend", "—"),
            "model": (data.get("model") if isinstance(data, dict) else None) or send_model,
            "alias": send_model, "response": data,
            "reasoning": r.headers.get("x-reasoning-control"),
        })
    except httpx.HTTPError as e:
        result = f"<h2>Response</h2><p class='bad'>Error: {_esc(f'{type(e).__name__}: {e}')}</p>"
    return HTMLResponse(_page("Chat Playground", _chatplay_body(vals, result), "playground",
                              subnav=_subnav("playground", "chat")))


# ── Tab: Mapping ────────────────────────────────────────────────────────────────

def _register_form() -> str:
    backend_opts = [b["name"] for b in _gen_backends()] or [("", "(no generation backends)")]
    return ('<form action="/ui/mapping/register" method="post" enctype="multipart/form-data">'
            f'<div class="formbar"><h2>Register Workflow</h2>{_btn("Register", submit=True)}'
            f'{_btn("Cancel", "/ui/mapping?sub=media", "secondary")}</div>'
            "<p class='hint'>The gateway <b>owns</b> the API JSON once registered — independent of "
            "later ComfyUI-GUI edits. You'll map fields after registering. For a <b>cloud</b> backend "
            "(meshy, tripo) no JSON is needed — the alias is created with that vendor's defaults and "
            "edited next.</p>"
            + _field("alias", _inp("alias", placeholder="flux"))
            + _field("backend", _select("backend", backend_opts))
            + _field("task", _task_select())
            + _field("API JSON file", '<input type="file" name="workflow_file" accept=".json,application/json">')
            + _field("…or share path", _inp("workflow_path", placeholder="/mnt/share/flux_api.json"))
            + "</form>")


def _mapping_list(cedit: str, iedit: str, sub: str = "chat") -> str:
    """The left column for ONE sub-tab: chat aliases (?cedit=) or media workflows
    (?edit=). They used to share a single scrolling list, which grew past the point
    where either was findable — the two are edited in completely different ways, so
    they get their own tab each."""
    if sub == "media":
        return _mapping_list_media(iedit)
    return _mapping_list_chat(cedit)


def _mapping_list_chat(cedit: str) -> str:
    # Chat group (config + UI-managed, merged)
    cfg, ui = _config_chat_aliases(), store.list_chat_aliases()
    chat_items = ""
    for name in sorted(set(cfg) | set(ui)):
        in_ui = name in ui
        val = ui.get(name) if in_ui else cfg.get(name)
        src = (_badge("ui", "ok", "Defined/edited in this UI (stored in the gateway; overrides config.yaml)")
               if in_ui else
               _badge("config", "muted", "From config.yaml (read-only base; Edit creates a UI override)"))
        specs = [("✎", f"/ui/mapping?cedit={_esc(name)}", "secondary", "Edit")]
        if in_ui:
            lbl = "Delete override (revert to config)" if name in cfg else f"Delete {name}?"
            specs.append(("✕", f"/ui/chat/delete?alias={_esc(name)}", "danger", "Delete", lbl))
        chat_items += _item(f"{_esc(name)} {src}", _chat_summary(val), _icon_acts(*specs),
                            sel=(name == cedit))
    chat_items = chat_items or "<p class='muted'>No chat aliases — + Chat alias.</p>"
    bar = ('<div class="bar"><h2>Chat aliases</h2>'
           f'<div style="display:flex;gap:8px">{_btn("+ Chat alias", "/ui/mapping?cnew=1")}</div></div>')
    legend = ("<p class='hint' style='margin:2px 0 6px'>"
              + _badge("config") + " from config.yaml · "
              + _badge("ui", "ok") + " created/edited here (overrides config)</p>")
    return bar + legend + chat_items


def _mapping_list_media(iedit: str) -> str:
    """Generation aliases (image/video/audio/mesh), grouped by task.

    Flat, the list runs to dozens of entries in which a text2img workflow sits
    between two rigging ones. Grouping by `task` is what you actually navigate by —
    you come here to edit "one of the mesh flows", not "the 14th alias". Groups
    follow _TASK_OPTIONS order (the pipeline's own order: image → video → mesh),
    anything unrecognised trails alphabetically so a typo'd task stays visible
    instead of silently vanishing."""
    by_task: dict[str, list] = {}
    for alias, cands in sorted(store.list_aliases().items()):
        by_task.setdefault((cands[0].get("task") or "").strip() or "—", []).append((alias, cands))

    order = [t for t in _TASK_OPTIONS if t in by_task]
    order += sorted(t for t in by_task if t not in _TASK_OPTIONS)

    body = ""
    for task in order:
        entries = by_task[task]
        body += (f'<div class="grouphdr">{_esc(task)} '
                 f'<span class="muted" style="font-weight:normal">{len(entries)}</span></div>')
        for alias, cands in entries:
            c = cands[0]
            k = adapters.cloud_kind(c)      # a cloud alias has an endpoint where a workflow has a mapping
            mapped = (f"{k} · {adapters.cloud_module(k).endpoint_of(c)}" if k
                      else ", ".join((c.get("mapping") or {}).keys()) or "auto")
            backends = ", ".join(x.get("backend", "") for x in cands)
            acts = _icon_acts(
                ("✎", f"/ui/mapping?edit={_esc(alias)}", "secondary", "Edit"),
                ("⧉", f"/ui/mapping/copy?alias={_esc(alias)}", "secondary", "Copy"),
                ("✕", f"/ui/mapping/delete?alias={_esc(alias)}", "danger", "Delete", f"Delete {alias}?"))
            # the task is the group header now — the row shows what differs within it
            body += _item(_esc(alias), f"{backends} · {mapped}", acts, sel=(alias == iedit))
    body = body or "<p class='muted'>No workflows — + Workflow.</p>"
    bar = ('<div class="bar"><h2>Media workflows</h2>'
           f'<div style="display:flex;gap:8px">'
           f'{_btn("⬇ Export all", "/ui/mapping/export-all", "secondary", title="Download all cleaned workflows as a zip")}'
           f'{_btn("+ Workflow", "/ui/mapping?new=1")}</div></div>')
    return bar + body


async def mapping_page(request: Request):
    if not store.is_active():
        return _inactive()
    qp = request.query_params
    cedit, iedit = qp.get("cedit", ""), qp.get("edit", "")
    # Which sub-tab: an explicit ?sub= wins, otherwise the edit target decides. That
    # keeps every existing action link working — they redirect to ?edit=/?cedit= and
    # land in the matching tab without carrying a sub of their own.
    sub = qp.get("sub", "")
    if sub not in ("chat", "media"):
        sub = "media" if (iedit or qp.get("new")) else "chat"
    list_html = _mapping_list(cedit, iedit, sub)

    def cols(*panels):
        return ('<div class="cols">'
                + "".join(f'<div class="col">{p}</div>' for p in panels) + "</div>")

    chat_names = set(_config_chat_aliases()) | set(store.list_chat_aliases())
    if cedit and cedit in chat_names:
        body = cols(list_html, _chat_editor(cedit))          # chat editor (2 cols)
    elif qp.get("cnew"):
        body = cols(list_html, _chat_new_form())
    elif iedit and store.get(iedit):
        # image editor (3 cols); the post-Save confirmation rides in the form bar so
        # Save leaves the editor's scroll position untouched
        editor, available = await _alias_editor(iedit, saved=bool(qp.get("saved")))
        # wider editor (col 2), narrower Available fields (col 3) — see .cols.map3 CSS
        body = ('<div class="cols map3">'
                f'<div class="col">{list_html}</div>'
                f'<div class="col">{editor}</div>'
                f'<div class="col">{available}</div></div>')
    elif qp.get("new"):
        body = cols(list_html, _register_form())
    else:
        what = ("a <b>+ Chat alias</b>" if sub == "chat" else "a <b>+ Workflow</b>")
        detail = (f"<h2>Details</h2><p class='hint'>Pick an entry to <b>Edit</b>, or add {what}.</p>")
        body = cols(list_html, detail)
    return HTMLResponse(_page("Mapping", body, "mapping", subnav=_subnav("mapping", sub)))


async def register_post(request: Request):
    f = await _multipart(request)
    alias = str(f.get("alias", "")).strip()
    backend = str(f.get("backend", "")).strip()
    picked_task = str(f.get("task", "")).strip()      # "" = field absent (API caller)
    task = picked_task or "text2img"
    upload = f.get("workflow_file")
    path = str(f.get("workflow_path", "")).strip()

    def err(msg):
        return HTMLResponse(_page("Register", f'<p class="bad">{_esc(msg)}</p>'
                            f'<div class="actions" style="padding-left:0">{_btn("← Back", "/ui/mapping?sub=media", "secondary")}</div>', "mapping"))
    if not alias or not backend:
        return err("alias and backend are required")
    # A cloud alias (Meshy, Tripo) has no workflow at all: its request fields are the
    # fixed label table in the kind's module, its per-backend half is a set of admin
    # options. Registering creates the default candidate; everything else is edited in
    # the alias editor. The KIND comes from the picked backend, and a name is unique only
    # per type (a ComfyUI and a Meshy backend may both be called "gpu") — so match on
    # (name, type) instead of trusting whichever same-named backend comes first.
    bt = next((b.get("type") for b in _gen_backends()
               if b["name"] == backend and b.get("type") in adapters.CLOUD_TYPES), None)
    if bt:
        cand = adapters.cloud_module(bt).default_candidate(backend)
        # The task dropdown defaults to `text2img`, which no cloud backend can do at all
        # (they only turn images into 3D). So the form's untouched default — like a
        # missing field — keeps default_candidate's `img2mesh`; only a task the user
        # actually PICKED overrides it. Without this the alias landed in the text2img group.
        if picked_task and picked_task != "text2img":
            cand["task"] = picked_task
        store.upsert(alias, [cand])
        logger.info(f"ui: registered '{alias}' -> {backend} ({bt}, no workflow)")
        return RedirectResponse(f"/ui/mapping?edit={quote(alias)}", status_code=303)
    try:
        if isinstance(upload, (bytes, bytearray)) and upload.strip():
            wf = json.loads(upload.decode("utf-8"))
        elif path:
            with open(path) as fh:
                wf = json.load(fh)
        else:
            return err("upload an API JSON file or give a share path")
    except Exception as e:
        return err(f"invalid workflow JSON: {e}")
    if not isinstance(wf, dict):
        return err("workflow JSON must be an object of nodes (API format)")
    oi = await _object_info(backend, wf)
    cand = {"backend": backend, "task": task, "workflow_json": wf,
            "mapping": adapters.suggest_mapping(wf), "fixed": _detect_model_bindings(wf, oi)}
    # Pre-fill the delivery node when the workflow declares its main export by title
    # (`Output` / `output_final`) — same node the auto mode would pick, but visible
    # and editable here, and a run producing nothing then fails clearly. Note the
    # mesh export classes report nothing themselves; final_output_node redirects to
    # the Preview3D node that consumes them (chain export_node is NOT this one).
    if (out_node := adapters.final_output_node(wf)):
        cand["output_node"] = out_node
    store.upsert(alias, [cand])
    logger.info(f"ui: registered '{alias}' → {backend} ({len(wf)} nodes, {len(cand['fixed'])} model slots)")
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def update_workflow(request: Request):
    """Replace an existing alias's workflow_json, **keeping** mapping + fixed +
    backends/retries. Lets the user re-export a tweaked ComfyUI workflow without
    redoing the mapping — only bindings whose node id changed need re-pointing."""
    f = await _multipart(request)
    alias = str(f.get("alias", "")).strip()
    cands = store.get(alias)

    def err(msg):
        return HTMLResponse(_page("Update workflow", f'<p class="bad">{_esc(msg)}</p>'
                            f'<div class="actions" style="padding-left:0">'
                            f'{_btn("← Back", f"/ui/mapping?edit={quote(alias)}", "secondary")}</div>', "mapping"))
    if not cands:
        return err(f"alias '{alias}' not found")
    upload = f.get("workflow_file")
    path = str(f.get("workflow_path", "")).strip()
    try:
        if isinstance(upload, (bytes, bytearray)) and upload.strip():
            wf = json.loads(upload.decode("utf-8"))
        elif path:
            with open(path) as fh:
                wf = json.load(fh)
        else:
            return err("upload an API JSON file or give a share path")
    except Exception as e:
        return err(f"invalid workflow JSON: {e}")
    if not isinstance(wf, dict):
        return err("workflow JSON must be an object of nodes (API format)")
    base_mapping = cands[0].get("mapping") or {}
    for c in cands:
        c["workflow_json"] = wf                          # workflow + mapping are backend-independent →
        c["mapping"] = base_mapping                      # keep synced; `fixed` stays per-backend
    # Pull the node TITLE into the display label (input name) for image-loader slots
    # whose label is still empty — so a re-exported workflow that (re)titles its
    # LoadImage nodes updates their field names. A hand-set label is left untouched.
    for m in base_mapping.values():
        if m and not (m.get("label") or "").strip() and adapters.is_image_field(wf, m.get("node")):
            title = ((wf.get(m.get("node"), {}) or {}).get("_meta") or {}).get("title", "").strip()
            if title:
                m["label"] = title
    store.upsert(alias, cands)
    mapping = base_mapping
    stale = [p for p, m in mapping.items() if (m or {}).get("node") not in wf]
    stale_byp = sorted({str(n) for c in cands for n in (c.get("bypass") or []) if str(n) not in wf})
    logger.info(f"ui: workflow updated for '{alias}' ({len(wf)} nodes); {len(stale)} stale binding(s): {stale}"
                + (f"; {len(stale_byp)} stale bypass node(s): {stale_byp}" if stale_byp else ""))
    return RedirectResponse(f"/ui/mapping?edit={quote(alias)}&saved=1", status_code=303)


def _reorder_js(alias: str) -> str:
    """Vanilla drag-to-reorder for the request-fields rows. On drop, if the order
    changed, persist it via /ui/mapping/field-order (one ?order= per param) and the
    redirect reloads the editor — the same order then drives the Playground."""
    a = json.dumps(alias)        # safe JS string literal
    return ("<script>(function(){"
            "var tb=document.getElementById('reqfields');if(!tb)return;"
            "function ord(){return [].map.call(tb.querySelectorAll('tr[data-p]'),"
            "function(r){return r.getAttribute('data-p')});}"
            "var start=ord().join('\\u0001'),drag=null;"
            "tb.querySelectorAll('tr[draggable]').forEach(function(tr){"
            "tr.addEventListener('dragstart',function(){drag=tr;tr.classList.add('dragging');});"
            "tr.addEventListener('dragend',function(){tr.classList.remove('dragging');});"
            "tr.addEventListener('dragover',function(e){e.preventDefault();"
            "if(!drag||drag===tr)return;var b=tr.getBoundingClientRect();"
            "tb.insertBefore(drag,(e.clientY-b.top)/b.height>0.5?tr.nextSibling:tr);});"
            "tr.addEventListener('drop',function(e){e.preventDefault();"
            "if(ord().join('\\u0001')===start)return;"
            "location.href='/ui/mapping/field-order?alias='+encodeURIComponent(" + a + ")+"
            "'&'+ord().map(function(p){return 'order='+encodeURIComponent(p);}).join('&');});"
            "});})();</script>")


def _same_kind(cands: list, backend_name: str) -> bool:
    """An alias is homogeneous: candidates of ONE kind only — ComfyUI, or one cloud kind
    (the editor, schema and playground read the FIRST candidate as the alias's shape).

    Backends are keyed (name, type), so a bare-name lookup could answer about the
    same-named backend of the OTHER kind — match on the wanted kind directly."""
    want = adapters.cand_kind(cands[0]) if cands else "comfyui"
    return any(x["name"] == backend_name and adapters.backend_kind(x) == want
               for x in _gen_backends())


def _backends_section(alias: str, cands: list) -> str:
    """Allowed backends for an alias — a flat list (no primary/fallback). A job takes
    the fastest free unpaid one of them; on error the job runner moves to the next.
    Add/remove only; adding copies the existing workflow + mapping onto that backend."""
    used = [c.get("backend") for c in cands]
    rows = ""
    for bn in used:
        rm = (_btn("✕", f"/ui/mapping/cand-del?alias={_esc(alias)}&backend={_esc(bn)}",
                   "danger", sm=True, icon=True, title="Remove this backend")
              if len(used) > 1 else "<span class='muted' title='an alias needs ≥1 backend'>—</span>")
        rows += f"<tr><td>{_esc(bn)}</td><td class='acts'>{rm}</td></tr>"
    add_opts = [b["name"] for b in _gen_backends()
                if b["name"] not in used and _same_kind(cands, b["name"])]
    add_sel = ""
    if add_opts:
        opts = "".join(f"<option>{_esc(o)}</option>" for o in add_opts)
        add_sel = ('<select class="addsel" onchange="if(this.value)location.href=\'/ui/mapping/cand-add?alias='
                   f"{_esc(alias)}&amp;backend='+encodeURIComponent(this.value)\">"
                   f'<option value="">+ Add backend…</option>{opts}</select>')
    return (f"<table class='pins'><tr><th>allowed backend</th><th></th></tr>{rows}</table>{add_sel}")


_PIN_CSS_JS = ("<style>.ptabs{display:flex;gap:4px;margin:10px 0 0;flex-wrap:wrap}"
               ".ptab{padding:6px 14px;background:#0c0e12;border:1px solid #242a33;border-bottom:none;"
               "border-radius:8px 8px 0 0;color:#8b97a4;cursor:pointer;font:inherit}"
               ".ptab.on{color:#dce4ec;background:#11151b;border-color:#3b82f6}"
               ".ppanel{border:1px solid #242a33;border-radius:0 8px 8px 8px;padding:10px 12px}"
               ".ovrow{margin:9px 0}.ovrow label{display:block;font-size:12px;color:#8b97a4;margin-bottom:3px}"
               ".ovrow input,.ovrow select{width:100%;max-width:560px;box-sizing:border-box}"
               ".ovrow.inherited{opacity:.5}.ovrow.inherited:focus-within{opacity:1}"
               ".ovact{float:right}.ovtag{font-size:10px;padding:1px 6px;border-radius:6px;margin-left:6px}"
               ".ovtag.inh{background:#1a1f27;color:#7e8b99}.ovtag.set{background:#15301f;color:#7fd6a0}"
               ".ntitle{color:#cbd4de;font-style:italic}</style>"
               "<script>function pinTab(b,p){var f=b.closest('form');"
               "f.querySelectorAll('.ptab').forEach(function(x){x.classList.toggle('on',x===b);});"
               "f.querySelectorAll('.ppanel').forEach(function(x){x.style.display="
               "x.getAttribute('data-pt')===p?'':'none';});}</script>")


def _map_del_btn(alias: str, qs: str) -> str:
    return _btn("✕", f"/ui/mapping/field-del?alias={_esc(alias)}&{qs}", "danger", sm=True,
                icon=True, title="Remove")


def _req_fields_rows(alias: str, wf: dict, mapping: dict, oi: dict) -> str:
    """Request fields: one row PER MAPPED param (dynamic — promoted from the
    Available fields list via →; NOT a fixed list). Rendered in mapping order (NOT
    sorted) so drag-to-reorder sticks; that order drives the Playground. node/field
    stay editable, the `=` column EDITS the workflow default (the value a request
    without this field runs with), ∅ clears it, ✕ removes the param."""
    rows = ""
    for p in mapping:
        m = mapping.get(p) or {}
        node, fld = m.get("node", ""), m.get("field", "")
        is_img = adapters.is_image_field(wf, node)
        cur = (wf.get(node, {}).get("inputs") or {}).get(fld)
        if is_img:
            mode = adapters.slot_empty_mode(m)
            eopts = [("placeholder", "8×8 if empty"), ("required", "required"),
                     ("disable", "disable branch if empty")]
            esel = "".join(f'<option value="{v}"{" selected" if v == mode else ""}>{l}</option>'
                           for v, l in eopts)
            cur_cell = ('image upload <select name="empty__' + _esc(p) + '" style="width:auto" '
                        'title="what to do when the request sends no image for this slot: '
                        '8×8 black placeholder · required (error if missing) · disable the loader '
                        'node AND the dead branch behind it — every node that requires that '
                        'input dies with it, a node whose socket is optional keeps running '
                        'without the image (no depth to configure, it follows the workflow)">'
                        + esel + '</select>'
                        # extra ids for the disable mode: nodes the dead-branch cascade does NOT
                        # reach (optional socket, main path) but that are pointless without the
                        # image — bypassed mode-4, so the path behind them stays connected
                        + ' <input type="text" style="width:9em" name="bypass__' + _esc(p) + '" '
                        'value="' + _esc(", ".join(str(x) for x in (m.get("on_empty_bypass") or [])))
                        + '" placeholder="also bypass: 58,61" '
                        'title="only for &quot;disable branch if empty&quot;: node ids that are '
                        'additionally BYPASSED (ComfyUI mode 4 — consumers reconnect to the '
                        'node\'s same-typed input) when this slot gets no image. For nodes the '
                        'dead-branch cascade cannot take because their image socket is optional, '
                        'but which do nothing useful without it. Comma separated.">')
        elif isinstance(cur, list):
            cur_cell = "(linked)"                        # wired to another node — not editable
        elif node and fld and node in wf:
            cur_cell = _value_control("default__" + p, node, fld, None, wf, oi)
        else:
            cur_cell = ""                                # stale/incomplete binding — nothing to edit
        tag = " <span class='tag'>image</span>" if is_img else ""
        if node and node not in wf:                      # node vanished after a workflow update
            tag += " <span class='badge bad' title='this node no longer exists in the workflow'>stale</span>"
        actions = ((_btn("∅", f"/ui/mapping/field-clear?alias={_esc(alias)}&param={_esc(p)}",
                         "secondary", sm=True, icon=True, title="Clear the workflow default")
                    if not is_img else "")
                   + _map_del_btn(alias, "param=" + _esc(p)))
        rows += (f'<tr draggable="true" data-p="{_esc(p)}">'
                 f"<td><span class='grip' title='Drag to reorder'>⠿</span> {_esc(p)}{tag}</td>"
                 f"<td>{_inp('label__' + p, m.get('label', ''), placeholder=p)}</td>"
                 f"<td>{_inp('node__' + p, node)}</td>"
                 f"<td>{_inp('field__' + p, fld)}</td>"
                 f"<td class='muted'>{cur_cell}</td>"
                 f"<td class='acts'>{actions}</td></tr>")
    return rows or ("<tr><td colspan=6 class='muted'>none yet — promote a field "
                    "from Available fields below (→)</td></tr>")


def _pin_tab_rows(alias: str, c: dict, is_primary: bool, fixed: list, wf: dict, oi_bn: dict) -> str:
    """One pinned-values tab — SAME layout in every tab (.ovrow). The PRIMARY (first
    backend) tab is the editor: value + ✕ delete per slot. Extra backends override
    only the VALUE (no delete; the slot set is shared). A value equal to the
    primary's is flagged "inherited" and dimmed; a differing one "override"."""
    rows = ""
    for b in fixed:
        nid, fld = str(b["node"]), str(b["field"])
        cls = wf.get(nid, {}).get("class_type", "")
        if is_primary:
            cur, name, inherited = b.get("value"), f"fixed__{nid}__{fld}", False
            acts = (f"<span class='ovact'>"
                    f"{_map_del_btn(alias, 'node=' + _esc(nid) + '&field=' + _esc(fld))}</span>")
        else:
            cv = next((x.get("value") for x in (c.get("fixed") or [])
                       if str(x.get("node")) == nid and str(x.get("field")) == fld), None)
            inherited = cv is None or cv == b.get("value")
            cur, name, acts = (cv if cv is not None else b.get("value")), \
                "ovr__" + str(c.get("backend")) + "__" + nid + "__" + fld, ""
        ctl = _value_control(name, nid, fld, cur, wf, oi_bn)
        tag = "" if is_primary else (" <span class='ovtag inh'>inherited</span>" if inherited
                                     else " <span class='ovtag set'>override</span>")
        title = wf.get(nid, {}).get("_meta", {}).get("title", "")
        ttl_html = f' <span class="ntitle">“{_esc(title)}”</span>' if title else ""
        rows += (f'<div class="ovrow{" inherited" if inherited else ""}">'
                 f'<label><code>{_esc(nid)}</code> {_esc(cls)}{ttl_html} '
                 f'<span class="muted">· {_esc(fld)}</span>{tag}{acts}</label>{ctl}</div>')
    return rows or "<p class='muted'>none pinned — add from Available fields below</p>"


def _tf_opts(cands: list) -> str:
    """Options for the delivered-texture-format select (blank = as produced)."""
    cur = next((str(c.get("texture_format")) for c in cands if c.get("texture_format")), "")
    return "".join(
        f'<option value="{v}"{" selected" if cur == v else ""}>{lbl}</option>'
        for v, lbl in (("", "as produced (png)"), ("jpeg", "jpeg — smaller, no alpha")))


def _output_section(wf: dict, cands: list) -> str:
    """Which node's artifacts a job returns — rendered under Pinned values.
    Default (auto) keeps the legacy behaviour: every output-producing node, or
    the one the workflow titles as its main export (`output_final` / `Output`).
    Workflows that export intermediate files (Trellis: meshes at several nodes,
    the rigged model at one) pin the REAL result node here — it producing nothing
    then fails the job instead of silently returning leftovers."""
    cur = next((str(c.get("output_node")) for c in cands if c.get("output_node")), "")
    opts = ('<option value="">auto — every output node '
            '(or the one titled "Output" / "output_final")</option>')
    for nid, n in sorted(wf.items(), key=lambda kv: (len(kv[0]), kv[0])):
        title = ((n.get("_meta") or {}).get("title") or "")
        lbl = f"{nid} — {n.get('class_type', '')}" + (f" “{title}”" if title else "")
        opts += f'<option value="{_esc(nid)}"{" selected" if nid == cur else ""}>{_esc(lbl)}</option>'
    if cur and cur not in wf:                        # keep a stale choice visible instead of silently clearing
        opts += f'<option value="{_esc(cur)}" selected>{_esc(cur)} — (stale: node missing)</option>'
    cur_ext = next((str(c.get("output_ext")) for c in cands if c.get("output_ext")), "")
    cur_globs = next((c.get("output_globs") for c in cands if c.get("output_globs")), []) or []
    cur_cases = next((c.get("output_cases") for c in cands if c.get("output_cases")), []) or []
    globs_text = "\n".join([f"{c.get('rig','')}: {', '.join(c.get('globs') or [])}" for c in cur_cases]
                           + list(cur_globs))        # cases and plain globs coexist
    return ("<h2>Output</h2>"
            "<p class='hint'>Which node's artifacts the job returns. <b>auto</b> collects from every "
            "output-producing node. Pin the final node when the workflow also exports intermediates — "
            "the job then fails clearly if that node produces nothing.</p>"
            + _field("output node", f'<select name="output_node">{opts}</select>')
            + _field("deliver file type", _inp("output_ext", cur_ext, placeholder="as reported (e.g. glb)"), short=True)
            + "<p class='hint'>Fetch the SIBLING file with this extension instead of the one the node "
              "reports — some nodes register only one format but write several next to it (UniRig reports "
              "the <code>.fbx</code> but also writes <code>.glb</code>). Blank = deliver what the node "
              "reports. The job fails clearly if the sibling doesn't exist.</p>"
            + _field("deliver textures as", f'<select name="texture_format">{_tf_opts(cands)}</select>', short=True)
            + "<p class='hint'>For <b>generic</b>-rig deliveries: <b>jpeg</b> transcodes the baked texture "
              "PNGs to JPEG (quality 90) at delivery — ComfyUI has no JPEG export, so the gateway shrinks "
              "the multi-MB bake here. A texture with a real alpha channel keeps PNG. On a chain set this "
              "on the client-facing (stage-1) alias.</p>"
            + _field("texture check", _checkbox("dummy_check",
                     all(c.get("dummy_check") is not False for c in cands),
                     "fail on a 2×2-dummy texture (known export-node bug)"))
            + "<p class='hint'>On by default: a flat (non-case) GLB whose only embedded texture is a "
              "2×2 dummy is treated as a failed generation. <b>Uncheck</b> for a workflow that legitimately "
              "exports a 1×1/2×2 constant-colour texture (uniform models).</p>"
            + _field("output files", _textarea("output_globs", globs_text, 4,
                     "plain globs (deliver all), or 'rig: globs' lines for conditional cases:\n"
                     "mixamo: *.glb\ngeneric: *.fbx, *_basecolor*.png"), wide=True)
            + "<p class='hint'><b>Multi-file delivery</b>. Plain glob lines deliver EVERY match across all "
              "output nodes, including a same-stem sibling with a glob's extension (so <code>*.glb</code> "
              "grabs the .glb next to a reported <code>.fbx</code>). ComfyUI appends a counter to save/export "
              "nodes (<code>&lt;name&gt;_basecolor_00001_.png</code>), so end an artifact glob with "
              "<code>*</code> before the extension. Without an <b>output node</b> the "
              "globs replace it as the whole delivery; WITH one, the node's result stays authoritative and "
              "plain globs ship as unconditional extras on top (e.g. a baked <code>*_metallic*.png</code> "
              "next to the node's GLB). "
              "<br><b>Cases</b> — <code>rig: glob, glob</code> per line: the FIRST case whose first glob "
              "actually exists wins, and only its files ship, tagged with that <b>rig</b> type "
              "(<code>mixamo</code> → validated as a 52-joint humanoid GLB with embedded texture; "
              "<code>generic</code> → a rigged FBX + its basecolor PNG). Plain glob lines may be MIXED "
              "with cases — they ship unconditionally on top of the matched case. Unmatched cases/globs "
              "are simply absent, so a split workflow still works; the job fails if nothing matches or "
              "validation fails (e.g. the 2×2-dummy-texture bug).</p>"
            + _chain_section(wf, cands))


def _chain_section(wf: dict, cands: list) -> str:
    """Workflow chain: this alias is stage 1 (the client interface); on success the
    gateway runs a `successor` alias on the SAME backend, feeding it this stage's
    exported mesh (by full path) + this stage's params, and delivers ONLY the
    successor's result. Both runs are back-to-back (queue-isolated)."""
    s = next((c.get("successor") for c in cands if c.get("successor")), None) or {}
    cur_exp = str(s.get("export_node") or "")
    node_opts = '<option value="">— export node —</option>'
    listed = False
    for nid, n in sorted(wf.items(), key=lambda kv: (len(kv[0]), kv[0])):
        cls = n.get("class_type", "")
        # shortlist writer-ish classes, but ALWAYS include the stored choice — a
        # custom writer class without export/save in its name would otherwise render
        # unselected and the next unrelated Save would silently clear it.
        if "export" in cls.lower() or "save" in cls.lower() or nid == cur_exp:
            title = ((n.get("_meta") or {}).get("title") or "")
            lbl = f"{nid} — {cls}" + (f" “{title}”" if title else "")
            sel = " selected" if cur_exp == nid else ""
            listed = listed or cur_exp == nid
            node_opts += f'<option value="{_esc(nid)}"{sel}>{_esc(lbl)}</option>'
    if cur_exp and not listed:                       # keep a stale choice visible instead of silently clearing
        node_opts += f'<option value="{_esc(cur_exp)}" selected>{_esc(cur_exp)} — (stale: node missing)</option>'
    cur_relay = (s.get("relay") or "path").strip().lower()
    # A cloud successor (Meshy, Tripo) never reads a shared disk — the mesh always travels
    # as bytes. Offering a hand-off choice there would only let the user pick one that is
    # ignored.
    succ_kind = adapters.cloud_kind((store.get(str(s.get("alias") or "").strip()) or [{}])[0])
    if succ_kind:
        relay_field = ('<input type="hidden" name="chain_relay" value="upload">'
                       '<p class="hint" style="margin:0">upload — forced: the successor runs on '
                       f'{adapters.cloud_module(succ_kind).VENDOR}</p>')
    else:
        relay_opts = "".join(
            f'<option value="{v}"{" selected" if cur_relay == v else ""}>{lbl}</option>'
            for v, lbl in (("path", "path — same backend, shared disk (default)"),
                           ("upload", "upload — relay bytes to a stage-2 backend")))
        relay_field = f'<select name="chain_relay">{relay_opts}</select>'
    return ("<h2 style='margin-top:18px'>Chain (successor)</h2>"
            "<p class='hint'>Optional: run a second alias after this one and deliver only its result — "
            "e.g. mesh here, rigging as the successor (a ComfyUI rigger on the same or another backend, "
            "or a cloud rigging alias — Meshy-Rig, Tripo-Rig). The path hand-off needs the backend's "
            "<b>comfy output dir</b>; "
            "the upload hand-off does not. Leave the successor blank for a normal single-stage alias.</p>"
            + _field("successor alias", _inp("successor", s.get("alias", ""),
                     placeholder="e.g. mesh-reg-mia"), short=True)
            + _field("mesh export node", f'<select name="chain_export_node">{node_opts}</select>')
            + _field("successor mesh param", _inp("chain_mesh_param", s.get("mesh_param", ""),
                     placeholder="mesh_path (mesh workflows: input_mesh_path)"), short=True)
            + _field("mesh hand-off", relay_field)
            + "<p class='hint'>The gateway pins that export node's filename, so it knows the mesh, and passes "
              "it to the successor under the <b>mesh param</b> (blank = <code>mesh_path</code>; the mesh "
              "workflows label their mesh input <code>input_mesh_path</code>) — must be a request field "
              "(param or label) on the successor. <b>Hand-off</b>: <code>path</code> keeps both stages on ONE "
              "backend (shared disk) and passes the mesh's absolute output path. <code>upload</code> lets the "
              "successor run on a DIFFERENT backend — the gateway fetches the mesh, uploads it into that "
              "backend's <b>input</b> dir and passes its absolute path there (the backend's <b>comfy input "
              "dir</b>, blank = derived from the output dir), so the successor loads it exactly like a path "
              "hand-off — no special loader needed. Stage 2 runs on an <b>allowed</b> "
              "successor candidate, preferring the same backend as stage 1; so only list the successor on "
              "backends that can actually run it. This stage's other params (name, no_fingers, …) are "
              "threaded to the successor by matching param name. The successor may also be a "
              "<b>cloud alias</b> (e.g. <code>Meshy-Rig</code> endpoint <code>rigging</code>, "
              "<code>Tripo-Rig</code> endpoint <code>rig</code>) — the mesh then travels to that vendor as "
              "bytes (the hand-off setting is ignored), the <b>mesh param</b> must be one "
              "of that alias's file fields (<code>input_mesh_path</code>) and the <b>delivered rig type</b> is "
              "<code>meshy</code> / <code>tripo</code>.</p>"
            + _field("keep from this stage", _inp("chain_keep", ", ".join(s.get("keep_from_mesh") or []),
                     placeholder="e.g. *_basecolor*.png"), short=True)
            + _field("delivered rig type", _inp("chain_rig", s.get("rig", ""),
                     placeholder="blank · mixamo · generic · meshy · tripo"), short=True)
            + "<p class='hint'><b>keep from this stage</b>: globs for files THIS (mesh) stage produces that "
              "must ship with the successor's result — e.g. the <code>*_basecolor*.png</code> the texturing "
              "bakes here (the UniRig fbx only references its texture). <b>delivered rig type</b>: set "
              "<code>generic</code>/<code>mixamo</code> to tag + validate the COMBINED delivery at the chain "
              "level (generic needs fbx + basecolor); <code>meshy</code>/<code>tripo</code> (a cloud rig) are "
              "only tagged, never re-normalized or validated. Blank = trust the successor's own output "
              "config.</p>"
            + _chain_rig_warning(wf, s))


def _chain_rig_warning(wf: dict, s: dict) -> str:
    """Static pre-flight for a `generic` (FBX) chain: the rigger's FBX only references
    its texture by a temp path, so the basecolor PNG has to come from THIS stage —
    `validate_delivery` fails the (long) job at the very end if it doesn't. Two
    misconfigurations are visible from the workflow alone: a stage that bakes no
    texture at all (triposplat — SplatToMesh emits the core MESH type, the texture
    tool chain is TRIMESH-only, so its texture stays embedded in the GLB), and an
    empty `keep from this stage`."""
    if (s.get("rig") or "").strip().lower() != "generic":
        return ""
    if not any((n or {}).get("class_type") == "SaveImage" for n in wf.values()):
        why = ("this stage exports no texture PNG (no <code>SaveImage</code> node), so a "
               "<code>generic</code> delivery can never be complete — put a texturing stage "
               "(e.g. <code>mesh-shrink</code>) in between, or deliver via <code>mixamo</code>")
    elif not [g for g in (s.get("keep_from_mesh") or []) if str(g).strip()]:
        why = ("<b>keep from this stage</b> is empty — add <code>*_basecolor*.png</code>, else the "
               "chain fails validation after stage 2 has already run")
    else:
        return ""
    return f"<p class='hint'><span class='bad'>⚠ generic rig: {why}.</span></p>"


async def _pinned_block(alias: str, cands: list, fixed: list, wf: dict, oi: dict,
                        oi_from: Optional[str] = None) -> str:
    """Pinned values as per-backend tabs (primary edits; extras override values).
    `oi_from` names the sibling backend whose node defs stood in because the primary
    could not answer — said out loud, because the choices then list what is installed
    THERE (see `_editor_object_info`)."""
    primary_rows = _pin_tab_rows(alias, cands[0], True, fixed, wf, oi)
    if oi_from:
        primary_rows = (f"<p class='hint'>Primary <b>{_esc(str(cands[0].get('backend')))}</b> is not "
                        f"answering — the choices below list what is installed on "
                        f"<b>{_esc(oi_from)}</b>.</p>") + primary_rows
    if len(cands) == 1:
        return f"<h2>Pinned values</h2><div class='ppanel'>{primary_rows}</div>"
    bn0 = str(cands[0].get("backend"))
    tabs = "".join(
        f'<button type="button" class="ptab{" on" if i == 0 else ""}" '
        f"onclick=\"pinTab(this,'{_esc(str(c.get('backend')))}')\">{_esc(str(c.get('backend')))}</button>"
        for i, c in enumerate(cands))
    panels = f'<div class="ppanel" data-pt="{_esc(bn0)}">{primary_rows}</div>'
    # override backends' own models fetched in PARALLEL (cold /object_info is slow)
    oi_others = await asyncio.gather(*[_object_info(str(c.get("backend")), wf) for c in cands[1:]])
    for c, oi_bn in zip(cands[1:], oi_others):
        bn = str(c.get("backend"))
        rows = _pin_tab_rows(alias, c, False, fixed, wf, oi_bn or oi)  # fall back to primary's models
        panels += (f'<div class="ppanel" data-pt="{_esc(bn)}" style="display:none">'
                   f"<p class='hint'>Models/values installed on <b>{_esc(bn)}</b> "
                   f"(slots from primary <b>{_esc(bn0)}</b>; “inherited” = same as primary).</p>{rows}</div>")
    return (f"<h2>Pinned values <span class='muted' style='font-weight:normal'>— tab per backend</span></h2>"
            f'<div class="ptabs">{tabs}</div>{panels}')


def _bypass_block(alias: str, cands: list, wf: dict) -> str:
    """Per-backend node bypass as a node×backend checkbox grid (ComfyUI mode-4: the node
    is removed and its consumers reconnect to its same-typed input). The managed node set
    is the UNION of every candidate's `bypass`; a hidden `bypass_nodes` field carries it
    so `update` rebuilds each backend's list from present/absent checkboxes."""
    managed = sorted({str(n) for c in cands for n in (c.get("bypass") or [])},
                     key=lambda n: (len(n), n))
    out_nodes = {str(c.get("output_node")) for c in cands if c.get("output_node")}
    hdr = "".join(f"<th>{_esc(str(c.get('backend')))}</th>" for c in cands)
    rows = ""
    for nid in managed:
        node = wf.get(nid) or {}
        title = (node.get("_meta") or {}).get("title", "")
        label = f"<code>{_esc(nid)}</code> {_esc(node.get('class_type', ''))}" + (f" · {_esc(title)}" if title else "")
        if nid not in wf:
            label += " <span class='badge bad' title='this node no longer exists in the workflow'>stale</span>"
        elif nid in out_nodes:
            label += " <span class='badge bad' title='this is the output node — bypassing it yields no result'>output!</span>"
        cells = "".join(
            f"<td><input type='checkbox' value='1' "
            f"name='byp__{_esc(str(c.get('backend')))}__{_esc(nid)}'"
            f"{' checked' if nid in {str(x) for x in (c.get('bypass') or [])} else ''}></td>"
            for c in cands)
        rm = _btn("⊘", f"/ui/mapping/bypass-del?alias={_esc(alias)}&node={_esc(nid)}",
                  "danger", sm=True, icon=True, title="Remove from bypass")
        rows += f"<tr><td>{label}</td>{cells}<td class='acts'>{rm}</td></tr>"
    body = (f"<table class='pins'><tr><th>node</th>{hdr}<th></th></tr>{rows}</table>"
            if managed else "<p class='muted'>No nodes bypassed — add one below.</p>")
    avail = [nid for nid in sorted(wf, key=lambda n: (len(n), n)) if nid not in set(managed)]
    add_sel = ""
    if avail:
        def _opt(nid):
            t = ((wf[nid] or {}).get("_meta") or {}).get("title", "")
            return (f"<option value='{_esc(nid)}'>{_esc(nid)} — {_esc((wf[nid] or {}).get('class_type', ''))}"
                    + (f" · {_esc(t)}" if t else "") + "</option>")
        add_sel = ('<select class="addsel" onchange="if(this.value)location.href=\'/ui/mapping/bypass-add?alias='
                   f"{_esc(alias)}&amp;node='+encodeURIComponent(this.value)\">"
                   f'<option value="">+ Bypass a node…</option>{"".join(_opt(n) for n in avail)}</select>')
    return (f"<input type='hidden' name='bypass_nodes' value='{_esc(','.join(managed))}'>"
            "<h2 style='margin-top:18px'>Bypass "
            "<span class='muted' style='font-weight:normal'>— skip a node per backend</span></h2>"
            "<p class='hint'>Remove a node from the graph on the checked backends (ComfyUI bypass): its "
            "consumers reconnect to the node's same-typed input. Use for a post-process/switch node some "
            "backends should skip. A <b>required</b> downstream input left unconnected makes ComfyUI error.</p>"
            f"{body}{add_sel}")


def _option_rows(mod, opts: dict, ep: str) -> str:
    """The vendor option block of a cloud alias editor, rendered from `mod.OPTION_FIELDS`.

    Consecutive bool fields whose label is "" share the previous field's row (Meshy's
    `texture` row = should_texture + enable_pbr). `rig_only` fields carry a marker
    outside the rig endpoint but are still rendered and saved — switching an alias's
    endpoint back and forth must not silently drop what the other endpoint needs."""
    rows, pending_label, pending_ctrls, pending_hint = [], None, [], ""

    def flush():
        if pending_ctrls:
            rows.append(_field(pending_label or "", "".join(pending_ctrls)))
            if pending_hint:
                rows.append(f"<p class='hint' style='margin:-4px 0 10px'>{pending_hint}</p>")

    for fld in mod.OPTION_FIELDS:
        k, t, label = fld["key"], fld["type"], fld.get("label", fld["key"])
        # plain text, not markup: both _field and _checkbox escape their label. It rides
        # on the row label, except for a blank-label box that HAS no row label of its own.
        rig_mark = " (rig only)" if fld.get("rig_only") and ep != mod.RIG_ENDPOINT else ""
        if t == "bool":
            txt = fld.get("checkbox_text") or k
            if label == "" and pending_ctrls:
                pending_ctrls.append(_checkbox(f"opt__{k}", bool(opts.get(k)), txt + rig_mark))
                pending_hint = fld.get("hint") or pending_hint
                continue
            flush()
            pending_label = label + rig_mark
            pending_ctrls = [_checkbox(f"opt__{k}", bool(opts.get(k)), txt)]
            pending_hint = fld.get("hint") or ""
            continue
        flush()
        pending_label, pending_ctrls, pending_hint = None, [], ""
        if t == "select":
            # _select treats a bare list as a scalar option — its choices must be TUPLES
            choices = [tuple(c) if isinstance(c, (tuple, list)) else (c, c) for c in fld["choices"]]
            ctrl = _select(f"opt__{k}", choices, cloudtask.field_value_str(fld, opts.get(k)))
        elif t == "tristate":
            ctrl = _select(f"opt__{k}", [("", "model default"), ("true", "always"), ("false", "never")],
                           cloudtask.field_value_str(fld, opts.get(k)))
        else:                                   # int | text | list
            ctrl = _inp(f"opt__{k}", cloudtask.field_value_str(fld, opts.get(k)),
                        placeholder=fld.get("placeholder", ""), typ="number" if t == "int" else "text")
        rows.append(_field(label + rig_mark, ctrl, short=(t == "int")))
        if fld.get("hint"):
            rows.append(f"<p class='hint' style='margin:-4px 0 10px'>{fld['hint']}</p>")
    flush()
    return "".join(rows)


def _cloud_editor(kind: str, alias: str, cands: list, saved: bool = False) -> str:
    """Editor for a cloud alias (Meshy, Tripo): endpoint, ai model and the admin option
    defaults — no workflow, no mapping, no pins (the public fields are a fixed table, see
    the vendor module). Everything vendor-specific is READ from that module (OPTION_FIELDS,
    the endpoint/model/format tuples, the two hints), so a second cloud kind gets this
    editor instead of a fork of it — and a vendor option can never be offered here without
    the request builder knowing it.
    It DOES carry the chain successor: a cloud alias can be stage 1 (mesh here, rigging in
    the successor) — without an export node or a hand-off choice, because the mesh comes
    back as a result blob and always travels to stage 2 as bytes."""
    mod = adapters.cloud_module(kind)
    vendor = mod.VENDOR
    cand = cands[0]
    s = next((c.get("successor") for c in cands if c.get("successor")), None) or {}
    keep = [g for g in (s.get("keep_from_mesh") or []) if str(g).strip()]
    rig_cur = (s.get("rig") or "").strip()
    rig_opts: list = [("", "blank — trust the successor"), "mixamo", "generic", "meshy", "tripo"]
    if rig_cur and rig_cur not in rig_opts:      # keep an unknown stored value visible
        rig_opts.append((rig_cur, f"{rig_cur} — (unknown)"))   # …a Save would clear it otherwise
    ep = mod.endpoint_of(cand)
    opts = mod.options_of(cand)
    model = cand.get("model") if cand.get("model") in mod.AI_MODELS else mod.AI_MODELS[0]
    retries = next((c.get("retries") for c in cands if c.get("retries") not in (None, "")), "")
    cur_task = next((c.get("task") for c in cands if c.get("task")), "") or "img2mesh"
    # the rig endpoint answers with a narrower format set (options_of filters either way,
    # so a stored alias that is switched TO it cannot keep an impossible format)
    fmts = "".join(f'<label style="margin-right:10px"><input type="checkbox" name="fmt__{_esc(f)}"'
                   f'{" checked" if f in opts["target_formats"] else ""}> {_esc(f)}</label>'
                   for f in (mod.RIG_FORMATS if ep == mod.RIG_ENDPOINT else mod.FORMATS))
    params, images, files = adapters.public_fields(cand)
    fields = "".join(f"<tr><td><code>{_esc(i['name'])}</code></td><td>image · {_esc(i['on_empty'])}</td></tr>"
                     for i in images)
    fields += "".join(f"<tr><td><code>{_esc(x['name'])}</code></td><td>file · "
                      f"{'required' if x.get('required') else 'optional'}"
                      f"{' · ' + '/'.join(_esc(a) for a in x['accept']) if x.get('accept') else ''}"
                      "</td></tr>" for x in files)
    fields += "".join(f"<tr><td><code>{_esc(p['name'])}</code></td><td>{_esc(p['type'])}"
                      f"{' · default ' + _esc(str(p['default'])) if p.get('default') not in (None, '') else ''}"
                      f"{' · ' + '/'.join(_esc(c or 'none') for c in p['choices']) if p.get('choices') else ''}"
                      "</td></tr>" for p in params)
    ignored = getattr(mod, "IGNORED_PARAMS", ())
    ign_hint = (" " + " / ".join(f"<code>{_esc(n)}</code>" for n in ignored)
                + " are accepted and ignored.") if ignored else ""
    return (f'<form action="/ui/mapping/cloud-update" method="post"><input type="hidden" name="alias" value="{_esc(alias)}">'
            f'<div class="formbar"><h2 style="margin:0">{_esc(alias)}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/mapping?sub=media", "secondary")}'
            + ("<span class='ok-chip fade'>✓ Saved</span>" if saved else "") + "</div>"
            + _field("alias name", _inp("new_alias", alias), short=True)
            + _field("task", _task_select(cur_task), short=True)
            + f'<h2 style="margin-top:18px">{_esc(vendor)}</h2>'
            + _field("endpoint", _select("cloud_endpoint", list(mod.ENDPOINTS), ep))
            + f"<p class='hint' style='margin:-4px 0 10px'>{mod.ENDPOINT_HINT}</p>"
            + _field("ai model", _select("cloud_model", list(mod.AI_MODELS), model))
            + _option_rows(mod, opts, ep)
            + _field("deliver formats", fmts)
            + _field("retries", _inp("retries", str(retries), placeholder="blank = try all backends",
                                     typ="number"), short=True)
            + '<h2 style="margin-top:18px">Chain (successor)</h2>'
            + "<p class='hint'>Optional: run a second alias (a rigger) on this alias's mesh and deliver "
              "ONLY its result. Leave blank for a normal single-stage alias.</p>"
            + _field("successor alias", _inp("successor", s.get("alias", ""),
                     placeholder="e.g. mesh-mia"), short=True)
            + _field("successor mesh param", _inp("chain_mesh_param", s.get("mesh_param", ""),
                     placeholder="input_mesh_path"), short=True)
            + _field("keep from this stage", _inp("chain_keep", ", ".join(keep),
                     placeholder="e.g. preview.png"), short=True)
            + _field("delivered rig type", _select("chain_rig", rig_opts, rig_cur), short=True)
            + "<p class='hint'>The mesh (glb) is relayed as <b>bytes</b> to the successor's backend — no "
              f"export node, no hand-off choice (a {_esc(vendor)} stage shares no disk with anything). It "
              "arrives under the <b>mesh param</b> (blank = <code>input_mesh_path</code>, what the mesh "
              "workflows label their mesh input), which must be a request field (param or label) of the "
              "successor. <b>keep from this stage</b>: globs for files THIS stage produces that must ship "
              "with the successor's result. <b>delivered rig type</b> tags the delivery; "
              "<code>generic</code>/<code>mixamo</code> are additionally normalized and validated at chain "
              "level, <code>meshy</code>/<code>tripo</code> are only tagged (a cloud vendor rigs to its own "
              "conventions). Requires <code>glb</code> in <b>deliver formats</b> — the job is refused up "
              f"front otherwise, before credits are spent. {mod.CHAIN_HINT}</p>"
            + '<h2 style="margin-top:18px">Request fields</h2>'
            + f"<p class='hint'>Fixed for {_esc(vendor)} aliases — what "
              f"<code>GET /v1/generations/{{alias}}/schema</code> advertises.{ign_hint}</p>"
            + f"<table class='pins'><tr><th>name</th><th>type</th></tr>{fields}</table>"
            + "</form>"
            + '<h2 style="margin-top:18px">Backends</h2>'
            + "<p class='hint'>Allowed backends for this alias — a job takes the fastest free one; on a "
              f"connection error the next one is used. Only {_esc(vendor)} backends can be added to a "
              f"{_esc(vendor)} alias.</p>"
            + _backends_section(alias, cands))


def _cloud_side(kind: str) -> str:
    """The editor's third column ("Available fields") is a workflow view — a cloud alias
    has no workflow, so it carries the reason instead of rendering empty."""
    vendor = adapters.cloud_module(kind).VENDOR
    return (f"<h2>Available fields</h2><p class='hint'>A {_esc(vendor)} alias has no workflow: its request "
            "fields are the fixed table in the editor, and everything else is an admin option "
            "set on the left.</p>")


async def _alias_editor(alias: str, saved: bool = False) -> str:
    """The alias editor as a single-column fragment for the master-detail right
    side (request fields + pinned values in one form, available fields below).
    `saved` renders the post-Save confirmation inside the sticky form bar (see
    .ok-chip — a stacked banner would shift the restored scroll position)."""
    cands = store.get(alias)
    if not cands:
        return f'<p class="bad">alias \'{_esc(alias)}\' not found</p>'
    cand = cands[0]
    kind = adapters.cloud_kind(cand)             # no workflow, no mapping, no /object_info
    if kind:
        return _cloud_editor(kind, alias, cands, saved), _cloud_side(kind)
    wf = cand.get("workflow_json")
    if wf is None and cand.get("workflow"):
        try:
            with open(cand["workflow"]) as fh:
                wf = json.load(fh)
        except Exception:
            wf = {}
    wf = wf or {}
    oi, oi_from = await _editor_object_info(cands, wf, cand.get("mapping"))
    mapping = cand.get("mapping") or {}
    fixed = cand.get("fixed") or []
    mapped = ({(m["node"], m["field"]) for m in mapping.values()}
              | {(b["node"], b["field"]) for b in fixed})

    req_rows = _req_fields_rows(alias, wf, mapping, oi)
    pinned_block = await _pinned_block(alias, cands, fixed, wf, oi, oi_from)
    pin_extra = _PIN_CSS_JS

    retries = next((c.get("retries") for c in cands if c.get("retries") not in (None, "")), "")
    cur_task = next((c.get("task") for c in cands if c.get("task")), "") or "text2img"
    # video aliases registered before the task dropdown carry the old free-text
    # default ('text2img') yet have fps configured — configured fps must never
    # render its own editor fields invisible.
    is_video = "vid" in cur_task.lower() or any(c.get("fps") for c in cands)
    form = (f'<form action="/ui/mapping/update" method="post"><input type="hidden" name="alias" value="{_esc(alias)}">'
            f'<div class="formbar"><h2 style="margin:0">{_esc(alias)}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/mapping?sub=media", "secondary")}'
            f'{_btn("⬇ Export", "/ui/mapping/export?alias=" + quote(alias), "secondary", title="Download the gateway-cleaned workflow JSON")}'
            + ("<span class='ok-chip fade'>✓ Saved</span>" if saved else "")
            + '</div>'
            + _field("alias name", _inp("new_alias", alias), short=True)
            + _field("task", _task_select(cur_task, _TASK_VIDEO_JS), short=True)
            + '<h2 style="margin-top:18px">Update workflow</h2>'
            + "<p class='hint'>Replace the ComfyUI API JSON — request fields + pinned values are kept; "
              "bindings whose node vanished are flagged <span class='badge bad'>stale</span> in Request fields.</p>"
            + _field("API JSON", '<input type="file" name="workflow_file" accept="application/json,.json">')
            + _field("…or share path", _inp("workflow_path", "", placeholder="/mnt/share/flux_api.json"))
            + '<div class="field"><label></label><div class="control">'
              '<button class="btn secondary" formaction="/ui/mapping/update-workflow" '
              'formenctype="multipart/form-data">Update workflow</button></div></div>'
            + "<h2>Backends</h2>"
            + "<p class='hint'>Allowed backends for this alias — a job takes the fastest free "
              "unpaid one; on a connection error the next one is used.</p>"
            + _backends_section(alias, cands)
            + _field("retries", _inp("retries", str(retries), typ="number"), short=True)
            + "<p class='hint'>Backends to try after the first on error. Blank = try all eligible · 0 = no failover.</p>"
            + f'<div id="gw-video" style="display:{"" if is_video else "none"}">'
            + '<h2 style="margin-top:18px">Video</h2>'
            + _field("fps", _inp("fps", str(next((c.get("fps") for c in cands
                                                  if c.get("fps")), "") or ""), typ="number"), short=True)
            + _field("frames raster", _inp("frames_snap", str(next((c.get("frames_snap") for c in cands
                                                                    if c.get("frames_snap")), "") or ""),
                                           typ="number"), short=True)
            + "<p class='hint'><b>fps</b> enables <code>params.seconds</code> → frames "
              "(needs a request field named/labelled <code>frames</code>) and shows up in "
              "<code>/v1/generations/&lt;alias&gt;/schema</code>. <b>frames raster</b> S snaps computed "
              "frames onto S·k+1 (Wan: 4). Blank = off.</p>"
            + '</div>'
            + f'<h2>Request fields <span class="muted" style="font-weight:normal">— drag ⠿ to set Playground order</span></h2>'
            "<p class='hint'>label overrides the Playground label / external API field name (blank = param).</p>"
            f'<table class="reqf"><thead><tr><th>param</th>'
            f'<th title="Playground label / API field name (blank = param)">label</th>'
            f'<th>node</th><th>field</th>'
            f'<th title="workflow default — used when a request omits the field '
            f'(a seed field still gets a random value unless the request sends one)">=</th>'
            f'<th></th></tr></thead>'
            f'<tbody id="reqfields">{req_rows}</tbody></table>'
            + pinned_block
            + _bypass_block(alias, cands, wf)
            + _output_section(wf, cands)
            + '</form>' + pin_extra + _reorder_js(alias))
    return form, _available_fields(alias, wf, mapped, oi)


_HEAD_MAX = 60


def _node_head(nid: str, n: dict) -> str:
    """Node header for the Available fields pane (mapping column 3): `<id> <class> · <title>`,
    SHORTENED once that runs past `_HEAD_MAX` chars — a wide header widens the whole
    (narrow) column. Custom node packs name a class and then title it with the same words
    again (`Trellis2MeshWithVoxelMultiViewGenerator · Trellis2 - Mesh With Voxel Multi-View
    Generator`), so the shortening drops the class — the title is the readable half — and
    the title's own vendor prefix up to the first ` - `. Whatever is left is hard-capped;
    the full text always stays in the tooltip, so nothing becomes unrecoverable."""
    cls = str(n.get("class_type") or "")
    title = str((n.get("_meta") or {}).get("title") or "")
    full = f"{nid} {cls}" + (f" · {title}" if title else "")
    if len(full) <= _HEAD_MAX:
        return f"<code>{_esc(nid)}</code> {_esc(cls)}" + (f" · {_esc(title)}" if title else "")
    short = (title.split(" - ", 1)[1] if " - " in title else title) or cls
    if len(nid) + 1 + len(short) > _HEAD_MAX:
        short = short[:max(1, _HEAD_MAX - len(nid) - 2)].rstrip() + "…"
    return f'<code>{_esc(nid)}</code> <span title="{_esc(full)}">{_esc(short)}</span>'


def _available_fields(alias: str, wf: dict, mapped: set, oi: dict) -> str:
    """Available scalar fields (not yet mapped) with + (pin) / → (request field)
    actions — stacked below the editor form."""
    avail = ""
    for nid, n in sorted(wf.items(), key=lambda kv: (len(kv[0]), kv[0])):
        arows = ""
        for f2, v2 in (n.get("inputs") or {}).items():
            if isinstance(v2, list) or (nid, f2) in mapped:    # links / already-mapped → skip
                continue
            add = _btn("+", f"/ui/mapping/field-add?alias={_esc(alias)}&node={_esc(nid)}&field={_esc(f2)}",
                       "secondary", sm=True, icon=True, title="Add as pinned value")
            req_btn = _btn("→", f"/ui/mapping/field-map?alias={_esc(alias)}&node={_esc(nid)}"
                           f"&field={_esc(f2)}", "secondary", sm=True, icon=True,
                           title="Add as request field")
            fopts = oi.get(n.get("class_type", ""), {}).get(f2)   # discovery hint: dropdown? bool?
            if isinstance(fopts, list) and _is_model_field(fopts, v2):
                fhint = f" <span class='badge ok' title='becomes a discovery dropdown (e.g. LoRA / model) when pinned or promoted'>▾ {len(fopts)}</span>"
            elif isinstance(fopts, dict) and fopts.get("_num"):
                fhint = f" <span class='muted'>{fopts['_num'].lower()}</span>"
            elif isinstance(v2, bool):
                fhint = " <span class='muted'>bool</span>"
            else:
                fhint = ""
            arows += (f"<tr><td>{_esc(f2)} <span class='muted'>= {_esc(str(v2))[:22]}</span>{fhint}</td>"
                      f"<td class='acts'>{add}{req_btn}</td></tr>")
        if arows:
            avail += f"<tr class='node'><td colspan=2>{_node_head(nid, n)}</td></tr>{arows}"
    return (f"<h2>Available fields</h2>"
            f"<p class='hint'>Add a field to pin it (Switch boolean, reference image, …). "
            f"Unmapped fields keep the workflow's value.</p>"
            f"<table class='avail'>{avail or '<tr><td>all scalar fields mapped</td></tr>'}</table>")


async def edit_add(request: Request):
    alias = _qp(request, "alias")
    node, fld = _qp(request, "node"), _qp(request, "field")
    cands = store.get(alias)
    if cands and node and fld:
        cand = cands[0]
        wf = cand.get("workflow_json") or {}
        cur = (wf.get(node, {}).get("inputs") or {}).get(fld)
        fixed = cand.get("fixed") or []
        if not any(b["node"] == node and b["field"] == fld for b in fixed):
            fixed.append({"node": node, "field": fld, "value": cur})
            cand["fixed"] = fixed
            store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def edit_del(request: Request):
    alias = _qp(request, "alias")
    cands = store.get(alias)
    if cands:
        cand = cands[0]
        param, node, fld = _qp(request, "param"), _qp(request, "node"), _qp(request, "field")
        if param:
            (cand.get("mapping") or {}).pop(param, None)
        elif node and fld:
            cand["fixed"] = [b for b in (cand.get("fixed") or [])
                             if not (b["node"] == node and b["field"] == fld)]
        store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def bypass_add(request: Request):
    """Add a node to the bypass set — defaults to bypassed on the PRIMARY backend (so it
    joins the union and appears in the grid); toggle per backend + Save afterwards."""
    alias, node = _qp(request, "alias"), _qp(request, "node")
    cands = store.get(alias)
    if cands and node:
        bl = [str(x) for x in (cands[0].get("bypass") or [])]
        if node not in bl:
            cands[0]["bypass"] = bl + [node]
            store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def bypass_del(request: Request):
    """Drop a node from the bypass set on EVERY backend (else the union re-adds it)."""
    alias, node = _qp(request, "alias"), _qp(request, "node")
    cands = store.get(alias)
    if cands and node:
        changed = False
        for c in cands:
            bl = [x for x in (c.get("bypass") or []) if str(x) != node]
            if len(bl) != len(c.get("bypass") or []):
                changed = True
                if bl:
                    c["bypass"] = bl
                else:
                    c.pop("bypass", None)
        if changed:
            store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


def _slug_param(s: str) -> str:
    """A ComfyUI node title → a safe request-param name (lowercase, non-alnum → _)."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", (s or "").lower())).strip("_")


async def field_map(request: Request):
    """Promote an available node.field to a request param. For image loaders the param
    name (and label) default to the node TITLE — so 4 titled reference-image nodes get
    distinct, readable names instead of image / image_<id> — else the field name;
    node-qualified on collision. Rename afterwards by editing the cells."""
    alias, node = _qp(request, "alias"), _qp(request, "node")
    fld = _qp(request, "field")
    cands = store.get(alias)
    if cands and node and fld:
        cand = cands[0]
        wf = cand.get("workflow_json") or {}
        title = ((wf.get(node, {}) or {}).get("_meta") or {}).get("title", "").strip()
        is_img = adapters.is_image_field(wf, node)
        base = (_slug_param(title) if (is_img and title) else fld) or fld
        mp = cand.setdefault("mapping", {})
        param = base if base not in mp else f"{base}_{node}"
        entry = {"node": node, "field": fld}
        if is_img and title:
            entry["label"] = title
        mp[param] = entry
        store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def field_clear(request: Request):
    """Clear the workflow default of a mapped request field so the playground field
    starts blank. For a combo field whose valid options include "None" (e.g. a LoRA
    loader), clear to "None" — an empty lora_name makes ComfyUI error; "None" is the
    valid 'off' value. Text fields still clear to "". Applied to every candidate so no
    backend keeps the bad value."""
    alias, param = _qp(request, "alias"), _qp(request, "param")
    cands = store.get(alias)
    if cands and param:
        cand = cands[0]
        m = (cand.get("mapping") or {}).get(param)
        wf = cand.get("workflow_json")
        if m and wf and m.get("node") in wf:
            node, field = m["node"], m["field"]
            cls = wf.get(node, {}).get("class_type", "")
            opts = (await _object_info(cand.get("backend", ""), wf, cand.get("mapping"))).get(cls, {}).get(field)
            if isinstance(opts, list):                   # authoritative: does the combo offer "None"?
                none_ok = "None" in opts
            else:                                        # oi unreachable → fall back to a lora name/class hint
                none_ok = "lora" in (field + cls).lower()
            blank = "None" if none_ok else ""
            for c in cands:                              # keep it consistent across backends
                w = c.get("workflow_json")
                if w and node in w:
                    w[node].setdefault("inputs", {})[field] = blank
            store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def cand_add(request: Request):
    """Add an allowed backend to an alias — an independent snapshot of the existing
    workflow+mapping on that backend (deduped by backend name)."""
    alias, backend = _qp(request, "alias"), _qp(request, "backend")
    cands = store.get(alias)
    valid = {b["name"] for b in _gen_backends() if _same_kind(cands or [], b["name"])}
    if cands and backend in valid and backend not in [c.get("backend") for c in cands]:
        new = json.loads(json.dumps(cands[0]))     # snapshot workflow+mapping (+retries)
        new["backend"] = backend
        new.pop("fallback", None)                  # legacy flag, no longer used
        cands.append(new)
        store.upsert(alias, cands)
        logger.info(f"ui: alias '{alias}' + backend '{backend}'")
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def cand_del(request: Request):
    """Remove an allowed backend (an alias must keep at least one)."""
    alias, backend = _qp(request, "alias"), _qp(request, "backend")
    cands = store.get(alias)
    if cands and len(cands) > 1:
        kept = [c for c in cands if c.get("backend") != backend]
        if kept and len(kept) != len(cands):
            store.upsert(alias, kept)
            logger.info(f"ui: alias '{alias}' − backend '{backend}'")
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def field_order(request: Request):
    """Reorder the request fields (drag & drop). Rebuilds the mapping dict in the
    given param order; that order is what the editor and Playground render."""
    alias = _qp(request, "alias")
    order = request.query_params.getlist("order")
    cands = store.get(alias)
    if cands and order:
        cand = cands[0]
        m = cand.get("mapping") or {}
        new = {p: m[p] for p in order if p in m}
        for p in m:                       # keep any param not in the posted order
            new.setdefault(p, m[p])
        cand["mapping"] = new
        store.upsert(alias, cands)
    return RedirectResponse(f"/ui/mapping?edit={alias}", status_code=303)


async def update(request: Request):
    f = await _form(request)
    alias = f.get("alias", "").strip()
    cands = store.get(alias)
    if not alias or not cands:
        raise HTTPException(404, "alias not found")
    cand = cands[0]
    task = (f.get("task", "") or "").strip()          # task dropdown (blank keeps the stored value)
    if task:
        for c in cands:
            c["task"] = task
    # Request fields are dynamic: one node__<param>/field__<param> pair per mapped
    # param the editor rendered (no fixed list). Slice after the prefix so param
    # names may themselves contain underscores.
    mapping = {}
    for key, val in f.items():
        if key.startswith("node__"):
            p = key[len("node__"):]
            node = (val or "").strip()
            fld = (f.get(f"field__{p}", "") or "").strip()
            if node and fld:
                entry = {"node": node, "field": fld}
                lbl = (f.get(f"label__{p}", "") or "").strip()
                if lbl:                       # label overrides Playground label / API field name
                    entry["label"] = lbl
                emode = (f.get(f"empty__{p}", "") or "").strip()   # image slot empty-behaviour
                if emode in ("placeholder", "required", "disable"):
                    entry["on_empty"] = emode
                # extra node ids bypassed when THIS slot is empty — only stored for the
                # disable mode, so switching the slot back to placeholder/required leaves
                # no invisible rule behind (adapters.slot_empty_bypass ignores it anyway)
                if emode == "disable":
                    extra = list(dict.fromkeys(
                        x for x in re.split(r"[,\s]+", f.get(f"bypass__{p}", "") or "") if x))
                    if extra:
                        entry["on_empty_bypass"] = extra
                mapping[p] = entry
    # Editable workflow defaults (the "=" column): default__<param> writes the
    # value a request-without-this-field runs with into the workflow JSON at the
    # mapped node/field — on EVERY candidate (the workflow is backend-independent,
    # same rule as field_clear). Coerced to the field's current type like pins
    # (adapters._coerce); linked inputs are never touched; a blank only clears
    # string fields (number/bool widgets never submit blank deliberately).
    for key, val in f.items():
        if not key.startswith("default__"):
            continue
        m = mapping.get(key[len("default__"):])
        if not m:
            continue
        for c in cands:
            w = c.get("workflow_json")
            if not w or m["node"] not in w:
                continue
            inputs = w[m["node"]].setdefault("inputs", {})
            old = inputs.get(m["field"])
            if isinstance(old, list):                    # wired to another node — hands off
                continue
            if val == "":
                if isinstance(old, str) and old:
                    inputs[m["field"]] = ""
                continue
            new = adapters._coerce(val, old)
            if new != old:
                inputs[m["field"]] = new
    fixed = []
    for key, val in f.items():
        if key.startswith("fixed__") and val != "":
            _, nid, fld = key.split("__", 2)
            fixed.append({"node": nid, "field": fld, "value": val})
    # per-backend overrides of pinned values: ovr__<backend>__<node>__<field>
    ovr = {}
    for key, val in f.items():
        if key.startswith("ovr__") and (val or "").strip() != "":
            parts = key.split("__", 3)
            if len(parts) == 4:
                ovr[(parts[1], parts[2], parts[3])] = val
    # mapping is backend-independent → sync to every candidate. `fixed` (pinned model
    # values) is per-backend: the primary holds the edited values; extra backends use
    # the same slots with optional per-backend value overrides (blank = inherit).
    for c in cands:
        c["mapping"] = mapping
    cands[0]["fixed"] = fixed
    for c in cands[1:]:
        bn = c.get("backend")
        c["fixed"] = [{"node": b["node"], "field": b["field"],
                       "value": ovr.get((bn, b["node"], b["field"]), b["value"])} for b in fixed]
    # per-backend node bypass: a byp__<backend>__<node> checkbox grid. The hidden
    # bypass_nodes (union managed set) is authoritative — an unchecked (=absent) box clears.
    managed_byp = [n for n in (f.get("bypass_nodes", "") or "").split(",") if n]
    for c in cands:
        bl = [n for n in managed_byp if f.get(f"byp__{c.get('backend')}__{n}")]
        if bl:
            c["bypass"] = bl
        else:
            c.pop("bypass", None)
    # retries is a per-alias setting (blank = try all) — keep it on every candidate so
    # it survives whichever backend is removed first.
    retries = (f.get("retries", "") or "").strip()
    for c in cands:
        c["retries"] = retries
    # video metadata (schema + seconds→frames), same per-candidate persistence
    for key in ("fps", "frames_snap"):
        v = (f.get(key, "") or "").strip()
        for c in cands:
            if v.isdigit() and int(v) > 0:
                c[key] = int(v)
            else:
                c.pop(key, None)
    # explicit output node (Output section; blank = auto/legacy collection)
    out_node = (f.get("output_node", "") or "").strip()
    for c in cands:
        if out_node:
            c["output_node"] = out_node
        else:
            c.pop("output_node", None)
    # deliver-as extension (fetch the sibling with this ext; blank = as reported)
    out_ext = (f.get("output_ext", "") or "").strip().lstrip(".").lower()
    for c in cands:
        if out_ext:
            c["output_ext"] = out_ext
        else:
            c.pop("output_ext", None)
    # delivered texture format (generic-rig deliveries; blank = as produced)
    tf = (f.get("texture_format", "") or "").strip().lower()
    for c in cands:
        if tf in ("jpeg", "jpg"):
            c["texture_format"] = "jpeg"
        else:
            c.pop("texture_format", None)
    # flat-mode 2x2-dummy safety net: on by default (drop the key); unchecked → opt out
    dummy_on = "dummy_check" in f
    for c in cands:
        if dummy_on:
            c.pop("dummy_check", None)
        else:
            c["dummy_check"] = False
    # output files: 'rig: globs' lines → conditional cases, plain glob lines →
    # multi-file delivery (the whole delivery without an output node; extras on
    # top of the node's result with one).
    out_cases, out_flat = [], []
    for line in re.split(r"[\r\n]+", f.get("output_globs", "") or ""):
        line = line.strip()
        if not line:
            continue
        if ":" in line and not line.startswith("*"):
            label, _, rest = line.partition(":")
            gl = [g.strip() for g in rest.split(",") if g.strip()]
            if gl:
                out_cases.append({"rig": label.strip().lower(), "globs": gl})
        else:
            out_flat += [g.strip() for g in line.split(",") if g.strip()]
    for c in cands:
        c.pop("output_globs", None)
        c.pop("output_cases", None)
        if out_cases:
            c["output_cases"] = out_cases
        if out_flat:                       # plain globs may accompany cases: unconditional extras
            c["output_globs"] = out_flat
    # chain successor (blank alias → not a chain)
    succ_alias = (f.get("successor", "") or "").strip()
    keep = [g.strip() for g in re.split(r"[\r\n,]+", f.get("chain_keep", "") or "") if g.strip()]
    relay = (f.get("chain_relay", "") or "path").strip().lower()
    # A cloud successor (Meshy, Tripo) has exactly one file field and no mapping to rename
    # it, so a blank mesh param must default to ITS name (`input_mesh_path`) — ComfyUI's
    # `mesh_path` would be a param the cloud stage cannot bind. Read off the successor's
    # own file fields rather than hard-coding a name per vendor.
    succ_cand = (store.get(succ_alias) or [{}])[0]
    succ_kind = adapters.cloud_kind(succ_cand)
    files = adapters.public_fields(succ_cand)[2] if succ_kind else []
    succ = ({"alias": succ_alias,
             "export_node": (f.get("chain_export_node", "") or "").strip(),
             "mesh_param": ((f.get("chain_mesh_param", "") or "").strip()
                            or (files[0]["name"] if files else
                                ("input_mesh_path" if succ_kind else "mesh_path"))),
             **({"relay": relay} if relay == "upload" else {}),
             **({"keep_from_mesh": keep} if keep else {}),
             **({"rig": (f.get("chain_rig", "") or "").strip()} if (f.get("chain_rig", "") or "").strip() else {})}
            if succ_alias else None)
    for c in cands:
        if succ:
            c["successor"] = succ
        else:
            c.pop("successor", None)
    new_alias = (f.get("new_alias", "") or "").strip()
    if new_alias and new_alias != alias and not store.get(new_alias):
        store.delete(alias)            # rename: move under the new name
        alias = new_alias
    store.upsert(alias, cands)         # cand is cands[0] — keeps the other allowed backends
    logger.info(f"ui: updated '{alias}' ({len(mapping)} params, {len(fixed)} pinned, retries={retries or 'all'})")
    # Save keeps the editor open (mapping work is iterative); a transient banner
    # confirms the write instead of the editor closing.
    return RedirectResponse(f"/ui/mapping?edit={quote(alias)}&saved=1", status_code=303)


def _cloud_update_apply(kind: str, cands: list, f: dict) -> None:
    """Apply a cloud alias form to EVERY candidate (they are the alias's shape, not
    per-backend). Options go through `parse_options` (the schema) and then the module's
    `options_of` (the same normalization the request builder applies), so what is stored
    is exactly what will be sent — a combination the vendor refuses cannot survive a Save
    and turn up as a 400 on a paid request."""
    mod = adapters.cloud_module(kind)
    ep = (f.get("cloud_endpoint", "") or "").strip()
    ep = ep if ep in mod.ENDPOINTS else mod.ENDPOINTS[0]
    model = (f.get("cloud_model", "") or "").strip()
    model = model if model in mod.AI_MODELS else mod.AI_MODELS[0]
    opts = cloudtask.parse_options(mod.OPTION_FIELDS, f, mod.OPTION_DEFAULTS)
    # ASSIGN, never mutate: parse_options shallow-copies the defaults, so appending here
    # would rewrite OPTION_DEFAULTS["target_formats"] for every future alias.
    opts["target_formats"] = [x for x in mod.FORMATS if f.get(f"fmt__{x}")] or ["glb"]
    opts = mod.options_of({mod.KIND: {"endpoint": ep, "options": opts}, "model": model})
    task = (f.get("task", "") or "").strip()
    retries = (f.get("retries", "") or "").strip()
    # chain successor (blank alias → not a chain). No export_node and no relay: a cloud
    # stage's mesh is a result blob and always travels to stage 2 as bytes (_run_chain
    # forces `upload`), so those two ComfyUI fields would only be misleading here.
    succ_alias = (f.get("successor", "") or "").strip()
    keep = [g.strip() for g in re.split(r"[\r\n,]+", f.get("chain_keep", "") or "") if g.strip()]
    rig = (f.get("chain_rig", "") or "").strip()
    succ = ({"alias": succ_alias,
             "mesh_param": (f.get("chain_mesh_param", "") or "").strip() or "input_mesh_path",
             **({"keep_from_mesh": keep} if keep else {}),
             **({"rig": rig} if rig else {})}
            if succ_alias else None)
    for c in cands:
        # a fresh options dict per candidate — one shared object would let a later
        # in-place edit of one candidate rewrite the others
        c[mod.KIND] = {"endpoint": ep, "options": json.loads(json.dumps(opts))}
        c["model"] = model
        c["retries"] = retries
        if succ:
            c["successor"] = json.loads(json.dumps(succ))    # own copy per candidate (see above)
        else:
            c.pop("successor", None)
        if task:
            c["task"] = task


async def cloud_update(request: Request):
    """Save a cloud alias (Meshy, Tripo): endpoint + ai model + the admin options, on
    every candidate. Refuses a ComfyUI alias — its fields live in /ui/mapping/update."""
    f = await _form(request)
    alias = (f.get("alias", "") or "").strip()
    cands = store.get(alias) if alias else []
    kind = adapters.cloud_kind(cands[0]) if cands else None
    if not kind:
        raise HTTPException(404, "cloud alias not found")
    _cloud_update_apply(kind, cands, f)
    new_alias = (f.get("new_alias", "") or "").strip()
    if new_alias and new_alias != alias and not store.get(new_alias):
        store.delete(alias)            # rename: move under the new name
        alias = new_alias
    store.upsert(alias, cands)
    logger.info(f"ui: updated {kind} alias '{alias}' "
                f"({cands[0][kind]['endpoint']}, {cands[0]['model']})")
    return RedirectResponse(f"/ui/mapping?edit={quote(alias)}&saved=1", status_code=303)


async def delete(request: Request):
    alias = request.query_params.get("alias", "").strip()
    if alias:
        store.delete(alias)
    return RedirectResponse("/ui/mapping?sub=media", status_code=303)


async def copy(request: Request):
    alias = _qp(request, "alias")
    cands = store.get(alias)
    if not cands:
        return RedirectResponse("/ui/mapping?sub=media", status_code=303)
    new = f"{alias}-copy"
    i = 2
    while store.get(new):
        new, i = f"{alias}-copy{i}", i + 1
    store.upsert(new, json.loads(json.dumps(cands)))     # deep copy of the candidate(s)
    logger.info(f"ui: copied alias '{alias}' → '{new}'")
    return RedirectResponse(f"/ui/mapping?edit={new}", status_code=303)


# ── Tab: Playground ─────────────────────────────────────────────────────────────

# Uploaded inputs stick across generations: stashed in memory PER USER as
# {param: (filename, bytes)} so the file-input (which the browser can't pre-fill)
# doesn't have to be re-picked each time — and so switching the model keeps them:
# same-named slots carry over to the new alias (generate() filters to the alias's
# actual slots). A new upload replaces; the "clear" checkbox drops it. Lost on
# restart. The filename rides along because a MESH's type lives in its extension
# alone (a .fbx and a .glb are both "some binary"); images carry theirs in the bytes.
_pg_images: dict = {}

# Mesh uploads the file picker offers, and what a stashed extension means on the
# wire. Deliberately NOT adapters._mime_and_kind: that maps .fbx/.ply to
# application/octet-stream (right for storage, wrong here — the API recovers the
# file's extension from the data-URI MIME, and octet-stream would lose it).
_PG_FILE_ACCEPT = ".glb,.gltf,.obj,.fbx,.stl,.ply"
_PG_FILE_MIME = {"glb": "model/gltf-binary", "gltf": "model/gltf+json", "obj": "model/obj",
                 "stl": "model/stl", "ply": "model/ply", "fbx": "model/fbx"}


def _pg_file_mime(name: str) -> str:
    """The data-URI MIME for a stashed mesh, derived from its filename extension —
    that MIME is the ONLY thing the API has to rebuild the extension with (see
    main._EXT_BY_MIME), so an unknown one costs the file its type."""
    return _PG_FILE_MIME.get(os.path.splitext(name or "")[1].lstrip(".").lower(),
                             "application/octet-stream")


def _pg_history_opts() -> dict:
    """{"image": [(value, label)…], "file": […]} — every artifact of the recent jobs,
    newest first, as picker options. `value` is "<job>:r:<n>" (a result) or
    "<job>:i:<n>" (a stored input image); generate() resolves it back to bytes.

    An image slot may take any earlier image — a result OR another job's reference
    input (re-running the same picture against a second alias is the common case);
    a mesh param takes result FILES (glb/fbx/…), which is what a mesh stage produces."""
    out = {"image": [], "file": []}
    for j in jobs.recent_artifacts(limit=60):
        when = time.strftime("%H:%M", time.localtime(int(j.get("created") or 0)))
        head = f"{when} {j.get('alias') or '?'}"
        for r in j["results"]:
            kind = r.get("kind")
            if kind in out:
                out[kind].append((f"{j['id']}:r:{r['n']}", f"{head} · #{r['n']} {r.get('name') or ''}"))
        for i in j["inputs"]:
            out["image"].append((f"{j['id']}:i:{i['n']}",
                                 f"{head} · in#{i['n']} {i.get('slot') or i.get('filename') or ''}"))
    return out


def _pg_history_blob(ref: str):
    """A picked "<job>:r|i:<n>" → (filename, bytes), or None when the artifact is gone
    (TTL pruning deletes a job's files while the row still lists them)."""
    jid, _, rest = (ref or "").partition(":")
    which, _, ns = rest.partition(":")
    if which not in ("r", "i") or not ns.isdigit():
        return None
    n = int(ns)
    p = jobs.result_path(jid, n) if which == "r" else jobs.input_path(jid, n)
    if not p:
        return None
    try:
        with open(p[0], "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    # a result's original name (mesh.glb) if it has one, else the on-disk name —
    # either way the EXTENSION is what the upload's mime is derived from
    name = (p[2] if which == "r" and len(p) > 2 and p[2] else os.path.basename(p[0]))
    return name, data


def _alias_defaults(cand: dict) -> dict:
    """Default value per request param, read from the workflow at the mapped node."""
    wf = cand.get("workflow_json") or {}
    out = {}
    for param, m in (cand.get("mapping") or {}).items():
        v = (wf.get(m.get("node"), {}).get("inputs") or {}).get(m.get("field"))
        if v is not None and not isinstance(v, list):
            out[param] = v
    return out


def _playground_form(aliases: list, vals: dict, cand: Optional[dict], oi: Optional[dict] = None,
                     kept: Optional[dict] = None) -> str:
    """`kept` is the user's upload stash ({param: (filename, bytes)}) — membership drives
    the '✓ kept' badge, the name is shown for mesh uploads (which one is loaded?)."""
    v = lambda k: str(vals.get(k) if vals.get(k) is not None else "")
    # Job-artifact picker options, loaded AT MOST ONCE per render (one SELECT) and only
    # when a row actually needs them — a text-only alias must not pay for the query.
    hist_cache: list = []

    def hist(param: str, kind: str) -> str:
        if not hist_cache:
            hist_cache.append(_pg_history_opts())
        return _select(f"hist__{param}", [("", "(from a job…)")] + hist_cache[0][kind])

    def kept_badge(param: str, show_name: bool = False) -> str:
        name = (kept or {}).get(param)
        name = name[0] if isinstance(name, tuple) else ""
        return (' <span class="badge ok">✓ kept</span>'
                + (f' <span class="muted">{_esc(name)}</span>' if show_name and name else "")
                + ' <label class="muted" style="font-weight:normal">'
                  f'<input type="checkbox" name="clear__{_esc(param)}"> clear</label>')
    # selecting an alias reloads with its workflow defaults pre-filled
    opts = "".join(f'<option value="{_esc(a)}"{" selected" if a == vals.get("model") else ""}>{_esc(a)}</option>'
                   for a in aliases)
    # switching alias carries the current scalar fields over, so same-named params
    # (prompt, lora_01, …) survive; the new alias just ignores the ones it lacks.
    alias_select = f'<select name="model" onchange="pgSwitch(this)">{opts}</select>'
    # one input per MAPPED param of the selected alias (dynamic — mirrors the request
    # fields configured in Mapping, in drag-set order). Image-loader fields render as
    # file uploads (img__<param>, empty → 8×8 placeholder); scalars as p__<param>
    # (number when the default is numeric, else text). The label override (else the
    # param name) is what the user sees.
    mapping = (cand.get("mapping") if cand else {}) or {}
    wf = (cand.get("workflow_json") if cand else {}) or {}
    imgset = set(adapters.image_params(wf, mapping))
    defaults = _alias_defaults(cand) if cand else {}
    rows = ""
    # A cloud alias (Meshy, Tripo) has no workflow to read fields off — its public fields
    # are the fixed label table (same source the schema endpoint uses), so they are
    # rendered from adapters.public_fields and the workflow-driven loop below is skipped.
    if cand and adapters.cloud_kind(cand):
        params, images, mfiles = adapters.public_fields(cand)
        for x in mfiles:
            # Same row as a ComfyUI mesh param (upload or an earlier job's artifact) —
            # WITHOUT the "path on the backend" field: Meshy/Tripo read the bytes out of
            # the request, there is no backend disk a path could point into.
            acc = ",".join("." + a for a in (x.get("accept") or [])) or _PG_FILE_ACCEPT
            extra = (kept_badge(x["name"], show_name=True) if kept and x["name"] in kept else
                     ' <span class="muted">required</span>' if x.get("required") else "")
            rows += _field(x["name"], f'<input type="file" name="file__{_esc(x["name"])}" '
                                      f'accept="{_esc(acc)}"> {hist(x["name"], "file")}{extra}')
        for i in images:
            extra = (kept_badge(i["name"])
                     if kept and i["name"] in kept else
                     ' <span class="muted">required</span>' if i["required"] else
                     ' <span class="muted">optional · empty → not sent</span>')
            rows += _field(i["name"], f'<input type="file" name="img__{_esc(i["name"])}" '
                                      f'accept="image/png,image/jpeg"> {hist(i["name"], "image")}{extra}')
        for p in params:
            cur = v(p["name"]) or ("" if p.get("default") in (None, "") else str(p["default"]))
            if p.get("choices"):
                rows += _field(p["name"], _select(f"p__{p['name']}",
                                                  [(c, c or "none") for c in p["choices"]], cur))
            elif p["type"] == "bool":
                rows += _field(p["name"], _checkbox(f"p__{p['name']}",
                                                    cur.lower() in ("true", "1", "on"), p["name"]))
            else:
                rows += _field(p["name"], _inp(f"p__{p['name']}", cur,
                                               typ="number" if p["type"] in ("int", "float") else "text"))
        mapping = {}                       # skip the workflow-driven loop below
    for p, m in mapping.items():
        label = (m or {}).get("label") or p
        if p in imgset:
            emode = adapters.slot_empty_mode(m)
            if kept and p in kept:
                extra = kept_badge(p)
            elif emode == "required":
                extra = ' <span class="muted">required — no placeholder</span>'
            elif emode == "disable":
                extra = ' <span class="muted">empty → loader node disabled</span>'
            else:
                extra = ' <span class="muted">empty → 8×8 placeholder</span>'
            rows += _field(label, f'<input type="file" name="img__{_esc(p)}" accept="image/*"> '
                                  f'{hist(p, "image")}{extra}')
        elif adapters.is_file_param(p, m):
            # A mesh param takes a real file (uploaded, or an earlier job's result) —
            # sent as the API's `files`. The old text field stays as the second way in:
            # a path that already exists ON THE BACKEND needs no upload at all. Upload
            # wins over path when both are set (generate() drops the path then).
            extra = kept_badge(p, show_name=True) if kept and p in kept else ""
            rows += _field(label,
                           f'<input type="file" name="file__{_esc(p)}" accept="{_PG_FILE_ACCEPT}"> '
                           f'{hist(p, "file")}{extra}'
                           f'<div style="margin-top:4px">'
                           + _inp(f"p__{p}", v(p), placeholder="…or a path on the backend")
                           + '</div>')
        else:
            dv = defaults.get(p)
            node, field = (m or {}).get("node"), (m or {}).get("field")
            opts = (oi or {}).get((wf.get(node) or {}).get("class_type", ""), {}).get(field)
            if isinstance(opts, list) and opts:          # ANY combo (model, skeleton_template, …) → dropdown
                cur = v(p) or (str(dv) if dv is not None else "")
                o = list(opts)
                if cur and cur not in o:
                    o = [cur] + o
                rows += _field(label, _select(f"p__{p}", o, cur))
            elif isinstance(opts, dict) and opts.get("_num"):    # FLOAT/INT → bounded number input
                cur = v(p) or (str(dv) if dv is not None else "")
                rows += _field(label, _num_input(f"p__{p}", cur, opts))
            elif "prompt" in p.lower():                          # prompt / negative_prompt → multi-line
                rows += _field(label, _textarea(f"p__{p}", v(p), 4), wide=True)
            else:
                typ = "number" if isinstance(dv, (int, float)) and not isinstance(dv, bool) else "text"
                rows += _field(label, _inp(f"p__{p}", v(p), typ=typ))
    rows = rows or "<p class='hint'>This alias has no request fields — add some in Mapping.</p>"
    bk_list = [c.get("backend") for c in (store.get(vals.get("model")) or [])]
    sel_bk = vals.get("backend", "")
    bk_opts = ('<option value="">(auto · scheduler picks)</option>'
               + "".join(f'<option value="{_esc(b)}"{" selected" if b == sel_bk else ""}>{_esc(b)}</option>'
                         for b in bk_list))
    backend_field = _field("backend", f'<select name="backend">{bk_opts}</select>') if bk_list else ""
    pg_switch_js = ("<script>function pgSwitch(sel){var f=sel.form,"
                    "q='sub=media&model='+encodeURIComponent(sel.value);"
                    "f.querySelectorAll('[name^=\"p__\"]').forEach(function(el){"
                    "if(el.value!=='')q+='&'+encodeURIComponent(el.name)+'='+encodeURIComponent(el.value);});"
                    "location.href='/ui/playground?'+q;}</script>")
    return ('<form action="/ui/playground/generate" method="post" enctype="multipart/form-data">'
            f'<div class="formbar"><h2>Media Playground</h2>{_btn("Generate", submit=True)}</div>'
            + _field("alias", alias_select)
            + backend_field
            + rows
            + '<p class="hint">synchronous; image backends ~1 min · each image reuses one slot</p></form>'
            + pg_switch_js)


def _playground_body(aliases: list, vals: dict, cand: Optional[dict], result_html: str,
                     oi: Optional[dict] = None, kept: Optional[dict] = None) -> str:
    # model-viewer loads with the page and stays loaded: _LIVE_JS never re-inserts a
    # <script>, so a viewer arriving through a live update upgrades against the
    # definition that is already there.
    return (f'<script type="module" src="{_MODELVIEWER_SRC}"></script>'
            f'<div class="cols"><div class="col">{_playground_form(aliases, vals, cand, oi, kept)}</div>'
            f'<div class="col"><div id="resultcol">{result_html}</div></div></div>')


def _job_result_html(job_id: str, job: Optional[dict]):
    """(right-column HTML, refresh-seconds-or-None) for a generation job."""
    if not job:
        return f"<h2>Result</h2><p class='bad'>job {_esc(job_id)} not found</p>", None
    st = job.get("status")
    if st in ("queued", "running"):
        return (f"<h2>Result</h2><p>⏳ <b>Generating…</b></p>"
                f"<p class='muted'>job {_esc(job_id)} · {st} · this view auto-updates</p>"), 2
    if st == "failed":
        return (f"<h2>Result</h2><p class='bad'>✗ failed · job {_esc(job_id)}</p>"
                f"<pre class='err'>{_esc(job.get('error'))}</pre>"), None
    cells = []
    for r in job.get("results", []):
        src = f"/ui/playground/result/{_esc(job_id)}/{r['n']}"
        extra = ""
        if (r.get("mime") or "").lower() == "model/gltf-binary":
            rp = jobs.result_path(job_id, r["n"])
            extra = _glb_stats_html(rp[0] if rp else None)
        cells.append(f"<div>{_media_tag(src, r.get('mime'), r.get('kind'), cls='result', autoplay=True)}{extra}</div>")
    imgs = "".join(cells)
    meta = job.get("meta") or {}
    tags = ""
    if meta.get("rig"):
        tags += " · " + _badge(f"rig: {meta['rig']}", "ok" if meta["rig"] == "mixamo" else "warn")
    wrn = ("<p class='hint'>⚠ " + "; ".join(_esc(w) for w in meta["warnings"]) + "</p>") if meta.get("warnings") else ""
    return (f"<h2>Result</h2><p>✓ done · job {_esc(job_id)} · "
            f"backend {_esc(job.get('backend'))}{tags}</p>{wrn}{imgs or '<p class=muted>No artifacts.</p>'}"), None


async def playground_page(request: Request):
    """Parent tab: dispatches on ?sub= (see SUBTABS; first child = default)."""
    qp = request.query_params
    sub = qp.get("sub") or SUBTABS["playground"][0][0]
    if sub == "chat":
        vals = {k: qp.get(k, "") for k in _CHATPLAY_KEYS}
        result_html = "<h2>Response</h2><p class='hint'>Send a message to see the reply here.</p>"
        return HTMLResponse(_page("Chat Playground", _chatplay_body(vals, result_html), "playground",
                                  subnav=_subnav("playground", "chat")))
    if sub == "voice":
        vals = {k: qp.get(k, "") for k in _VOICEPLAY_KEYS}
        vu_user = _session_user(request) or "default"
        prog = _voice_upload_prog.get(vu_user)
        # A running upload makes the page live at 1s; the final tick renders the
        # finished checklist AND the library table with the new entry, then drops
        # data-live to stop the poller — which is all the old reload ever did.
        # Consumed once it has been SHOWN in its terminal state: _voice_upload_prog
        # lives for the whole process and nothing else clears it, so without this the
        # last upload's verdict is what every later visit to the tab renders instead of
        # the hint — a 303 landing from ship/delete included, where it can describe a
        # reference that no longer exists. `seen` is set on the render that shows the
        # finished checklist (the final live tick still gets it); the NEXT GET drops the
        # entry. A new upload installs a fresh dict, so it is never popped early.
        if prog is not None and prog.get("seen"):
            _voice_upload_prog.pop(vu_user, None)
            prog = None
        vu_refresh = 1 if (prog and not prog.get("done")) else None
        if prog is not None and prog.get("done"):
            prog["seen"] = True
        result_html = (_vu_fragment(prog) if prog else
                       "<h2>Result</h2><p class='hint'>Synthesize to hear the result here.</p>")
        return HTMLResponse(_page("Voice", _voiceplay_body(vals, result_html), "playground",
                                  refresh=vu_refresh, subnav=_subnav("playground", "voice")))
    if not store.is_active():
        return _inactive()
    aliases = list(store.list_aliases().keys())
    if not aliases:
        return HTMLResponse(_page("Media Playground",
            "<h2>Media Playground</h2><p class='hint'>Register an alias in "
            "the <a href='/ui/mapping'>Mapping</a> tab first.</p>", "playground",
            subnav=_subnav("playground", "media")))
    model = qp.get("model", "") or aliases[0]   # first load: pick the first alias
    cand = (store.get(model) or [None])[0]
    # values per request param, keyed by param name; query (p__<param>) overrides the
    # alias's workflow default. Params are dynamic — whatever Mapping configured.
    vals = {"model": model, "backend": qp.get("backend", "")}
    defaults = _alias_defaults(cand) if cand else {}
    # the alias's public param names: a mapping for a ComfyUI alias, the fixed label
    # table for a cloud one (Meshy, Tripo) — without this a p__<field> from the URL
    # (Send to Playground, or the post-Generate redirect) would never reach the form
    if cand and adapters.cloud_kind(cand):
        _pf, _, _ff = adapters.public_fields(cand)
        pnames = [x["name"] for x in _pf] + [x["name"] for x in _ff]
    else:
        pnames = list((cand.get("mapping") if cand else {}) or {})
    for p in pnames:
        q = qp.get(f"p__{p}", "")
        vals[p] = q if q != "" else ("" if defaults.get(p) is None else str(defaults[p]))
    job_id = qp.get("job", "")
    refresh = None
    if job_id:
        result_html, refresh = _job_result_html(job_id, jobs.get(job_id))
    else:
        result_html = "<h2>Result</h2><p class='hint'>Generate to see the result here.</p>"
    wf = (cand.get("workflow_json") if cand else {}) or {}
    oi = await _object_info(cand.get("backend", ""), wf, cand.get("mapping")) if cand else {}
    kept = dict(_pg_images.get(_session_user(request) or "default", {}))
    return HTMLResponse(_page("Media Playground",
                              _playground_body(aliases, vals, cand, result_html, oi, kept),
                              "playground", refresh=refresh,
                              subnav=_subnav("playground", "media")))


async def generate(request: Request):
    f = await _multipart(request)
    model = str(f.get("model", ""))
    force_bk = str(f.get("backend", "")).strip()       # pin to one backend (testing per-backend)
    # dynamic request fields arrive as p__<param>. prompt/negative_prompt feed the
    # request's inputs; everything else goes into params (numeric-coerced when it
    # parses). The mapping in the alias decides which node each lands on.
    submitted = {k[len("p__"):]: str(f.get(k, "")) for k in f if k.startswith("p__")}
    body = {"model": model, "mode": "async", "params": {}}     # async → instant job id
    if force_bk:
        body["backend"] = force_bk
    for p, raw in submitted.items():
        if raw.strip() == "":
            continue
        if p in ("prompt", "negative_prompt"):
            body[p] = raw
        else:
            body["params"][p] = _num(raw)
    # per-field uploads: images (img__<param>, empty → the 8×8 placeholder downstream,
    # so simply omitted here) and mesh files (file__<param>). Both persist across
    # generations AND model switches (one stash per user, keyed by slot/param name —
    # same-named slots carry over to another alias); a new upload replaces, a checked
    # clear__<param> drops the kept one, and hist__<param> takes an earlier job's
    # artifact instead of a fresh upload.
    user = _session_user(request) or "default"
    stash = _pg_images.setdefault(user, {})
    cand = (store.get(model) or [None])[0]
    wf_i = (cand.get("workflow_json") if cand else {}) or {}
    map_i = (cand.get("mapping") if cand else {}) or {}
    vals = {"model": model, "backend": force_bk, **submitted}

    def _err(msg: str, status: int = 404):
        aliases = list(store.list_aliases().keys())
        result_html = f'<h2>Result</h2><p class="bad">Error {status}: {_esc(msg)}</p>'
        return HTMLResponse(_page("Media Playground",
                                  _playground_body(aliases, vals, cand, result_html, kept=dict(stash)),
                                  "playground", subnav=_subnav("playground", "media")))

    # One pass PER PARAM, so the three ways an input arrives have a defined precedence:
    # clear drops what was kept, a fresh upload beats it, a history pick fills what is
    # then still empty. (`__filename` companions are skipped — they are not params.)
    for p in sorted({k.split("__", 1)[1] for k in f
                     if k.startswith(("img__", "file__", "hist__", "clear__"))
                     and not k.endswith("__filename")}):
        if f.get(f"clear__{p}") is not None:          # checkbox present ⇒ checked
            stash.pop(p, None)
        up = next((k for k in (f"file__{p}", f"img__{p}")
                   if isinstance(f.get(k), (bytes, bytearray)) and f[k].strip()), None)
        if up:
            # the browser's filename, because a mesh's type lives in its extension —
            # and a fallback that HAS one: a name without a suffix would go out as
            # application/octet-stream and the API could not name the file's kind.
            stash[p] = (str(f.get(f"{up}__filename") or "")
                        or (f"{p}.glb" if up.startswith("file__") else f"{p}.png"), bytes(f[up]))
            continue
        ref = str(f.get(f"hist__{p}", "") or "").strip()
        if ref:
            got = _pg_history_blob(ref)
            if not got:                               # never silently generate without it
                jid, _, rest = ref.partition(":")
                return _err(f"job {jid} #{rest.partition(':')[2]} is no longer available — "
                            f"its files were deleted (job TTL)")
            stash[p] = got
    # only the slots/params this alias actually has ride along — the stash may carry
    # another alias's inputs (that is the point of it surviving a model switch).
    if cand and adapters.cloud_kind(cand):
        _, m_imgs, m_files = adapters.public_fields(cand)
        slots = {i["name"] for i in m_imgs}
        fset = {x["name"] for x in m_files}           # a rigging alias: input_mesh_path
    else:
        slots = set(adapters.image_params(wf_i, map_i)) if wf_i else set()
        fset = set(adapters.file_params(wf_i, map_i))
    if slots or fset:
        images = {p: v for p, v in stash.items() if p in slots}
        files = {p: v for p, v in stash.items() if p in fset}
    else:
        # An alias the store knows nothing about (no mapping at all): send the stash
        # as images and let downstream ignore extras — EXCEPT a file-ish param. A mesh
        # left in the stash by another alias must never ride as an image; the alias
        # cannot consume it either way, and `images` means images.
        images = {p: v for p, v in stash.items() if not adapters.is_file_param(p, None)}
        files = {}
    # A REAL API call through POST /v1/generations (reference images as the API's
    # per-field base64 `images` dict, meshes as `files` data-URIs) — the playground
    # tests the API, bypassing nothing.
    if images:
        body["images"] = {p: base64.b64encode(d).decode() for p, (_, d) in images.items()}
    if files:
        body["files"] = {p: f"data:{_pg_file_mime(n)};base64,{base64.b64encode(d).decode()}"
                         for p, (n, d) in files.items()}
        for p in files:
            # an upload beats the typed backend path — sending both would bind the
            # same node twice, and the API would reject the pair
            body["params"].pop(p, None)
    try:
        try:
            r = await _self_api(request, "POST", "/v1/generations", json=body)
        except httpx.HTTPError as e:
            raise HTTPException(502, f"{type(e).__name__}: {e}")
        if r.status_code not in (200, 202):
            try:
                detail = str((r.json() or {}).get("detail") or "")[:300] or r.text[:300]
            except Exception:
                detail = r.text[:300]
            raise HTTPException(r.status_code, detail)
        view = r.json()
    except HTTPException as e:
        return _err(str(e.detail), e.status_code)
    # Redirect to the GET view (form re-populated + live-updating) — instant feedback.
    q = urlencode({"sub": "media", "model": model, "backend": force_bk, "job": view.get("job_id", ""),
                   **{f"p__{p}": v for p, v in submitted.items() if v}})
    return RedirectResponse(f"/ui/playground?{q}", status_code=303)


_STATIC_DIR = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"))
_STATIC_MIME = {".js": "text/javascript", ".css": "text/css", ".wasm": "application/wasm",
                ".json": "application/json"}   # served extensions (bundled JS libs)


async def static_asset(path: str):
    """Serve a bundled /ui static asset (model-viewer, three.js + FBXLoader). Only
    known JS/CSS extensions under the static dir; the resolved real path must stay
    inside it (no traversal). Long immutable cache — files are content-versioned."""
    ext = os.path.splitext(path)[1].lower()
    full = os.path.realpath(os.path.join(_STATIC_DIR, path))
    if (ext not in _STATIC_MIME or not full.startswith(_STATIC_DIR + os.sep)
            or not os.path.isfile(full)):
        raise HTTPException(404, "not found")
    return FileResponse(full, media_type=_STATIC_MIME[ext],
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


async def result(job_id: str, n: int, anim: str = ""):
    rp = await asyncio.to_thread(jobs.result_path, job_id, n)
    if rp is None:
        raise HTTPException(404, "result not found")
    path, mime, name = rp
    if anim == "idle" and mime == "model/gltf-binary":   # preview-only: inject a diagnostic idle clip
        import previewanim
        def _read_and_anim():                            # file read + full GLB rebuild (30 MB) off-loop —
            with open(path, "rb") as fh:                 # otherwise it stalls chat streams and /v1 dispatch
                return previewanim.add_idle(fh.read())
        data = await asyncio.to_thread(_read_and_anim)
        return Response(content=data, media_type=mime)
    headers = None
    if name:
        headers = {"Content-Disposition": jobs.content_disposition(name)}
    return FileResponse(path, media_type=mime, headers=headers)


# ── Playground sub-tab: Voice (direct /v1/audio/speech, synchronous) ─────────────
# A real API client like chatplay: POSTs the gateway's own /v1/audio/speech and
# plays the returned WAV. The result bytes live in a per-user stash (like the
# media playground's reference images) served by /ui/playground/voice-audio.

_VOICEPLAY_KEYS = ("model", "input", "voice", "ref_text")
_voice_out: dict = {}          # user → (bytes, mime) — last synthesis result

_VOICEPLAY_JS = ("<script>function vpSending(f){"
                 "var r=document.getElementById('vpresult');"
                 "if(r)r.innerHTML=\"<h2>Result</h2><p class='muted'>\\u23f3 <b>Synthesizing\\u2026</b> · "
                 "routing + waiting for the backend (model load can take a while)</p>\";"
                 "var b=f.querySelector('button[type=submit]');if(b){b.disabled=true;b.textContent='Synthesizing\\u2026';}"
                 "return true;}</script>")


def _voiceplay_form(vals: dict) -> str:
    v = lambda k: vals.get(k, "")
    lib = store.get_voice_library() if store.is_active() else {}
    vopts = '<option value="">— custom path / model default —</option>' + "".join(
        f'<option value="lib:{_esc(n)}"{" selected" if v("voice") == f"lib:{n}" else ""}>'
        f'📚 {_esc(n)}{"" if (e or {}).get("shipped") else " (not shipped!)"}</option>'
        for n, e in sorted(lib.items()))
    return ('<form action="/ui/playground/voice" method="post" onsubmit="return vpSending(this)">'
            f'<div class="formbar"><h2>Voice</h2>{_btn("Synthesize", submit=True)}</div>'
            + _field("model", _dl_input("model", v("model"), "vpmodels", "TTS model id or alias"))
            + _field("text", _textarea("input", v("input"), 5, "the text to speak"))
            + _field("voice (library)", f'<select name="voice_sel">{vopts}</select>')
            + _field("…or path", _inp("voice", "" if v("voice").startswith("lib:") else v("voice"),
                                      placeholder="raw backend-side path, e.g. /opt/voices/kai.wav"))
            + _field("ref text", _textarea("ref_text", v("ref_text"), 2,
                                           "transcript override (library entries carry their own)"))
            + "<p class='hint'>Synchronous <code>POST /v1/audio/speech</code> — routed like chat. "
              "Pick a <b>library voice</b> (its reference + transcript are applied by the gateway) "
              "or give a raw path; empty = the model's default voice.</p>"
            + "</form>" + _datalist("vpmodels", _chat_models()) + _VOICEPLAY_JS)


def _voice_lib_panel(status_html: str = "") -> str:
    """Library management under the Voice form: scp target, upload, entries table."""
    lib = store.get_voice_library() if store.is_active() else {}
    rows = ""
    for n, e in sorted(lib.items()):
        e = e or {}
        hosts_state = " · ".join(f"{h.split('@')[-1]}: {v if v == 'ok' else v[:60]}"
                                 for h, v in (e.get("hosts") or {}).items())
        shipped = (_badge("shipped", "ok", f"{e.get('remote', '')} — {hosts_state}") if e.get("shipped")
                   else _badge("pending", "warn", hosts_state or "not on the backend hosts yet — retry ship"))
        rt = (e.get("ref_text") or "")[:60]
        play = (f'<a class="btn secondary sm icon" href="#" data-v="{_esc(n)}" '
                f'onclick="return vlPlay(this)" title="Play the reference in the result column">▶</a>')
        acts = play + _icon_acts(
            ("↻", f"/ui/playground/voice-ship?name={quote(n)}", "secondary", "Ship to the backend host (scp)"),
            ("✕", f"/ui/playground/voice-del?name={quote(n)}", "danger", "Delete", f"Delete voice '{n}'?"))
        rows += (f"<tr><td><code>lib:{_esc(n)}</code></td><td>{shipped}</td>"
                 f"<td class='muted' title='{_esc(e.get('ref_text', ''))}'>{_esc(rt)}</td>"
                 f"<td class='acts'>{acts}</td></tr>")
    rows = rows or "<tr><td colspan=4 class='muted'>no voices yet — upload one below</td></tr>"
    hosts, rdir = _voice_ship_config()
    wm = str((store.get_settings() or {}).get("whisper_model") or "small") if store.is_active() else "small"
    wm_opts = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
    if wm not in wm_opts:
        wm_opts = [wm] + wm_opts
    return (
        "<div style='margin-top:18px;padding-top:12px;border-top:1px solid #272b33'>"
        "<h2>Voice library</h2>"
        "<p class='hint'>Uploaded references are stored on the gateway (master copy) and <b>shipped via "
        "scp to every TTS host</b> — LocalAI has no upload API; the model reads <code>voice</code> only "
        "as a local file, and failover may route to any host serving the model. Empty <b>ref text</b> is "
        "auto-transcribed by the gateway's local faster-whisper (CPU).</p>"
        + status_html
        + f"<table><tr><th>voice</th><th>state</th><th>ref text</th><th></th></tr>{rows}</table>"
        + '<form action="/ui/playground/voice-upload" method="post" enctype="multipart/form-data" '
          'onsubmit="return vuSending(this)" style="margin-top:10px">'
        + _field("name", _inp("name", "", placeholder="kai"), short=True)
        + _field("wav file", '<input type="file" name="file" accept="audio/*">')
        + _field("ref text", _textarea("ref_text", "", 2, "leave empty → auto-transcribe (whisper)"))
        + f'<div class="field"><label></label><div class="control">{_btn("⬆ Upload voice", submit=True)}</div></div>'
        + "</form>"
        + '<form action="/ui/playground/voice-target" method="post" style="margin-top:6px">'
        + _field("scp targets", _inp("hosts", ", ".join(hosts),
                                     placeholder="user@tts-host:/abs/host/dir, … (comma-separated)"))
        + _field("voice dir (model view)", _inp("dir", rdir,
                                                placeholder="the path the MODEL sees, e.g. its container mount"))
        + _field("whisper model", _select("whisper_model", wm_opts, wm), short=True)
        + f'<div class="field"><label></label><div class="control">{_btn("Save settings", submit=True, kind="secondary")}'
          "<span class='hint' style='margin-left:10px'>one scp target per cloning host (host-side dir — for a "
          "dockerized LocalAI the bind-mount source); <b>voice dir</b> is what goes into <code>voice</code> — the "
          "path from the model's view, identical on every host. Key once per host: <code>ssh-copy-id</code> from "
          "the gateway.</span></div></div>"
        + "</form>"
        + "<script>function vlPlay(a){var r=document.getElementById('vpresult');if(!r)return false;"
          "var n=a.getAttribute('data-v');"
          "r.innerHTML=\"<h2>Result</h2><p class='muted'>\\ud83d\\udcda reference: <b>\"+n+\"</b></p>"
          "<audio class='result' controls autoplay src='/ui/playground/voice-lib/\"+encodeURIComponent(n)+"
          "\"?t=\"+Date.now()+\"'></audio>\";return false;}"
          "function vuSending(f){var r=document.getElementById('vpresult');"
          "if(r)r.innerHTML=\"<h2>Result</h2><p class='muted'>\\u23f3 <b>Uploading voice\\u2026</b> · "
          "transcribing (local whisper) + shipping to the TTS hosts (scp) — the first run can take a "
          "while (whisper model load)</p>\";"
          "var b=f.querySelector('button[type=submit]');if(b){b.disabled=true;b.textContent='Uploading\\u2026';}"
          "return true;}</script></div>")


def _voiceplay_body(vals: dict, result_html: str, lib_status: str = "") -> str:
    return (f'<div class="cols"><div class="col">{_voiceplay_form(vals)}{_voice_lib_panel(lib_status)}</div>'
            f'<div class="col" id="vpresult">{result_html}</div></div>')


async def voiceplay_send(request: Request):
    f = await _form(request)
    vals = {k: (f.get(k, "") or "") for k in _VOICEPLAY_KEYS}
    if (f.get("voice_sel", "") or "").strip():         # picked library voice wins over the path box
        vals["voice"] = f["voice_sel"].strip()
    model, text = vals["model"].strip(), vals["input"].strip()
    if not model or not text:
        result = "<h2>Result</h2><p class='bad'>model and text are required</p>"
        return HTMLResponse(_page("Voice", _voiceplay_body(vals, result), "playground",
                                  subnav=_subnav("playground", "voice")))
    body = {"model": model, "input": text}
    if vals["voice"].strip():
        body["voice"] = vals["voice"].strip()
    if vals["ref_text"].strip():
        body["params"] = {"ref_text": vals["ref_text"].strip()}
    try:
        r = await _self_api(request, "POST", "/v1/audio/speech", json=body)
        ct = r.headers.get("content-type", "")
        if r.status_code == 200 and not ct.startswith(("application/json", "text/")):
            _voice_out[_session_user(request) or "default"] = (r.content, ct or "audio/wav")
            meta = (f"backend <b>{_esc(r.headers.get('x-gateway-backend', '—'))}</b> · "
                    f"{len(r.content) // 1024} KB · {_esc(ct)}")
            src = f"/ui/playground/voice-audio?t={int(time.time())}"   # cache-buster per synthesis
            result = (f"<h2>Result</h2><p class='muted'>✓ {meta}</p>"
                      f"<audio class='result' controls autoplay src='{src}'></audio>"
                      f"<p>{_btn('⬇ Download', src, 'secondary')}</p>")
        else:
            try:
                detail = str((r.json() or {}).get("detail") or "")[:600] or r.text[:600]
            except Exception:
                detail = r.text[:600]
            result = (f"<h2>Result</h2><p class='bad'>HTTP {r.status_code}</p>"
                      f"<pre class='err'>{_esc(detail)}</pre>")
    except httpx.HTTPError as e:
        result = f"<h2>Result</h2><p class='bad'>Error: {_esc(f'{type(e).__name__}: {e}')}</p>"
    return HTMLResponse(_page("Voice", _voiceplay_body(vals, result), "playground",
                              subnav=_subnav("playground", "voice")))


async def voice_audio(request: Request):
    """Serve the current user's last synthesis result (stash — no persistence)."""
    stash = _voice_out.get(_session_user(request) or "default")
    if not stash:
        raise HTTPException(404, "no synthesis result")
    data, mime = stash
    return Response(data, media_type=mime)


def _voice_status(kind: str, msg: str) -> str:
    cls = {"ok": "muted", "bad": "bad"}.get(kind, "muted")
    return f"<p class='{cls}'>{msg}</p>"


_voice_upload_prog: dict = {}   # user → {name, steps: [(kind, text)], done, ok} — live upload progress


def _vu_fragment(prog: dict) -> str:
    """Progress checklist for the result column: ✓/✗ per finished step,
    ⏳ for the one currently running (superseded 'run' markers are dropped)."""
    icons = {"ok": "✓", "err": "✗"}
    steps = prog.get("steps") or []
    rows = ""
    for i, (k, t) in enumerate(steps):
        if k == "run":
            if i == len(steps) - 1 and not prog.get("done"):
                rows += f"<p class='muted'>⏳ <b>{_esc(t)}…</b></p>"
            continue
        rows += f"<p class='{'bad' if k == 'err' else 'muted'}'>{icons.get(k, '·')} {_esc(t)}</p>"
    head = f"<h2>Result</h2><p class='muted'>⬆ upload <b>lib:{_esc(prog.get('name', ''))}</b></p>"
    if not prog.get("done"):
        return head + rows
    tail = ("<p>✓ <b>done — shipped to all hosts</b></p>" if prog.get("ok") else
            "<p class='bad'>finished with problems — see the steps above (↻ retries the ship)</p>")
    return head + rows + tail


async def voice_upload(request: Request):
    """Start the upload as a background task and hand over to the GET view, which
    shows LIVE progress in the result column (store → whisper transcription → scp
    per host) while the live morph keeps the library table in step with it."""
    f = await _multipart(request)
    name = str(f.get("name", "")).strip()
    data = f.get("file")
    ref_text = str(f.get("ref_text", "") or "")
    vals = {k: "" for k in _VOICEPLAY_KEYS}
    if not name or not isinstance(data, (bytes, bytearray)) or not data.strip() or _voice_lib_save is None:
        status = _voice_status("bad", "name and a WAV file are required"
                               if _voice_lib_save else "library unavailable (gateway not bound)")
        result = "<h2>Result</h2><p class='hint'>Synthesize to hear the result here.</p>"
        return HTMLResponse(_page("Voice", _voiceplay_body(vals, result, status), "playground",
                                  subnav=_subnav("playground", "voice")))
    user = _session_user(request) or "default"
    prog = {"name": name, "steps": [], "done": False, "ok": False}
    _voice_upload_prog[user] = prog
    blob = bytes(data)

    async def _run():
        try:
            res = await _voice_lib_save(name, blob, ref_text,
                                        lambda k, t: prog["steps"].append((k, t)))
            prog["ok"] = bool(res.get("shipped"))
        except Exception as ex:
            prog["steps"].append(("err", f"{type(ex).__name__}: {ex}"))
        finally:
            prog["done"] = True
            logger.info(f"ui: voice ref '{name}' uploaded (ok={prog['ok']})")

    asyncio.create_task(_run())
    # Redirect to the GET view like every other voice action: _LIVE_JS polls
    # location.href, and this POST-only URL would answer a GET with 405 — the page
    # would render the first checklist and then never move again.
    return RedirectResponse("/ui/playground?sub=voice", status_code=303)


async def voice_target(request: Request):
    f = await _form(request)
    store.set_settings({"voice_ref_hosts": (f.get("hosts", "") or "").strip(),
                        "voice_ref_dir": (f.get("dir", "") or "").strip(),
                        "whisper_model": (f.get("whisper_model", "") or "").strip() or "small"})
    return RedirectResponse("/ui/playground?sub=voice", status_code=303)


async def voice_ship(request: Request):
    name = (request.query_params.get("name", "") or "").strip()
    if name and _voice_lib_ship is not None:
        ok, msg = await _voice_lib_ship(name)
        logger.info(f"ui: voice ref '{name}' ship → {'ok' if ok else msg}")
    return RedirectResponse("/ui/playground?sub=voice", status_code=303)


async def voice_del(request: Request):
    name = (request.query_params.get("name", "") or "").strip()
    if name and _voice_lib_delete is not None:
        _voice_lib_delete(name)
        logger.info(f"ui: voice ref '{name}' deleted")
    return RedirectResponse("/ui/playground?sub=voice", status_code=303)


async def voice_lib_play(name: str):
    """Serve a library reference's gateway blob (listen before/after shipping)."""
    e = (store.get_voice_library() if store.is_active() else {}).get(name)
    p = (e or {}).get("file")
    if not p or not os.path.isfile(p):
        raise HTTPException(404, "voice not found")
    return FileResponse(p, media_type="audio/wav")


# ── Media Jobs tab (G1): inspect a generation's inputs + outputs within its TTL ──

_JOB_TICK = ("<script>function _fd(ms){ms=ms|0;if(ms<1000)return ms+' ms';var s=ms/1000;"
             "return s<60?s.toFixed(1)+' s':(s/60).toFixed(1)+' min';}"
             "function _td(){var n=Date.now()/1000;document.querySelectorAll('.jdur[data-since]')"
             ".forEach(function(e){e.textContent=_fd((n-parseFloat(e.getAttribute('data-since')))*1000);});}"
             "setInterval(_td,1000);_td();</script>")
_JOB_SCLS = {"done": "ok", "failed": "bad", "running": "warn", "queued": "warn"}


def _job_status_text(j: dict) -> str:
    """Status label with the multi-stage sub-step appended while running — a chain in
    stage 1 shows 'running 1/2', in stage 2 'running 2/2'. Single-stage jobs are plain.

    Plus the backend's own step counter when its websocket feed is reporting one:
    'running 25/35' is the answer to "is it stuck?", and it is the same number ComfyUI
    prints in its log. Absent for a backend that does not report — never faked."""
    st = j.get("status") or ""
    stg = j.get("stage")
    txt = f"{st} {stg}" if st == "running" and stg else st
    if st == "running":
        live = _job_progress(j.get("id") or "") or {}
        if live.get("step") is not None and live.get("steps"):
            txt += f" {live['step']}/{live['steps']}"
    return txt


_DUR_EST: dict = {}          # "alias|backend" → (cached_at, median seconds | None)
_DUR_EST_TTL = 60            # the median of the last 10 runs does not move by the second


def _expected_dur_s(alias: str, backend: str):
    """Expected runtime of a job on this alias, or None without history.

    Same basis as the job detail view's ETA (`jobs.median_duration`: median of the
    alias's last 10 done jobs, narrowed to this backend when it has its own history,
    falling back to the alias across all backends) — the two must not disagree.
    Memoised for `_DUR_EST_TTL`, because a job LIST renders one cell per running row
    and each one would otherwise be its own SQLite query on every live tick."""
    key = f"{alias}|{backend or ''}"
    hit = _DUR_EST.get(key)
    now = time.time()
    if hit and now - hit[0] < _DUR_EST_TTL:
        return hit[1]
    try:
        med = jobs.median_duration(alias, backend) or jobs.median_duration(alias)
    except Exception:                                  # a job store hiccup must not
        med = None                                     # take the whole page down
    _DUR_EST[key] = (now, med)
    return med


def _job_dur_cell(j: dict, now: int) -> str:
    """Duration cell: fixed for finished jobs, JS-ticking (`.jdur` + _JOB_TICK) while
    running, em-dash while queued — each of the two live states followed by the
    EXPECTED runtime ("12.0 s / ~40 s"), so the list answers "is this one stuck or just
    slow?" without opening the job. Shared by Media Jobs and the dashboard.

    The estimate is a SIBLING of `.jdur`, never inside it: _JOB_TICK overwrites that
    element's textContent once a second and would eat it."""
    st = j["status"]
    cr, upd = int(j.get("created") or 0), int(j.get("updated") or 0)
    if st in ("done", "failed") and upd >= cr:
        return f"<td class='muted'>{_dur((upd - cr) * 1000)}</td>"
    est = _expected_dur_s(j.get("alias") or "", j.get("backend") or "") \
        if st in ("running", "queued") else None
    exp = f" <span class='muted'>/ ~{_dur(est * 1000)}</span>" if est else ""
    if st == "running":
        # The backend's own step counter beats the median whenever we have it: it knows
        # THIS run's seconds per step, so "8.9 min left" is measured, not typical.
        live = _job_progress(j["id"]) or {}
        if live.get("eta_s") is not None:
            exp = (f" <span class='muted'>/ ~{_dur(live['eta_s'] * 1000)} left</span>")
        return (f"<td class='muted'><span class='jdur' data-since='{cr}'>"
                f"{_dur((now - cr) * 1000)}</span>{exp}</td>")
    if st == "queued" and est:
        return f"<td class='muted'>—<span class='muted'> / ~{_dur(est * 1000)}</span></td>"
    return "<td class='muted'>—</td>"


def _job_row(j: dict, now: int, *, task_col: bool = False, count_col: bool = False,
             actions: bool = False, time_col: bool = False) -> str:
    """One job table row ([time] / id-link / [task] / alias / backend / status /
    [imgs] / age / dur / owner / [actions]) — the ONE template behind Media Jobs
    and the dashboard (which stays compact: age only, no time column)."""
    st, jid = j["status"], j["id"]
    cells = []
    if time_col:
        cells.append(f"<td class='muted'>{_ts(j.get('created'))}</td>")
    cells.append(f"<td><a href='/ui/job/{_esc(jid)}'><code>{_esc(jid[:8])}</code></a></td>")
    if task_col:
        cells.append(f"<td>{_esc(j.get('task'))}</td>")
    cells += [f"<td>{_esc(j.get('alias'))}</td>", f"<td>{_esc(j.get('backend'))}</td>",
              f"<td><span class='badge {_JOB_SCLS.get(st, 'muted')}'>{_esc(_job_status_text(j))}</span></td>"]
    if count_col:
        cells.append(f"<td class='muted'>{j.get('result_count') or 0}</td>")
    cells += [f"<td class='muted'>{_age(j.get('created'))}</td>", _job_dur_cell(j, now),
              f"<td class='muted'>{_esc(j.get('owner'))}</td>"]
    if actions:
        acts = ((_btn('✕', f'/ui/job/{jid}/cancel', 'danger', sm=True, icon=True, confirm='Cancel this job?')
                 if st in ('queued', 'running') else '')
                + _btn('view', f'/ui/job/{jid}', 'secondary', sm=True))
        cells.append(f"<td style='text-align:right;white-space:nowrap'>{acts}</td>")
    # data-k keys this row for the live morph: the media job list is newest-first,
    # so a new job shifts every row — without a key the reconciler would match
    # positionally and rewrite every cell instead of inserting one row.
    return f"<tr data-k=\"job-{_esc(jid)}\">" + "".join(cells) + "</tr>"


def _call_kind(endpoint: str) -> str:
    """Which sub-tab a call-log row belongs to: 'voice' | 'media' | 'llm'.

    The stats table is ONE log, but the console shows it in three places, and every
    row must land in exactly one of them. Media endpoints used to have no partition
    of their own, so a refused image request surfaced under LLM Calls."""
    p = str(endpoint or "")
    if p.startswith("/v1/audio"):
        return "voice"
    if p.startswith("/v1/images") or p.startswith("/v1/generations") or p.startswith("/v1/jobs"):
        return "media"
    return "llm"


async def _refused_media_table(user, aliases) -> str:
    """Media requests the gateway turned away, for the Media Jobs sub-tab.

    A refusal on a media endpoint happens BEFORE a job row exists (no eligible
    backend, quota, a malformed request), so the job list cannot show it — and
    without a home here it would either vanish or, as before, be filed under LLM
    Calls. Once a job exists the call log stays out of it: the job owns the outcome
    (see run_generation), which is why a failed generation is NOT in this table."""
    if not stats.is_active():
        return ""
    s = await asyncio.to_thread(stats.summary, recent_limit=300, user=user)
    rows = [r for r in s["recent"] if _call_kind(r[7]) == "media"]
    if not rows:
        return ""
    return (f"<h2 style='margin-top:26px'>Refused media requests "
            f"<span class='muted' style='font-weight:normal'>· last {len(rows)}</span></h2>"
            "<p class='hint'>Requests turned away before a job existed — they have no job row "
            "above. A generation that ran and <b>failed</b> is a job, not a refusal.</p>"
            + _recent_calls_table(rows, aliases, src="media"))


async def _jobs_media_body(request: Request) -> tuple[str, Optional[int]]:
    """(body, refresh) — generation jobs (image/video/audio), newest first; excludes
    parked-chat / background-response rows, followed by the media requests that were
    refused before they became a job. Same user picker + row-filter input as the call
    lists (`?user=` filters by job owner)."""
    user = (request.query_params.get("user") or "").strip() or None
    aliases = store.get_ip_aliases()
    refused = await _refused_media_table(user, aliases)
    if not jobs.is_active():
        return ("<h2>Media Jobs</h2><p class='hint'>Job store is off — set <code>image_models</code> "
                "or <code>jobs.enabled: true</code> in config.</p>" + refused + _FILTER_JS, None)
    rows = jobs.recent(200, media_only=True, owner=user)
    if not rows and not user:
        return ("<h2>Media Jobs</h2><p class='hint'>No generation jobs yet. Run one in the "
                "<a href='/ui/playground?sub=media'>Media Playground</a>.</p>"
                + refused + _FILTER_JS, None)
    scope, bar = _user_filter_bar("/ui/jobs?sub=media", user,
                                  [(o,) for o in jobs.owners()], aliases)
    now = int(time.time())
    tr = "".join(_job_row(j, now, task_col=True, count_col=True, actions=True, time_col=True)
                 for j in rows)
    tbl = ((f"<table class='filterable'><tr><th>time</th><th>id</th><th>task</th><th>alias</th>"
            f"<th>backend</th><th>status</th><th>imgs</th><th>age</th><th>dur</th><th>owner</th>"
            f"<th></th></tr>{tr}</table>") if rows
           else "<p class='muted'>no media jobs for this user</p>")
    refresh = 5 if any(j["status"] in ("running", "queued") for j in rows) else None
    head = (f"<h2>Media Jobs{scope} <span class='muted' style='font-weight:normal'>"
            f"· last {len(rows)}</span></h2>{bar}")
    return (f"{head}{tbl}{refused}{_JOB_TICK}{_FILTER_JS}", refresh)


async def _calls_view_body(request: Request, kind: str) -> str:
    """Per-call history body for ONE partition (see _call_kind) — LLM Calls or Voice
    Calls. Same table/filter machinery, split by endpoint. Media rows live in the
    Media Jobs sub-tab next to the jobs they belong with."""
    title = "Voice Calls" if kind == "voice" else "LLM Calls"
    if not stats.is_active():
        return (f"<h2>{title}</h2><p class='hint'>Call recording is off. Enable <b>stats</b> in the "
                "<a href='/ui/server'>Server</a> tab (needs a restart) to log per-call history here.</p>")
    user = (request.query_params.get("user") or "").strip() or None
    s = await asyncio.to_thread(stats.summary, recent_limit=300, user=user)
    rows = [r for r in s["recent"] if _call_kind(r[7]) == kind]
    aliases = store.get_ip_aliases()
    scope, bar = _user_filter_bar(f"/ui/jobs?sub={kind}", user, s["by_source"], aliases)
    head = (f"<h2>{title}{scope} <span class='muted' style='font-weight:normal'>· last {len(rows)}</span></h2>"
            f"{bar}")
    return head + _recent_calls_table(rows, aliases, src=kind) + _FILTER_JS


async def jobs_page(request: Request):
    """Parent tab Jobs & Calls: sub-tabs Media Jobs | LLM Calls | Voice Calls
    (?sub=, first child = default)."""
    sub = request.query_params.get("sub") or SUBTABS["jobs"][0][0]
    refresh = None
    if sub == "llm":
        title, body = "LLM Calls", await _calls_view_body(request, "llm")
    elif sub == "voice":
        title, body = "Voice Calls", await _calls_view_body(request, "voice")
    else:
        sub = "media"
        title, (body, refresh) = "Media Jobs", await _jobs_media_body(request)
    return HTMLResponse(_page(title, body, "jobs", refresh=refresh, subnav=_subnav("jobs", sub)))


# three.js FBX preview: the generic (UniRig) result is an FBX whose texture is only
# a dead temp-path reference, so the sibling basecolor PNG is applied as the material
# map. Import map + module init emitted ONCE per page, HOISTED by the page that can
# ever show an FBX (job_detail_page) rather than appended next to the viewer div:
# _LIVE_JS strips every <script> out of a subtree it morphs in, so a block that only
# arrives WITH the finished artifact never executes (see the invariant in _page).
# All bundled locally under /ui/static/three (no CDN).
# The scan is a named global and a post-morph hook, because hoisting alone only gets
# the code onto the page: the module body runs once at load, when a still-running job
# has no `.fbxview` at all. Each morph that brings one in re-runs gwFbxScan, which
# picks up exactly the ones not yet initialised.
_FBX_VIEWER_JS = (
    '<script type="importmap">{"imports":{"three":"/ui/static/three/three.module.min.js"}}</script>'
    '<script type="module">'
    "import * as THREE from 'three';"
    "import { FBXLoader } from '/ui/static/three/jsm/loaders/FBXLoader.js';"
    "import { OrbitControls } from '/ui/static/three/jsm/controls/OrbitControls.js';"
    "window.gwFbxScan = function(){"
    "document.querySelectorAll('.fbxview:not([data-init])').forEach(function(el){"
    " el.dataset.init='1';"
    " var W=el.clientWidth||480,H=el.clientHeight||420;"
    " var sc=new THREE.Scene(); sc.background=new THREE.Color(0x0c0e12);"
    " var cam=new THREE.PerspectiveCamera(45,W/H,0.1,1e5);"
    " var rn=new THREE.WebGLRenderer({antialias:true}); rn.setPixelRatio(devicePixelRatio); rn.setSize(W,H);"
    " el.appendChild(rn.domElement);"
    " sc.add(new THREE.HemisphereLight(0xffffff,0x333344,2.2));"
    " var dl=new THREE.DirectionalLight(0xffffff,1.4); dl.position.set(2,3,2); sc.add(dl);"
    " var ct=new OrbitControls(cam,rn.domElement); ct.enableDamping=true;"
    " var tex=null;"
    " if(el.dataset.tex){tex=new THREE.TextureLoader().load(el.dataset.tex); tex.colorSpace=THREE.SRGBColorSpace; tex.flipY=false;}"
    " new FBXLoader().load(el.dataset.src,function(o){"
    "  o.traverse(function(c){ if(c.isMesh&&tex){ c.material=new THREE.MeshStandardMaterial({map:tex,roughness:0.85,metalness:0.0,side:THREE.DoubleSide}); }});"
    "  var b=new THREE.Box3().setFromObject(o),s=b.getSize(new THREE.Vector3()),ce=b.getCenter(new THREE.Vector3());"
    "  var md=Math.max(s.x,s.y,s.z)||1; o.position.sub(ce);"
    "  cam.position.set(0,md*0.25,md*1.9); cam.near=md/200; cam.far=md*200; cam.updateProjectionMatrix();"
    "  ct.target.set(0,0,0); ct.update(); sc.add(o);"
    " },undefined,function(e){ el.innerHTML='<p style=\"padding:14px;color:#e89\">FBX-Vorschau fehlgeschlagen</p>'; });"
    " (function loop(){ requestAnimationFrame(loop); ct.update(); rn.render(sc,cam); })();"
    "});"
    "};"
    "window.gwFbxScan();"
    "window.gwLiveHooks = window.gwLiveHooks || [];"
    "window.gwLiveHooks.push(window.gwFbxScan);"
    '</script>')


def _job_thumbs(jid: str, kind: str, entries: list) -> str:
    """Gallery of artifact thumbnails (kind = 'input'|'result'). Images link to the
    full file; video/audio play inline; GLB → <model-viewer>; FBX → a three.js 3D
    viewer textured with the sibling basecolor PNG; other files → a download card.

    Emits no import map and no viewer INIT of its own: the three.js module and its
    import map (_FBX_VIEWER_JS) and the model-viewer module are hoisted by the calling
    page, because this gallery is exactly the markup a live morph inserts mid-session
    and the morph drops every <script> it would insert. A GLB cell still carries
    _media_tag's own <script type="module" src=...> for model-viewer — byte-identical
    to the hoisted one and turned into a comment by adopt(), so it is inert either
    way."""
    base = f"/ui/job/{_esc(jid)}/input/" if kind == "input" else f"/ui/playground/result/{_esc(jid)}/"
    style = "max-width:260px;max-height:260px;border:1px solid #313a46;border-radius:8px"
    box3d = "width:720px;max-width:100%;height:640px"
    # a basecolor/texture PNG to feed the FBX viewer (prefer one named *basecolor*)
    tex_url = None
    for r in entries:
        if (r.get("mime") or "").startswith("image/"):
            u = f"{base}{r['n']}"
            if "basecolor" in (r.get("name") or "").lower():
                tex_url = u
                break
            tex_url = tex_url or u
    cells = ""
    for r in entries:
        src = f"{base}{r['n']}"
        m, mk = (r.get("mime") or "").lower(), (r.get("kind") or "").lower()
        name = r.get("name") or ""
        dl = f' download="{_esc(name)}"' if name else " download"
        label = name or f"artifact {r['n']}"
        if mk in ("video", "audio") or m.startswith("video/") or m.startswith("audio/"):
            cells += f"<div>{_media_tag(src, r.get('mime'), r.get('kind'), style=style)}</div>"
        elif m in ("model/gltf-binary", "model/gltf+json"):   # GLB → <model-viewer> + download
            stats = ""
            if kind == "result" and m == "model/gltf-binary":
                rp = jobs.result_path(jid, r["n"])
                stats = _glb_stats_html(rp[0] if rp else None)
            cells += (f"<div>{_media_tag(src, r.get('mime'), 'file', style=box3d)}{stats}"
                      f"<div>{_dl_card(src, label, dl=dl, compact=True)}</div></div>")
        elif name.lower().endswith(".fbx"):               # FBX → three.js viewer + download
            tex_attr = f' data-tex="{_esc(tex_url)}"' if tex_url else ""
            # data-live-skip: the server renders this div EMPTY and gwFbxScan fills it
            # client-side (data-init + a three.js canvas). A morph would strip both back
            # out — the attribute is not in the server's markup and the canvas is not in
            # its children — and the viewer would go black with nothing in any log.
            cells += (f'<div><div class="fbxview" data-live-skip data-src="{_esc(src)}"{tex_attr} '
                      f'style="{box3d};background:#0c0e12;border:1px solid #313a46;border-radius:10px"></div>'
                      f"<div>{_dl_card(src, label, dl=dl, compact=True)}</div></div>")
        elif mk == "file":                                # other file artifacts → download card
            cells += _dl_card(src, label, dl=dl, mime=(r.get("mime") or "file"))
        else:
            cells += (f"<a href='{src}' target='_blank'><img src='{src}' style='{style}'></a>")
    return f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:8px 0'>{cells}</div>"


def _cloud_table(title: str, m: dict) -> str:
    """One cloud run (Meshy, Tripo) in the job view: the body actually sent (image data
    replaced by its size) plus the task id — the id is what the vendor's own dashboard is
    searched by, and the body answers "which options did this run use?" without
    re-deriving them from today's config. '' when `m` is not a cloud run (no task id), so
    a chain renders a table per cloud STAGE: the top-level meta is the last stage's, a
    stage-1 cloud run is kept beside it under `meta.chain_stage1`.

    `title` names the stage generically ("Cloud · stage 1"); the VENDOR is substituted
    from the run's own meta, because the two stages of a chain may be different vendors.
    `meshy_task_id` is read as well: rows written before the kind existed carry only that
    key and no `cloud`, and a job view that suddenly renders nothing for them would look
    like the run was never recorded."""
    tid = m.get("cloud_task_id") or m.get("meshy_task_id")
    if not tid or not m.get("request"):
        return ""
    kind = m.get("cloud") or "meshy"
    vendor = adapters.cloud_module(kind).VENDOR if kind in adapters.CLOUD_MODULES else str(kind)
    rows = "".join(f"<tr><td><code>{_esc(str(k))}</code></td>"
                   f"<td>{_esc(json.dumps(v) if isinstance(v, (list, dict)) else str(v))}</td></tr>"
                   for k, v in (m.get("request") or {}).items())
    cr = m.get("consumed_credits")
    # A vendor may run SEVERAL tasks for one generation (Tripo: rig-check, the main task,
    # every convert and animation clip). Each is billed and each has its own id in the
    # vendor's dashboard, so they are listed instead of hidden behind the last one.
    tasks = m.get("tasks") if isinstance(m.get("tasks"), list) else []
    trows = "".join(f"<tr><td>{_esc(str((t or {}).get('role') or ''))}</td>"
                    f"<td><code>{_esc(str((t or {}).get('task_id') or ''))}</code></td>"
                    f"<td>{_esc(str((t or {}).get('credits')))}</td></tr>" for t in tasks)
    sub = (f"<p class='hint' style='margin:8px 0 2px'>tasks</p>"
           f"<table><tr><th>role</th><th>task id</th><th>credits</th></tr>{trows}</table>"
           if trows else "")
    return (f"<h3>{_esc(title.replace('Cloud', vendor))} "
            f"<span class='muted' style='font-weight:normal'>· task "
            f"<code>{_esc(str(tid))}</code> · {_esc(str(m.get('endpoint') or ''))}"
            f"{' · ' + _esc(str(cr)) + ' credits' if cr is not None else ''}</span></h3>"
            f"<table>{rows}</table>{sub}")


def _stage2_section(s2: dict) -> str:
    """The chain hand-off, for the job view: what stage 2 was actually HANDED and which
    of it the successor mapped. Stage-1 params are threaded to the successor by mapping
    LABEL, and `_apply_mapping` silently skips a name the successor does not bind — so a
    param that never arrives (a renamed label on either side) is invisible in the result
    and used to be answerable only from the backend's own ComfyUI history.

    `s2` is recorded at run time by `_run_chain` (meta.chain_stage2); only the node/field
    DETAIL is looked up in the alias config live, and marked as such — the mapping may
    have changed since the run. `applied` is absent when stage 2 never reported back
    (a failed hand-off); the rows then say what the CURRENT config would bind, and say so.

    A **cloud** successor (Meshy, Tripo) has no mapping at all — its request fields are a
    fixed label table (the kind's public_fields) and it reports no `applied` set. Its rows
    therefore say "handed", never "dropped": claiming a loss nobody measured would send
    you hunting a mapping that does not exist."""
    alias2, b2 = s2.get("alias") or "?", s2.get("backend") or "?"
    params = s2.get("params") or {}
    mesh_param = s2.get("mesh_param") or ""
    applied = s2.get("applied")               # None → stage 2 never got that far
    mapping2, cloud2 = {}, None
    if store.is_active():
        cs = store.get(alias2) or []
        c2 = next((x for x in cs if x.get("backend") == b2), cs[0] if cs else None) or {}
        mapping2 = c2.get("mapping") or {}
        cloud2 = adapters.cloud_kind(c2)
    vendor2 = adapters.cloud_module(cloud2).VENDOR if cloud2 else ""

    def target(k):                            # (node, field) the successor binds `k` to, per CURRENT config
        for p, m in mapping2.items():
            m = m or {}
            if p == k or ((m.get("label") or "").strip() == k):
                return str(m.get("node")), str(m.get("field") or "")
        return None, None

    rows, n_app, n_drop = [], 0, 0
    for k, v in params.items():
        if k == mesh_param:                   # the mesh itself gets its own line below
            continue
        node, field = target(k)
        if cloud2:                            # no mapping to look up — and none to miss
            hit = True
            tag = (f"<span class='muted' title='{_esc(vendor2)} binds a fixed label table; a name "
                   f"it does not know is ignored'>handed · {_esc(vendor2)} fixed table</span>")
        elif applied is None:                 # unverified: describe the binding, don't claim it ran
            hit = node is not None
            tag = (f"<span class='muted' title='from the successor alias config as it "
                   f"stands now'>→ node {_esc(node)}.{_esc(field)}</span>" if hit else
                   "<span class='muted'>not mapped by successor</span>")
        else:
            hit = k in applied
            # node None while applied says otherwise = the mapping moved since the run;
            # still say "applied", never leave the cell blank.
            tag = ((f"<span class='muted' title='from the successor alias config as it "
                    f"stands now'>→ node {_esc(node)}.{_esc(field)}</span>" if node else
                    "<span class='muted'>applied</span>")
                   if hit else "<span class='muted'>dropped · not mapped by successor</span>")
        n_app, n_drop = (n_app + 1, n_drop) if hit else (n_app, n_drop + 1)
        s = "" if hit else " style='text-decoration:line-through;opacity:.55'"
        rows.append(f"<tr><td{s}><code>{_esc(k)}</code></td><td{s}>{_esc(str(v))}</td>"
                    f"<td>{tag}</td></tr>")
    verb = "applied" if applied is not None else "mapped"
    tally = (f"{len(rows)} param(s) handed · {n_app} {verb} · {n_drop} dropped"
             if rows else "no params handed besides the mesh")
    warn = ("" if applied is not None else
            " <span class='bad'>· stage 2 did not report back — bindings shown are from "
            "the current alias config</span>")
    if cloud2:                                # nothing to tally against: no mapping, no `applied`
        tally = (f"{len(rows)} param(s) handed · {vendor2} takes its fixed label table and "
                 f"ignores the rest" if rows else "no params handed besides the mesh")
        warn = "" if applied is not None else " <span class='bad'>· stage 2 did not report back</span>"
    mesh = (f"<p style='margin:6px 0'><code>{_esc(mesh_param)}</code> "
            f"<span class='muted'>← the relayed mesh</span><br>"
            f"<code style='font-size:12px'>{_esc(str(s2.get('mesh_ref') or ''))}</code></p>")
    tbl = f"<table>{''.join(rows)}</table>" if rows else ""
    return (f"<h3>Successor <span class='muted' style='font-weight:normal'>· stage 2 · "
            f"{_esc(alias2)} · {_esc(b2)} · {_esc(s2.get('relay') or 'path')} hand-off</span></h3>"
            f"<p class='hint' style='margin:2px 0 8px'>{tally}{warn}</p>{mesh}{tbl}")


async def job_detail_page(job_id: str, request: Request):
    """Input (prompt/params/reference images) + output artifacts of one job."""
    if not jobs.is_active():
        return _inactive()
    job = jobs.get(job_id)
    back = _btn("← Back to Media Jobs", "/ui/jobs?sub=media", "secondary")
    if job is None:
        return HTMLResponse(_page("Job", f"<div class='bar'><h2>Job</h2>{back}</div>"
            f"<p class='bad'>job {_esc(job_id)} not found (or pruned past its TTL).</p>", "jobs"), status_code=404)
    st = job["status"]
    meta = job.get("meta") or {}
    inp = meta.get("inputs") or {}
    prompt, neg = inp.get("prompt") or "", inp.get("negative_prompt") or ""
    params = inp.get("params") or {}
    # candidate for the backend this job ran on → mapping (param→node), pinned (fixed),
    # bypass set and workflow. Params + pinned that target a BYPASSED node are struck
    # (they had no effect — the node was removed); a bypassed node with no param/pin is
    # listed separately. Bypass comes from the alias config (matches the pinned source).
    cand = {}
    if store.is_active():
        cs = store.get(job["alias"]) or []
        cand = next((x for x in cs if x.get("backend") == job["backend"]), cs[0] if cs else None) or {}
    mapping = cand.get("mapping") or {}
    pinned = cand.get("fixed") or []
    wf = cand.get("workflow_json") or {}
    bypassed = {str(x) for x in (cand.get("bypass") or [])}
    covered = set()                          # bypassed nodes surfaced (struck) via a param/pin
    _STRIKE = " style='text-decoration:line-through;opacity:.55'"

    def _byp(nid):                           # (td-style, tag) if node is bypassed, marking it covered
        if nid and nid in bypassed:
            covered.add(nid)
            return _STRIKE, " <span class='muted'>· bypassed</span>"
        return "", ""

    def _param_node(key):                    # the node a params-table key maps to (param OR label)
        for p, m in mapping.items():
            if p == key or ((m or {}).get("label") or "").strip() == key:
                return str((m or {}).get("node"))
        return None

    prow = []
    for k, v in params.items():
        s, tag = _byp(_param_node(k))
        prow.append(f"<tr><td{s}><code>{_esc(k)}</code></td><td{s}>{_esc(str(v))}{tag}</td></tr>")
    prows = "".join(prow)
    frow = []
    for b in pinned:
        s, tag = _byp(str(b.get("node")))
        frow.append(f"<tr><td{s}><code>{_esc(b.get('field'))}</code></td>"
                    f"<td{s}>{_esc(str(b.get('value')))}{tag}</td></tr>")
    frows = "".join(frow)
    in_imgs = meta.get("input_images", [])
    inbox = ""
    if prompt:
        inbox += f"<h3>Prompt</h3><pre class='chatout'>{_esc(prompt)}</pre>"
    if neg:
        inbox += f"<h3>Negative</h3><pre class='chatout'>{_esc(neg)}</pre>"
    ptbl = f"<h3>Params</h3><table>{prows}</table>" if prows else ""
    ftbl = (f"<h3>Pinned values <span class='muted' style='font-weight:normal'>· {_esc(job['backend'])}</span></h3>"
            f"<table>{frows}</table>") if frows else ""
    if ptbl or ftbl:
        inbox += (f"<div style='display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start'>"
                  f"<div>{ptbl}</div><div>{ftbl}</div></div>")
    # bypassed nodes with no param/pin of their own → list them as skipped
    extra_byp = sorted((n for n in bypassed if n not in covered), key=lambda n: (len(n), n))
    if extra_byp:
        def _byp_li(nid):
            n = wf.get(nid) or {}
            t = (n.get("_meta") or {}).get("title", "")
            cls = n.get("class_type", "") or "(stale — not in workflow)"
            return f"<li><code>{_esc(nid)}</code> {_esc(cls)}" + (f" · {_esc(t)}" if t else "") + "</li>"
        inbox += ("<h3>Bypassed nodes <span class='muted' style='font-weight:normal'>· skipped, no param/pin</span></h3>"
                  f"<ul class='muted' style='margin:4px 0 0 18px'>{''.join(_byp_li(n) for n in extra_byp)}</ul>")
    if in_imgs:
        # the content hash identifies WHICH bytes this run actually processed — the
        # same guarantee `results[]` gives for the output side.
        ids = "".join(f"<li><code>{_esc(r.get('slot') or r['n'])}</code> · "
                      f"sha256 <code>{_esc((r.get('sha256') or '?')[:16])}</code>"
                      + (f" · {r['bytes'] / 1024:.0f} kB" if r.get("bytes") else "") + "</li>"
                      for r in in_imgs)
        inbox += (f"<h3>Reference images</h3>{_job_thumbs(job_id, 'input', in_imgs)}"
                  f"<ul class='muted' style='margin:4px 0 0 18px;font-size:12px'>{ids}</ul>")
    if not inbox:
        inbox = "<p class='muted'>No stored inputs (job predates this feature).</p>"
    # Chain hand-off: what stage 2 was handed. Recorded per run; a job from before that
    # says so rather than being reconstructed from today's config, which may have moved.
    if meta.get("chain_stage2"):
        inbox += _stage2_section(meta["chain_stage2"])
    elif meta.get("chain") or cand.get("successor"):
        inbox += ("<h3>Successor <span class='muted' style='font-weight:normal'>· stage 2</span></h3>"
                  "<p class='muted'>Hand-off params not recorded (job predates this feature).</p>")
    # Cloud runs (task id + the body actually sent + credits — see _cloud_table): a
    # chain's stage-1 cloud run FIRST, then the top-level meta (stage 2's on a chain),
    # so a cloud→cloud chain reads in the order it ran. Each table names its OWN vendor.
    inbox += _cloud_table("Cloud · stage 1", meta.get("chain_stage1") or {})
    inbox += _cloud_table("Cloud", meta)
    if st in ("queued", "running"):
        # A long render used to be a spinner and nothing else — the one question it
        # could not answer was whether anything was still happening. The backend's step
        # feed answers it: a bar that moves, the node that is counting, and an ETA from
        # the seconds per step measured in THIS run. No feed → the plain line as before.
        outbox = f"<p>⏳ <b>{_esc(_job_status_text(job))}</b> · this view auto-updates</p>"
        live = _job_progress(job_id) or {}
        if live.get("fraction") is not None:
            pct = min(100, int(live["fraction"] * 100))
            eta = (f" · ~{_dur(live['eta_s'] * 1000)} left"
                   if live.get("eta_s") is not None else "")
            node = f" · node {_esc(live['node'])}" if live.get("node") else ""
            nodes = (f" · {live['nodes_done']}/{live['nodes_total']} nodes"
                     if live.get("nodes_total") else "")
            outbox += (
                f"<div style='max-width:520px;margin:6px 0 2px'>"
                f"<div style='height:8px;background:#232a35;border-radius:4px;overflow:hidden'>"
                f"<div style='height:8px;width:{pct}%;background:#6cb0ef'></div></div>"
                f"<p class='muted' style='margin:4px 0 0'>step {live.get('step')}/"
                f"{live.get('steps')} ({pct}%){eta}{node}{nodes}</p></div>")
    elif st == "failed":
        outbox = f"<p class='bad'>✗ failed</p><pre class='err'>{_esc(job.get('error'))}</pre>"
    elif job.get("results"):
        outbox = _job_thumbs(job_id, "result", job["results"])
    else:
        outbox = "<p class='muted'>No artifacts.</p>"
    exp = " · <span class='bad'>expired</span>" if job.get("expired") else ""
    info = (f"<p class='muted'>task {_esc(job['task'])} · alias {_esc(job['alias'])} · backend "
            f"{_esc(job['backend'])} · owner {_esc(job['owner'])} · {_age(job['created'])}{exp}</p>")
    cancel_btn = (_btn("✕ Cancel", f"/ui/job/{_esc(job_id)}/cancel", "danger", confirm="Cancel this job?")
                  if st in ("queued", "running") else "")
    # Load this job's inputs (prompt/params + reference images) into the Media Playground.
    to_pg = (_btn("→ Send to Playground", f"/ui/job/{_esc(job_id)}/to-playground",
                  title="Copy this job's prompt, params and reference images into the Media Playground")
             if store.is_active() and job.get("task") != "response" else "")
    # prev/next in the Media Jobs list (newest first); hidden at the ends.
    newer, older = jobs.neighbors(job_id)
    nav = ((_btn("‹ Prev", f"/ui/job/{_esc(newer)}", "secondary", title="Newer job") if newer else "")
           + (_btn("Next ›", f"/ui/job/{_esc(older)}", "secondary", title="Older job") if older else ""))
    # Both 3D viewers are hoisted UNCONDITIONALLY, exactly as _playground_body hoists
    # model-viewer: this page is live while the job runs, and _LIVE_JS strips every
    # <script> out of the subtree it morphs in. A GLB or FBX that only appears when the
    # job finishes would otherwise arrive without its viewer — an un-upgraded custom
    # element or an uninitialised div, i.e. a permanently black box, and `data-live`
    # vanishes in that same response so the poller stops and it never self-heals.
    # Custom elements upgrade on their own once model-viewer is defined; the FBX side
    # needs the gwFbxScan hook on top (the module body runs when there is nothing to
    # scan yet). Both files are local static and browser-cached.
    # ORDER IS LOAD-BEARING: _FBX_VIEWER_JS carries the import map, and an import map
    # inserted after a module script's load has been triggered is REJECTED outright by
    # every engine without the "multiple import maps" support (Chrome < 133, older
    # Firefox/Safari) — the model-viewer <script type="module" src> starts loading the
    # moment it is parsed. The FBX module would then die on "Failed to resolve module
    # specifier 'three'", gwFbxScan would never exist and the viewer stays a black box
    # with nothing in any log. So the import map goes FIRST. Pinned by
    # test_admin_live.LiveScriptInvariant.test_import_map_precedes_any_module_script.
    page = (f'{_FBX_VIEWER_JS}<script type="module" src="{_MODELVIEWER_SRC}"></script>'
            f"<div class='bar'><h2>Job <code>{_esc(job_id[:12])}</code> "
            f"<span class='badge {_JOB_SCLS.get(st, 'muted')}'>{_esc(_job_status_text(job))}</span></h2>"
            f"<div style='display:flex;gap:8px'>{cancel_btn}{to_pg}{nav}{back}</div></div>{info}"
            f"<div style='display:flex;gap:28px;flex-wrap:wrap;align-items:flex-start'>"
            f"<div style='flex:1;min-width:320px'><h2>Input</h2>{inbox}</div>"
            f"<div style='flex:1;min-width:320px'><h2>Output</h2>{outbox}</div></div>"
            + (_JOB_TICK if st in ("queued", "running") else ""))
    return HTMLResponse(_page("Job", page, "jobs", refresh=(2 if st in ("queued", "running") else None)))


async def job_input(job_id: str, n: int):
    ip = jobs.input_path(job_id, n)
    if ip is None:
        raise HTTPException(404, "input not found")
    path, mime = ip
    return FileResponse(path, media_type=mime)


async def job_to_playground(job_id: str, request: Request):
    """Load a job's stored inputs into the Media Playground and jump there: scalar
    params ride the URL (`p__<param>`, so same-named fields land), reference images go
    into the per-user stash the Playground already reads — same store a real upload
    uses, so they show as '✓ kept'."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    alias = job.get("alias", "")
    meta = job.get("meta") or {}
    inp = meta.get("inputs") or {}
    # Stored params/slots are keyed by the EXTERNAL name (a mapping label, e.g.
    # "remove background"), because a client sends them under the schema's label.
    # The playground reads p__<param>/img__<param>, so translate external→param;
    # else nothing lands (this was the "Send to Playground does nothing" for mesh).
    # A cloud alias (Meshy, Tripo) has no mapping at all: its external names ARE its param
    # names, so the empty table below leaves the .get(k, k) lookups as the identity.
    mapping = ((store.get(alias) or [{}])[0]).get("mapping") or {}
    ext2param = {}
    for p, m in mapping.items():
        ext2param[p] = p
        lbl = ((m or {}).get("label") or "").strip()
        if lbl:
            ext2param[lbl] = p
    q = {"sub": "media", "model": alias}
    if inp.get("prompt"):
        q["p__prompt"] = inp["prompt"]
    if inp.get("negative_prompt"):
        q["p__negative_prompt"] = inp["negative_prompt"]
    for k, val in (inp.get("params") or {}).items():
        if val is not None and str(val) != "":
            q[f"p__{ext2param.get(k, k)}"] = str(val)
    user = _session_user(request) or "default"
    stash = _pg_images.setdefault(user, {})
    for r in meta.get("input_images", []):
        ip = jobs.input_path(job_id, r.get("n"))
        if ip:
            try:
                with open(ip[0], "rb") as fh:
                    slot = ext2param.get(r.get("slot"), r.get("slot"))
                    # (filename, bytes) — the stash's shape since mesh uploads joined it
                    stash[slot] = (r.get("filename") or f"{slot}.png", fh.read())
            except OSError:
                pass
    return RedirectResponse(f"/ui/playground?{urlencode(q)}", status_code=303)


async def job_cancel(job_id: str):
    if _cancel_generation:
        await _cancel_generation(job_id)
    return RedirectResponse(f"/ui/job/{job_id}", status_code=303)


# ── Tabs: stubs ─────────────────────────────────────────────────────────────────

def _cost(v) -> str:
    return f"${float(v or 0):.4f}"


def _ts(ts) -> str:
    return time.strftime("%m-%d %H:%M:%S", time.localtime(int(ts)))


def _age(ts) -> str:
    s = max(0, int(time.time()) - int(ts or 0))
    return f"{s}s" if s < 60 else (f"{s // 60}m" if s < 3600 else f"{s // 3600}h")


def _dur(ms) -> str:
    """Duration in a fitting unit: <1s → ms, <60s → seconds (1 dp), else minutes (1 dp)."""
    ms = int(ms or 0)
    if ms < 1000:
        return f"{ms} ms"
    s = ms / 1000
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


def _looks_like_ip(s: str) -> bool:
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        return False


def _src_name(src: str, aliases: dict) -> str:
    """Display name for a stats source: its IP alias if set, else the raw source."""
    return aliases.get(src) or src


async def _reverse_dns(ip: str) -> str:
    try:
        host = await asyncio.wait_for(asyncio.to_thread(socket.gethostbyaddr, ip), timeout=1.5)
        return host[0]
    except Exception:
        return ""


async def _autoresolve_ips(by_source: list) -> None:
    """Best-effort: for caller IPs seen in stats with no alias yet, reverse-DNS them
    and persist the hostname (or '' to mark 'attempted', so we don't retry forever).
    Takes the caller's already-fetched summary()['by_source'] rows (one query/render)."""
    aliases = store.get_ip_aliases()
    seen = [r[0] for r in by_source]
    todo = [s for s in seen if _looks_like_ip(s) and s not in aliases][:20]
    if not todo:
        return
    names = await asyncio.gather(*[_reverse_dns(ip) for ip in todo])
    for ip, name in zip(todo, names):
        aliases[ip] = name
    store.save_ip_aliases(aliases)


def _dash_cards(d: dict, bes: list) -> str:
    """Headline counters (in-flight / parked / active jobs / backends up / 24h)."""
    up = sum(1 for b in bes if b.get("healthy"))
    jc = d.get("jobs_counts", {})
    active_jobs = (jc.get("queued", 0) or 0) + (jc.get("running", 0) or 0)
    card = lambda num, lbl: f"<div class='card'><div class='cnum'>{num}</div><div class='clbl'>{_esc(lbl)}</div></div>"
    return ("<div class='cards'>"
            + card(d.get("llm_inflight", 0), "LLM in flight")
            + card(d.get("parked", 0), "calls parked")
            + card(active_jobs, "media jobs active")
            + card(f"{up}/{len(bes)}", "backends up")
            + (card(d.get("calls_24h", 0), "calls · 24h") if d.get("stats_active") else "")
            + "</div>")


def _dash_backends(bes: list, offline: list) -> str:
    """Per-backend live table, sorted ready → busy → off → disabled."""
    def srank(b):
        if not b.get("enabled"):
            return 3
        if not b.get("healthy"):
            return 2
        return 1 if b.get("busy") else 0

    def bstatus(b):
        if b.get("draining"):
            return _badge(f"⏳ draining · {b.get('inflight', 0)} in-flight", "warn")
        if not b.get("enabled"):
            return _badge("⏻ offline", "warn")
        if not b.get("healthy"):
            return _badge("off", "bad")
        return _badge("busy", "warn") if b.get("busy") else _badge("ready", "ok")

    brows = ""
    for b in sorted(bes, key=lambda x: (srank(x), x.get("name", "").lower())):
        cap = b.get("max_concurrent")
        inf = f"{b.get('inflight', 0)}" + (f" / {cap}" if cap else "")
        r1h = b.get("reqs_1h", 0)
        r1h_cell = f"{r1h}" if r1h else "<span class='muted'>0</span>"
        # data-k: _bid (type:name) is this row's identity, and the panel re-sorts
        # itself (ready → busy → off), so a backend changing state moves its row —
        # positional matching would rewrite the neighbours it stepped over. The bare
        # name is NOT unique: _bid exists because an LLM and a ComfyUI backend may
        # share one (main.rebuild_backends keys on (name, type)), and two rows with the
        # same key collapse into one pool entry — each tick would then rewrite one row
        # from the other's content, the exact per-tick full-row rewrite keys prevent.
        brows += (f"<tr data-k=\"bk-{_esc(_bid(b))}\"><td>{_esc(b['name'])}</td><td>{_type_badge(b.get('type'))}</td>"
                  f"<td>{bstatus(b)}</td><td>{inf}</td><td>{r1h_cell}</td>"
                  f"<td>{b.get('models', 0)}</td></tr>")
    off_hint = (f" · {len(offline)} offline hidden (<a href='/ui/backends'>manage</a>)" if offline else "")
    return (f"<h2>Backends <span class='muted' style='font-weight:normal;font-size:12px'>"
            f"· click a header to sort{off_hint}</span></h2>"
            f"<table class='sortable' data-sk='dash-backends'><tr><th>backend</th><th>type</th><th>status</th>"
            f"<th>in flight</th><th title='requests handled in the last hour'>req · 1h</th>"
            f"<th>models</th></tr>{brows}</table>")


def _dash_jobs(d: dict, now: int) -> str:
    """Media jobs panel: running/queued now + finished within the last 5 min."""
    if not d.get("jobs_active"):
        return "<h2>Media jobs</h2><p class='hint'>Media generation is off.</p>"
    order = (("running", "warn"), ("queued", "warn"), ("done", "ok"), ("failed", "bad"))
    recent_jobs = [j for j in d.get("jobs_recent", [])
                   if j.get("status") in ("running", "queued") or int(j.get("updated") or 0) > now - 300]
    # Badges count this same window — not lifetime totals — so they can't claim
    # "done 12" while the list shows nothing recent (lifetime is in the Media Jobs tab).
    wc = {k: sum(1 for j in recent_jobs if j.get("status") == k) for k, _ in order}
    badges = " ".join(_badge(f"{k} {wc[k]}", kind) for k, kind in order if wc[k])
    jr = "".join(_job_row(j, now, time_col=True) for j in recent_jobs)
    return (f"<h2>Media jobs <span class='muted' style='font-weight:normal'>· running + last 5 min</span> "
            f"{badges}</h2>"
            + (f"<table class='sortable' data-sk='dash-jobs'><tr><th>time</th><th>id</th><th>alias</th>"
               f"<th>backend</th><th>status</th><th>age</th><th>dur</th><th>owner</th></tr>{jr}</table>" if jr
               else "<p class='muted'>nothing running or recently finished</p>"))


def _dash_llm(d: dict, now: int) -> str:
    """Recent LLM calls: currently running (live registry) + finished within the last
    5 min (stats, via the shared _call_row template — same columns as LLM Calls)."""
    aliases = store.get_ip_aliases()
    lr = ""
    for c in d.get("llm_running", []):
        started = int(c.get("started") or 0)
        model, alias = c.get("model"), c.get("alias")
        am = (_esc(alias) or "") + (("→" + _esc(model)) if model else "")
        lr += (f"<tr><td class='muted'>now</td><td>{_esc(_src_name(c.get('source', ''), aliases))}</td>"
               f"<td>{_esc(c.get('backend'))}</td><td>{am}</td>"
               f"<td class='muted'>{_esc((c.get('endpoint') or '').replace('/v1/', ''))}</td>"
               f"<td><span class='badge warn'>running</span></td>"
               f"<td class='muted jdur' data-since='{started}'>{_dur((now - started) * 1000)}</td>"
               f"<td class='muted'>—</td><td class='muted'>—</td><td class='muted'>—</td>"
               f"<td class='muted'>—</td></tr>")
    lr += "".join(_call_row(r, aliases) for r in d.get("llm_recent", []))
    nrun = len(d.get("llm_running", []))
    runbadge = _badge(f"running {nrun}", "warn") if nrun else ""
    head = (f"<h2>Recent LLM calls <span class='muted' style='font-weight:normal'>· running + last 5 min</span> "
            f"{runbadge}</h2>")
    if lr:
        return head + _calls_table(lr, sk="dash-llm")
    if not d.get("stats_active") and not nrun:
        return (head + "<p class='hint'>Call recording is off — only currently-running calls show here. "
                "Enable <b>stats</b> in the <a href='/ui/server'>Server</a> tab for the 5-minute history "
                "and the full <a href='/ui/llmcalls'>LLM Calls</a> log.</p>")
    return head + "<p class='muted'>nothing running or in the last 5 min</p>"


def _dash_parked(d: dict) -> str:
    """Parked calls panel: chat requests queued because all their backends are busy,
    waiting for a free slot up to the alias's park time (empty when nothing parks)."""
    parked = d.get("parked_calls", [])
    if not parked:
        return ""
    prows = "".join(f"<tr><td>{_esc(p.get('alias'))}</td><td>{_esc(p.get('source'))}</td>"
                    f"<td class='muted'>{_dur(float(p.get('waited_s') or 0) * 1000)}</td>"
                    f"<td class='muted'>{_dur(float(p.get('remaining_s') or 0) * 1000)}</td></tr>"
                    for p in parked)
    return (f"<h2>Parked calls <span class='muted' style='font-weight:normal'>· queued, waiting for a "
            f"free backend</span> {_badge(f'parked {len(parked)}', 'warn')}</h2>"
            f"<table class='sortable' data-sk='dash-parked'><tr><th>alias</th><th>user</th>"
            f"<th>waited</th><th>park left</th></tr>{prows}</table>")


async def dashboard_page(request: Request):
    d = await asyncio.to_thread(_dashboard_snapshot)   # runs several stats/jobs queries off-loop
    # Backends taken offline (disabled) are intentionally out of rotation — hide them
    # from the live view (they stay manageable in the Backends tab). Draining ones stay
    # enabled until idle, so they remain visible while finishing in-flight work.
    bes_all = d.get("backends", [])
    offline = [b for b in bes_all if not b.get("enabled")]
    bes = [b for b in bes_all if b.get("enabled")]
    now = int(time.time())
    body = ("<h2>Dashboard <span class='muted' style='font-weight:normal'>· live · auto-refresh 4s</span></h2>"
            + _dash_cards(d, bes) + _dash_backends(bes, offline) + _dash_parked(d)
            + _dash_llm(d, now) + _dash_jobs(d, now) + _JOB_TICK)
    return HTMLResponse(_page("Dashboard", body, "dashboard", refresh=4))


def _reasoning_cell(rsn) -> str:
    """Table cell for the applied reasoning control (off:prefill / on:enable_thinking /
    unsupported / — for auto). Says what the GATEWAY did with the requested switch,
    not whether the model can think."""
    if not rsn:
        return "<td class='muted' title='no reasoning switch sent (auto) — request untouched'>—</td>"
    rsn = str(rsn)
    if rsn == "unsupported":
        return ("<td><span class='badge muted' title='off/on was requested, but NO reasoning rule "
                "covers this model×backend — the request was forwarded UNCHANGED. Not a statement "
                "about the model: add a rule in the Reasoning tab.'>unsupported</span></td>")
    if rsn.endswith("noop"):
        kind, tip = "muted", "on requested — the model thinks by default, nothing to change"
    elif rsn.startswith("off"):
        kind, tip = "warn", "thinking switched OFF via this mechanism"
    else:
        kind, tip = "ok", "thinking switched ON via this mechanism"
    return f"<td><span class='badge {kind}' title='{tip}'>{_esc(rsn)}</span></td>"


def _call_row(r, aliases) -> str:
    """One finished-call row (15-col stats tuple) — the ONE template behind the
    LLM Calls list and the dashboard's running+5min panel. The detail link's `src`
    is derived from the endpoint so a row carries its own way back regardless of
    which list rendered it."""
    cid, ts, dur, backend, source, alias, model, endpoint, status, intk, outk, cost, prev, has_body, rsn = r
    scls = "ok" if (status and 200 <= int(status) < 300) else "bad"
    if has_body:
        src = _call_kind(endpoint)
        view = f"<a href='/ui/call/{cid}?src={src}' title='{_esc(prev or '')}'>view</a>"
    elif prev:
        view = f"<span class='muted' title='{_esc(prev)}'>{_esc(prev[:30])}…</span>"
    else:
        view = "<span class='muted'>—</span>"
    # data-k keys this row for the live morph — the call lists are newest-first, so a
    # finished call shifts every row down and positional matching would rewrite the
    # whole table instead of inserting one row. Prefixed to keep the key space apart
    # from the job rows', in case the two ever share a parent node.
    return (f"<tr data-k=\"call-{_esc(cid)}\">"
            f"<td class='muted'>{_ts(ts)}</td><td>{_esc(_src_name(source, aliases))}</td><td>{_esc(backend)}</td>"
            f"<td>{_esc(alias) or ''}{('→' + _esc(model)) if model else ''}</td>"
            f"<td class='muted'>{_esc((endpoint or '').replace('/v1/', ''))}</td>"
            f"<td><span class='badge {scls}'>{_esc(status)}</span></td>"
            f"<td>{_dur(dur)}</td><td>{intk}/{outk}</td><td>{_cost(cost)}</td>{_reasoning_cell(rsn)}<td>{view}</td></tr>")


def _calls_table(rows_html: str, sk: str) -> str:
    """Header + shell around _call_row rows (one column set everywhere)."""
    return (f"<table class='recent filterable sortable' data-sk='{sk}'><tr><th>time</th><th>source</th>"
            f"<th>backend</th><th>alias→model</th><th>endpoint</th><th>status</th><th>dur</th><th>tok i/o</th>"
            f"<th>cost</th><th title='normalized reasoning control applied to this call'>reasoning</th>"
            f"<th>req</th></tr>{rows_html}</table>")


def _recent_calls_table(rows, aliases, src: str = "llm") -> str:
    """The per-call history table (time/source/backend/alias→model/…/reasoning/req body).
    `src` only names the sort-key so the three lists keep their own sort state."""
    rec = "".join(_call_row(r, aliases) for r in rows)
    if not rec:
        return "<p class='muted'>no calls yet</p>"
    return _calls_table(rec, sk=f"{src}-calls")


_BOX_STYLE = "padding:7px 10px;background:#0c0e12;border:1px solid #242a33;border-radius:8px;color:#cdd6e0"
# Row filter for `table.filterable`, driven by the ONE `#sf` input a view renders.
# The typed text is persisted in sessionStorage per view and re-applied on load — like
# the sort order above, and for the same reason: it must survive REAL navigation (a tab
# switch, a form POST, F5), which the live morph does not cover.
# Focus and caret are deliberately NOT saved any more. That was pure compensation for
# the old full-page auto-refresh, which yanked the page out from under someone typing;
# the morph never replaces a focused or dirty control (see _LIVE_JS), so the input the
# caret sits in is exactly the node it was, and the 15-second "was someone typing"
# window it needed had itself been made meaningless by the hook below re-saving on
# every tick. Every write is now just the text, so a tick rewriting it is a no-op.
# A gwLiveHooks entry re-runs the filter after every live morph, because rows the
# morph brings in fresh carry no display style and would otherwise ignore it.
# The hook is registered outside the `if(i)` guard and no-ops when the view has no
# #sf input, so it is harmless on a page that never renders one.
_FILTER_JS = ("<script>(function(){var K='flt:'+location.pathname+location.search;"
              "function el(){return document.getElementById('sf');}"
              "function save(){var i=el();if(!i)return;try{sessionStorage.setItem(K,"
              "JSON.stringify({v:i.value}));}catch(e){}}"
              "window.sfRun=function(){var i=el();if(!i)return;"
              "var q=(i.value||'').toLowerCase();"
              "document.querySelectorAll('.filterable tr').forEach(function(r){"
              "if(r.getElementsByTagName('th').length)return;"
              "r.style.display=r.textContent.toLowerCase().indexOf(q)>-1?'':'none';});save();};"
              "var i=el();if(i){var s=null;"
              "try{s=JSON.parse(sessionStorage.getItem(K)||'null');}catch(e){}"
              "if(s&&s.v){i.value=s.v;window.sfRun();}}"
              "window.gwLiveHooks=window.gwLiveHooks||[];"
              "window.gwLiveHooks.push(function(){if(window.sfRun)window.sfRun();});"
              "})();</script>")


def _user_filter_bar(path: str, user, by_source, aliases) -> tuple[str, str]:
    """(scope_suffix, bar_html) — the user picker + row-filter input shared by the
    LLM Calls and Statistic pages. Append _FILTER_JS to the page body once."""
    opts = "<option value=''>all users</option>" + "".join(
        f"<option value='{_esc(r[0])}'{' selected' if r[0] == user else ''}>{_esc(_src_name(r[0], aliases))}</option>"
        for r in by_source)
    sep = "&" if "?" in path else "?"                  # path may already carry ?sub=…
    picker = (f"<select style=\"width:auto;{_BOX_STYLE}\" onchange=\"location.href='{path}'+"
              f"(this.value?('{sep}user='+encodeURIComponent(this.value)):'')\">{opts}</select>")
    search = (f"<input id='sf' autocomplete='off' oninput='sfRun()' "
              f"placeholder='filter rows: backend / alias / model / user…' "
              f"style=\"flex:1;min-width:220px;max-width:420px;{_BOX_STYLE}\">")
    scope = (f" · <span class='muted' style='font-weight:normal'>user <b>{_esc(user)}</b> · "
             f"<a href='{path}'>clear</a></span>") if user else ""
    bar = f"<div style='display:flex;gap:10px;align-items:center;margin:6px 0 10px'>{picker}{search}</div>"
    return scope, bar


async def llmcalls_page(request: Request):
    """Legacy URL — LLM Calls now lives under /ui/jobs?sub=llm."""
    q = dict(request.query_params)
    q["sub"] = "llm"
    return RedirectResponse(f"/ui/jobs?{urlencode(q)}", status_code=307)


def _cache_cells(in_tok: int, read: int, write: int, series: Optional[list]) -> str:
    """The prompt-cache columns of a By-backend row: cached / written / fresh, plus
    a 24h hit-rate sparkline. A backend that reported no cache numbers at all shows
    dashes rather than a row of zeros — zeros would read as "cache missed
    everything", which is a different statement from "does not do caching"."""
    in_tok, read, write = int(in_tok or 0), int(read or 0), int(write or 0)
    if not read and not write:
        return "<td class='muted'>—</td><td class='muted'>—</td>" \
               f"<td>{in_tok or '—'}</td><td class='muted'>—</td>"
    fresh = max(0, in_tok - read - write)
    pct = f" <span class='muted'>{read * 100 // in_tok}%</span>" if in_tok else ""
    return (f"<td>{read}{pct}</td><td>{write}</td><td>{fresh}</td>"
            f"<td>{_cache_spark(series)}</td>")


def _cache_spark(series: Optional[list]) -> str:
    """Inline-SVG sparkline of the hit rate (cache_read / input) per bucket. Bars
    only where the bucket saw traffic, so a gap means "no calls", not "no hits".
    Inline SVG keeps the console dependency-free (no chart library, no JS)."""
    if not series:
        return "<span class='muted'>—</span>"
    n = len(series)
    w, h, gap = 4, 18, 1
    bars = []
    for i, (in_tok, read, _write) in enumerate(series):
        if not in_tok:
            continue
        rate = min(1.0, read / in_tok)
        bh = max(1, round(rate * h))
        bars.append(f"<rect x='{i * (w + gap)}' y='{h - bh}' width='{w}' height='{bh}' "
                    f"fill='#5cb87f' opacity='{0.35 + 0.65 * rate:.2f}'></rect>")
    if not bars:
        return "<span class='muted'>—</span>"
    total_in = sum(i for i, _r, _w in series)
    total_read = sum(r for _i, r, _w in series)
    title = f"{(total_read * 100 // total_in) if total_in else 0}% of input served from cache (24h)"
    width = n * (w + gap)
    # A baseline + a full-height frame: without them a single bar reads as a stray
    # mark instead of "one hour out of twenty-four".
    axis = (f"<rect x='0' y='0' width='{width}' height='{h}' fill='#1b2028'></rect>"
            f"<rect x='0' y='{h - 1}' width='{width}' height='1' fill='#2a313c'></rect>")
    return (f"<svg width='{width}' height='{h}' viewBox='0 0 {width} {h}' "
            f"role='img' aria-label='{_esc(title)}'><title>{_esc(title)}</title>"
            f"{axis}{''.join(bars)}</svg>")


def _media_gen_panel() -> str:
    """"Media generation" — the aggregate for media jobs, and the ONLY place the media
    routing decision is legible.

    A ComfyUI/cloud generation is a job, not a forwarded call, so it never reaches
    `stats.calls` and every table on this page was blind to it — media looked entirely
    unmeasured. It is measured, and by exactly the number in the `routing` column: the
    scheduler orders candidates unpaid-first, then fastest, on that per alias+backend
    EMA. An unmeasured pair sorts FIRST once (probe-once), which is why a brand-new
    backend gets one job before the ordering settles."""
    rows = jobs.gen_stats_rows()
    if not rows:
        return ""
    info = _gen_speed_info() or {}
    speed, quar = info.get("speed") or {}, info.get("quarantine") or {}
    out = []
    for alias, backend, done, failed, avg_ms, last_ts in rows:
        key = f"{alias}|{backend}"
        secs = speed.get(key)
        routing = (f"{_dur(secs * 1000)}" if secs
                   else "<span class='muted'>unmeasured — tried first</span>")
        held = quar.get(key)
        note = (f"<b>⚠ quarantined, {max(1, held // 60)} min left</b>" if held
                else (f"<span class='muted'>{failed} failed</span>" if failed else ""))
        total = done + failed
        out.append((alias, secs if secs else 0.0,
                    f"<tr><td>{_esc(alias)}</td><td>{_esc(backend)}</td>"
                    f"<td>{done}</td><td>{failed}</td>"
                    f"<td>{round(100.0 * done / total) if total else 0}%</td>"
                    f"<td>{_dur(avg_ms) if avg_ms else '—'}</td>"
                    f"<td>{routing}</td><td class='muted'>{_age(last_ts)}</td>"
                    f"<td>{note}</td></tr>"))
    out.sort(key=lambda r: (r[0].lower(), r[1]))
    tr = "".join(r[2] for r in out)
    return (f"<h2>Media generation</h2>"
            f"<p class='hint'>Media jobs never reach the call log (a generation is a job, "
            f"not a forwarded call) — this is their aggregate. <b>routing</b> is what the "
            f"scheduler orders on: unpaid backends first, then the fastest measured "
            f"runtime for that alias+backend. A pair with no measurement yet is tried "
            f"first once, so a new backend gets one probe job. <b>avg</b> counts finished "
            f"jobs only — averaging a three-second failure in would make a broken backend "
            f"look like the fastest one.</p>"
            f"<table class='filterable sortable' data-sk='stat-media'>"
            f"<tr><th>alias</th><th>backend</th><th>done</th><th>failed</th><th>ok</th>"
            f"<th>avg</th><th title='the EMA the scheduler routes on'>routing</th>"
            f"<th>last</th><th></th></tr>{tr}</table>")


async def statistic_page(request: Request):
    if not stats.is_active():
        # Media aggregates live in the JOB store, not in stats.calls — they are there to
        # show even when call recording is off, and this is the page they belong on.
        media = await asyncio.to_thread(_media_gen_panel)
        return HTMLResponse(_page("Statistic", "<h2>Statistic</h2><p class='hint'>Call recording is off. "
            "Enable <b>stats</b> in the <a href='/ui/server'>Server</a> tab (needs a restart) to collect "
            "per-call stats here.</p>" + media + (_FILTER_JS if media else ""), "statistic"))
    user = (request.query_params.get("user") or "").strip() or None
    s = await asyncio.to_thread(stats.summary, user=user)
    aliases = store.get_ip_aliases()
    scope, bar = _user_filter_bar("/ui/statistic", user, s["by_source"], aliases)
    # Refused calls are counted in the totals but never in the tables below (they had no
    # backend and no model) — this card is where they stay visible, and it explains the
    # gap between "calls total" and the sum of the By-backend column. Deliberately NOT
    # a link: _call_kind() files each row under LLM Calls, Media Jobs or Voice Calls by
    # endpoint, so no single list holds them all (measured on prod: a whole day of 401s
    # on /v1/models next to media 503s). The tooltip names the three places instead.
    ref_tip = (f"{s['refused_count']} in total. Turned away before any backend saw them "
               f"(no healthy backend, park timeout, quota, unknown alias, bad key) — "
               f"counted in the totals above, but not in the tables below. The rows sit "
               f"under LLM Calls, Media Jobs or Voice Calls, by endpoint.")
    cards = (f"<div class='cards'>"
             f"<div class='card'><div class='cnum'>{s['total_count']}</div><div class='clbl'>calls total</div></div>"
             f"<div class='card'><div class='cnum'>{_cost(s['total_cost'])}</div><div class='clbl'>cost total</div></div>"
             f"<div class='card'><div class='cnum'>{s['h24_count']}</div><div class='clbl'>calls · 24h</div></div>"
             f"<div class='card'><div class='cnum'>{_cost(s['h24_cost'])}</div><div class='clbl'>cost · 24h</div></div>"
             f"<div class='card' title='{_esc(ref_tip)}'>"
             f"<div class='cnum'>{s['refused_24h']}</div>"
             f"<div class='clbl'>refused · 24h</div></div>"
             f"</div>")
    trend = await asyncio.to_thread(stats.cache_trend, user=user)
    be = "".join(f"<tr><td>{_esc(r[0])}</td><td>{r[1]}</td><td>{r[2]}</td>"
                 + _cache_cells(r[2], r[6] if len(r) > 6 else 0, r[7] if len(r) > 7 else 0,
                                trend.get(r[0]))
                 + f"<td>{r[3]}</td><td>{_cost(r[4])}</td><td>{_dur(r[5])}</td></tr>"
                 for r in s["by_backend"])
    by_backend = (f"<h2>By backend</h2>"
                  f"<p class='hint'>Prompt cache: <b>cached</b> = input served from the backend's "
                  f"cache (a fraction of the fresh price), <b>written</b> = input stored into it "
                  f"(a surcharge, paid once), <b>fresh</b> = the rest, billed in full. The trend is "
                  f"the hit rate over the last 24h — a session whose cache stops being hit starts "
                  f"paying full price for its whole context again.</p>"
                  f"<table class='filterable sortable' data-sk='stat-backend'>"
                  f"<tr><th>backend</th><th>calls</th><th>in tok</th>"
                  f"<th title='input served out of the prompt cache'>cached</th>"
                  f"<th title='input written into the prompt cache'>written</th>"
                  f"<th title='input processed fresh — neither read nor written'>fresh</th>"
                  f"<th title='cache hit rate per hour, last 24h'>24h trend</th>"
                  f"<th>out tok</th><th>cost</th><th>avg</th></tr>{be}</table>" if be
                  # "no calls yet" would be a lie when every call was refused — that is
                  # exactly the state this table can no longer show by itself.
                  else "<h2>By backend</h2><p class='muted'>" + (
                      f"no forwarded calls — all {s['refused_count']} were refused"
                      if s["refused_count"] else "no calls yet") + "</p>")
    mo = "".join(f"<tr><td>{_esc(r[0]) or '—'}</td><td><code>{_esc(r[1])}</code></td><td>{r[2]}</td>"
                 f"<td>{r[3]}</td><td>{r[4]}</td><td>{_cost(r[5])}</td></tr>" for r in s["by_model"])
    by_model = (f"<h2>By alias / model</h2><table class='filterable sortable' data-sk='stat-model'>"
                f"<tr><th>alias</th><th>model</th><th>calls</th>"
                f"<th>in</th><th>out</th><th>cost</th></tr>{mo}</table>" if mo else "")
    so = "".join(f"<tr><td>{_esc(_src_name(r[0], aliases))}</td><td>{r[1]}</td><td>{_cost(r[2])}</td></tr>" for r in s["by_source"])
    by_source = (f"<h2>By user / source</h2><table class='filterable sortable' data-sk='stat-source'>"
                 f"<tr><th>source</th><th>calls</th><th>cost</th></tr>"
                 f"{so}</table>" if so else "")
    # Per-call history lives in its own tab now (the LLM Calls list) — keep Statistic
    # to the aggregates. Point there so the link is discoverable.
    recent = ("<p class='hint' style='margin-top:18px'>Per-call history (with request/response bodies) "
              "moved to the <a href='/ui/llmcalls'>LLM Calls</a> tab.</p>")
    media = await asyncio.to_thread(_media_gen_panel)
    head = f"<h2>Statistic{scope}</h2>{bar}"
    body = head + cards + by_backend + by_model + by_source + media + recent + _FILTER_JS
    return HTMLResponse(_page("Statistic", body, "statistic"))


async def call_view(call_id: int, request: Request):
    """Full stored request + response body for one call (E3). Binary audio
    responses (/v1/audio/speech) render as an inline player instead of JSON."""
    body = stats.get_body(call_id)
    if body is None:
        inner = "<p class='muted'>No stored body for this call (predates the feature, or pruned).</p>"
    else:
        req = json.dumps(body.get("request"), indent=2, ensure_ascii=False)
        respobj = body.get("response")
        if isinstance(respobj, dict) and respobj.get("_audio"):
            resp_html = (f"<p class='muted'>binary audio · {int(respobj.get('bytes', 0)) // 1024} KB · "
                         f"{_esc(respobj['_audio'])}</p>"
                         f"<audio class='result' controls src='/ui/call/{call_id}/audio'></audio>"
                         f"<p>{_btn('⬇ Download', f'/ui/call/{call_id}/audio', 'secondary')}</p>")
        else:
            resp_html = f"<pre class='chatout'>{_esc(json.dumps(respobj, indent=2, ensure_ascii=False))}</pre>"
        inner = (f"<h3>Request</h3><pre class='chatout'>{_esc(req)}</pre>"
                 f"<h3>Response</h3>{resp_html}")
    src = request.query_params.get("src") or "llm"
    if src not in ("voice", "media"):
        src = "llm"
    back = _btn({"voice": "← Back to Voice Calls", "media": "← Back to Media Jobs"}
                .get(src, "← Back to LLM Calls"), f"/ui/jobs?sub={src}", "secondary")
    # prev/next within the same list partition (newest first: prev = newer row
    # above, next = older row below); hidden at the list ends. `call_neighbors`
    # only partitions voice vs. rest, so media rows — a handful of refusals, listed
    # under their jobs — get the Back button and no stepping.
    nav = ""
    if src != "media":
        newer, older = stats.call_neighbors(call_id, src == "voice")
        nav = ((_btn("‹ Prev", f"/ui/call/{newer}?src={src}", "secondary", title="Newer call") if newer else "")
               + (_btn("Next ›", f"/ui/call/{older}?src={src}", "secondary", title="Older call") if older else ""))
    page = (f"<div class='bar'><h2>Call #{call_id}</h2>"
            f"<div style='display:flex;gap:8px'>{nav}{back}</div></div>{inner}")
    return HTMLResponse(_page("Call", page, "jobs"))


async def call_audio(call_id: int):
    """Serve a call's stored binary audio response (within the stats retention)."""
    hit = stats.get_audio(call_id)
    if hit is None:
        raise HTTPException(404, "no audio stored for this call")
    path, mime = hit
    return FileResponse(path, media_type=mime)


# ── Reasoning tab: normalized thinking toggle (per-model × per-backend rules) ────

_REASON_ADAPTERS = ["enable_thinking", "reasoning_effort", "nothink_token", "prefill", "none"]
_REASON_HINT = {
    "enable_thinking": "Adds <code>chat_template_kwargs: {enable_thinking:false}</code> to the request. Works "
                       "<b>only if the backend forwards chat_template_kwargs</b> to the model template — vLLM: yes · "
                       "llama.cpp: only with <code>--jinja</code> · <b>LocalAI: usually NOT</b> (then use "
                       "<b>nothink_token</b> or <b>prefill</b> instead). <b>param:</b> leave empty.",
    "reasoning_effort": "Sets the OpenAI <code>reasoning_effort</code> field: off → the param value (default "
                        "<code>minimal</code>), on → <code>high</code>. For gpt-oss / harmony models. "
                        "<b>param:</b> the off value, e.g. <code>minimal</code>.",
    "nothink_token": "Appends a soft-switch token to the last <b>user</b> message — message-level, so it always "
                     "reaches the model. <b>Qwen3.x: <code>/no_think</code></b> · GLM: <code>/nothink</code>. "
                     "<b>param:</b> the token (default <code>/nothink</code> — set <code>/no_think</code> for Qwen).",
    "prefill": "Appends an assistant turn with a closed, empty think block so the model skips reasoning. "
               "Message-level, works regardless of backend, but some backends reject a trailing assistant message. "
               "<b>param:</b> the block content (default <code>&lt;think&gt;\\n\\n&lt;/think&gt;</code>).",
    "none": "Do nothing — reasoning off/on is reported as <code>unsupported</code> (for models that can't toggle "
            "thinking). <b>param:</b> unused.",
}
_REASON_PKEY = {"reasoning_effort": "off", "nothink_token": "token", "prefill": "content"}


def _reason_pval(rule: dict) -> str:
    k = _REASON_PKEY.get(rule.get("adapter", ""))
    return str((rule.get("param") or {}).get(k, "")) if k else ""


_REASON_TEST_PROMPT = "Say hello in one short sentence."

# Probe verdict → badge kind. ok = mechanism works; bad = no-op / still thinks;
# warn = suppressed but answer empty; error = backend rejected / unreachable;
# info = model doesn't think by default (nothing to suppress).
_VERDICT_KIND = {"ok": "ok", "warn": "warn", "bad": "bad", "error": "bad", "info": "muted"}


def _reason_verdict(res: dict) -> tuple:
    """(kind, headline) summarising a probe result for the tested adapter. Only 'ok'
    (baseline provably thinks → adapter suppressed it, answer intact) offers auto-save."""
    if res.get("error"):
        return ("error", res["error"])
    req = res.get("requested", "off")
    base, cand = res["baseline"], res["candidate"]
    if cand["status"] != 200:
        return ("error", f"backend returned HTTP {cand['status']} with this adapter"
                + (" — some backends reject a trailing assistant turn (prefill)"
                   if res.get("adapter") == "prefill" else ""))
    if req == "off":
        thinks_by_default = base["status"] == 200 and base["reasoning_len"] > 0
        if cand["reasoning_len"] > 0:
            return ("bad", "no-op — the model still emitted reasoning")
        if cand["content_len"] == 0:                       # suppressed but broke the answer
            return ("warn", "no reasoning, but this adapter left the answer empty")
        if thinks_by_default:                              # proven: thought → suppressed → answer intact
            return ("ok", "reasoning suppressed, answer intact — this adapter works")
        return ("info", "model didn't reason by default here — adapter is harmless but unproven")
    # requested == "on"
    if cand["reasoning_len"] > 0:
        return ("ok", "reasoning present (on)")
    return ("warn", "no reasoning produced with 'on'")


def _reason_test_result(res: dict) -> str:
    """Render a probe result: verdict badge, baseline vs. adapter numbers, preview,
    and — on a positive verdict — an inline 'Save this rule' submit (auto-create)."""
    kind, headline = _reason_verdict(res)
    badge = _badge({"ok": "✓ works", "warn": "⚠ partial", "bad": "✗ no-op",
                    "error": "✗ error", "info": "· n/a"}[kind], _VERDICT_KIND[kind])
    if res.get("error"):
        return f"<h2 style='margin-top:18px'>Test result</h2><p>{badge} <b>{_esc(headline)}</b></p>"
    base, cand = res["baseline"], res["candidate"]
    tbl = ("<table style='margin:8px 0'><tr><th></th><th>HTTP</th><th>reasoning</th><th>answer</th></tr>"
           f"<tr><td class='muted'>baseline (no adapter)</td><td>{base['status']}</td>"
           f"<td>{base['reasoning_len']} chars</td><td>{base['content_len']} chars</td></tr>"
           f"<tr><td><b>with {_esc(res['adapter'])}</b> "
           f"<code>{_esc(res.get('control') or '—')}</code></td><td>{cand['status']}</td>"
           f"<td><b>{cand['reasoning_len']}</b> chars</td><td>{cand['content_len']} chars</td></tr></table>")
    prev = _esc(cand.get("content_preview") or "")
    prev_html = f"<p class='muted' style='margin:6px 0'>answer preview: <code>{prev or '(empty)'}</code></p>"
    save = ("" if kind not in ("ok",) else
            "<p style='margin-top:10px'>" + _btn("✔ Save this rule", submit=True) +
            " <span class='muted'>— persists the rule exactly as tested (match · backends · "
            "adapter · param) and applies it live.</span></p>")
    return (f"<h2 style='margin-top:18px'>Test result</h2><p>{badge} <b>{_esc(headline)}</b> "
            f"<span class='muted'>· {_esc(res['model'])} on {_esc(res['backend'])} · requested "
            f"{_esc(res['requested'])}</span></p>{tbl}{prev_html}{save}")


def _reasoning_form(rule: Optional[dict], idx, test: Optional[dict] = None,
                    tvals: Optional[dict] = None) -> str:
    r = rule or {}
    tv = tvals or {}
    hidden = f'<input type="hidden" name="idx" value="{idx}">' if idx is not None else ""
    sel = set(r.get("backends") or [])
    allck = " checked" if ("*" in sel or not sel) else ""
    bk = (f'<label class="ckbox"><input type="checkbox" name="bk_all" value="1"{allck} onclick="rAll(this)"> '
          f'<b>all backends</b></label><div style="margin:6px 0">')
    for n in _llm_backend_names():
        ck = " checked" if (n in sel and "*" not in sel) else ""
        bk += (f'<label class="ckbox" style="margin:0 12px 4px 0"><input type="checkbox" name="bk" '
               f'value="{_esc(n)}" class="rbk"{ck}> {_esc(n)}</label>')
    bk += "</div>"
    ad = r.get("adapter", "enable_thinking")
    aopts = "".join(f'<option{" selected" if a == ad else ""}>{a}</option>' for a in _REASON_ADAPTERS)
    asel = f'<select name="adapter" onchange="rAd(this.value)">{aopts}</select>'
    hints = "".join(f'<div class="hint rh" data-a="{a}" style="display:{"block" if a == ad else "none"};'
                    f'margin:-4px 0 10px">{_REASON_HINT[a]}</div>' for a in _REASON_ADAPTERS)
    js = ("<script>function rAll(c){document.querySelectorAll('.rbk').forEach(function(x){"
          "x.disabled=c.checked; if(c.checked) x.checked=false;});}"
          "function rAd(a){document.querySelectorAll('.rh').forEach(function(e){"
          "e.style.display=e.getAttribute('data-a')===a?'block':'none';});}"
          "function rTest(f){var r=document.getElementById('rtresult');"
          "if(r)r.innerHTML=\"<h2 style='margin-top:18px'>Test result</h2><p class='muted'>\\u23f3 "
          "<b>Testing\\u2026</b> · calling the backend twice (baseline + adapter)</p>\";return true;}"
          "rAll(document.querySelector('input[name=bk_all]'));</script>")
    # Live test panel — shares the form, so 'Run test' and 'Save' submit the same
    # rule fields (match · backends · adapter · param); the backend is called directly
    # with the chosen adapter applied, so a candidate mechanism is validated before save.
    tb_pre = (tv.get("test_backend") or next((n for n in (r.get("backends") or []) if n != "*"), ""))
    bopts = '<option value="">backend…</option>' + "".join(
        f"<option{' selected' if n == tb_pre else ''}>{_esc(n)}</option>" for n in _llm_backend_names())
    ropts = "".join(f'<option value="{x}"{" selected" if (tv.get("test_requested") or "off") == x else ""}>{x}</option>'
                    for x in ("off", "on"))
    # Picking a backend filters the model datalist to THAT backend's models (like the
    # Chat Playground) — a rule is per (model×backend), so the model list should follow.
    rt_bk = {b["name"]: sorted(b.get("models", [])) for b in _llm_backends()}
    rt_all = _chat_models()
    rt_js = ("<script>var RT_BK=%s,RT_ALL=%s;function rtFilter(bk){"
             "var dl=document.getElementById('rtmodels');if(!dl)return;"
             "var ms=(bk&&RT_BK[bk])?RT_BK[bk]:RT_ALL;"
             "dl.innerHTML=ms.map(function(m){var o=document.createElement('option');"
             "o.value=m;return o.outerHTML;}).join('');}</script>") % (json.dumps(rt_bk), json.dumps(rt_all))
    test_panel = (
        "<div style='margin-top:16px;padding-top:12px;border-top:1px solid #272b33'>"
        "<h2>Live test</h2><p class='hint'>Fire a real call to one backend with the "
        "<b>adapter above</b> applied, and see if the model's reasoning is actually "
        "suppressed. A green result → <b>Save this rule</b>.</p>"
        + _field("test on backend",
                 f"<select name='test_backend' onchange='rtFilter(this.value)'>{bopts}</select>", short=True)
        + _field("test model / alias",
                 _dl_input("test_model", tv.get("test_model", ""), "rtmodels", "alias or real model id"), short=True)
        + _field("requested", f"<select name='test_requested'>{ropts}</select>", short=True)
        + _field("prompt", _inp("test_prompt", tv.get("test_prompt", "") or _REASON_TEST_PROMPT))
        + '<button type="submit" class="btn secondary" formaction="/ui/reasoning/test" '
          'formnovalidate onclick="return rTest(this.form)">▶ Run test</button>'
        + f"<div id='rtresult'>{_reason_test_result(test) if test else ''}</div>"
        + "</div>" + _datalist("rtmodels", rt_bk.get(tb_pre) or rt_all))
    return ('<form action="/ui/reasoning/save" method="post">' + hidden +
            f'<div class="formbar"><h2>{"Edit rule" if rule else "New rule"}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/reasoning", "secondary")}</div>'
            + _field("enabled", _checkbox("enabled", r.get("enabled", True), "rule is active"))
            + _field("order", _inp("order", str(r.get("order", "")), placeholder="1", typ="number"), short=True)
            + _field("match (model glob)", _inp("match", r.get("match", ""),
                     placeholder="qwen3.5*  ·  gemma-4-12b-it  ·  *"), short=True)
            + _field("backends", bk)
            + _field("adapter (off)", asel) + hints
            + _field("param", _inp("param", _reason_pval(r), placeholder="optional — see the adapter hint above"))
            + test_panel
            + "</form>" + js + rt_js)


def _reasoning_testbox(rules: list, qp) -> str:
    tm = (qp.get("test_model") or "").strip()
    tb = (qp.get("test_backend") or "").strip()
    res = ""
    if tm and tb:
        # Accept an ALIAS like the API does: resolve it to this backend's real model
        # first (rules always match against real model ids, never alias names).
        real = _resolve_for_backend(tm, tb)
        if real is None:
            res = (f"<p style='margin-top:8px'><code>{_esc(tm)}</code> on <b>{_esc(tb)}</b> → "
                   + _badge("not mapped", "bad")
                   + " <span class='muted'>(this alias has no entry for this backend — "
                     "a call would never route there)</span></p>")
        else:
            via = f" → <code>{_esc(real)}</code>" if real != tm else ""
            rule = reasoning.resolve(rules, tb, real)
            res = (f"<p style='margin-top:8px'><code>{_esc(tm)}</code>{via} on <b>{_esc(tb)}</b> → "
                   + (f"{_badge(rule.get('adapter'), 'ok')} <span class='muted'>(rule #{rules.index(rule) + 1})</span>"
                      if rule else _badge("unsupported", "muted") + " <span class='muted'>(no matching rule)</span>")
                   + "</p>")
    bopts = "".join(f"<option{' selected' if n == tb else ''}>{_esc(n)}</option>" for n in _llm_backend_names())
    return ("<h2 style='margin-top:22px'>Test resolution</h2>"
            "<p class='hint'>See which rule would apply for a (model, backend) — without a real call. "
            "Takes an <b>alias or a real model id</b>; aliases resolve to the backend's model first, "
            "exactly like the API.</p>"
            "<form method='get' action='/ui/reasoning' style='display:flex;gap:8px;align-items:center;flex-wrap:wrap'>"
            f"{_inp('test_model', tm, placeholder='alias or model id, e.g. tool / qwen3.5-9b')}"
            f"<select name='test_backend'><option value=''>backend…</option>{bopts}</select>"
            f"{_btn('Resolve', submit=True, kind='secondary')}</form>{res}")


def _reasoning_list_html(rules: list, edit_i: Optional[int]) -> str:
    rows = ""
    for i, r in enumerate(rules):
        bks = r.get("backends") or ["*"]
        bstr = "all" if "*" in bks else ", ".join(bks)
        on = r.get("enabled", True)
        state = _badge("on", "ok") if on else _badge("off", "muted")
        toggle = (("⏸", f"/ui/reasoning/toggle?idx={i}", "secondary", "Disable this rule") if on
                  else ("▶", f"/ui/reasoning/toggle?idx={i}", "secondary", "Enable this rule"))
        acts = _icon_acts(toggle, ("✎", f"/ui/reasoning?edit={i}", "secondary", "Edit"),
                          ("✕", f"/ui/reasoning/delete?idx={i}", "danger", "Delete", "Delete this rule?"))
        cls = "sel" if edit_i == i else ""
        if not on:
            cls = (cls + " muted").strip()
        cls = f" class='{cls}'" if cls else ""
        rows += (f"<tr{cls}><td>{_esc(r.get('order', ''))}</td><td><code>{_esc(r.get('match', '*'))}</code></td>"
                 f"<td>{_esc(bstr)}</td><td>{_esc(r.get('adapter', 'none'))}</td>"
                 f"<td class='muted'>{_esc(_reason_pval(r))}</td><td>{state}</td><td class='acts'>{acts}</td></tr>")
    rows = rows or ("<tr><td colspan=7 class='muted'>no rules — reasoning off/on is reported "
                    "'unsupported' for every model</td></tr>")
    return (f'<div class="bar"><h2>Reasoning</h2>{_btn("+ New rule", "/ui/reasoning?new=1")}</div>'
            "<p class='hint'>Clients send <code>reasoning: \"off\" | \"on\" | \"auto\"</code> (default auto → "
            "unchanged). The first <b>enabled</b> rule whose <b>model glob</b> matches the real model <b>and</b> "
            "whose <b>backend set</b> contains the serving backend is applied; no match → "
            "<code>unsupported</code> (never fails). Applied control shows in <a href='/ui/llmcalls'>LLM Calls</a> "
            "+ the <code>x-reasoning-control</code> header.</p>"
            f"<table class='sortable' data-sk='reason'><tr><th>#</th><th>match</th><th>backends</th>"
            f"<th>adapter</th><th>param</th><th>state</th><th></th></tr>{rows}</table>")


def _reasoning_shell(rules: list, detail: str, edit_i: Optional[int]) -> HTMLResponse:
    body = (f'<div class="cols"><div class="col">{_reasoning_list_html(rules, edit_i)}</div>'
            f'<div class="col">{detail}</div></div>')
    return HTMLResponse(_page("Reasoning", body, "reasoning"))


def _rule_from_form(qs: dict) -> tuple:
    """Build a rule dict + edit-index from a submitted reasoning form (shared by save
    and the live test). `qs` is the _form_multi mapping ({k:[v,…]})."""
    g = lambda k, d="": (qs.get(k) or [d])[-1]
    adapter = (g("adapter", "none").strip() or "none")
    if adapter not in _REASON_ADAPTERS:
        adapter = "none"
    o = g("order").strip()
    backends = ["*"] if qs.get("bk_all") else ([b for b in qs.get("bk", []) if b] or ["*"])
    param = {}
    pv = g("param").strip()
    key = _REASON_PKEY.get(adapter)
    if key and pv:
        param[key] = pv
    rule = {"order": (int(o) if o.lstrip("-").isdigit() else None),
            "match": (g("match").strip() or "*"), "backends": backends,
            "adapter": adapter, "param": param, "enabled": bool(qs.get("enabled"))}
    idx = g("idx")
    return rule, (int(idx) if idx.isdigit() else None)


async def reasoning_page(request: Request):
    rules = store.get_reasoning_rules() if store.is_active() else []
    qp = request.query_params
    edit_i = int(qp["edit"]) if (qp.get("edit", "").isdigit() and int(qp["edit"]) < len(rules)) else None
    if qp.get("new"):
        detail = _reasoning_form(None, None)
    elif edit_i is not None:
        detail = _reasoning_form(rules[edit_i], edit_i)
    else:
        detail = ("<h2>Details</h2><p class='hint'>Select a rule's <b>✎</b>, or <b>+ New rule</b> to add one.</p>"
                  + _reasoning_testbox(rules, qp))
    return _reasoning_shell(rules, detail, edit_i)


async def reasoning_test(request: Request):
    """Live-test the (unsaved) rule in the form against one backend+model, then
    re-render the form with the verdict + an inline Save (auto-create on success)."""
    qs = await _form_multi(request)
    g = lambda k, d="": (qs.get(k) or [d])[-1]
    rule, idx = _rule_from_form(qs)
    if rule.get("order") is None:
        rule["order"] = ""
    rules = store.get_reasoning_rules() if store.is_active() else []
    tvals = {"test_backend": g("test_backend").strip(), "test_model": g("test_model").strip(),
             "test_prompt": g("test_prompt"), "test_requested": (g("test_requested", "off").strip() or "off")}
    tb, tm = tvals["test_backend"], tvals["test_model"]
    if not tb or not tm:
        res = {"error": "pick a test backend and a model/alias first"}
    elif _probe_reasoning is None:
        res = {"error": "probe unavailable (gateway not bound)"}
    else:
        real = _resolve_for_backend(tm, tb)      # alias → this backend's real model, like the API
        if real is None:
            res = {"error": f"'{tm}' is not mapped on backend '{tb}' — a call would never route there"}
        else:
            res = await _probe_reasoning(tb, real, rule["adapter"], rule["param"],
                                         tvals["test_requested"], tvals["test_prompt"])
    detail = _reasoning_form(rule, idx, test=res, tvals=tvals)
    return _reasoning_shell(rules, detail, idx)


async def reasoning_save(request: Request):
    qs = await _form_multi(request)                     # backends arrive as a checkbox list
    g = lambda k, d="": (qs.get(k) or [d])[-1]
    rules = store.get_reasoning_rules() if store.is_active() else []
    adapter = (g("adapter", "none").strip() or "none")
    if adapter not in _REASON_ADAPTERS:
        adapter = "none"
    o = g("order").strip()
    order = int(o) if o.lstrip("-").isdigit() else (len(rules) + 1)
    backends = ["*"] if qs.get("bk_all") else ([b for b in qs.get("bk", []) if b] or ["*"])
    param = {}
    pv = g("param").strip()
    key = _REASON_PKEY.get(adapter)
    if key and pv:
        param[key] = pv
    rule = {"order": order, "match": (g("match").strip() or "*"),
            "backends": backends, "adapter": adapter, "param": param,
            "enabled": bool(qs.get("enabled"))}
    idx = g("idx")
    if idx.isdigit() and int(idx) < len(rules):
        rules[int(idx)] = rule
    else:
        rules.append(rule)
    rules.sort(key=lambda r: r.get("order") if isinstance(r.get("order"), int) else 999)
    store.set_reasoning_rules(rules)
    _apply_reasoning()
    logger.info(f"ui: reasoning rule saved — {rule['match']} @ {backends} → {adapter} "
                f"({'on' if rule['enabled'] else 'off'})")
    return RedirectResponse("/ui/reasoning", status_code=303)


async def reasoning_toggle(request: Request):
    idx = request.query_params.get("idx") or ""
    rules = store.get_reasoning_rules() if store.is_active() else []
    if idx.isdigit() and int(idx) < len(rules):
        rules[int(idx)]["enabled"] = not rules[int(idx)].get("enabled", True)
        store.set_reasoning_rules(rules)
        _apply_reasoning()
    return RedirectResponse("/ui/reasoning", status_code=303)


async def reasoning_del(request: Request):
    idx = request.query_params.get("idx") or ""
    rules = store.get_reasoning_rules() if store.is_active() else []
    if idx.isdigit() and int(idx) < len(rules):
        rules.pop(int(idx))
        store.set_reasoning_rules(rules)
        _apply_reasoning()
    return RedirectResponse("/ui/reasoning", status_code=303)


def _show_user_keys() -> bool:
    """Whether the user editor pre-fills an existing user's API key so it can be
    copied again. Default ON; turn it off in Server → flags to keep the key out of
    the rendered page and show only keys generated right there in the form."""
    return bool(store.get_setting("show_user_keys", True)) if store.is_active() else True


def _user_form(u: Optional[dict]) -> str:
    g = lambda k, d="": str((u or {}).get(k) if (u or {}).get(k) is not None else d)
    has_key = bool((u or {}).get("api_key"))
    # A stored key is only prefilled for an EXISTING user (a new one has none) and
    # only with the setting on. Masked as a password field; the Copy button reveals
    # it, exactly as it does for a freshly generated key.
    show_key = has_key and _show_user_keys()
    key_val = (u or {}).get("api_key", "") if show_key else ""
    orig = f'<input type="hidden" name="orig" value="{_esc((u or {}).get("name", ""))}">' if u else ""
    # model allow-list as a table — ALIASES only (chat + image generation aliases),
    # since access is granted at the alias level; raw model ids would be noise.
    # Empty selection = all allowed.
    allowed = set((u or {}).get("models") or [])
    chat_al = sorted(set(_gateway_info().get("virtual_models", [])))
    img_al = sorted(store.list_aliases().keys()) if store.is_active() else []
    # Backend grants apply to LLM backends only (generation backends aren't in
    # /v1/models; image access is granted via image aliases). Filtering them out also
    # removes the confusing duplicate when an LLM and a media backend share a name.
    bk_al = sorted({b["name"] for b in _gateway_info().get("backends", [])
                    if b.get("name") and b.get("type", "openai") not in adapters.GEN_TYPES})
    # chat/image aliases granted by name; a backend grants ALL of its models (and filters
    # what this user's key sees in /v1/models). Each kind gets a "select all" header row.
    rows = ""
    for kind, items in (("chat", chat_al), ("image", img_al), ("backend", bk_al)):
        if not items:
            continue
        all_ck = " checked" if all(a in allowed for a in items) else ""
        rows += (f'<tr style="background:#13161c"><td><input type="checkbox"{all_ck} '
                 f'onclick="gwTogAll(this,\'{kind}\')" title="select all {kind}"></td>'
                 f'<td colspan="2"><b>all {kind}</b> <span class="muted">({len(items)})</span></td></tr>')
        for a in items:
            ck = " checked" if a in allowed else ""
            rows += (f'<tr><td><input type="checkbox" name="model" value="{_esc(a)}" data-grp="{kind}"{ck}></td>'
                     f'<td><code>{_esc(a)}</code></td><td class="muted">{kind}</td></tr>')
    acc = ((f'<div class="acctbl"><table><thead><tr><th title="allow this entry">✓</th>'
            f'<th>name</th><th>kind</th></tr></thead><tbody>{rows}</tbody></table></div>'
            "<script>function gwTogAll(c,g){var s='input[name=model][data-grp=\"'+g+'\"]';"
            "document.querySelectorAll(s).forEach(function(x){x.checked=c.checked;});}</script>") if rows
           else "<p class='muted'>no aliases or backends yet</p>")
    return (f'<form action="/ui/users/save" method="post">{orig}'
            f'<div class="formbar"><h2>{"Edit User" if u else "Add User"}</h2>'
            f'{_btn("Save", submit=True)}{_btn("Cancel", "/ui/users", "secondary")}</div>'
            + _field("name", _inp("name", g("name"), placeholder="alice"))
            + _field("API key",
                     _inp("api_key", key_val, typ=("password" if show_key else "text"),
                          placeholder=("" if show_key else
                                       ("•••• set — blank keeps it" if has_key
                                        else "the user's bearer token")))
                     + ' <button type="button" class="btn secondary sm" onclick="gwGenKey(this)" '
                       'title="generate a random key">🔑 Generate</button>'
                     + ' <button type="button" class="btn secondary sm" onclick="gwCopyKey(this)" '
                       'title="copy to clipboard">📋 Copy</button>'
                     + ("<p class='hint' style='margin:4px 0 0'>This user's key is filled in and hidden — "
                        "<b>📋 Copy</b> reveals and copies it. Overwrite the field to change the key. "
                        "Turn off <code>show_user_keys</code> in <a href='/ui/server'>Server</a> to keep "
                        "stored keys out of this page.</p>" if show_key else
                        "<p class='hint' style='margin:4px 0 0'>The key is shown once here — copy it now; "
                        "after Save it is stored encrypted and no longer displayed.</p>")
                     + "<script>function _gwKeyInp(b){return b.closest('.control').querySelector('input[name=api_key]');}"
                       "function gwGenKey(b){var a=new Uint8Array(24);crypto.getRandomValues(a);"
                       "var k='sk-'+Array.from(a).map(function(x){return ('0'+x.toString(16)).slice(-2);}).join('');"
                       "var i=_gwKeyInp(b);i.type='text';i.value=k;i.focus();i.select();}"
                       "function gwCopyKey(b){var i=_gwKeyInp(b);i.type='text';i.focus();i.select();"
                       "var d=function(){b.textContent='✓ Copied';setTimeout(function(){b.textContent='📋 Copy';},1200);};"
                       "if(navigator.clipboard&&navigator.clipboard.writeText){"
                       "navigator.clipboard.writeText(i.value).then(d,function(){document.execCommand('copy');d();});}"
                       "else{document.execCommand('copy');d();}}</script>")
            + _field("role", _select("role", ["user", "admin"], g("role", "user")))
            + _field("enabled", _checkbox("enabled", (u or {}).get("enabled", True), "enabled"))
            + _field("quota req/day", _inp("quota_req_day", g("quota_req_day"),
                                           placeholder="blank = unlimited", typ="number"))
            + _field("quota cost/month ($)", _inp("quota_cost_month", g("quota_cost_month"),
                                                  placeholder="blank = unlimited, e.g. 5.00"))
            + "<h2>Model access</h2>"
            + "<p class='hint'>Leave <b>all unchecked</b> to allow every model (default). Checking rows "
              "restricts this user to them <b>and</b> filters what their key sees in <code>/v1/models</code>. "
              "A <b>backend</b> grants all of its models; an <b>image</b> alias is what an image client "
              "(e.g. anima-verse) should be limited to.</p>"
            + acc
            + "</form>")


async def users_page(request: Request):
    qp = request.query_params
    edit = qp.get("edit", "")
    open_auth = (not store.list_users()) and not _server_info().get("effective", {}).get("api_key_set")
    items = ""
    for u in store.list_users():
        role_b = _badge(u.get("role", "user"), "ok" if u.get("role") == "admin" else "muted")
        st = "" if u.get("enabled", True) else _badge("disabled")
        key_b = _badge("key set", "ok") if u.get("api_key") else _badge("no key", "warn")
        acts = _icon_acts(("✎", f"/ui/users?edit={_esc(u['name'])}", "secondary", "Edit"),
                          ("✕", f"/ui/users/delete?name={_esc(u['name'])}", "danger", "Delete",
                           f"Delete user {u['name']}?"))
        allow = u.get("models") or []
        cq = u.get('quota_cost_month')
        sub = (f"quota/day {u.get('quota_req_day') or '∞'} · "
               f"cost/mo {('$' + format(float(cq), '.2f')) if cq else '∞'} · "
               f"models {', '.join(allow) if allow else 'all'}")
        items += _item(f"{_esc(u['name'])} {role_b}{st} {key_b}", sub, acts, sel=(u["name"] == edit))
    items = items or "<p class='muted'>No users — the gateway is open (bootstrap). Add one to require keys.</p>"
    warn = ("<p class='ok-banner'>Bootstrap mode: no users and no master key → the API and /ui are "
            "open. Add an <b>admin</b> user (or set a master API key in Server) to lock it down.</p>"
            if open_auth else "")
    list_html = (f'<div class="bar"><h2>Users</h2>{_btn("+ New user", "/ui/users?new=1")}</div>'
                 "<p class='hint'>Each user authenticates with their API key; calls are attributed to "
                 "them (stats, job ownership). Empty model list = all allowed.</p>" + items)
    if edit and store.get_user(edit):
        detail = _user_form(store.get_user(edit))
    elif qp.get("new"):
        detail = _user_form(None)
    else:
        detail = "<h2>Details</h2><p class='hint'>Select a user's <b>Edit</b>, or <b>+ New user</b>.</p>"
    by_source = ((await asyncio.to_thread(stats.summary))["by_source"]
                 if stats.is_active() else [])              # ONE summary per render
    await _autoresolve_ips(by_source)
    ipa = store.get_ip_aliases()
    seen_ips = sorted({r[0] for r in by_source if _looks_like_ip(r[0])} | set(ipa.keys()))
    iprows = ""
    for ip in seen_ips:
        iprows += (f"<tr><td><code>{_esc(ip)}</code></td>"
                   f"<td><form action='/ui/ipalias/save' method='post' style='display:flex;gap:6px;align-items:center;margin:0'>"
                   f"<input type='hidden' name='ip' value='{_esc(ip)}'>{_inp('name', ipa.get(ip, ''), placeholder='alias')}"
                   f"{_btn('Save', submit=True)}</form></td>"
                   f"<td style='text-align:right;white-space:nowrap'>"
                   f"{_icon_acts(('✕', f'/ui/ipalias/delete?ip={quote(ip)}', 'danger', 'Delete', f'Delete IP alias {ip}?'))}</td></tr>")
    ip_section = (f"<h2 style='margin-top:26px'>IP aliases</h2>"
                  f"<p class='hint'>Friendly names for caller IPs (unauthenticated / <code>x-source</code> calls) as shown in "
                  f"Statistic. Hostnames are auto-resolved via reverse DNS on load — edit or clear as needed.</p>"
                  + (f"<table><tr><th>IP</th><th>alias</th><th></th></tr>{iprows}</table>" if iprows
                     else "<p class='muted'>No caller IPs seen yet (calls are currently attributed to authenticated users).</p>"))
    # Design convention (mirrors Mapping): the master-detail .cols is the SOLE full-height
    # block — intro/extra sections live INSIDE a column, never as a sibling after it — so
    # the detail form's sticky Save/Cancel bar stays visible while scrolling.
    body = (warn + f'<div class="cols"><div class="col">{list_html}{ip_section}</div>'
            f'<div class="col">{detail}</div></div>')
    return HTMLResponse(_page("Users", body, "users"))


async def ipalias_save(request: Request):
    f = await _form(request)
    ip = (f.get("ip", "") or "").strip()
    name = (f.get("name", "") or "").strip()
    if ip:
        store.set_ip_alias(ip, name)
        logger.info(f"ui: ip alias '{ip}' → '{name or '(cleared)'}'")
    return RedirectResponse("/ui/users", status_code=303)


async def ipalias_del(request: Request):
    ip = (request.query_params.get("ip", "") or "").strip()
    if ip:
        store.delete_ip_alias(ip)
        logger.info(f"ui: ip alias '{ip}' deleted")
    return RedirectResponse("/ui/users", status_code=303)


async def users_save(request: Request):
    qs = await _form_multi(request)                     # model allow-list is a checkbox list
    g = lambda k, d="": (qs.get(k) or [d])[-1]
    name = g("name").strip()
    if not name:
        return RedirectResponse("/ui/users?new=1", status_code=303)
    orig = g("orig").strip()
    u = dict(store.get_user(orig) or store.get_user(name) or {})
    u["name"] = name
    u["role"] = (g("role", "user") or "user").strip()
    u["enabled"] = bool(qs.get("enabled"))
    q = g("quota_req_day").strip()
    u["quota_req_day"] = int(q) if q.isdigit() else None
    qc = g("quota_cost_month").strip()
    try:
        u["quota_cost_month"] = float(qc) if qc else None
    except ValueError:
        u["quota_cost_month"] = None
    u["models"] = [m for m in qs.get("model", []) if m]
    ak = g("api_key").strip()
    if ak:
        u["api_key"] = ak
    if orig and orig != name and store.get_user(orig) is not None:
        store.delete_user(orig)
    store.upsert_user(u)
    _apply_users()
    logger.info(f"ui: user '{name}' saved (role={u['role']}, models={len(u['models'])})")
    return RedirectResponse("/ui/users", status_code=303)


async def users_del(request: Request):
    name = (request.query_params.get("name", "") or "").strip()
    if name:
        store.delete_user(name)
        _apply_users()
        logger.info(f"ui: user '{name}' deleted")
    return RedirectResponse("/ui/users", status_code=303)


# Server settings, split by when they take effect. Each: (key, kind, label, note).
_SRV_RUNTIME = [
    ("health_check_interval", "int", "health check interval", "seconds"),
    ("max_concurrent", "int", "default max_concurrent", "blank = unlimited"),
    ("park_timeout_s", "int", "default park time", "seconds a call waits for a free backend when all are busy (blank = 60; per-alias override in Mapping; 0 = off)"),
    ("max_parked", "int", "max parked calls", "queue cap — beyond this a busy call gets 503 (blank = 100)"),
    ("affinity_max_wait_s", "float", "affinity max wait",
     "seconds — a queued request older than this beats the same-type preference and takes "
     "the next free backend"),
    ("fast_probe_interval_s", "int", "fast probe interval",
     "seconds — how often an UNHEALTHY backend is re-checked while calls or jobs wait for "
     "capacity, so one that came back is picked up in seconds instead of a whole health "
     "check interval (blank = 3; 0 = off)"),
]
_SRV_RESTART = [
    ("__grp", "", "AI-Hub", ""),
    ("port", "int", "port", "set by launch cmd (uvicorn --port / systemd)"),
    ("__grp", "", "Stats (call log)", ""),
    ("stats_enabled", "bool", "enabled", "record calls (dashboard in Statistic tab)"),
    ("stats_db_path", "text", "db path", ""),
    ("stats_retention_days", "int", "retention days", "0 = keep forever"),
    ("__grp", "", "Jobs (image/video generation)", ""),
    ("jobs_enabled", "bool", "enabled", "auto-on when image models exist"),
    ("jobs_db_path", "text", "db path", ""),
    ("jobs_blob_dir", "text", "blob dir", ""),
    ("jobs_default_ttl_s", "int", "default TTL", "hours — how long a generation job + its images are kept before deletion"),
    ("jobs_prune_interval_s", "int", "prune interval", "minutes — how often expired jobs are purged"),
]
_SRV_RESTART_KEYS = [k for k, kind, *_ in _SRV_RESTART if kind]
# These settings are stored (and used) in SECONDS but shown/edited in a friendlier unit.
_SRV_UNITS = {"jobs_default_ttl_s": 3600, "jobs_prune_interval_s": 60}


def _srv_disp(k, v):
    """Seconds-stored setting → its display unit (TTL→hours, prune→minutes)."""
    if k in _SRV_UNITS and isinstance(v, (int, float)) and v not in ("", None):
        q = v / _SRV_UNITS[k]
        return int(q) if q == int(q) else round(q, 2)
    return v


async def server_page(request: Request):
    si = _server_info()
    eff, rt = si.get("effective", {}), si.get("runtime", {})
    running_port = request.url.port      # the port THIS request hit = the gateway port
    saved = request.query_params.get("saved")

    def rdiff(key):                      # configured value differs from what's running
        return key in rt and str(eff.get(key)) != str(rt.get(key))
    port_diff = bool(running_port and str(eff.get("port")) != str(running_port))
    any_restart = port_diff or any(rdiff(k) for k in _SRV_RESTART_KEYS if k != "port")
    mark = lambda cond: (" " + _badge("↻ restart", "warn")) if cond else ""
    note = lambda n: f" <span class='muted'>{_esc(n)}</span>" if n else ""

    banner = ""
    if saved == "1":
        banner = "<p class='ok-banner'>✓ Saved — runtime settings applied live.</p>"
    elif saved == "restart":
        banner = "<p class='bad'>✓ Saved — port/stats/jobs changes need a <b>restart</b> to apply.</p>"

    runtime_rows = (
        _field("API key (client auth)",
               _inp("api_key", "", placeholder=("•••• set — blank keeps it" if eff.get("api_key_set")
                                                else "unset — clients need no key")))
        + "".join(_field(lbl, _inp(k, "" if eff.get(k) in (None, "") else eff.get(k), typ="number") + note(n))
                  for k, kind, lbl, n in _SRV_RUNTIME)
        + _field("flags",
                 _checkbox("log_per_call", bool(eff.get("log_per_call")), "log_per_call",
                           "one log line per forwarded request")
                 + _checkbox("model_prefix", bool(eff.get("model_prefix")), "model_prefix",
                             "list models as backend/model in /v1/models")
                 + _checkbox("show_user_keys", _show_user_keys(), "show_user_keys",
                             "let an existing user's API key be copied again in the user "
                             "editor (off: only a key generated right there is shown)")))
    runtime_form = (
        '<form action="/ui/server/save" method="post"><input type="hidden" name="_form" value="runtime">'
        f'<div class="formbar"><h2>Runtime settings</h2>{_btn("Save", submit=True)}</div>'
        "<p class='hint'>Applied immediately on Save (no restart).</p>" + runtime_rows + "</form>")

    restart_rows = ""
    for k, kind, lbl, n in _SRV_RESTART:
        if kind == "":                                   # subsection header
            restart_rows += f'<div class="grouphdr">{_esc(lbl)}</div>'
        elif kind == "bool":
            restart_rows += _field(lbl, _checkbox(k, bool(eff.get(k)), "enabled", n) + mark(rdiff(k)))
        else:
            d = port_diff if k == "port" else rdiff(k)
            restart_rows += _field(lbl, _inp(k, _srv_disp(k, eff.get(k, "")), typ=("number" if kind == "int" else "text"))
                                   + mark(d) + note(n))
    restart_form = (
        '<form action="/ui/server/save" method="post"><input type="hidden" name="_form" value="restart">'
        f'<div class="formbar"><h2>Restart-required{mark(any_restart)}</h2>{_btn("Save", submit=True)}</div>'
        f"<p class='hint'>AI-Hub is listening on port <b>{_esc(running_port or '?')}</b>. "
        "These take effect on the next restart.</p>" + restart_rows + "</form>")

    info = ("<h2>Server</h2><p class='hint'>These override <code>config.yaml</code> and are stored in "
            "the gateway (API key encrypted at rest), so config.yaml only needs backends, aliases and "
            "the launch port.</p>")
    # intro/banner live inside the column (not as a sibling before .cols) so the sticky
    # Save bars stay visible — see the Users design-convention note.
    body = (f'<div class="cols"><div class="col">{info}{banner}{runtime_form}</div>'
            f'<div class="col">{restart_form}</div></div>')
    return HTMLResponse(_page("Server", body, "server"))


async def server_save(request: Request):
    f = await _form(request)
    which = f.get("_form", "")
    spec = _SRV_RUNTIME if which == "runtime" else _SRV_RESTART
    bools = {"log_per_call", "model_prefix", "show_user_keys"} if which == "runtime" else set()
    vals = {}
    for k, kind, *_ in spec:
        if not kind:
            continue
        if kind == "bool":
            vals[k] = bool(f.get(k))
        elif kind == "int":
            raw = (f.get(k, "") or "").strip()
            try:                                          # unit fields (TTL/prune) edited in hours/min → store seconds
                vals[k] = int(round(float(raw) * _SRV_UNITS.get(k, 1)))
            except ValueError:
                vals[k] = ""
        elif kind == "float":
            raw = (f.get(k, "") or "").strip()
            try:
                vals[k] = float(raw)
            except ValueError:
                vals[k] = ""
        else:
            vals[k] = (f.get(k, "") or "").strip()
    if which == "runtime":
        for b in bools:
            vals[b] = bool(f.get(b))
        ak = (f.get("api_key", "") or "").strip()
        if ak:
            vals["api_key"] = ak
    store.set_settings(vals)
    _apply_server_settings()
    logger.info(f"ui: server settings saved ({which}: {', '.join(vals)})")
    return RedirectResponse(f"/ui/server?saved={'1' if which == 'runtime' else 'restart'}",
                            status_code=303)


# ── Registration ────────────────────────────────────────────────────────────────

def _wf_filename(s: str) -> str:
    """Filesystem-safe name for a workflow download."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "workflow"


async def mapping_export(request: Request):
    """Download an alias's gateway-owned (Mapping-cleaned) workflow JSON — the
    request-field defaults you cleared in Mapping are already gone from it."""
    alias = _qp(request, "alias")
    cands = store.get(alias) if store.is_active() else None
    wf = cands[0].get("workflow_json") if cands else None
    if not wf:
        raise HTTPException(404, f"no stored workflow for alias '{alias}'")
    body = json.dumps(wf, indent=2, ensure_ascii=False)
    return Response(body, media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="{_wf_filename(alias)}.json"'})


async def mapping_export_all(request: Request):
    """Download every generation alias's cleaned workflow as one zip (bundle)."""
    if not store.is_active():
        raise HTTPException(404, "generation store inactive")
    import io
    import zipfile
    buf, n = io.BytesIO(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for alias in store.list_aliases():
            cands = store.get(alias) or []
            wf = cands[0].get("workflow_json") if cands else None
            if wf:
                z.writestr(f"{_wf_filename(alias)}.json", json.dumps(wf, indent=2, ensure_ascii=False))
                n += 1
    if not n:
        raise HTTPException(404, "no workflows to export")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="comfyui_workflows.zip"'})


def register(app) -> None:
    app.middleware("http")(_ui_guard)          # admin-login guard for /ui
    app.add_api_route("/", root_redirect, methods=["GET"], include_in_schema=False)
    app.add_api_route("/ui", ui_root, methods=["GET"], include_in_schema=False)
    app.add_api_route("/ui/login", login_page, methods=["GET"])
    app.add_api_route("/ui/login", login_post, methods=["POST"])
    app.add_api_route("/ui/static/{path:path}", static_asset, methods=["GET"])
    app.add_api_route("/ui/logout", logout, methods=["GET"])
    app.add_api_route("/ui/backends", backends_page, methods=["GET"])
    app.add_api_route("/ui/backends/save", backend_save, methods=["POST"])
    app.add_api_route("/ui/backends/host-save", host_save, methods=["POST"])
    app.add_api_route("/ui/backends/delete", backend_del, methods=["GET"])
    app.add_api_route("/ui/backends/drain", backend_drain, methods=["GET"])
    app.add_api_route("/ui/backends/undrain", backend_undrain, methods=["GET"])
    app.add_api_route("/ui/backends/restart", backend_restart, methods=["GET"])
    app.add_api_route("/ui/backends/enable", backend_enable, methods=["GET"])
    app.add_api_route("/ui/input", input_page, methods=["GET"])
    app.add_api_route("/ui/routing", routing_page, methods=["GET"])
    app.add_api_route("/ui/chat/create", chat_create, methods=["POST"])
    app.add_api_route("/ui/chat/save", chat_save, methods=["POST"])
    app.add_api_route("/ui/chat/badd", chat_badd, methods=["GET"])
    app.add_api_route("/ui/chat/bdel", chat_bdel, methods=["GET"])
    app.add_api_route("/ui/chat/delete", chat_del, methods=["GET"])
    app.add_api_route("/ui/chatplay", chatplay_page, methods=["GET"])
    app.add_api_route("/ui/chatplay/send", chatplay_send, methods=["POST"])
    app.add_api_route("/ui/mapping", mapping_page, methods=["GET"])
    app.add_api_route("/ui/mapping/register", register_post, methods=["POST"])
    app.add_api_route("/ui/mapping/update-workflow", update_workflow, methods=["POST"])
    app.add_api_route("/ui/mapping/field-add", edit_add, methods=["GET"])
    app.add_api_route("/ui/mapping/field-map", field_map, methods=["GET"])
    app.add_api_route("/ui/mapping/field-clear", field_clear, methods=["GET"])
    app.add_api_route("/ui/mapping/field-del", edit_del, methods=["GET"])
    app.add_api_route("/ui/mapping/cand-add", cand_add, methods=["GET"])
    app.add_api_route("/ui/mapping/cand-del", cand_del, methods=["GET"])
    app.add_api_route("/ui/mapping/bypass-add", bypass_add, methods=["GET"])
    app.add_api_route("/ui/mapping/bypass-del", bypass_del, methods=["GET"])
    app.add_api_route("/ui/mapping/field-order", field_order, methods=["GET"])
    app.add_api_route("/ui/mapping/update", update, methods=["POST"])
    app.add_api_route("/ui/mapping/cloud-update", cloud_update, methods=["POST"])
    app.add_api_route("/ui/mapping/export", mapping_export, methods=["GET"])
    app.add_api_route("/ui/mapping/export-all", mapping_export_all, methods=["GET"])
    app.add_api_route("/ui/mapping/copy", copy, methods=["GET"])
    app.add_api_route("/ui/mapping/delete", delete, methods=["GET"])
    app.add_api_route("/ui/reasoning", reasoning_page, methods=["GET"])
    app.add_api_route("/ui/reasoning/save", reasoning_save, methods=["POST"])
    app.add_api_route("/ui/reasoning/test", reasoning_test, methods=["POST"])
    app.add_api_route("/ui/reasoning/toggle", reasoning_toggle, methods=["GET"])
    app.add_api_route("/ui/reasoning/delete", reasoning_del, methods=["GET"])
    app.add_api_route("/ui/playground", playground_page, methods=["GET"])
    app.add_api_route("/ui/playground/voice", voiceplay_send, methods=["POST"])
    app.add_api_route("/ui/playground/voice-audio", voice_audio, methods=["GET"])
    app.add_api_route("/ui/playground/voice-upload", voice_upload, methods=["POST"])
    app.add_api_route("/ui/playground/voice-target", voice_target, methods=["POST"])
    app.add_api_route("/ui/playground/voice-ship", voice_ship, methods=["GET"])
    app.add_api_route("/ui/playground/voice-del", voice_del, methods=["GET"])
    app.add_api_route("/ui/playground/voice-lib/{name}", voice_lib_play, methods=["GET"])
    app.add_api_route("/ui/playground/generate", generate, methods=["POST"])
    app.add_api_route("/ui/playground/result/{job_id}/{n}", result, methods=["GET"])
    app.add_api_route("/ui/jobs", jobs_page, methods=["GET"])
    app.add_api_route("/ui/job/{job_id}", job_detail_page, methods=["GET"])
    app.add_api_route("/ui/job/{job_id}/input/{n}", job_input, methods=["GET"])
    app.add_api_route("/ui/job/{job_id}/to-playground", job_to_playground, methods=["GET"])
    app.add_api_route("/ui/job/{job_id}/cancel", job_cancel, methods=["GET"])
    app.add_api_route("/ui/dashboard", dashboard_page, methods=["GET"])
    app.add_api_route("/ui/llmcalls", llmcalls_page, methods=["GET"])
    app.add_api_route("/ui/statistic", statistic_page, methods=["GET"])
    app.add_api_route("/ui/call/{call_id}", call_view, methods=["GET"])
    app.add_api_route("/ui/call/{call_id}/audio", call_audio, methods=["GET"])
    app.add_api_route("/ui/users", users_page, methods=["GET"])
    app.add_api_route("/ui/users/save", users_save, methods=["POST"])
    app.add_api_route("/ui/users/delete", users_del, methods=["GET"])
    app.add_api_route("/ui/ipalias/save", ipalias_save, methods=["POST"])
    app.add_api_route("/ui/ipalias/delete", ipalias_del, methods=["GET"])
    app.add_api_route("/ui/server", server_page, methods=["GET"])
    app.add_api_route("/ui/server/save", server_save, methods=["POST"])
