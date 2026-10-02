"""AI-Hub's RunPod Serverless worker: runs ONE ComfyUI prompt per job inside the
container and hands back ComfyUI's /history outputs plus every file they name.

Our own code — nothing is copied from runpod-workers/worker-comfyui (AGPL-3.0; ai-hub is
Apache-2.0). The contract (adapters.RunpodAdapter reads it):
  {"op": "prompt", "workflow", "inputs": [{name, b64}], "deliver": {"sibling_exts"}}
    → {"outputs", "manifest": {"<type>/<subfolder>/<file>": {b64, size, sha256} | null},
       "worker_version"}  |  {"error": "<text>"}  (RunPod then reports FAILED)
  {"op": "info"} → {"object_info_gz", "models": {"image": {…}, "volume": {…}}, "worker_version"}
Everything but `main()` is plain functions over a base URL, testable without the runpod
SDK and without a GPU (tests/test_runpod_worker.py)."""
import base64
import glob
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

COMFY = "http://127.0.0.1:8188"
COMFY_DIR = "/comfyui"
DIRS = {t: f"{COMFY_DIR}/{t}" for t in ("input", "output", "temp")}
MODEL_ROOTS = {"image": f"{COMFY_DIR}/models", "volume": "/runpod-volume/models"}
OUT_MAX_B64 = 7 * 1024 * 1024        # /run results are capped at 10 MB by RunPod
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$")
# What adapters._view_params accepts as an artifact path (its _MIME_BY_EXT keys) —
# kept equal by tests/test_runpod_worker.py::test_artifact_extensions_equal_the_gateways.
ARTIFACT_EXTS = frozenset({".3mf", ".fbx", ".flac", ".gif", ".glb", ".gltf", ".jpeg", ".jpg", ".mp3", ".mp4", ".obj", ".ply", ".png", ".stl", ".usdz", ".wav", ".webm", ".webp"})


class HandlerError(Exception):
    pass


def _worker_version() -> str:
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.json")) as f:
            return str(json.load(f).get("version") or "?")
    except Exception:
        return "?"


def _http(method: str, url: str, body=None, timeout: float = 30.0):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def view_params(item):
    """(filename, {filename, type, subfolder?}) for one /history output item — the SAME
    rule as adapters._view_params (dict items, or a bare path string with an artifact
    extension whose output/temp/input segment names type + subfolder)."""
    if isinstance(item, dict):
        fn = item.get("filename")
        if not fn:
            return None
        view = {"filename": fn, "type": item.get("type", "output")}
        if item.get("subfolder"):
            view["subfolder"] = item["subfolder"]
        return fn, view
    if isinstance(item, str) and item:
        p = item.replace("\\", "/")
        if os.path.splitext(p)[1].lower() not in ARTIFACT_EXTS:
            return None
        typ = "output"
        rel = p.rsplit("/", 1)[-1] if (p.startswith("/") or ":" in p.split("/", 1)[0]) else p
        for t in ("output", "temp", "input"):
            i = p.rfind(f"/{t}/")
            if i != -1:
                typ, rel = t, p[i + len(t) + 2:]
                break
        fn = rel.rsplit("/", 1)[-1]
        sub = rel[:len(rel) - len(fn)].strip("/")
        view = {"filename": fn, "type": typ}
        if sub:
            view["subfolder"] = sub
        return fn, view
    return None


def view_key(params: dict) -> str:
    return "/".join(p for p in (params.get("type") or "output", params.get("subfolder") or "",
                                params.get("filename") or "") if p)


def write_inputs(inputs: list, input_dir: str) -> None:
    for it in inputs or []:
        name = str(it.get("name") or "")
        if not NAME_RE.match(name) or ".." in name:
            raise HandlerError(f"input name refused: {name[:80]!r}")
        with open(os.path.join(input_dir, name), "wb") as f:
            f.write(base64.b64decode(it.get("b64") or ""))


def history_error(entry: dict) -> str:
    msgs = (entry.get("status") or {}).get("messages") or []
    for m in msgs:
        if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == "execution_error":
            d = m[1] or {}
            return (f"node {d.get('node_id')} ({d.get('node_type', '?')}): "
                    f"{d.get('exception_message') or d}")[:4000]
    return json.dumps(msgs)[:4000] or "ComfyUI reported an error"


def _entry(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return {"b64": base64.b64encode(data).decode(), "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def build_manifest(outputs: dict, sibling_exts: list, dirs: dict, max_b64: int) -> dict:
    """Every file the outputs name (None when it is not on disk) plus `<stem>.<ext>`
    for each requested sibling extension that exists (a glob-style ext like `???` is
    matched against the directory). Raises HandlerError past `max_b64`."""
    man, total = {}, 0

    def add(key, path):
        nonlocal total
        if key in man:
            return
        if not os.path.isfile(path):
            man[key] = None
            return
        e = _entry(path)
        total += len(e["b64"])
        if total > max_b64:
            raise HandlerError(f"output too large ({total / 1048576:.1f} MB base64) — "
                               "milestone 1 returns base64 only (≤ 7 MB)")
        man[key] = e

    for out in (outputs or {}).values():
        for items in (out or {}).values():
            if not isinstance(items, list):
                continue
            for item in items:
                vp = view_params(item)
                if vp is None:
                    continue
                fn, view = vp
                base = os.path.join(dirs.get(view["type"], dirs["output"]),
                                    view.get("subfolder", ""))
                add(view_key(view), os.path.join(base, fn))
                stem = fn.rsplit(".", 1)[0] if "." in fn else fn
                for ext in sibling_exts or []:
                    pat = os.path.join(glob.escape(base), glob.escape(stem) + "." + ext)
                    hits = sorted(glob.glob(pat)) if any(c in ext for c in "*?[") else [
                        os.path.join(base, f"{stem}.{ext}")]
                    for h in hits:
                        if os.path.isfile(h):
                            add(view_key({**view, "filename": os.path.basename(h)}), h)
    return man


def model_index(roots: dict) -> dict:
    out = {}
    for label, root in roots.items():
        idx = {}
        if os.path.isdir(root):
            for d, _subs, files in os.walk(root):
                for fn in files:
                    p = os.path.join(d, fn)
                    rel = "models/" + os.path.relpath(p, root).replace(os.sep, "/")
                    try:
                        idx[rel] = os.path.getsize(p)
                    except OSError:
                        pass
        out[label] = idx
    return out


def _follow_progress(base: str, client_id: str, prompt_id: str, report, stop: threading.Event):
    """ComfyUI /ws step counter → report({step, steps, node}). Best effort: no
    websockets module, a refused socket or an odd message just reports nothing."""
    try:
        from websockets.sync.client import connect
    except Exception:
        return
    try:
        ws_url = base.replace("http", "ws", 1) + f"/ws?clientId={client_id}"
        with connect(ws_url, open_timeout=10) as ws:
            while not stop.is_set():
                try:
                    raw = ws.recv(timeout=1.0)
                except TimeoutError:
                    continue
                if not isinstance(raw, str):
                    continue
                msg = json.loads(raw)
                d = msg.get("data") or {}
                if msg.get("type") == "progress" and d.get("prompt_id") == prompt_id:
                    report({"step": d.get("value"), "steps": d.get("max"),
                            "node": str(d.get("node") or "")})
    except Exception:
        return


def run_prompt(job_input: dict, base: str, dirs: dict, report, poll_s: float = 1.0) -> dict:
    try:
        write_inputs(job_input.get("inputs") or [], dirs["input"])
        client_id = f"gw-{uuid.uuid4().hex[:12]}"
        st, body = _http("POST", f"{base}/prompt",
                         {"prompt": job_input.get("workflow") or {}, "client_id": client_id})
        reply = json.loads(body or b"{}")
        if st != 200 or reply.get("node_errors"):
            return {"error": f"ComfyUI refused the prompt ({st}): "
                             f"{json.dumps({k: reply.get(k) for k in ('error', 'node_errors')})[:4000]}"}
        pid = reply.get("prompt_id")
        if not pid:
            return {"error": "ComfyUI returned no prompt_id"}
        stop = threading.Event()
        threading.Thread(target=_follow_progress, args=(base, client_id, pid, report, stop),
                         daemon=True).start()
        try:
            while True:
                time.sleep(poll_s)
                st, body = _http("GET", f"{base}/history/{pid}")
                if st != 200:
                    continue
                hist = json.loads(body or b"{}")
                if pid not in hist:
                    continue
                entry = hist[pid]
                if (entry.get("status") or {}).get("status_str") == "error":
                    return {"error": history_error(entry)}
                outputs = entry.get("outputs") or {}
                break
        finally:
            stop.set()
        man = build_manifest(outputs, (job_input.get("deliver") or {}).get("sibling_exts") or [],
                             dirs, OUT_MAX_B64)
        return {"outputs": outputs, "manifest": man, "worker_version": _worker_version()}
    except HandlerError as e:
        return {"error": str(e)}


def run_info(base: str, roots: dict) -> dict:
    st, body = _http("GET", f"{base}/object_info", timeout=120.0)
    if st != 200:
        return {"error": f"/object_info → HTTP {st}"}
    return {"object_info_gz": base64.b64encode(gzip.compress(body)).decode(),
            "models": model_index(roots), "worker_version": _worker_version()}


def wait_ready(base: str, deadline_s: float = 180.0) -> bool:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            if _http("GET", f"{base}/object_info", timeout=10.0)[0] == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    import runpod                                   # only inside the image
    proc = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1",
                             "--port", "8188", "--disable-auto-launch"], cwd=COMFY_DIR)
    ready = wait_ready(COMFY)

    def handle(job):
        if not ready or proc.poll() is not None:
            return {"error": "ComfyUI did not start in the worker"}
        inp = job.get("input") or {}
        if inp.get("op") == "info":
            return run_info(COMFY, MODEL_ROOTS)
        if inp.get("op") == "prompt":
            return run_prompt(inp, COMFY, DIRS,
                              report=lambda p: runpod.serverless.progress_update(job, p))
        return {"error": f"unknown op {inp.get('op')!r}"}

    runpod.serverless.start({"handler": handle})


if __name__ == "__main__":
    main()
