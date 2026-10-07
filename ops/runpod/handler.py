"""AI-Hub's RunPod Serverless worker: runs ONE ComfyUI prompt per job inside the
container and hands back ComfyUI's /history outputs plus every file they name.

Our own code — nothing is copied from runpod-workers/worker-comfyui (AGPL-3.0; ai-hub is
Apache-2.0). The contract (adapters.RunpodAdapter reads it):
  {"op": "prompt", "workflow", "inputs": [{name, b64}], "deliver": {"sibling_exts"}}
    → {"outputs", "manifest": {"<type>/<subfolder>/<file>": {b64, size, sha256} | null},
       "worker_version"}  |  {"error": "<text>"}  (RunPod then reports FAILED)
  {"op": "info"} → {"object_info_gz", "models": {"image": {…}, "volume": {…}},
    "worker_version", "volume": {total, used, free} | null}
  {"op": "fetch", "items": [{path, url, size, sha256?}]}
    → {"results": [{path, ok, size, sha256, error}]}
  {"op": "link", "links": [{path, target}]} → {"results": [{path, ok, error}]}
Everything but `main()` is plain functions over a base URL or volume root, testable without the runpod
SDK and without a GPU (tests/test_runpod_worker.py)."""
import base64
import binascii
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
import urllib.parse
import urllib.request
import uuid
from typing import Optional

VOLUME = "/runpod-volume"
_ROOTS = ("models/", "hf-cache/")
_HF_HOSTS = ("huggingface.co", "hf.co")
_CHUNK = 8 * 1024 * 1024

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

    def inside(path, root):
        """realpath containment: a `..` subfolder, a relative bare path, a sibling ext
        holding `/`, or a symlink out of the type dir names a file this job did not write
        (the volume's models, the worker's own files) — never read, never shipped."""
        real, rroot = os.path.realpath(path), os.path.realpath(root)
        return real != rroot and os.path.commonpath([real, rroot]) == rroot

    def add(key, path, root):
        nonlocal total
        if key in man:
            return
        if not inside(path, root) or not os.path.isfile(path):
            man[key] = None
            return
        n = os.path.getsize(path)
        need = total + 4 * ((n + 2) // 3)       # base64 size, judged BEFORE reading
        if need > max_b64:
            raise HandlerError(f"output too large ({need / 1048576:.1f} MB base64) — "
                               "milestone 1 returns base64 only (≤ 7 MB)")
        e = _entry(path)
        total += len(e["b64"])
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
                root = dirs.get(view["type"], dirs["output"])
                base = os.path.join(root, view.get("subfolder", ""))
                add(view_key(view), os.path.join(base, fn), root)
                stem = fn.rsplit(".", 1)[0] if "." in fn else fn
                for ext in sibling_exts or []:
                    pat = os.path.join(glob.escape(base), glob.escape(stem) + "." + ext)
                    hits = sorted(glob.glob(pat)) if any(c in ext for c in "*?[") else [
                        os.path.join(base, f"{stem}.{ext}")]
                    for h in hits:
                        if os.path.isfile(h) or not inside(h, root):
                            add(view_key({**view, "filename": os.path.basename(h)}), h, root)
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


def run_prompt(job_input: dict, base: str, dirs: dict, report, poll_s: float = 1.0,
               alive=lambda: True) -> dict:
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
                if not alive():
                    return {"error": "ComfyUI died during the prompt"}
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
    except (OSError, ValueError, binascii.Error, AttributeError, TypeError) as e:
        return {"error": f"{type(e).__name__}: {e}"[:4000]}


def safe_rel(path) -> Optional[str]:
    if not isinstance(path, str) or not path or len(path) > 512 or "\x00" in path:
        return None
    if path.startswith("/") or "\\" in path:
        return None
    norm = os.path.normpath(path)
    if norm != path.rstrip("/") or norm.startswith("..") or "/../" in f"/{norm}/":
        return None
    return norm if norm.startswith(_ROOTS) else None


def _hf_host(url: str) -> bool:
    h = (urllib.parse.urlsplit(url).hostname or "").lower()
    return any(h == x or h.endswith("." + x) for x in _HF_HOSTS)


def _under(path: str, root: str) -> bool:
    """Existing symlinks must obey the same fence as job-supplied path segments."""
    real, base = os.path.realpath(path), os.path.abspath(root)
    return real != base and os.path.commonpath([real, base]) == base


def run_fetch(job_input: dict, root: str, opener=urllib.request.urlopen, env=os.environ,
              ismount=os.path.ismount) -> dict:
    """Publish only verified bytes; retain interrupted downloads for the next job."""
    if not ismount(root):
        return {"error": "network volume not mounted at /runpod-volume"}
    results = []
    for item in job_input.get("items") or []:
        result = {"path": item.get("path") if isinstance(item, dict) else None,
                  "ok": False, "size": 0, "sha256": "", "error": ""}
        results.append(result)
        part, size = None, None
        try:
            path = safe_rel(result["path"])
            if path is None:
                raise HandlerError("path refused")
            url = item.get("url")
            if not isinstance(url, str) or urllib.parse.urlsplit(url).scheme != "https":
                raise HandlerError("URL must use https")
            size = item.get("size")
            if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
                raise HandlerError("size must be a nonnegative integer")
            final = os.path.join(root, path)
            candidate = final + ".gw-part"
            fence = os.path.join(root, path.split("/", 1)[0])
            if not _under(fence, root) or not _under(final, fence) or not _under(candidate, fence):
                raise HandlerError("path escapes volume root")
            part = candidate
            os.makedirs(os.path.dirname(final), exist_ok=True)
            if os.path.isfile(final) and (size is None or os.path.getsize(final) == size):
                result.update(ok=True, size=os.path.getsize(final), skipped=True)
                continue
            if size is None and os.path.exists(part):
                os.unlink(part)
            n, digest = 0, hashlib.sha256()
            if size is not None and os.path.isfile(part) and os.path.getsize(part) < size:
                with open(part, "rb") as f:
                    while chunk := f.read(_CHUNK):
                        digest.update(chunk)
                        n += len(chunk)
            # A stale range gets one fresh request in this job, never an endless loop.
            for attempt in range(2):
                req = urllib.request.Request(url, headers={"User-Agent": "ai-hub-worker"})
                if _hf_host(url) and env.get("HF_TOKEN"):
                    req.add_unredirected_header("Authorization", "Bearer " + env["HF_TOKEN"])
                if n:
                    req.add_header("Range", f"bytes={n}-")
                try:
                    response = opener(req, timeout=60)
                    if response.status >= 400:
                        response.close()
                        raise urllib.error.HTTPError(url, response.status, "fetch refused",
                                                     response.headers, None)
                except urllib.error.HTTPError as exc:
                    if exc.code == 416 and n and attempt == 0:
                        os.unlink(part)
                        n, digest = 0, hashlib.sha256()
                        continue
                    raise
                with response:
                    announced = response.headers.get('Content-Length')
                    announced = int(announced) if announced is not None else None
                    if response.status == 206:
                        span = re.fullmatch(r'bytes (\d+)-(\d+)/(?:\d+|\*)',
                                            response.headers.get('Content-Range', ''))
                        if not span or int(span[1]) != n or int(span[2]) < int(span[1]):
                            if attempt == 0:
                                if os.path.exists(part):
                                    os.unlink(part)
                                n, digest = 0, hashlib.sha256()
                                continue
                            raise HandlerError('invalid Content-Range')
                        announced = int(span[2]) - int(span[1]) + 1
                    elif n:
                        n, digest = 0, hashlib.sha256()
                    resumed = n > 0
                    received = 0
                    with open(part, "ab" if n else "wb") as f:
                        while chunk := response.read(_CHUNK):
                            f.write(chunk)
                            digest.update(chunk)
                            n += len(chunk)
                            received += len(chunk)
                    if announced is not None and received != announced:
                        result['error'] = f'short read ({received} of {announced} bytes)'
                        if size is None:
                            os.unlink(part)
                        break
                    if size is None and announced is None and not item.get('sha256'):
                        os.unlink(part)
                        result['error'] = 'size unknown and no sha256 — not published'
                        break
                break
            if result['error']:
                continue
            result.update(size=n, sha256=digest.hexdigest())
            if (size is not None and n != size) or (item.get("sha256") and item["sha256"] != digest.hexdigest()):
                os.unlink(part)
                why = "size mismatch" if size is not None and n != size else "sha256 mismatch"
                # a resumed part may be a stale file's head: the part is gone now, so the
                # next attempt starts from byte 0 — only a fresh download's mismatch is final
                result["error"] = ("" if resumed else "final: ") + why
                continue
            os.replace(part, final)
            result["ok"] = True
        except Exception as e:
            if size is None and part is not None and os.path.isfile(part):
                os.unlink(part)
            prefix = "final: " if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 else ""
            detail = f"HTTP {e.code}" if isinstance(e, urllib.error.HTTPError) else f"{type(e).__name__}: {e}"
            result["error"] = (prefix + detail)[:300]
    return {"results": results}


def run_link(job_input: dict, root: str, ismount=os.path.ismount) -> dict:
    """Replace stale entries atomically so a sync never exposes a half-made link."""
    if not ismount(root):
        return {"error": "network volume not mounted at /runpod-volume"}
    results = []
    for item in job_input.get("links") or []:
        result = {"path": item.get("path") if isinstance(item, dict) else None,
                  "ok": False, "error": ""}
        results.append(result)
        tmp = None
        try:
            path = safe_rel(result["path"])
            target = item.get("target")
            if path is None or not path.startswith("hf-cache/"):
                raise HandlerError("link path refused")
            if (not isinstance(target, str) or not target or os.path.isabs(target)
                    or "\x00" in target or "\\" in target):
                raise HandlerError("link target must be relative")
            final = os.path.join(root, path)
            cache = os.path.join(root, "hf-cache")
            real = os.path.realpath(os.path.join(os.path.dirname(final), target))
            if (not _under(cache, root) or not _under(os.path.dirname(final), root)
                    or not _under(real, cache)
                    or (not _under(os.path.dirname(final), cache)
                        and os.path.dirname(final) != cache)):
                raise HandlerError("link escapes hf-cache")
            os.makedirs(os.path.dirname(final), exist_ok=True)
            if not os.path.islink(final) or os.readlink(final) != target:
                tmp = final + ".gw-part-" + uuid.uuid4().hex
                os.symlink(target, tmp)
                os.replace(tmp, final)
            result["ok"] = True
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if tmp is not None and os.path.lexists(tmp):
                os.unlink(tmp)
    return {"results": results}


def volume_space(root: str) -> Optional[dict]:
    try:
        st = os.statvfs(root)
        return {"total": st.f_blocks * st.f_frsize,
                "used": (st.f_blocks - st.f_bfree) * st.f_frsize,
                "free": st.f_bavail * st.f_frsize}
    except OSError:
        return None


def run_info(base: str, roots: dict) -> dict:
    st, body = _http("GET", f"{base}/object_info", timeout=120.0)
    if st != 200:
        return {"error": f"/object_info → HTTP {st}"}
    return {"object_info_gz": base64.b64encode(gzip.compress(body)).decode(),
            "models": model_index(roots), "worker_version": _worker_version(),
            "volume": volume_space(VOLUME)}


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
        inp = job.get("input") or {}
        if inp.get("op") == "fetch":
            return run_fetch(inp, VOLUME)
        if inp.get("op") == "link":
            return run_link(inp, VOLUME)
        if not ready or proc.poll() is not None:
            return {"error": "ComfyUI did not start in the worker"}
        if inp.get("op") == "info":
            return run_info(COMFY, MODEL_ROOTS)
        if inp.get("op") == "prompt":
            return run_prompt(inp, COMFY, DIRS,
                              report=lambda p: runpod.serverless.progress_update(job, p),
                              alive=lambda: proc.poll() is None)
        return {"error": f"unknown op {inp.get('op')!r}"}

    runpod.serverless.start({"handler": handle})


if __name__ == "__main__":
    main()
