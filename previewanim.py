"""Inject a diagnostic idle animation into a rigged GLB, for the /ui preview only.

Rigged meshes ship in bind pose, where bad skin weights are invisible — the mesh only
spikes/rings once it deforms. model-viewer plays an animation if the GLB carries one, so
this appends a short looping range-of-motion clip (a walk-like scissor of the limbs +
spine sway) targeting the standard Mixamo joints by name. It stresses every major joint
so a mis-bound vertex visibly shoots out (spike) or the crotch/armpit webs (ring).

Pure struct/json on the glTF binary — no 3D engine, no deps. Only APPENDS (new buffer
data + accessors + one animation); existing accessors keep their offsets. Applied on the
fly to the previewed copy, never to the delivered file. A GLB without a skin is returned
unchanged.
"""
import json
import math
import struct

# joint (Mixamo bone suffix), local rotation axis, amplitude (deg), phase (rad). Legs run
# opposite phase to the arms → a walk-like scissor that separates the thighs (reveals the
# crotch ring) and swings every limb (reveals spikes). Diagnostic, not anatomically exact.
_X = (1.0, 0.0, 0.0)
_PI = math.pi
_IDLE = [
    ("Spine1", _X, 7, 0.0), ("Spine2", _X, 6, 0.0), ("Neck", _X, 6, _PI),
    ("LeftArm", _X, 32, 0.0), ("RightArm", _X, 32, _PI),
    ("LeftForeArm", _X, 22, 0.0), ("RightForeArm", _X, 22, _PI),
    ("LeftUpLeg", _X, 30, _PI), ("RightUpLeg", _X, 30, 0.0),
    ("LeftLeg", _X, 26, _PI), ("RightLeg", _X, 26, 0.0),
]
_PERIOD = 4.0          # seconds per loop
_NKEY = 17             # keyframes (LINEAR between)


def _bone_key(name) -> str:
    """Normalise a node name to its bare Mixamo bone key for the _IDLE lookup. Strips a
    namespace prefix (`Armature:mixamorig:Hips`) AND the `mixamorig` prefix in all its
    common FBX→GLB spellings — `mixamorig:Hips`, `mixamorig_Hips`, `mixamorigHips` —
    which the colon-split alone would miss (leaving `mixamorighips` ≠ `hips`)."""
    nm = (name or "").split(":")[-1].lower()
    for pre in ("mixamorig_", "mixamorig"):
        if nm.startswith(pre):
            return nm[len(pre):]
    return nm


def _parse(glb: bytes):
    """(gltf-dict, BIN bytearray, had_bin_chunk) or None. `had_bin_chunk` lets the
    caller refuse to corrupt a GLB whose buffer 0 is external (no BIN chunk). Reuses
    the shared GLB chunk-walk from adapters (imported lazily — this module stays a
    dependency-free leaf at import time; the parse itself is main-free)."""
    from adapters import glb_chunks
    parsed = glb_chunks(glb)
    if parsed is None:
        return None
    gltf, bc = parsed
    return gltf, bytearray(bc or b""), bc is not None


def _build(gltf: dict, binb: bytearray) -> bytes:
    j = json.dumps(gltf, separators=(",", ":")).encode()
    j += b" " * ((4 - len(j) % 4) % 4)
    b = bytes(binb) + b"\x00" * ((4 - len(binb) % 4) % 4)
    total = 12 + 8 + len(j) + 8 + len(b)
    out = bytearray(b"glTF")
    out += struct.pack("<II", 2, total)
    out += struct.pack("<I", len(j)) + b"JSON" + j
    out += struct.pack("<I", len(b)) + b"BIN\x00" + b
    return bytes(out)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _axis_angle(axis, ang):
    s = math.sin(ang / 2.0)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(ang / 2.0))


def add_idle(glb: bytes) -> bytes:
    """Return the GLB with a looping 'idle' animation appended, or unchanged on any
    problem / when there is no skinned skeleton to drive."""
    try:
        parsed = _parse(glb)
        if parsed is None:
            return glb
        gltf, binb, had_bin = parsed
        nodes = gltf.get("nodes") or []
        bufs = gltf.get("buffers") or []
        if not (gltf.get("skins") and nodes and bufs):
            return glb
        # We only ever append into buffer 0 as the embedded BIN chunk. If buffer 0 is
        # external (has a `uri`) or the file carries no BIN chunk, appending would point
        # the new accessors at the wrong bytes and clobber buffer 0's byteLength — so
        # return the original untouched (the preview just plays no idle clip).
        if not had_bin or bufs[0].get("uri"):
            return glb
        by_name = {}
        for i, n in enumerate(nodes):
            by_name.setdefault(_bone_key(n.get("name")), i)
        targets = [(by_name[b.lower()], ax, amp, ph) for b, ax, amp, ph in _IDLE
                   if b.lower() in by_name]
        if not targets:
            return glb

        times = [round(_PERIOD * k / (_NKEY - 1), 5) for k in range(_NKEY)]
        bufviews = gltf.setdefault("bufferViews", [])
        accessors = gltf.setdefault("accessors", [])

        def _append(data: bytes) -> int:
            if len(binb) % 4:
                binb.extend(b"\x00" * (4 - len(binb) % 4))
            off = len(binb)
            binb.extend(data)
            bufviews.append({"buffer": 0, "byteOffset": off, "byteLength": len(data)})
            return len(bufviews) - 1

        # shared time (input) accessor — glTF requires min/max on animation inputs
        tv = _append(struct.pack("<%df" % len(times), *times))
        t_acc = len(accessors)
        accessors.append({"bufferView": tv, "componentType": 5126, "count": len(times),
                          "type": "SCALAR", "min": [times[0]], "max": [times[-1]]})

        samplers, channels = [], []
        for node_i, axis, amp_deg, phase in targets:
            rest = tuple(nodes[node_i].get("rotation") or (0.0, 0.0, 0.0, 1.0))
            amp = math.radians(amp_deg)
            quats = []
            for t in times:
                ang = amp * math.sin(2 * _PI * t / _PERIOD + phase)
                quats.extend(_qmul(rest, _axis_angle(axis, ang)))
            ov = _append(struct.pack("<%df" % len(quats), *quats))
            o_acc = len(accessors)
            accessors.append({"bufferView": ov, "componentType": 5126,
                              "count": len(times), "type": "VEC4"})
            samplers.append({"input": t_acc, "output": o_acc, "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": node_i, "path": "rotation"}})

        gltf["buffers"][0]["byteLength"] = len(binb)
        gltf.setdefault("animations", []).append(
            {"name": "idle", "samplers": samplers, "channels": channels})
        return _build(gltf, binb)
    except Exception:
        return glb
