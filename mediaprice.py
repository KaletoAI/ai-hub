"""*Pure*: the "from" price of one media generation candidate — what the cheapest request
on it costs at least, read off what the provider publishes. No I/O; imports neither
`main` nor `adapters` (the caller passes the candidate's kind and its backend).

A price is `{"usd": x, "unit": u}` (or `{"credits": n, "unit": "run"}` where only the
credit count is known — Meshy's USD per credit depends on the account's plan). Units:
`image` / `video` / `run` are PER REQUEST; `second`, `1k chars`, `1M chars` and `hour`
are rates whose total depends on the request. `min_s` (seconds) rides along when the
shortest billable duration is known, so a per-second rate still yields a per-request
floor. `None` = unknown: never 0 — a paid generation shown as free is the silent failure.

Why "from": a media price depends on the request (size, duration, resolution, quality,
references, variant); the catalog's cheapest entry is a true lower bound and needs no
guess at what the client will send.
"""
from __future__ import annotations

import math
from typing import Optional

REQUEST_UNITS = ("image", "video", "run")
_SURCHARGE = ("reference", "input", "source", "minimum", "extra", "fee", "transition", "addon",
              "multiplier", "override",
              "default", "frames", "duration")      # settings, not prices: default_frames_per_second
_PER_VIDEO = ("per_video", "base_price", "base_prices_by_resolution", "per_resolution",
              "with_audio", "without_audio")
TRIPO_USD_PER_CREDIT = 0.01        # tripo.py: "1 credit = 0.01 USD"


def _num(v) -> Optional[float]:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, dict):                       # Venice: {"usd": 0.01, "diem": 0.01}
        v = v.get("usd")
        if isinstance(v, bool) or v is None:
            return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def _min_of(values) -> Optional[float]:
    """Smallest usable number in a (possibly nested) dict / list of prices."""
    found = []

    def walk(v):
        x = _num(v)
        if x is not None:
            found.append(x)
        elif isinstance(v, dict):
            for w in v.values():
                walk(w)
        elif isinstance(v, (list, tuple)):
            for w in v:
                walk(w)
    walk(values)
    return min(found) if found else None


def _price(usd: Optional[float], unit: str, min_s: Optional[float] = None) -> Optional[dict]:
    if usd is None:
        return None
    out = {"usd": usd, "unit": unit}
    if min_s:
        out["min_s"] = min_s
    return out


def nanogpt_from(pricing) -> Optional[dict]:
    """NanoGPT media `pricing` (USD): per_image (by size), per_run(_by_variant) (3D),
    per_generation, per_duration (video lengths), per_second(_by_resolution),
    per_thousand_chars (TTS), per_billing_interval, starting_at / minimum(_price)."""
    if not isinstance(pricing, dict) or pricing.get("currency", "USD") != "USD":
        return None
    p = pricing
    durs = [d for d in (_num(x) for x in p.get("supported_durations") or []) if d]
    min_s = min(durs) if durs else next(
        (x for x in (_num(p.get(k)) for k in ("minimum_billable_duration", "min_duration",
                                                "minimum_duration_seconds")) if x), None)
    # Each group: every field that prices the same thing; the cheapest counts (a TURBO
    # rendering speed, a 3D variant without texture are as callable as the default).
    for keys, unit in ((("per_image", "by_rendering_speed"), "image"),
                       (("per_run", "per_run_by_variant"), "run"),
                       (("per_generation",), "run"), (("per_duration",), "video")):
        lo = _min_of([p.get(k) for k in keys])
        if lo is not None:
            return _price(lo, unit)
    for key in ("per_second", "per_second_by_resolution"):
        lo = _min_of(p.get(key))
        if lo is not None:
            return _price(lo, "second", min_s)
    if _num(p.get("per_thousand_chars")) is not None:
        return _price(_num(p["per_thousand_chars"]), "1k chars")
    # The video long tail (2026-11: ~60 of 208 shapes): a flat per-video price, else any
    # `…per_second…` rate. Surcharge keys (references, input/source video, minimums,
    # fees, transitions, add-ons) are extras on top of a base, never the base itself.
    base = {k: v for k, v in p.items()
            if not any(s in k for s in _SURCHARGE) and k not in ("currency",)}
    lo = _min_of([base.get(k) for k in _PER_VIDEO])
    if lo is not None:
        return _price(lo, "video")
    lo = _min_of([v for k, v in base.items() if "per_second" in k])
    if lo is not None:
        return _price(lo, "second", min_s)
    for key in ("starting_at", "minimum", "minimum_price", "per_billing_interval"):
        x = _num(p.get(key))
        if x is not None and (x > 0 or key == "starting_at"):
            return _price(x, "run")
    return None


def venice_from(pricing) -> Optional[dict]:
    """Venice `model_spec.pricing` (values `{"usd": …}`): generation (flat), resolutions /
    quality (images), durations (music), per_second, per_thousand_characters, input
    (TTS, per 1M characters). Upscale prices are an extra, never the base."""
    if not isinstance(pricing, dict):
        return None
    for keys, unit in ((("generation",), "run"), (("resolutions", "quality"), "image"),
                       (("durations",), "run")):
        lo = _min_of([pricing.get(k) for k in keys])
        if lo is not None:
            return _price(lo, unit)
    if _num(pricing.get("per_second")) is not None:
        return _price(_num(pricing["per_second"]), "second")
    if _num(pricing.get("per_thousand_characters")) is not None:
        return _price(_num(pricing["per_thousand_characters"]), "1k chars")
    if _num(pricing.get("input")) is not None:
        return _price(_num(pricing["input"]), "1M chars")
    return None


def from_price(kind: str, cand: dict, backend: dict) -> Optional[dict]:
    """The candidate's from-price. `kind` = its cloud kind (nanogpt, venice, tripo, meshy,
    arliai) or the backend type for a workflow backend (comfyui, runpod)."""
    cand, backend = cand or {}, backend or {}
    meta = cand.get("model_metadata") or {}
    found = None
    if kind == "nanogpt":
        found = nanogpt_from(meta.get("pricing"))
    elif kind == "venice":
        found = venice_from((meta.get("model_spec") or {}).get("pricing"))
        # A flat `generation` price on an image endpoint is per image (one per request).
        if found and found["unit"] == "run" and (cand.get("venice") or {}).get("endpoint") in ("images", "edit"):
            found["unit"] = "image"
    elif kind == "tripo":
        # tripo.py: 20 credits per generation without texture (30 with), 25 per rig.
        rig = ((cand.get("tripo") or {}).get("endpoint") or "") == "rig"
        found = _price((25 if rig else 20) * TRIPO_USD_PER_CREDIT, "run")
    elif kind == "meshy":
        # meshy.py: 20 credits (no texture) per Meshy-6/7 task, 5 per rigging.
        rig = ((cand.get("meshy") or {}).get("endpoint") or "") == "rigging"
        found = {"credits": 5 if rig else 20, "unit": "run"}
    elif kind == "runpod":
        found = _price(_num(backend.get("cost_per_hour")), "hour")
    if found is None and not backend.get("paid"):
        return {"usd": 0.0, "unit": "run", "free": True}
    return found


def request_usd(price: Optional[dict]) -> Optional[float]:
    """The per-REQUEST floor in USD, or None when the price is a rate with no known
    minimum, credits without a USD rate, or unknown."""
    if not price or "usd" not in price:
        return None
    if price.get("free"):
        return 0.0
    if price["unit"] in REQUEST_UNITS:
        return price["usd"]
    if price["unit"] == "second" and price.get("min_s"):
        return price["usd"] * price["min_s"]
    return None


def usd_str(x: float) -> str:
    """A USD amount as OpenRouter writes it: plain decimal, never an exponent."""
    s = f"{x:.8f}".rstrip("0").rstrip(".")
    return s or "0"


def cheapest(prices) -> Optional[dict]:
    """Over an alias's candidates: the cheapest known price (by per-request floor where
    there is one), with `unknown` = how many candidates have no price. None when there
    are no candidates at all."""
    prices = list(prices)
    if not prices:
        return None
    known = [p for p in prices if p]
    if not known:
        return {"unknown": len(prices)}

    def key(p):
        r = request_usd(p)
        return (0, r) if r is not None else (1, p.get("usd", p.get("credits", 0)))
    best = dict(min(known, key=key))
    best["unknown"] = len(prices) - len(known)
    return best
