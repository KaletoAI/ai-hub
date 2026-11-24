"""Catalog capabilities shared by API filters and console pickers, not routing.

Provider declarations take precedence over conservative legacy name recognition.
Input vision is still chat; output modalities can give a model several types.
"""
from __future__ import annotations

import re

MEDIA_TYPES = frozenset({"image", "video", "3d", "tts", "audio", "asr"})
# "decision": System One models (POST /v1/systemone only) — set by the backend's
# `decision_models`, never derived from a name or provider label here.
MODEL_TYPES = MEDIA_TYPES | {"chat", "embedding", "rerank", "decision"}

_LABELS = {
    "chat": "chat", "text": "chat", "llm": "chat", "vision": "chat",
    "chat-completion": "chat", "chat-completions": "chat", "text-generation": "chat",
    "text2text-generation": "chat", "image-text-to-text": "chat", "conversational": "chat",
    "image": "image", "images": "image", "image-generation": "image", "text-to-image": "image",
    "inpaint": "image", "edit": "image", "upscale": "image", "background-remove": "image",
    "video": "video", "text-to-video": "video", "image-to-video": "video",
    "3d": "3d", "mesh": "3d", "text-to-3d": "3d", "image-to-3d": "3d",
    "tts": "tts", "speech": "tts", "text-to-speech": "tts", "speech-generation": "tts",
    "audio": "audio", "music": "audio", "voice-changer": "audio", "audio-generation": "audio",
    "asr": "asr", "transcription": "asr", "speech-to-text": "asr",
    "automatic-speech-recognition": "asr", "audio-to-text": "asr",
    "embedding": "embedding", "embeddings": "embedding", "feature-extraction": "embedding",
    "rerank": "rerank", "reranking": "rerank", "reranker": "rerank",
}
_CAPABILITIES = {
    "chat": "chat", "chat_completion": "chat", "chat_completions": "chat",
    "text_generation": "chat", "vision": "chat", "supports_vision": "chat",
    "image_generation": "image", "text_to_image": "image", "image_to_image": "image",
    "video_generation": "video", "text_to_video": "video", "image_to_video": "video",
    "video_to_video": "video", "text_to_3d": "3d", "image_to_3d": "3d",
    "model_3d_generation": "3d",
    "text_to_speech": "tts", "tts": "tts", "speech_to_text": "asr", "asr": "asr",
    "audio_generation": "audio", "music_generation": "audio", "text_to_music": "audio",
    "audio_to_audio": "audio",
    "embedding": "embedding", "embeddings": "embedding", "rerank": "rerank",
}


def _labels(value) -> set[str]:
    if isinstance(value, str):
        label = value.lower().replace("_", "-").strip()
        return {_LABELS[label]} if label in _LABELS else set()
    if isinstance(value, (list, tuple, set, frozenset)):
        return set().union(*(_labels(item) for item in value))
    return set()


def _metadata_types(metadata: dict) -> set[str]:
    kinds = set()
    architecture = metadata.get("architecture") or {}
    outputs = architecture.get("output_modalities") if isinstance(architecture, dict) else None
    if outputs is None:
        outputs = metadata.get("output_modalities")
    output_types = _labels(outputs)
    for key in ("type", "kind", "task", "pipeline_tag", "model_type", "model_types", "catalog_type"):
        declared = _labels(metadata.get(key))
        # OpenRouter may label its entire catalog chat despite image-only outputs.
        if output_types and "chat" not in output_types:
            declared.discard("chat")
        kinds.update(declared)
    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, dict):
        for name, value in capabilities.items():
            if value is True and name in _CAPABILITIES:
                kinds.add(_CAPABILITIES[name])
    else:
        kinds.update(_labels(capabilities))
    for key in ("model_spec", "meta", "llamaswap"):
        nested = metadata.get(key)
        if isinstance(nested, dict):
            kinds.update(_metadata_types(nested))
    # Audio is an output modality of speech, not an extra music/voice-conversion API.
    if "tts" in kinds:
        output_types.discard("audio")
    # Text output from recognition/embeddings is not automatically conversation.
    if kinds & {"asr", "embedding", "rerank"}:
        output_types.discard("chat")
    # Legacy servers may describe every model as text->text, including embeddings.
    # Without a declared task, that generic modality must not suppress name inference.
    if kinds or output_types != {"chat"}:
        kinds.update(output_types)
    return kinds


def model_types(model_id: str, metadata: dict | None = None) -> set[str]:
    """Return all known capabilities; untyped/unrecognized legacy IDs remain chat."""
    if isinstance(metadata, dict):
        declared = _metadata_types(metadata)
        if declared:
            return declared
    name = model_id.lower()
    # Boundaries keep e.g. reflection / audiophile chat models out of media.
    def named(pattern: str) -> bool:
        return re.search(r"(?:^|[/_. :@-])(?:" + pattern + r")(?:$|[/_. :@-])", name) is not None
    if named(r"rerank(?:er|ing)?"):
        return {"rerank"}
    if named(r"embed(?:ding|dings)?|nomic-embed|bge|e5|mxbai-embed|embeddinggemma|all-minilm|gte"):
        return {"embedding"}
    if named(r"tts(?:-\d+)?|kokoro|piper|chatterbox|xtts|f5-tts|fish-speech|orpheus|speecht5|cosyvoice|bark"):
        return {"tts"}
    if named(r"whisper|distil-whisper|parakeet|wav2vec2|asr|transcribe|transcription|canary"):
        return {"asr"}
    if named(r"flux(?:\d+)?|sdxl|sd[123](?:\.\d+)?|stable-diffusion|dall-e|gpt-image|qwen-image|z-image|ideogram|imagen"):
        return {"image"}
    if named(r"veo|sora|kling|hunyuanvideo|ltx-video|wan(?:\d+(?:\.\d+)*)?"):
        return {"video"}
    return {"chat"}


_SFX_NAME_RE = re.compile(
    r"(?:^|[/_. :@-])(?:sfx|sound[-_ ]?effects?|foley|mmaudio|audiogen)(?:$|[/_. :@-])",
    re.IGNORECASE,
)
_MUSIC_NAME_RE = re.compile(
    r"(?:^|[/_. :@-])(?:ace[-_]?step|acestep|music|musicgen|song|diffrhythm|stable[-_]?audio|"
    r"lyria|heartmula|audioldm|suno|udio|yue)(?:$|[/_. :@-])",
    re.IGNORECASE,
)
_STT_NAME_RE = re.compile(
    r"(?:^|[/_. :@-])(?:whisper|distil-whisper|parakeet|wav2vec2|asr|stt|transcribe|transcription|canary)(?:$|[/_. :@-])",
    re.IGNORECASE,
)
_TTS_NAME_RE = re.compile(
    r"(?:^|[/_. :@-])(?:tts(?:-\d+)?|zonos\d*|ming(?:-\d+)?|kokoro|piper|chatterbox|xtts|"
    r"f5-tts|fish-speech|orpheus|speecht5|cosyvoice|bark|voice[-_]?clone)(?:$|[/_. :@-])",
    re.IGNORECASE,
)
_MUSIC_CLASS_HINTS = ("acestep", "ace_step", "musicgen", "diffrhythm", "stableaudio", "audioldm")
_SFX_CLASS_HINTS = ("mmaudio", "audiogen", "soundeffect", "foley")
_STT_CLASS_HINTS = ("whisper", "parakeet", "transcrib")
_MUSIC_PARAM_KEYS = frozenset({"lyrics", "lyrics_prompt", "bpm", "keyscale", "timesignature",
                               "time_signature", "instrumental", "force_instrumental"})
_TTS_PARAM_KEYS = frozenset({"voice", "speaking_rate", "reference_text", "ref_text",
                             "voice_id", "speaker", "speaker_name"})


def _workflow_classes(workflow: dict | None) -> list[str]:
    if not isinstance(workflow, dict):
        return []
    return [str(n.get("class_type") or "") for n in workflow.values()
            if isinstance(n, dict) and n.get("class_type")]


def _mapping_keys(candidate: dict) -> set[str]:
    mapping = candidate.get("mapping")
    if not isinstance(mapping, dict):
        return set()
    keys: set[str] = set()
    for p, m in mapping.items():
        if p:
            keys.add(str(p).strip().lower())
        if isinstance(m, dict):
            lbl = str(m.get("label") or "").strip().lower()
            if lbl:
                keys.add(lbl)
            fld = str(m.get("field") or "").strip().lower()
            if fld:
                keys.add(fld)
    return keys


def _wiro_categories(candidate: dict) -> set[str]:
    meta = candidate.get("model_metadata")
    if not isinstance(meta, dict):
        return set()
    raw = meta.get("categories")
    if isinstance(raw, str):
        items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = raw
    else:
        return set()
    out: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            val = item.get("slug") or item.get("name") or item.get("id") or ""
        else:
            val = item or ""
        s = str(val).strip().lower().replace("_", "-")
        if s:
            out.add(s)
    return out


def audio_task(candidate: dict, *, alias: str = "", workflow: dict | None = None) -> str | None:
    """Fine-grained audio task (`tts` | `music` | `stt` | `sfx`) for `/v1/models` and
    TTS schema discovery, or None for non-audio / unclassified models."""
    cand = candidate or {}
    for vendor in ("meshy", "tripo", "arliai", "nanogpt"):
        if vendor in cand:
            return None
    names = " ".join(s for s in (alias, str(cand.get("model") or "")) if s)
    if "lulabs" in cand:
        ep = str((cand.get("lulabs") or {}).get("endpoint") or "image")
        if ep == "speech":
            return "tts"
        if ep == "music":
            return "sfx" if _SFX_NAME_RE.search(names) else "music"
        return None
    if "venice" in cand:
        ep = str((cand.get("venice") or {}).get("endpoint") or "images")
        if ep == "speech":
            return "tts"
        if ep == "transcription":
            return "stt"
        if ep == "music":
            return "sfx" if _SFX_NAME_RE.search(names) else "music"
        return None
    if "wiro" in cand:
        ep = str((cand.get("wiro") or {}).get("endpoint") or "image")
        if ep == "speech":
            return "tts"
        if ep == "asr":
            return "stt"
        if ep == "audio":
            cats = _wiro_categories(cand)
            if _SFX_NAME_RE.search(names) or any("sfx" in c or "sound-effect" in c or "foley" in c for c in cats):
                return "sfx"
            if cats & {"music-generation", "text-to-music", "text-to-song", "image-to-song"} or _MUSIC_NAME_RE.search(names):
                return "music"
            if (cats & {"text-to-speech", "voice-clone"} or _TTS_NAME_RE.search(names)) and "speech-to-speech" not in cats:
                return "tts"
        return None
    task = str(cand.get("task") or "").lower().strip()
    if task in {"music", "text2music", "text-to-music", "song", "text2song", "music-generation"}:
        return "music"
    if task in {"sfx", "text2sfx", "sound-effects", "sound_effects", "sound-effect"}:
        return "sfx"
    if task in {"stt", "asr", "audio2text", "audio-to-text", "speech-to-text", "transcription"}:
        return "stt"
    if task in {"tts", "speech", "text2speech", "text-to-speech"}:
        return "tts"
    if task in {"text2audio", "audio", "audio2audio"}:
        wf = workflow if isinstance(workflow, dict) else (
            cand.get("workflow_json") if isinstance(cand.get("workflow_json"), dict) else None
        )
        classes = " ".join(_workflow_classes(wf)).lower()
        mkeys = _mapping_keys(cand)
        if _SFX_NAME_RE.search(names) or any(h in classes for h in _SFX_CLASS_HINTS):
            return "sfx"
        if (_MUSIC_NAME_RE.search(names)
                or any(h in classes for h in _MUSIC_CLASS_HINTS)
                or (bool(mkeys & _MUSIC_PARAM_KEYS) and not (mkeys & _TTS_PARAM_KEYS))):
            return "music"
        if _STT_NAME_RE.search(names) or any(h in classes for h in _STT_CLASS_HINTS):
            return "stt"
        if task == "text2audio":
            return "tts"
        if task == "audio" and _TTS_NAME_RE.search(names):
            return "tts"
    return None


def audio_task_from_types(kinds: set[str], model_id: str = "") -> str | None:
    """Audio task for LLM-catalog models and virtual aliases from their capability set."""
    if "tts" in kinds:
        return "tts"
    if "asr" in kinds:
        return "stt"
    if "audio" in kinds:
        if _SFX_NAME_RE.search(model_id or ""):
            return "sfx"
        if _MUSIC_NAME_RE.search(model_id or ""):
            return "music"
        # Multimodal chat models (e.g. gpt-audio) carry {"chat", "audio"} — not music.
        if "chat" not in kinds:
            return "music"
    return None


def generation_types(candidate: dict, *, alias: str = "", workflow: dict | None = None) -> set[str]:
    """Classify the output of one configured workflow/cloud generation candidate."""
    for vendor in ("meshy", "tripo"):
        if vendor in candidate:
            return {"3d"}
    if "lulabs" in candidate:
        # LU endpoints are `<kind>[-<op>]` (image-upscale, video-lipsync) plus speech / music.
        endpoint = str((candidate.get("lulabs") or {}).get("endpoint") or "image")
        return {"speech": {"tts"}, "music": {"audio"}}.get(endpoint) or (
            {"video"} if endpoint.startswith("video") else {"image"})
    for vendor in ("nanogpt", "venice", "wiro"):
        if vendor in candidate:
            endpoint = (candidate.get(vendor) or {}).get("endpoint") or "images"
            return _labels(endpoint) or {"image"}
    task = str(candidate.get("task") or "").lower().strip()
    if "mesh" in task or "3d" in task:
        return {"3d"}
    if "vid" in task:
        return {"video"}
    if task in {"audio2text", "audio-to-text", "speech-to-text", "stt", "asr", "transcription"}:
        return {"asr"}
    if task in {"audio2audio", "audio-to-audio", "audio", "music", "text2music", "text-to-music",
                "song", "text2song", "sfx", "text2sfx", "sound-effects", "sound_effects"}:
        return {"audio"}
    if task == "text2audio":
        at = audio_task(candidate, alias=alias, workflow=workflow)
        if at in {"music", "sfx"}:
            return {"audio"}
        if at == "stt":
            return {"asr"}
        return {"tts"}
    if "tts" in task or "audio" in task or "speech" in task:
        return {"tts"}
    return {"image"}
