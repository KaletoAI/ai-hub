"""Catalog capabilities shared by API filters and console pickers, not routing.

Provider declarations take precedence over conservative legacy name recognition.
Input vision is still chat; output modalities can give a model several types.
"""
from __future__ import annotations

import re

MEDIA_TYPES = frozenset({"image", "video", "3d", "tts", "audio", "asr"})
MODEL_TYPES = MEDIA_TYPES | {"chat", "embedding", "rerank"}

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


def generation_types(candidate: dict) -> set[str]:
    """Classify the output of one configured workflow/cloud generation candidate."""
    for vendor in ("meshy", "tripo"):
        if vendor in candidate:
            return {"3d"}
    for vendor in ("nanogpt", "venice"):
        if vendor in candidate:
            endpoint = (candidate.get(vendor) or {}).get("endpoint") or "images"
            return _labels(endpoint) or {"image"}
    task = str(candidate.get("task") or "").lower()
    if "mesh" in task or "3d" in task:
        return {"3d"}
    if "vid" in task:
        return {"video"}
    if task in {"audio2text", "audio-to-text", "speech-to-text"}:
        return {"asr"}
    if task in {"audio2audio", "audio-to-audio", "audio", "music"}:
        return {"audio"}
    if "tts" in task or "audio" in task or "speech" in task:
        return {"tts"}
    return {"image"}
