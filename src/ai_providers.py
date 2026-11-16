"""Isolated text-generation and audio-transcription adapters for Shorts."""
import base64
import json
import os
import re
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path


def _post(url, headers, body):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"AI API HTTP {exc.code}: {detail}") from exc


def generate(provider, prompt):
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        model = os.getenv("OPENAI_MODEL", "gpt-6-luna")
        data = _post(
            "https://api.openai.com/v1/responses",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            {"model": model, "input": prompt, "max_output_tokens": 5000},
        )
        parts = []
        for item in data.get("output", []):
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    parts.append(part.get("text", ""))
        result = "\n".join(parts) or data.get("output_text", "")
        if not result:
            raise RuntimeError("OpenAI returned no text")
        return result, model

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": key, "Content-Type": "application/json"},
            {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 5000},
            },
        )
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        result = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        if not result:
            raise RuntimeError("Gemini returned no text")
        return result, model

    raise RuntimeError(f"Unknown AI provider: {provider}")


def _transcribe_openai(path):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    boundary = "----ShortsBoundary" + uuid.uuid4().hex
    chunks = []
    def field(name, value):
        chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    model = os.getenv("TRANSCRIPTION_MODEL", "whisper-1")
    field("model", model)
    field("response_format", "verbose_json")
    field("timestamp_granularities[]", "segment")
    chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: audio/mpeg\r\n\r\n".encode())
    chunks.extend([path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()])
    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions", data=b"".join(chunks),
        headers={"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"OpenAI transcription HTTP {exc.code}: {detail}") from exc
    segments = data.get("segments", [])
    if not segments and data.get("text"):
        segments = [{"start": 0, "end": 1, "text": data["text"]}]
    if not segments:
        raise RuntimeError("OpenAI returned no speech segments")
    return segments, model


def _duration_seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)s\s*", str(value))
    if not match:
        raise ValueError(f"Unrecognized Gemini word timestamp: {value}")
    return float(match.group(1))


_GEMINI_LAST_REQUEST_AT = None


def _gemini_generate_content(url, headers, body):
    """Respect Gemini RPM limits and wait/retry when a per-minute quota is hit."""
    global _GEMINI_LAST_REQUEST_AT
    rpm = int(os.getenv("GEMINI_REQUESTS_PER_MINUTE", "3"))
    if not 1 <= rpm <= 60:
        raise ValueError("GEMINI_REQUESTS_PER_MINUTE must be between 1 and 60.")
    interval = 60.0 / rpm + 1.0
    max_retries = int(os.getenv("GEMINI_RATE_LIMIT_RETRIES", "5"))

    for attempt in range(max_retries + 1):
        if _GEMINI_LAST_REQUEST_AT is not None:
            wait = interval - (time.monotonic() - _GEMINI_LAST_REQUEST_AT)
            if wait > 0:
                print(f"Gemini RPM pacing: waiting {wait:.1f}s before request.")
                time.sleep(wait)
        _GEMINI_LAST_REQUEST_AT = time.monotonic()
        try:
            return _post(url, headers, body)
        except RuntimeError as exc:
            message = str(exc)
            is_quota_429 = "HTTP 429" in message or "RESOURCE_EXHAUSTED" in message
            if not is_quota_429 or attempt >= max_retries:
                raise
            # Gemini may report RPM or rolling TPM exhaustion. Waiting a full
            # minute gives the rolling per-minute window time to replenish.
            retry_wait = 60
            print(
                f"Gemini quota limit reached; waiting {retry_wait}s "
                f"before retry {attempt + 1}/{max_retries}."
            )
            time.sleep(retry_wait)


def _transcribe_gemini(path):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.5-transcribe")
    audio_bytes = path.read_bytes()
    mime_type = "audio/mpeg"

    # Gemini recommends the Files API for audio longer than a few seconds.
    start_request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        data=json.dumps({"file": {"display_name": path.name}}).encode("utf-8"),
        headers={
            "x-goog-api-key": key,
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(start_request, timeout=120) as response:
            upload_url = response.headers.get("x-goog-upload-url")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Gemini file upload start HTTP {exc.code}: {detail}") from exc
    if not upload_url:
        raise RuntimeError("Gemini Files API did not return an upload URL")

    upload_request = urllib.request.Request(
        upload_url,
        data=audio_bytes,
        headers={
            "Content-Length": str(len(audio_bytes)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(upload_request, timeout=600) as response:
            uploaded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Gemini audio upload HTTP {exc.code}: {detail}") from exc
    file_uri = uploaded.get("file", {}).get("uri")
    if not file_uri:
        raise RuntimeError("Gemini Files API did not return an audio file URI")

    data = _gemini_generate_content(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {
            "contents": [{"role": "user", "parts": [
                {"fileData": {"fileUri": file_uri, "mimeType": mime_type}}
            ]}],
            "generationConfig": {
                "audioTranscriptionConfig": {"wordTimestamp": True}
            },
        },
    )
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    words = []
    for part in parts:
        transcription = part.get("audioTranscription", {})
        if not isinstance(transcription, dict):
            continue
        for word in transcription.get("words", []):
            spoken = str(word.get("word", "")).strip()
            if not spoken:
                continue
            words.append({
                "start": _duration_seconds(word.get("startOffset", "0s")),
                "end": _duration_seconds(word.get("endOffset", "0s")),
                "text": spoken,
            })
    if not words:
        raise RuntimeError("Gemini returned no word-level timestamps for the audio")
    # Combine word timestamps into short caption-sized segments.
    segments = []
    current = []
    segment_start = None
    segment_end = None
    for word in words:
        if segment_start is None:
            segment_start = word["start"]
            segment_end = word["end"]
            current = [word["text"]]
            continue
        if word["start"] - segment_end > 1.2 or word["end"] - segment_start > 4.5:
            segments.append({"start": segment_start, "end": segment_end, "text": " ".join(current)})
            segment_start = word["start"]
            current = [word["text"]]
        else:
            current.append(word["text"])
        segment_end = word["end"]
    if current:
        segments.append({"start": segment_start, "end": segment_end, "text": " ".join(current)})
    return segments, model



_LOCAL_WHISPER_MODEL = None
_LOCAL_WHISPER_MODEL_NAME = None


def _transcribe_local(path):
    """Run multilingual Whisper locally on the GitHub runner; no transcription API calls."""
    global _LOCAL_WHISPER_MODEL, _LOCAL_WHISPER_MODEL_NAME
    from faster_whisper import WhisperModel

    model_name = os.getenv("WHISPER_MODEL", "small")
    if _LOCAL_WHISPER_MODEL is None or _LOCAL_WHISPER_MODEL_NAME != model_name:
        print(f"Loading local Whisper model: {model_name}")
        _LOCAL_WHISPER_MODEL = WhisperModel(
            model_name,
            device="cpu",
            compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")),
            num_workers=1,
        )
        _LOCAL_WHISPER_MODEL_NAME = model_name

    detected_language = os.getenv("WHISPER_LANGUAGE", "").strip() or None
    segments, info = _LOCAL_WHISPER_MODEL.transcribe(
        str(path),
        language=detected_language,
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )
    rows = []
    for segment in segments:
        spoken = str(segment.text or "").strip()
        if spoken:
            rows.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": spoken,
            })
    if not rows:
        raise RuntimeError("Local Whisper returned no speech segments")
    language = getattr(info, "language", None) or "auto"
    return rows, f"faster-whisper-{model_name}-{language}"


def transcribe_audio(audio_path):
    """Use local Whisper by default; API transcription is opt-in only."""
    path = Path(audio_path)
    selected = os.getenv("TRANSCRIPTION_PROVIDER", "local").strip().lower()

    if selected in {"local", "whisper", "faster-whisper"}:
        try:
            segments, model = _transcribe_local(path)
            print(f"Audio transcription completed locally ({model})")
            return segments, "local", model
        except Exception as exc:
            if os.getenv("TRANSCRIPTION_ALLOW_API_FALLBACK", "false").lower() != "true":
                raise RuntimeError(
                    "Local Whisper transcription failed; API fallback is disabled. "
                    + str(exc)[:500]
                ) from exc
            print(f"Local Whisper failed; API fallback was explicitly enabled: {str(exc)[:300]}")

    if selected not in {"local", "whisper", "faster-whisper", "auto", "openai", "gemini"}:
        raise RuntimeError(f"Unknown TRANSCRIPTION_PROVIDER: {selected}")

    errors = []
    providers = (("openai", os.getenv("OPENAI_API_KEY"), _transcribe_openai),
                 ("gemini", os.getenv("GEMINI_API_KEY"), _transcribe_gemini))
    if selected in {"openai", "gemini"}:
        providers = tuple(item for item in providers if item[0] == selected)
    for provider, key, function in providers:
        if not key:
            errors.append(f"{provider}: API key is not configured")
            continue
        try:
            segments, model = function(path)
            print(f"Audio transcription completed with {provider} ({model})")
            return segments, provider, model
        except Exception as exc:
            message = str(exc).replace("\\n", " ")[:500]
            print(f"{provider} transcription failed; trying next provider: {message}")
            errors.append(f"{provider}: {message}")
    raise RuntimeError("Transcription providers failed. " + "; ".join(errors))
