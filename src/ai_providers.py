"""Isolated text-generation and audio-transcription adapters for Shorts."""
import base64
import json
import os
import re
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


def _parse_gemini_segments(text):
    cleaned = text.strip()
    try:
        obj = json.loads(cleaned)
        raw = obj.get("segments", []) if isinstance(obj, dict) else obj
        result = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                start = item.get("start", item.get("start_seconds"))
                end = item.get("end", item.get("end_seconds"))
                words = item.get("text", item.get("transcript", item.get("content", "")))
                if start is not None and end is not None and words:
                    result.append({"start": float(start), "end": float(end), "text": str(words)})
        if result:
            return result
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    pattern = re.compile(r"^\s*\[?(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\]?\s*(?:[-–]\s*\[?(\d{1,2}):(\d{2}):(\d{2}(?:\.\d+)?)\]?)?\s*[:\-]?\s*(.+?)\s*$")
    result = []
    for line in cleaned.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        h, m, s, h2, m2, s2, words = match.groups()
        start = int(h) * 3600 + int(m) * 60 + float(s)
        end = int(h2) * 3600 + int(m2) * 60 + float(s2) if h2 else start + 2
        if words.strip() and end > start:
            result.append({"start": start, "end": end, "text": words.strip()})
    return result


def _transcribe_gemini(path):
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_TRANSCRIPTION_MODEL", "gemini-3.5-transcribe")
    prompt = (
        "Transcribe this audio verbatim in the language spoken, primarily Hindi. "
        "Do not translate, summarize, correct, or omit words. Return only valid JSON "
        '{"segments":[{"start":0.0,"end":2.5,"text":"spoken words"}]}. '
        "All timestamps are seconds from the beginning of this audio chunk. "
        "Use accurate start/end times and short readable caption segments. No markdown."
    )
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {"contents": [{"role": "user", "parts": [
            {"text": prompt},
            {"inlineData": {"mimeType": "audio/mpeg",
                            "data": base64.b64encode(path.read_bytes()).decode("ascii")}},
        ]}],
         "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                              "maxOutputTokens": 16000,
                              "audioTranscriptionConfig": {"wordTimestamp": True}}},
    )
    candidates = data.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text_parts = [p["text"] for p in parts if p.get("text")]
    for part in parts:
        transcription = part.get("audioTranscription", {})
        if isinstance(transcription, dict) and transcription.get("text"):
            text_parts.append(transcription["text"])
    segments = _parse_gemini_segments("\n".join(text_parts))
    if not segments:
        raise RuntimeError("Gemini returned no usable timestamped transcript; timings were not guessed")
    return segments, model


def transcribe_audio(audio_path):
    """Try OpenAI first and automatically fall back to Gemini for audio transcription."""
    path = Path(audio_path)
    errors = []
    providers = (("openai", os.getenv("OPENAI_API_KEY"), _transcribe_openai),
                 ("gemini", os.getenv("GEMINI_API_KEY"), _transcribe_gemini))
    for provider, key, function in providers:
        if not key:
            errors.append(f"{provider}: API key is not configured")
            continue
        try:
            segments, model = function(path)
            print(f"Audio transcription completed with {provider} ({model})")
            return segments, provider, model
        except Exception as exc:
            message = str(exc).replace("\n", " ")[:500]
            print(f"{provider} transcription failed; trying next provider: {message}")
            errors.append(f"{provider}: {message}")
    raise RuntimeError("Both transcription providers failed. " + "; ".join(errors))
