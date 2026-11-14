"""Isolated text-generation and audio-transcription adapters for Shorts."""
import json
import os
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


def transcribe_audio(audio_path):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for speech-to-text from uploaded video audio."
        )
    path = Path(audio_path)
    boundary = "----ShortsBoundary" + uuid.uuid4().hex
    chunks = []

    def field(name, value):
        chunks.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode()
        )

    field("model", os.getenv("TRANSCRIPTION_MODEL", "whisper-1"))
    field("response_format", "verbose_json")
    field("timestamp_granularities[]", "segment")
    chunks.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\nContent-Type: audio/mpeg\r\n\r\n".encode()
    )
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        "https://api.openai.com/v1/audio/transcriptions",
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Speech transcription API HTTP {exc.code}: {detail}") from exc
    segments = data.get("segments", [])
    if not segments and data.get("text"):
        segments = [{"start": 0, "end": 1, "text": data["text"]}]
    return segments
