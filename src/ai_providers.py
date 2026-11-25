"""Small, credential-safe text generation adapters for the three SBV reviewers."""
import json
import os
import urllib.error
import urllib.request
import time
import re

# Conservative per-provider pacing: one request per minute by default.
_MIN_INTERVAL_SECONDS = max(1, int(os.getenv("AI_MIN_INTERVAL_SECONDS", "60")))
_MAX_RETRIES = max(0, int(os.getenv("AI_MAX_RETRIES", "3")))
_MAX_RETRY_WAIT_SECONDS = max(60, int(os.getenv("AI_MAX_RETRY_WAIT_SECONDS", "900")))
_LAST_CALL = {}


def _post(url, headers, body, timeout=300):
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1200]
        retry = exc.headers.get("Retry-After", "")
        suffix = f"; Retry-After: {retry}" if retry else ""
        raise RuntimeError(f"AI API HTTP {exc.code}{suffix}: {detail}") from exc


def _generate_once(provider, prompt):
    """Make one provider API request; return (text, model, usage)."""
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        model = os.getenv("OPENAI_MODEL") or "gpt-6-luna"
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
        return result, model, data.get("usage", {})

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        model = os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
        data = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"x-goog-api-key": key, "Content-Type": "application/json"},
            {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
             "generationConfig": {"maxOutputTokens": 5000}},
        )
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        result = "\n".join(p.get("text", "") for p in parts if p.get("text"))
        if not result:
            raise RuntimeError("Gemini returned no text")
        return result, model, data.get("usageMetadata", {})

    if provider == "cloudflare":
        token = os.getenv("CLOUDFLARE_API_TOKEN")
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account_id:
            raise RuntimeError("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID are required")
        model = os.getenv("CLOUDFLARE_MODEL") or "@cf/meta/llama-3.2-1b-instruct"
        data = _post(
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}",
            {"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            {"messages": [
                {"role": "system", "content": "You are an independent transcript reviewer. Be factual and careful."},
                {"role": "user", "content": prompt},
            ]},
        )
        if not data.get("success", False):
            raise RuntimeError(f"Cloudflare Workers AI returned errors: {data.get('errors', [])}")
        result = (data.get("result") or {}).get("response", "")
        if not result:
            raise RuntimeError("Cloudflare Workers AI returned no text")
        return result, model, {}

    raise RuntimeError(f"Unknown AI provider: {provider}")


def generate(provider, prompt):
    """Pace each provider independently and retry transient/rate-limit failures."""
    for attempt in range(_MAX_RETRIES + 1):
        previous = _LAST_CALL.get(provider)
        if previous is not None:
            wait = _MIN_INTERVAL_SECONDS - (time.monotonic() - previous)
            if wait > 0:
                print(f"{provider}: pacing limit; waiting {int(wait + 0.99)}s before next request.", flush=True)
                time.sleep(wait)
        _LAST_CALL[provider] = time.monotonic()
        try:
            return _generate_once(provider, prompt)
        except RuntimeError as exc:
            message = str(exc)
            match = re.search(r"AI API HTTP (429|500|502|503|504)", message)
            if not match or attempt >= _MAX_RETRIES:
                raise
            retry_match = re.search(r"Retry-After: ([0-9]+)", message)
            retry_wait = int(retry_match.group(1)) if retry_match else min(60 * (2 ** attempt), 300)
            if retry_wait > _MAX_RETRY_WAIT_SECONDS:
                raise RuntimeError(
                    f"{provider} requested a {retry_wait}s cooldown; this exceeds the "
                    f"configured safe wait ({_MAX_RETRY_WAIT_SECONDS}s). Stopping rather "
                    "than sending requests during the provider's cooldown."
                ) from exc
            print(
                f"{provider}: transient HTTP {match.group(1)}; retrying after {retry_wait}s "
                f"(attempt {attempt + 2}/{_MAX_RETRIES + 1}).",
                flush=True,
            )
            time.sleep(retry_wait)
    raise RuntimeError(f"{provider}: retry loop ended unexpectedly.")
