"""Approval-first Shorts preparation from a user-uploaded source video."""
import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
import unicodedata
from pathlib import Path

from src.ai_providers import generate, transcribe_audio, provider_error_kind, retry_after_seconds

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "shorts_work"


def run(args, cwd=None, timeout=1800):
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True,
                          capture_output=True, timeout=timeout)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout)[-2500:])
    return proc.stdout.strip()


def timecode(seconds):
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def get_video_duration(source):
    value = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(source)
    ])
    return float(value)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def transcribe(source, resume_dir=None):
    audio_dir = WORK / "audio_chunks"
    audio_dir.mkdir(parents=True)
    chunk_seconds = int(os.getenv("TRANSCRIPTION_CHUNK_SECONDS", "600"))
    if not 60 <= chunk_seconds <= 600:
        raise ValueError("TRANSCRIPTION_CHUNK_SECONDS must be between 60 and 600.")
    pattern = audio_dir / "part_%03d.mp3"
    print(
        f"STEP 1/5: Extracting audio into {chunk_seconds // 60}-minute parts...",
        flush=True,
    )
    run([
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1",
        "-ar", "16000", "-b:a", "64k", "-f", "segment",
        "-segment_time", str(chunk_seconds), "-reset_timestamps", "1",
        str(pattern)
    ], timeout=3600)
    audio_files = sorted(audio_dir.glob("part_*.mp3"))
    print(f"Audio extraction complete: {len(audio_files)} parts found.", flush=True)
    if not audio_files:
        raise RuntimeError("Could not extract audio for multilingual transcription.")
    rows = []
    transcription_providers = []
    transcription_models = []
    checkpoint_dir = WORK / "whisper_chunks"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    restored_dir = Path(resume_dir) / "whisper_chunks" if resume_dir else None
    import hashlib
    source_name = Path(source).name
    source_size = Path(source).stat().st_size
    source_hash = hashlib.sha256()
    with Path(source).open("rb") as source_stream:
        for block in iter(lambda: source_stream.read(1024 * 1024), b""):
            source_hash.update(block)
    source_sha256 = source_hash.hexdigest()
    for index, audio in enumerate(audio_files):
        offset = index * chunk_seconds
        print(
            f"TRANSCRIPTION: starting part {index + 1}/{len(audio_files)} "
            f"({offset // 60}-{(offset + chunk_seconds) // 60} min of source video).",
            flush=True,
        )
        checkpoint_file = checkpoint_dir / f"part_{index + 1:03}.json"
        restored_file = restored_dir / checkpoint_file.name if restored_dir else None
        saved_part = None
        if restored_file and restored_file.is_file():
            saved_part = json.loads(restored_file.read_text(encoding="utf-8"))
            if (saved_part.get("chunk_seconds") != chunk_seconds
                    or saved_part.get("part_index") != index + 1
                    or saved_part.get("source_video_name") != source_name
                    or saved_part.get("source_video_size_bytes") != source_size
                    or saved_part.get("source_video_sha256") != source_sha256):
                raise RuntimeError(f"Whisper checkpoint belongs to a different source or settings: {restored_file.name}")
            _atomic_json(checkpoint_file, saved_part)
            print(f"WHISPER RESUME: restored and re-saved part {index + 1}/{len(audio_files)}.", flush=True)
        elif checkpoint_file.is_file():
            saved_part = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            print(f"WHISPER RESUME: reusing local saved part {index + 1}/{len(audio_files)}.", flush=True)

        if saved_part is None:
            segments, provider, model = transcribe_audio(audio)
            part_rows = []
            for segment in segments or []:
                text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", str(segment.get("text", ""))).strip())
                if not text:
                    continue
                start = float(segment.get("start", 0)) + offset
                end = float(segment.get("end", start - offset)) + offset
                part_rows.append({"start": start, "duration": max(0.1, end - start), "text": text})
            saved_part = {
                "version": 1, "part_index": index + 1, "chunk_seconds": chunk_seconds,
                "source_audio": audio.name, "source_offset_seconds": offset,
                "source_video_name": source_name, "source_video_size_bytes": source_size,
                "source_video_sha256": source_sha256,
                "provider": provider, "model": model, "has_speech": bool(part_rows),
                "segments": part_rows,
            }
            _atomic_json(checkpoint_file, saved_part)
            print(f"WHISPER CHECKPOINT SAVED: {checkpoint_file.name} ({len(part_rows)} segments).", flush=True)
        provider = saved_part.get("provider", "restored-checkpoint")
        model = saved_part.get("model", "restored-checkpoint")
        transcription_providers.append(provider)
        transcription_models.append(model)
        part_rows = saved_part.get("segments", [])
        if not part_rows:
            print(
                f"TRANSCRIPTION: part {index + 1}/{len(audio_files)} contains no detected speech; "
                "saved as an empty/silent part and continuing.",
                flush=True,
            )
            continue
        print(
            f"TRANSCRIPTION: part {index + 1}/{len(audio_files)} complete "
            f"({len(part_rows)} speech segments).",
            flush=True,
        )
        rows.extend(part_rows)
    if not rows:
        raise RuntimeError(
            "No speech was detected in any audio part. Check that the source video contains audible speech."
        )
    print(
        f"TRANSCRIPTION COMPLETE: {len(rows)} timestamped speech segments collected; "
        f"{len(transcription_providers)} audio parts processed.",
        flush=True,
    )

    # Merge short ASR chunks into complete utterances. Boundaries are created only
    # after sentence punctuation or a meaningful pause, never at arbitrary 4-5s chunks.
    utterances = []
    current = None
    for row in rows:
        row_start = float(row["start"])
        row_end = row_start + float(row["duration"])
        if current is None:
            current = {"start": row_start, "end": row_end, "text": row["text"]}
            continue
        pause = row_start - current["end"]
        if pause >= 0.85:
            utterances.append({
                "start": current["start"],
                "duration": max(0.1, current["end"] - current["start"]),
                "text": unicodedata.normalize("NFC", current["text"]).strip(),
            })
            current = {"start": row_start, "end": row_end, "text": row["text"]}
            continue
        current["text"] += " " + row["text"]
        current["end"] = row_end
        if re.search(r"[।!?…][\"')\]]?$", row["text"]):
            utterances.append({
                "start": current["start"],
                "duration": max(0.1, current["end"] - current["start"]),
                "text": unicodedata.normalize("NFC", current["text"]).strip(),
            })
            current = None
    if current is not None:
        utterances.append({
            "start": current["start"],
            "duration": max(0.1, current["end"] - current["start"]),
            "text": unicodedata.normalize("NFC", current["text"]).strip(),
        })
    print(f"Transcript assembled into {len(utterances)} utterances.", flush=True)
    return utterances, sorted(set(transcription_providers)), sorted(set(transcription_models))


def select(rows, count, resume_state=None, transcription_providers=None, transcription_models=None):
    """Review transcript batches across OpenAI and Gemini, checkpointing every batch."""
    import hashlib
    import time

    # Gemini is the preferred first provider; OpenAI is the immediate fallback.
    providers = [
        ("gemini", "GEMINI_API_KEY"),
        ("openai", "OPENAI_API_KEY"),
    ]
    providers = [(name, key) for name, key in providers if os.getenv(key)]
    if not providers:
        raise RuntimeError("Neither OPENAI_API_KEY nor GEMINI_API_KEY is configured.")

    # Keep an immutable raw transcript for checkpoint identity. Corrected caption text
    # is stored separately so a resumed run can safely restore it without changing the hash.
    raw_rows = [dict(row) for row in rows]
    boundaries = sorted(set(round(float(x), 2) for row in raw_rows
                            for x in (row["start"], row["start"] + row["duration"])))
    transcript_rows = [
        f"ROW_ID={index} [{row['start']:.2f}s - {row['start'] + row['duration']:.2f}s] {row['text']}"
        for index, row in enumerate(raw_rows)
    ]
    signature = hashlib.sha256("\n".join(transcript_rows).encode("utf-8")).hexdigest()

    # Keep batches comfortably below the previous single 65k-character request.
    batches, current, current_size = [], [], 0
    for line in transcript_rows:
        if current and current_size + len(line) > 12000:
            batches.append(current)
            current, current_size = [], 0
        current.append(line)
        current_size += len(line)
    if current:
        batches.append(current)

    checkpoint_path = WORK / "ai_selection_checkpoint.json"
    checkpoint = {
        "version": 1,
        "transcript_sha256": signature,
        "total_batches": len(batches),
        "completed_batches": {},
        "token_usage": {"openai": {"input": 0, "output": 0, "total": 0},
                        "gemini": {"input": 0, "output": 0, "total": 0}},
        "provider_models": {},
        "transcript_rows": raw_rows,
        "transcription_providers": transcription_providers or [],
        "transcription_models": transcription_models or [],
    }
    if resume_state and Path(resume_state).is_file():
        saved = json.loads(Path(resume_state).read_text(encoding="utf-8"))
        if saved.get("transcript_sha256") != signature:
            raise RuntimeError("Resume checkpoint belongs to a different transcript; refusing to mix videos.")
        if int(saved.get("version", 0)) != 1:
            raise RuntimeError("Unsupported AI resume checkpoint version.")
        checkpoint.update(saved)
        print(
            f"RESUME: restoring {len(checkpoint.get('completed_batches', {}))}/"
            f"{len(batches)} completed transcript batches.",
            flush=True,
        )

    completed = checkpoint.setdefault("completed_batches", {})
    # Older checkpoints predate Hindi proofreading. Re-run those batches instead of
    # silently producing captions from their uncorrected Whisper transcript.
    for batch_key in list(completed):
        saved_batch = completed.get(batch_key, {})
        if not isinstance(saved_batch.get("corrected_rows"), list):
            print(f"RESUME: batch {int(batch_key) + 1} has no Hindi-proofreading result; it will be reviewed again.", flush=True)
            del completed[batch_key]
    for batch_index, batch in enumerate(batches):
        saved_batch = completed.get(str(batch_index))
        if saved_batch:
            corrected = saved_batch["corrected_rows"]
            if len(corrected) != len(batch):
                raise RuntimeError(f"Saved Hindi proofreading row count mismatch in batch {batch_index + 1}.")
            start_index = sum(len(part) for part in batches[:batch_index])
            for offset, corrected_text in enumerate(corrected):
                rows[start_index + offset]["text"] = corrected_text
    usage_totals = checkpoint.setdefault("token_usage", {})
    disabled_today = set()  # Daily quota resets are checked afresh on each new run.
    disabled_permanently = set()
    cooldown_until = {}

    def save_checkpoint():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint["updated_at_utc"] = __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat()
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def parse_response(raw):
        raw = raw.strip()
        if raw.startswith(chr(96) * 3):
            raw = "\n".join(raw.splitlines()[1:-1])
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            left, right = raw.find("{"), raw.rfind("}")
            if left < 0 or right <= left:
                raise ValueError("AI response did not contain JSON.")
            obj = json.loads(raw[left:right + 1])
        if not isinstance(obj, dict) or not isinstance(obj.get("clips"), list):
            raise ValueError("AI response JSON did not contain a clips list.")
        if not isinstance(obj.get("corrected_rows"), list):
            raise ValueError("AI response omitted mandatory corrected_rows Hindi proofreading.")
        return obj

    # Persist the full Whisper transcript before making the first paid API call.
    save_checkpoint()

    for batch_index, batch in enumerate(batches):
        batch_key = str(batch_index)
        if batch_key in completed:
            continue
        transcript = "\n".join(batch)
        prompt = f"""You are a specialist Hindi/Devanagari proofreader and senior editor of Hindu scripture discourses.
This batch has TWO mandatory tasks in one pass.

TASK A — PROOFREAD EVERY TRANSCRIPT ROW for Hindi spelling and clear speech-recognition errors.
- Preserve the speaker's exact spoken meaning, order, tone and wording. Correct spelling, matras, conjuncts and obvious ASR substitutions.
- Use context to recognize Sanskrit/Hindu terms and proper nouns, e.g. भगवान, भगवद्गीता, शिवपुराण, रुद्र संहिता, सांख्ययोग, समबुद्धि, निष्काम कर्म, अर्जुन, भीष्म, क्षत्रिय, धर्म, पुण्य, पाप, ज्ञान, वैराग्य, भक्ति, सनत्कुमार, जलंधर, विष्णु, महादेव, पार्वती.
- Do NOT paraphrase, summarize, modernize, add explanations, invent words, or complete speech that is not present.
- If a word cannot be confidently heard from the transcript context, leave that word unchanged; do not guess.
- Return exactly one corrected text for EVERY supplied ROW_ID, preserving row order and IDs. Keep timestamps outside the corrected text; never alter timestamps.
- Keep each corrected row concise and close in length to its source row.

TASK B — Select up to 3 distinct, self-contained Shorts from the corrected transcript.
Mandatory: compelling natural hook AND complete religious thought/context. Never sacrifice scriptural meaning for a hook. Do not cut a sentence, phrase, word, or thought in half. Use only exact transcript row start/end timestamps.
Duration: prefer 10-15 seconds; if no complete thought fits, try 20-25, then 26-40, then 41-60 seconds. Maximum 60 seconds. Completeness is more important than short duration. Avoid overlapping/repeated topics.
Score honestly 1-10: hook strength, context completeness, natural boundaries, Hindi/caption quality, technical suitability. Return only candidates scoring at least 8/10 and explain context completeness.

Return ONLY valid JSON in this exact structure:
{{"corrected_rows":[{{"id":0,"text":"सुधारा हुआ मूल वाक्य"}}],"clips":[{{"start_seconds":12.3,"end_seconds":25.4,"hook":"हिन्दी हुक","title":"हिन्दी शीर्षक","description":"विवरण","hashtags":["#..."],"tags":["..."],"reason":"हुक और प्रसंग की पूर्णता का कारण","quality_score":9,"score_breakdown":{{"completeness":2,"boundaries":2,"hook":2,"captions":2,"technical":1}},"improvements":"सुधार","duration_tier":"10-15"}}]}}
All corrected_rows.text and clip text fields must be in Hindi. Timestamps are source-video seconds. If no suitable complete moment exists, return an empty clips list but still proofread every row.
TRANSCRIPT BATCH {batch_index + 1}/{len(batches)}:
{transcript}"""

        batch_done = False
        while not batch_done:
            now = time.monotonic()
            eligible = [
                (name, key) for name, key in providers
                if name not in disabled_today
                and name not in disabled_permanently
                and now >= cooldown_until.get(name, 0)
            ]

            if not eligible:
                active = [
                    name for name, _ in providers
                    if name not in disabled_today and name not in disabled_permanently
                ]
                if not active:
                    save_checkpoint()
                    done_count = len(completed)
                    if len(disabled_today) == len(providers):
                        raise RuntimeError(
                            f"AI_DAILY_QUOTA_EXHAUSTED: both configured providers reached their daily "
                            f"quota. Saved {done_count}/{len(batches)} batches in "
                            f"{checkpoint_path}; resume after a provider quota resets."
                        )
                    raise RuntimeError(
                        "AI_PROVIDERS_UNAVAILABLE: all configured providers returned permanent "
                        "configuration/request errors. Check API keys, permissions, model names, "
                        "billing and request settings. Checkpoint saved."
                    )

                next_wait = min(
                    max(0, cooldown_until.get(name, now) - now) for name in active
                )
                if next_wait > 1200:
                    save_checkpoint()
                    raise RuntimeError(
                        f"AI_RETRY_DEFERRED: all remaining providers are cooling down for more "
                        f"than 20 minutes (earliest retry in about {int(next_wait)} seconds). "
                        f"Saved {len(completed)}/{len(batches)} batches; resume this run later."
                    )
                print(
                    "AI PROVIDERS TEMPORARILY UNAVAILABLE: both providers have been tried. "
                    "Saving progress and waiting 20 minutes before restarting with Gemini.",
                    flush=True,
                )
                save_checkpoint()
                time.sleep(1200)
                continue

            for provider, key in eligible:
                try:
                    print(
                        f"AI REVIEW: batch {batch_index + 1}/{len(batches)} "
                        f"({len(batch)} transcript rows), provider={provider}.",
                        flush=True,
                    )
                    raw, model, usage = generate(provider, prompt)
                    checkpoint.setdefault("provider_models", {})[provider] = model
                    provider_usage = usage_totals.setdefault(
                        provider, {"input": 0, "output": 0, "total": 0}
                    )
                    for field in ("input", "output", "total"):
                        value = usage.get(field)
                        if isinstance(value, (int, float)):
                            provider_usage[field] = provider_usage.get(field, 0) + int(value)
                    print(
                        f"AI USAGE: {provider}; input={usage.get('input') if usage.get('input') is not None else 'not reported'}, "
                        f"output={usage.get('output') if usage.get('output') is not None else 'not reported'}, "
                        f"total={usage.get('total') if usage.get('total') is not None else 'not reported'} tokens "
                        f"(API-reported={usage.get('reported_by_api', False)}).",
                        flush=True,
                    )
                    response_obj = parse_response(raw)
                    corrected_items = response_obj["corrected_rows"]
                    expected_ids = list(range(
                        sum(len(part) for part in batches[:batch_index]),
                        sum(len(part) for part in batches[:batch_index + 1])
                    ))
                    corrected_map = {}
                    for entry in corrected_items:
                        if not isinstance(entry, dict) or not isinstance(entry.get("id"), int) or not isinstance(entry.get("text"), str):
                            raise ValueError("Hindi proofreading returned a malformed row; refusing unverified captions.")
                        row_id = entry["id"]
                        corrected_text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", entry["text"]).strip())
                        if row_id in corrected_map or not corrected_text:
                            raise ValueError("Hindi proofreading returned duplicate IDs or empty caption text.")
                        original_text = raw_rows[row_id]["text"]
                        if len(corrected_text) > max(80, len(original_text) * 2):
                            raise ValueError(f"Hindi proofreading expanded row {row_id} too much; refusing possible paraphrase.")
                        corrected_map[row_id] = corrected_text
                    if sorted(corrected_map) != expected_ids:
                        raise ValueError("Hindi proofreading did not return exactly one correction for every row.")
                    corrected_rows = [corrected_map[row_id] for row_id in expected_ids]
                    batch_start_index = expected_ids[0]
                    for offset, corrected_text in enumerate(corrected_rows):
                        rows[batch_start_index + offset]["text"] = corrected_text
                    valid = []
                    for item in response_obj["clips"]:
                        try:
                            s0, e0 = float(item["start_seconds"]), float(item["end_seconds"])
                            s = min(boundaries, key=lambda x: abs(x - s0))
                            e = min(boundaries, key=lambda x: abs(x - e0))
                            score = float(item.get("quality_score", 0))
                            if (abs(s-s0) > 1.25 or abs(e-e0) > 1.25 or e <= s
                                    or not 10 <= e-s <= 60 or not 8 <= score <= 10
                                    or not item.get("title") or not item.get("hook")):
                                continue
                            item["start_seconds"], item["end_seconds"] = round(s, 2), round(e, 2)
                            item["quality_score"] = score
                            item["selected_by"] = provider
                            item["source_batch"] = batch_index + 1
                            valid.append(item)
                        except (KeyError, TypeError, ValueError):
                            continue
                    completed[batch_key] = {
                        "provider": provider,
                        "model": model,
                        "candidate_count": len(valid),
                        "transcript_batch": transcript,
                        "raw_response": raw,
                        "corrected_rows": corrected_rows,
                        "clips": valid,
                    }
                    save_checkpoint()
                    print(
                        f"AI REVIEW COMPLETE: batch {batch_index + 1}/{len(batches)}; "
                        f"{len(valid)} valid candidate(s) saved.",
                        flush=True,
                    )
                    batch_done = True
                    break
                except Exception as exc:
                    kind = provider_error_kind(exc)
                    message = str(exc).replace("\n", " ")[:700]
                    if kind == "daily":
                        disabled_today.add(provider)
                        print(
                            f"AI DAILY QUOTA EXHAUSTED: {provider}; disabling it for this run. "
                            f"Detail: {message}",
                            flush=True,
                        )
                    elif kind in ("permanent", "payload"):
                        disabled_permanently.add(provider)
                        print(
                            f"AI PROVIDER CONFIGURATION/REQUEST ERROR: {provider}; not retrying "
                            f"this provider in this run. Check key, permission, model, billing, "
                            f"or request size. Detail: {message}",
                            flush=True,
                        )
                    else:
                        retry_after = retry_after_seconds(exc)
                        cooldown = max(120, retry_after or 120)
                        cooldown_until[provider] = time.monotonic() + cooldown
                        print(
                            f"AI TEMPORARY ERROR: {provider}; next attempt for this provider "
                            f"will wait at least {cooldown} seconds. Switching provider. "
                            f"Detail: {message}",
                            flush=True,
                        )
                    save_checkpoint()
                    # Give the alternate provider a two-minute hand-off window.
                    if not batch_done:
                        time.sleep(120)

            if not batch_done:
                save_checkpoint()
                if len(disabled_today) == len(providers):
                    raise RuntimeError(
                        f"AI_DAILY_QUOTA_EXHAUSTED: both configured providers reached their daily "
                        f"quota. Saved {len(completed)}/{len(batches)} batches in "
                        f"{checkpoint_path}; resume after a provider quota resets."
                    )
                if all(
                    name in disabled_today or name in disabled_permanently
                    for name, _ in providers
                ):
                    raise RuntimeError(
                        "AI_PROVIDERS_UNAVAILABLE: all configured providers returned permanent "
                        "or daily-quota errors. Checkpoint saved."
                    )

                active = [
                    name for name, _ in providers
                    if name not in disabled_today and name not in disabled_permanently
                ]
                next_wait = min(
                    max(0, cooldown_until.get(name, time.monotonic()) - time.monotonic())
                    for name in active
                )
                if next_wait > 1200:
                    raise RuntimeError(
                        f"AI_RETRY_DEFERRED: all remaining providers are cooling down for more "
                        f"than 20 minutes (earliest retry in about {int(next_wait)} seconds). "
                        f"Saved {len(completed)}/{len(batches)} batches; resume this run later."
                    )
                print(
                    "AI PROVIDERS TEMPORARILY UNAVAILABLE: both providers have been tried. "
                    "Waiting 20 minutes, then retrying Gemini-first.",
                    flush=True,
                )
                time.sleep(1200)

    candidates = []
    for result in completed.values():
        candidates.extend(result.get("clips", []))
    candidates.sort(key=lambda item: float(item.get("quality_score", 0)), reverse=True)
    chosen = []
    for item in candidates:
        s, e = item["start_seconds"], item["end_seconds"]
        if any(s < old["end_seconds"] and e > old["start_seconds"] for old in chosen):
            continue
        chosen.append(item)
        if len(chosen) == count:
            break
    if not chosen:
        raise RuntimeError("Neither OpenAI nor Gemini found a valid complete Short in the transcript.")
    provider_summary = ", ".join(
        f"{name}:{len([v for v in completed.values() if v.get('provider') == name])} batches"
        for name, _ in providers
    )
    model_summary = ", ".join(
        f"{name}={model}" for name, model in checkpoint.get("provider_models", {}).items()
    )
    save_checkpoint()
    print(
        f"AI REVIEW SUMMARY: {len(completed)}/{len(batches)} batches complete; "
        f"{len(candidates)} candidates; {len(chosen)} final selection(s).",
        flush=True,
    )
    return {"clips": chosen}, provider_summary, model_summary

def create(file_path, count, minimum, maximum, allow_short_test=False, resume_state=None):
    source_input = Path(file_path)
    if not source_input.is_file():
        raise FileNotFoundError(f"Uploaded source video not found: {source_input}")
    if source_input.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported video type. Upload MP4, MOV, MKV, WEBM, or M4V.")
    if not 1 <= count <= 10 or minimum != 10 or maximum != 60:
        raise ValueError("Count 1–10; adaptive duration policy is 10–60 seconds.")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    source_dir = WORK / "source"
    source_dir.mkdir()
    source = source_dir / ("source" + source_input.suffix.lower())
    shutil.copy2(source_input, source)

    duration = get_video_duration(source)
    if duration <= 600 and not allow_short_test:
        raise ValueError("Source video must be longer than 10 minutes. Use --allow-short-test only for a short test source.")
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(source)
    ]))
    stream = probe.get("streams", [{}])[0]
    source_width = int(stream.get("width", 0))
    source_height = int(stream.get("height", 0))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Could not determine source video dimensions.")
    is_horizontal = source_width > source_height

    print(f"SOURCE: {source.name}; duration {duration / 60:.1f} minutes.", flush=True)
    resume_data = None
    if resume_state and Path(resume_state).is_file():
        resume_data = json.loads(Path(resume_state).read_text(encoding="utf-8"))
    if resume_data and isinstance(resume_data.get("transcript_rows"), list) and resume_data["transcript_rows"]:
        rows = resume_data["transcript_rows"]
        transcription_providers = resume_data.get("transcription_providers", ["restored-checkpoint"])
        transcription_models = resume_data.get("transcription_models", ["restored-checkpoint"])
        print(
            f"RESUME: restored {len(rows)} timestamped transcript rows; "
            "skipping Whisper re-transcription.",
            flush=True,
        )
    else:
        print("STEP 1/5: Starting audio transcription.", flush=True)
        resume_dir = Path(resume_state).parent if resume_state else None
        rows, transcription_providers, transcription_models = transcribe(source, resume_dir=resume_dir)
    print("STEP 2/5: Transcript ready; starting clip selection.", flush=True)
    selected, provider, model = select(
        rows, count, resume_state=resume_state,
        transcription_providers=transcription_providers,
        transcription_models=transcription_models,
    )
    print(f"CLIP SELECTION COMPLETE: {len(selected.get('clips', []))} candidates returned.", flush=True)
    clips = []
    for item in selected["clips"]:
        try:
            start, end = float(item["start_seconds"]), float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if (start < 0 or end > duration or not 10 <= end - start <= 60
                or float(item.get("quality_score", 0)) < 8
                or not item.get("title") or not item.get("hook")):
            continue
        clips.append({
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "duration_seconds": round(end - start, 2),
            "quality_score": float(item.get("quality_score", 0)),
            "score_breakdown": item.get("score_breakdown", {}),
            "improvements": str(item.get("improvements", ""))[:500],
            "duration_tier": str(item.get("duration_tier", ""))[:20],
            "hook": str(item["hook"])[:180],
            "title": str(item["title"])[:100],
            "description": str(item.get("description", ""))[:1600],
            "hashtags": item.get("hashtags", [])[:15],
            "tags": item.get("tags", [])[:30],
            "reason": str(item.get("reason", ""))[:400],
            "selected_by": str(item.get("selected_by", ""))[:30],
        })
        if len(clips) >= count:
            break
    if not clips:
        raise RuntimeError("AI returned no valid timestamp selections.")

    output = WORK / "shorts"
    output.mkdir()
    print(f"STEP 3/5: Rendering {len(clips)} Shorts for review.", flush=True)
    for index, clip in enumerate(clips, 1):
        print(
            f"RENDER: Short {index}/{len(clips)}; source "
            f"{clip['start_seconds']:.1f}s–{clip['end_seconds']:.1f}s.",
            flush=True,
        )
        subtitle = output / f"short_{index:02}.srt"
        cues = []
        for row in rows:
            begin = row["start"]
            finish = row["start"] + row["duration"]
            # Keep each transcript caption whole; do not clip a phrase at edit boundaries.
            if begin >= clip["start_seconds"] - 0.05 and finish <= clip["end_seconds"] + 0.05:
                cues.append(
                    f"{len(cues) + 1}\n"
                    f"{timecode(begin - clip['start_seconds'])} --> "
                    f"{timecode(finish - clip['start_seconds'])}\n"
                    f"{row['text']}\n"
                )
        if not cues:
            raise RuntimeError("No captions overlap selected clip.")
        subtitle.write_text("\n".join(cues), encoding="utf-8-sig")
        subtitle_filter = (
            f"subtitles={subtitle.name}:force_style="
            "'FontName=Noto Sans Devanagari,FontSize=20,Outline=2,"
            "Shadow=1,Alignment=2,MarginV=55'"
        )
        if is_horizontal:
            # Keep the complete original 16:9 frame centered and uncropped.
            # A blurred, darkened duplicate fills the unused 9:16 canvas.
            video_filter = (
                "[0:v]split=2[bgsrc][fgsrc];"
                "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
                "crop=1080:1920,boxblur=24:12,eq=brightness=-0.18[bg];"
                "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
                f"{subtitle_filter},setsar=1,format=yuv420p[v]"
            )
            map_args = ["-filter_complex", video_filter, "-map", "[v]", "-map", "0:a?"]
        else:
            # Existing vertical footage is scaled proportionally to fit; no crop.
            video_filter = (
                "scale=1080:1920:force_original_aspect_ratio=decrease,"
                "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
                f"{subtitle_filter},setsar=1,format=yuv420p"
            )
            map_args = ["-vf", video_filter]
        run([
            "ffmpeg", "-y", "-ss", str(clip["start_seconds"]),
            "-i", str(source), "-t", str(clip["duration_seconds"]),
            *map_args, "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(output / f"short_{index:02}.mp4")
        ], cwd=output, timeout=3600)
        clip.update({"source_video_file": source_input.name, "review_status": "Pending Approval"})
        print(f"RENDER COMPLETE: Short {index}/{len(clips)} written with synced Hindi captions.", flush=True)

    print("STEP 4/5: Writing SEO metadata and review report.", flush=True)
    checkpoint_report = {}
    checkpoint_path = WORK / "ai_selection_checkpoint.json"
    if checkpoint_path.is_file():
        saved_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_report = {
            "total_batches": saved_checkpoint.get("total_batches", 0),
            "completed_batches": len(saved_checkpoint.get("completed_batches", {})),
            "token_usage": saved_checkpoint.get("token_usage", {}),
            "provider_models": saved_checkpoint.get("provider_models", {}),
        }
    report = {
        "status": "success",
        "source_video_file": source_input.name,
        "source_duration_seconds": duration,
        "framing": ("Original horizontal frame fully preserved in center; blurred background fills 9:16 canvas" if is_horizontal else "Original vertical frame proportionally fitted; no crop"),
        "transcription": "Generated from uploaded video audio",
        "transcription_chunk_seconds": int(os.getenv("TRANSCRIPTION_CHUNK_SECONDS", "600")),
        "transcription_providers": transcription_providers,
        "transcription_models": transcription_models,
        "ai_provider": provider,
        "ai_model": model,
        "ai_batch_processing": checkpoint_report,
        "requested_shorts": count,
               "created_shorts": len(clips),
               "selection_policy": "10-15s first, then 20-25s, then longer complete thoughts up to 60s",
               "minimum_quality_score": 8,
        "shorts": clips,
        "publishing": "Not uploaded; each Short requires separate user approval.",
        "rights_notice": "Use only videos you own or are authorized to edit.",
    }
    (output / "shorts_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = WORK / "jyotirling_shorts_review.zip"
    print("STEP 5/5: Creating the review ZIP package.", flush=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in output.iterdir():
            if path.is_file():
                package.write(path, path.name)
    print(f"PACKAGE COMPLETE: {archive} ({archive.stat().st_size:,} bytes).", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to uploaded video file")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-seconds", type=int, default=10)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument(
        "--allow-short-test", action="store_true",
        help="Allow a source video of 10 minutes or less for a controlled test only."
    )
    parser.add_argument("--resume-state", help="Path to a previous AI checkpoint JSON file.")
    args = parser.parse_args()
    print("SHORTS PACKAGE:", create(
        args.file, args.count, args.min_seconds, args.max_seconds,
        allow_short_test=args.allow_short_test, resume_state=args.resume_state
    ))
    print("No YouTube upload, metadata change, or publication was performed.")


if __name__ == "__main__":
    main()
