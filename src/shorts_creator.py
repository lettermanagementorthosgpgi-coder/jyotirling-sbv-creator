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

from src.ai_providers import generate, transcribe_audio

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


def transcribe(source):
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
    for index, audio in enumerate(audio_files):
        offset = index * chunk_seconds
        print(
            f"TRANSCRIPTION: starting part {index + 1}/{len(audio_files)} "
            f"({offset // 60}-{(offset + chunk_seconds) // 60} min of source video).",
            flush=True,
        )
        segments, provider, model = transcribe_audio(audio)
        transcription_providers.append(provider)
        transcription_models.append(model)
        if not segments:
            print(
                f"TRANSCRIPTION: part {index + 1}/{len(audio_files)} contains no detected speech; "
                "skipping it and continuing.",
                flush=True,
            )
            continue
        print(
            f"TRANSCRIPTION: part {index + 1}/{len(audio_files)} complete "
            f"({len(segments)} speech segments).",
            flush=True,
        )
        for segment in segments:
            text = unicodedata.normalize("NFC", re.sub(r"\s+", " ", str(segment.get("text", ""))).strip())
            if not text:
                continue
            start = float(segment.get("start", 0)) + offset
            end = float(segment.get("end", start - offset)) + offset
            rows.append({
                "start": start,
                "duration": max(0.1, end - start),
                "text": text,
            })
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
    from src.ai_providers import provider_limit_kind

    providers = [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GEMINI_API_KEY"),
    ]
    providers = [(name, key) for name, key in providers if os.getenv(key)]
    if not providers:
        raise RuntimeError("Neither OPENAI_API_KEY nor GEMINI_API_KEY is configured.")

    boundaries = sorted(set(round(float(x), 2) for row in rows
                            for x in (row["start"], row["start"] + row["duration"])))
    transcript_rows = [
        f"[{row['start']:.2f}s - {row['start'] + row['duration']:.2f}s] {row['text']}"
        for row in rows
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
        "transcript_rows": rows,
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
    usage_totals = checkpoint.setdefault("token_usage", {})
    disabled_today = set()  # Daily quota resets are checked afresh on each new run.
    batch_failures = []

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
        return obj["clips"]

    # Persist the full Whisper transcript before making the first paid API call.
    save_checkpoint()

    for batch_index, batch in enumerate(batches):
        batch_key = str(batch_index)
        if batch_key in completed:
            continue
        transcript = "\n".join(batch)
        prompt = f"""Act as a senior professional Hindi short-video editor and religious-context reviewer.
Review ONLY this transcript batch from a Hindi discourse. Select up to 3 distinct, self-contained moments.
Mandatory conditions: (1) a compelling natural hook, and (2) the complete religious thought/context must remain understandable. Never sacrifice scriptural meaning for a hook. Do not invent speech or add facts. Do not cut a sentence, phrase, word, or thought in half. Use only exact transcript row start/end timestamps.
Duration: prefer 10-15 seconds; if no complete thought fits, try 20-25, then 26-40, then 41-60 seconds. Maximum 60 seconds. Completeness is more important than short duration. Avoid overlapping/repeated topics.
Score honestly 1-10: hook strength, context completeness, natural boundaries, Hindi/caption quality, technical suitability. Return only candidates scoring at least 8/10 and explain both hook and context completeness.
Return ONLY JSON:
{{"clips":[{{"start_seconds":12.3,"end_seconds":25.4,"hook":"हिन्दी हुक","title":"हिन्दी शीर्षक","description":"विवरण","hashtags":["#..."],"tags":["..."],"reason":"हुक और प्रसंग की पूर्णता का कारण","quality_score":9,"score_breakdown":{{"completeness":2,"boundaries":2,"hook":2,"captions":2,"technical":1}},"improvements":"सुधार","duration_tier":"10-15"}}]}}
All text fields in Hindi. Timestamps are source-video seconds. If no suitable complete moment exists, return an empty clips list.
TRANSCRIPT BATCH {batch_index + 1}/{len(batches)}:
{transcript}"""
        # Alternate the initial provider by batch. On quota/error, hand the same
        # unfinished batch to the other provider.
        preferred = providers[batch_index % len(providers):] + providers[:batch_index % len(providers)]
        batch_done = False
        tried = set()
        for provider, key in preferred:
            if provider in disabled_today or provider in tried:
                continue
            tried.add(provider)
            if not os.getenv(key):
                continue
            for attempt in range(9):
                try:
                    print(
                        f"AI REVIEW: batch {batch_index + 1}/{len(batches)} "
                        f"({len(batch)} transcript rows), provider={provider}, attempt={attempt + 1}.",
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
                    valid = []
                    for item in parse_response(raw):
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
                    kind = provider_limit_kind(exc)
                    message = str(exc).replace("\n", " ")[:500]
                    if kind == "minute" and attempt < 8:
                        wait_seconds = min(300, 60 * (attempt + 1))
                        print(
                            f"AI TPM/RPM LIMIT: {provider}; waiting {wait_seconds}s, "
                            f"then retrying the same batch. Detail: {message}",
                            flush=True,
                        )
                        __import__("time").sleep(wait_seconds)
                        continue
                    if kind == "daily":
                        disabled_today.add(provider)
                        print(
                            f"AI DAILY QUOTA EXHAUSTED: {provider}; remaining work will be "
                            f"offered to the other configured provider. Detail: {message}",
                            flush=True,
                        )
                    else:
                        print(
                            f"AI PROVIDER FAILED: {provider}; handing this batch to the next "
                            f"available provider. Detail: {message}",
                            flush=True,
                        )
                    batch_failures.append(f"{provider}: {message}")
                    break
            if batch_done:
                break

        if not batch_done:
            save_checkpoint()
            done_count = len(completed)
            if len(disabled_today) == len(providers):
                raise RuntimeError(
                    f"AI_DAILY_QUOTA_EXHAUSTED: both configured providers reached their daily "
                    f"quota. Saved {done_count}/{len(batches)} batches in "
                    f"{checkpoint_path}; resume on the next quota day."
                )
            raise RuntimeError(
                f"AI_BATCH_INCOMPLETE: could not process batch {batch_index + 1}. "
                f"Saved {done_count}/{len(batches)} batches for safe resume. "
                + "; ".join(batch_failures[-4:])
            )

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
        rows, transcription_providers, transcription_models = transcribe(source)
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
    args = parser.parse_args()
    print("SHORTS PACKAGE:", create(
        args.file, args.count, args.min_seconds, args.max_seconds,
        allow_short_test=args.allow_short_test, resume_state=args.resume_state
    ))
    print("No YouTube upload, metadata change, or publication was performed.")


if __name__ == "__main__":
    main()
