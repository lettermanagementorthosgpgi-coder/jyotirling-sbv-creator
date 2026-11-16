"""Approval-first Shorts preparation from a user-uploaded source video."""
import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
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
    pattern = audio_dir / "part_%03d.mp3"
    run([
        "ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1",
        "-ar", "16000", "-b:a", "64k", "-f", "segment",
        "-segment_time", "600", "-reset_timestamps", "1",
        str(pattern)
    ], timeout=3600)
    audio_files = sorted(audio_dir.glob("part_*.mp3"))
    if not audio_files:
        raise RuntimeError("Could not extract audio for multilingual transcription.")
    rows = []
    for index, audio in enumerate(audio_files):
        offset = index * 600
        print(f"Transcribing audio part {index + 1}/{len(audio_files)}")
        segments = transcribe_audio(audio)
        for segment in segments:
            text = re.sub(r"\s+", " ", str(segment.get("text", ""))).strip()
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
        raise RuntimeError("AI transcription returned no speech segments.")
    return rows


def select(rows, count, minimum, maximum):
    transcript = "\n".join(
        f"[{row['start']:.2f}s] {row['text']}" for row in rows
    )[:65000]
    prompt = f"""Choose up to {count} distinct, complete and compelling moments from this Hindi devotional transcript. Each must be {minimum}-{maximum} seconds. Do not invent speech. Return ONLY JSON: {{"clips":[{{"start_seconds":1.0,"end_seconds":44.0,"hook":"Hindi hook","title":"Hindi title","description":"Hindi description","hashtags":["#..."],"tags":["..."],"reason":"Hindi reason"}}]}}. Use Hindi for all text fields. Timestamps are seconds from source start:\n{transcript}"""
    errors = []
    for provider, key in (("openai", "OPENAI_API_KEY"), ("gemini", "GEMINI_API_KEY")):
        if not os.getenv(key):
            continue
        try:
            raw, model = generate(provider, prompt)
            raw = raw.strip()
            if raw.startswith(chr(96) * 3):
                raw = "\n".join(raw.splitlines()[1:-1])
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                left, right = raw.find("{"), raw.rfind("}")
                obj = json.loads(raw[left:right + 1])
            if isinstance(obj, dict) and isinstance(obj.get("clips"), list):
                return obj, provider, model
        except Exception as exc:
            errors.append(provider + ": " + str(exc)[:250])
    raise RuntimeError("No AI provider returned valid clip selections. " + "; ".join(errors))


def create(file_path, count, minimum, maximum, allow_short_test=False):
    source_input = Path(file_path)
    if not source_input.is_file():
        raise FileNotFoundError(f"Uploaded source video not found: {source_input}")
    if source_input.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}:
        raise ValueError("Unsupported video type. Upload MP4, MOV, MKV, WEBM, or M4V.")
    if not 1 <= count <= 10 or not 15 <= minimum <= maximum <= 60:
        raise ValueError("Count 1–10; duration 15–60 seconds.")
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
    if int(stream.get("height", 0)) <= int(stream.get("width", 0)):
        raise ValueError("Source is not vertical; no crop/reframing was applied.")

    rows = transcribe(source)
    selected, provider, model = select(rows, count, minimum, maximum)
    clips = []
    for item in selected["clips"]:
        try:
            start, end = float(item["start_seconds"]), float(item["end_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        if (start < 0 or end > duration or not minimum <= end - start <= maximum
                or not item.get("title") or not item.get("hook")):
            continue
        clips.append({
            "start_seconds": round(start, 2),
            "end_seconds": round(end, 2),
            "duration_seconds": round(end - start, 2),
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
    for index, clip in enumerate(clips, 1):
        subtitle = output / f"short_{index:02}.srt"
        cues = []
        for row in rows:
            begin = max(clip["start_seconds"], row["start"])
            finish = min(clip["end_seconds"], row["start"] + row["duration"])
            if finish > begin:
                cues.append(
                    f"{len(cues) + 1}\n"
                    f"{timecode(begin - clip['start_seconds'])} --> "
                    f"{timecode(finish - clip['start_seconds'])}\n"
                    f"{row['text']}\n"
                )
        if not cues:
            raise RuntimeError("No captions overlap selected clip.")
        subtitle.write_text("\n".join(cues), encoding="utf-8-sig")
        filter_value = (
            f"subtitles={subtitle.name}:force_style="
            "'FontName=Noto Sans Devanagari,FontSize=20,Outline=2,"
            "Shadow=1,Alignment=2,MarginV=55'"
        )
        run([
            "ffmpeg", "-y", "-ss", str(clip["start_seconds"]),
            "-i", str(source), "-t", str(clip["duration_seconds"]),
            "-vf", filter_value, "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "21", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(output / f"short_{index:02}.mp4")
        ], cwd=output, timeout=3600)
        clip.update({"source_video_file": source_input.name, "review_status": "Pending Approval"})

    report = {
        "status": "success",
        "source_video_file": source_input.name,
        "source_duration_seconds": duration,
        "framing": "Original vertical framing preserved",
        "transcription": "Generated from uploaded video audio",
        "ai_provider": provider,
        "ai_model": model,
        "shorts": clips,
        "publishing": "Not uploaded; each Short requires separate user approval.",
        "rights_notice": "Use only videos you own or are authorized to edit.",
    }
    (output / "shorts_metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    archive = WORK / "jyotirling_shorts_review.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for path in output.iterdir():
            if path.is_file():
                package.write(path, path.name)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to uploaded video file")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=60)
    parser.add_argument(
        "--allow-short-test", action="store_true",
        help="Allow a source video of 10 minutes or less for a controlled test only."
    )
    args = parser.parse_args()
    print("SHORTS PACKAGE:", create(
        args.file, args.count, args.min_seconds, args.max_seconds,
        allow_short_test=args.allow_short_test
    ))
    print("No YouTube upload, metadata change, or publication was performed.")


if __name__ == "__main__":
    main()
