"""Approval-first Shorts preparation from a user-authorized public vertical video."""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from src.ai_providers import generate

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "shorts_work"


def run(args, cwd=None):
    proc = subprocess.run(args, cwd=str(cwd) if cwd else None, text=True,
                          capture_output=True, timeout=1800)
    if proc.returncode:
        raise RuntimeError((proc.stderr or proc.stdout)[-2500:])
    return proc.stdout.strip()


def get_id(url):
    match = re.search(
        r"youtu\.be/([A-Za-z0-9_-]{11})|youtube\.com/(?:watch\?.*v=|shorts/|embed/)([A-Za-z0-9_-]{11})",
        url,
    )
    if not match:
        raise ValueError("Valid public YouTube URL required.")
    return next(value for value in match.groups() if value)


def download(url):
    folder = WORK / "source"
    folder.mkdir(parents=True)
    info = json.loads(run([sys.executable, "-m", "yt_dlp", "--no-playlist",
                           "--dump-single-json", url]))
    if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
        raise ValueError("Completed videos only.")
    run([sys.executable, "-m", "yt_dlp", "--no-playlist",
         "-f", "bv*[height<=1080]+ba/b[height<=1080]",
         "--merge-output-format", "mp4",
         "-o", str(folder / "source.%(ext)s"), url])
    candidates = [p for p in folder.glob("source.*")
                  if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov")]
    if not candidates:
        raise RuntimeError("Downloader did not produce a video file.")
    source = candidates[0]
    if source.suffix.lower() != ".mp4":
        converted = folder / "source.mp4"
        run(["ffmpeg", "-y", "-i", str(source), "-c", "copy", str(converted)])
        source = converted
    return source, info


def get_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi
    data = YouTubeTranscriptApi().fetch(video_id, languages=["hi", "sa", "ur", "bn", "en"])
    rows = []
    for item in getattr(data, "snippets", data):
        text = item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
        start = item.get("start", 0) if isinstance(item, dict) else getattr(item, "start", 0)
        duration = item.get("duration", 0) if isinstance(item, dict) else getattr(item, "duration", 0)
        text = re.sub(r"\s+", " ", str(text)).strip()
        if text:
            rows.append({"start": float(start), "duration": float(duration), "text": text})
    if not rows:
        raise RuntimeError("No public timed transcript/captions found.")
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


def timecode(seconds):
    milliseconds = int(round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def create(url, count, minimum, maximum):
    video_id = get_id(url)
    if not 1 <= count <= 10 or not 15 <= minimum <= maximum <= 60:
        raise ValueError("Count 1–10; duration 15–60 seconds.")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    source, info = download(url)
    rows = get_transcript(video_id)
    duration = float(info.get("duration") or run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(source)
    ]))
    if duration <= 600:
        raise ValueError("Source video must be longer than 10 minutes.")

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
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(source)
    ]))
    stream = probe.get("streams", [{}])[0]
    if int(stream.get("height", 0)) <= int(stream.get("width", 0)):
        raise ValueError("Source is not vertical; no crop/reframing was applied.")

    for index, clip in enumerate(clips, 1):
        subtitle = output / f"short_{index:02}.srt"
        cues = []
        for row in rows:
            begin = max(clip["start_seconds"], row["start"])
            finish = min(clip["end_seconds"], row["start"] + max(row["duration"], 1.2))
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
        ], cwd=output)
        clip.update({"source_video_url": url, "review_status": "Pending Approval"})

    report = {
        "status": "success",
        "source_video_id": video_id,
        "source_video_url": url,
        "source_title": info.get("title", ""),
        "source_duration_seconds": duration,
        "framing": "Original vertical framing preserved",
        "ai_provider": provider,
        "ai_model": model,
        "shorts": clips,
        "publishing": "Not uploaded; each Short requires separate user approval.",
        "rights_notice": "Use only videos you own or are authorized to download and edit.",
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
    parser.add_argument("--url", required=True)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=60)
    args = parser.parse_args()
    print("SHORTS PACKAGE:", create(
        args.url, args.count, args.min_seconds, args.max_seconds
    ))
    print("No YouTube upload, metadata change, or publication was performed.")


if __name__ == "__main__":
    main()
