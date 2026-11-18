"""Targeted repair of two approved Shorts using fresh, local Hindi Whisper transcription."""
import argparse
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.ai_providers import transcribe_audio
from src.shorts_creator import timecode

WORK = Path("shorts_work")
CORRECTIONS = {
    "जैनी": "ज्ञानी",
    "जैनीं": "ज्ञानी",
}


def run(command, cwd=None, timeout=1800):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result.stdout.strip()


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def correct_caption_text(value):
    text = clean_text(value)
    for wrong, right in CORRECTIONS.items():
        text = text.replace(wrong, right)
    return text


def merge_utterances(rows, pause_limit=0.85):
    utterances = []
    current = None
    for row in rows:
        start = float(row["start"])
        end = start + float(row["duration"])
        if current is None:
            current = {"start": start, "end": end, "text": row["text"]}
            continue
        if start - current["end"] >= pause_limit:
            utterances.append(current)
            current = {"start": start, "end": end, "text": row["text"]}
        else:
            current["end"] = end
            current["text"] += " " + row["text"]
        if re.search(r"[।!?…][\"')\]]?$", row["text"]):
            utterances.append(current)
            current = None
    if current:
        utterances.append(current)
    return utterances


def find_word_clip(utterances, target_start, target_end, duration_limit=60):
    candidates = [
        item for item in utterances
        if item["start"] <= target_end and item["end"] >= target_start
        and ("ज्ञानी" in correct_caption_text(item["text"]) or "जैनी" in item["text"])
    ]
    if not candidates:
        raise RuntimeError(
            "Could not locate the reported 'ज्ञानी' word in fresh transcription near the first Short. "
            "No guessed caption was written; inspect the repair transcript."
        )
    selected = candidates[0]
    start, end = selected["start"], selected["end"]
    # Include neighboring utterances only when needed to make a complete 10-second thought.
    for item in utterances:
        if item["end"] <= start and start - item["start"] < 8 and end - item["start"] <= duration_limit:
            start = item["start"]
        if end - start < 10 and item["start"] >= end and item["end"] - start <= duration_limit:
            end = item["end"]
    if not 10 <= end - start <= duration_limit:
        raise RuntimeError(f"First repaired thought is {end-start:.1f}s; expected 10-{duration_limit}s.")
    return {"start": start, "end": end, "title": "ज्ञानी का सही अर्थ", "description": "प्रवचन का सुधारा हुआ अंश।"}


def find_bhakti_clip(utterances, target_start, target_end, duration_limit=60):
    start_index = None
    for index, item in enumerate(utterances):
        overlaps = item["start"] <= target_end and item["end"] >= target_start
        text = item["text"]
        if overlaps and "भक्ति" in text and "पुत्र" in text:
            start_index = index
            break
    if start_index is None:
        raise RuntimeError(
            "Fresh transcription did not locate the 'भक्ति ... पुत्र' statement near the second Short. "
            "Refusing to create a possibly unrelated clip."
        )
    collected = []
    end = utterances[start_index]["end"]
    names_found = set()
    for item in utterances[start_index:]:
        if item["start"] - utterances[start_index]["start"] > duration_limit:
            break
        collected.append(item["text"])
        end = item["end"]
        joined = " ".join(collected)
        normalized = joined.replace("वैराग्य", "वैराग्य")
        if re.search(r"ज्ञान", normalized) and re.search(r"वैराग्य|वैराग्य", normalized):
            names_found = {"ज्ञान", "वैराग्य"}
            break
    if names_found != {"ज्ञान", "वैराग्य"}:
        excerpt = " ".join(collected)[-1200:]
        raise RuntimeError(
            "Could not verify both names (ज्ञान and वैराग्य) in the fresh speech transcript. "
            "No incomplete Short was rendered. Transcript excerpt: " + excerpt
        )
    start = utterances[start_index]["start"]
    if not 10 <= end - start <= duration_limit:
        raise RuntimeError(f"Bhakti explanation is {end-start:.1f}s; expected 10-{duration_limit}s.")
    return {
        "start": start, "end": end,
        "title": "भक्ति के दोनों पुत्र कौन हैं?",
        "description": "भक्ति के पुत्र ज्ञान और वैराग्य के विषय में प्रवचन।",
    }


def render_clip(source, output_dir, rows, clip, number, horizontal):
    start, end = clip["start"], clip["end"]
    subtitle = output_dir / f"short_{number:02}.srt"
    cues = []
    for row in rows:
        begin = float(row["start"])
        finish = begin + float(row["duration"])
        if begin >= start - 0.05 and finish <= end + 0.05:
            caption = correct_caption_text(row["text"])
            if caption:
                cues.append(
                    f"{len(cues)+1}\n{timecode(begin-start)} --> {timecode(finish-start)}\n{caption}\n"
                )
    if not cues:
        raise RuntimeError(f"No complete caption rows found for repaired Short {number}.")
    subtitle.write_text("\n".join(cues), encoding="utf-8-sig")
    subtitle_filter = (
        f"subtitles={subtitle.name}:force_style="
        "'FontName=Noto Sans Devanagari,FontSize=20,Outline=2,Shadow=1,Alignment=2,MarginV=55'"
    )
    if horizontal:
        vf = (
            "[0:v]split=2[bgsrc][fgsrc];"
            "[bgsrc]scale=1080:1920:force_original_aspect_ratio=increase,"
            "crop=1080:1920,boxblur=24:12,eq=brightness=-0.18[bg];"
            "[fgsrc]scale=1080:1920:force_original_aspect_ratio=decrease[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            f"{subtitle_filter},setsar=1,format=yuv420p[v]"
        )
        mapping = ["-filter_complex", vf, "-map", "[v]", "-map", "0:a?"]
    else:
        vf = (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,"
            f"{subtitle_filter},setsar=1,format=yuv420p"
        )
        mapping = ["-vf", vf]
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source),
        "-t", f"{end-start:.3f}", *mapping, "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "21", "-c:a", "aac",
        "-b:a", "128k", "-movflags", "+faststart",
        str(output_dir / f"short_{number:02}.mp4")
    ], cwd=output_dir, timeout=3600)
    clip.update({
        "number": number, "duration_seconds": round(end-start, 2),
        "captions_file": subtitle.name, "video_file": f"short_{number:02}.mp4",
        "review_status": "Pending Approval",
    })


def repair(source_path, first_start, first_end, second_start, second_end):
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    local_source = WORK / ("source" + source.suffix.lower())
    shutil.copy2(source, local_source)
    probe = json.loads(run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(local_source)
    ]))
    stream = probe.get("streams", [{}])[0]
    horizontal = int(stream.get("width", 0)) > int(stream.get("height", 0))
    duration = float(run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(local_source)
    ]))
    output = WORK / "repaired_shorts"
    output.mkdir()

    all_rows = []
    windows = [
        ("first", first_start, first_end),
        ("second", second_start, second_end),
    ]
    for label, target_start, target_end in windows:
        context_start = max(0, target_start - 12)
        context_end = min(duration, target_end + 48)
        audio = WORK / f"{label}_context.mp3"
        run([
            "ffmpeg", "-y", "-ss", f"{context_start:.3f}", "-i", str(local_source),
            "-t", f"{context_end-context_start:.3f}", "-vn", "-ac", "1",
            "-ar", "16000", "-b:a", "64k", str(audio)
        ], timeout=600)
        segments, provider, model = transcribe_audio(audio)
        rows = []
        for segment in segments:
            start = context_start + float(segment.get("start", 0))
            end = context_start + float(segment.get("end", segment.get("start", 0)) or 0)
            text = clean_text(segment.get("text", ""))
            if text and end > start:
                rows.append({"start": start, "duration": end-start, "text": text})
        if not rows:
            raise RuntimeError(f"No fresh speech transcription for {label} context.")
        all_rows.extend(rows)
        print(
            f"REPAIR TRANSCRIPTION {label}: {len(rows)} segments; provider={provider}; model={model}",
            flush=True,
        )
        (output / f"{label}_fresh_transcript.txt").write_text(
            "\n".join(f"[{r['start']:.2f}-{r['start']+r['duration']:.2f}] {r['text']}" for r in rows),
            encoding="utf-8",
        )
    all_rows.sort(key=lambda row: row["start"])
    utterances = merge_utterances(all_rows)
    first_clip = find_word_clip(utterances, first_start, first_end)
    second_clip = find_bhakti_clip(utterances, second_start, second_end)
    clips = [first_clip, second_clip]
    for clip in clips:
        if clip["start"] < 0 or clip["end"] > duration:
            raise RuntimeError("Repair clip boundary falls outside the source video.")
    print(
        "VERIFIED REPAIR BOUNDARIES: "
        + "; ".join(f"Short {i+1} {c['start']:.2f}-{c['end']:.2f}s" for i, c in enumerate(clips)),
        flush=True,
    )
    for index, clip in enumerate(clips, 1):
        render_clip(local_source, output, all_rows, clip, index, horizontal)
    report = {
        "status": "success",
        "source_video": source.name,
        "transcription": "Fresh local Hindi Whisper transcription of only the two target regions",
        "caption_correction": "Exact ASR variants जैनी/जैनीं replaced with ज्ञानी; all other words retained from fresh speech recognition",
        "second_clip_completeness_check": "Required both ज्ञान and वैराग्य to be detected after the Bhakti sons statement",
        "shorts": clips,
        "publishing": "Not uploaded; pending user review and approval",
    }
    (output / "repair_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = WORK / "jyotirling-repaired-two-shorts.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
        for item in output.iterdir():
            if item.is_file():
                package.write(item, item.name)
    print(f"REPAIR PACKAGE READY: {archive} ({archive.stat().st_size:,} bytes)", flush=True)
    return archive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--first-start", type=float, default=3888.8)
    parser.add_argument("--first-end", type=float, default=3899.6)
    parser.add_argument("--second-start", type=float, default=4358.2)
    parser.add_argument("--second-end", type=float, default=4372.5)
    args = parser.parse_args()
    print("REPAIR OUTPUT:", repair(
        args.file, args.first_start, args.first_end, args.second_start, args.second_end
    ))
    print("No YouTube upload or publication was performed.", flush=True)


if __name__ == "__main__":
    main()
