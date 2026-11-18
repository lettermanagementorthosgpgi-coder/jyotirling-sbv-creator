# Jyotirling YouTube Shorts

A standalone, manual, approval-first workflow. Upload one completed vertical source video as a GitHub Release asset; the workflow then transcribes the uploaded audio, selects compelling moments, and renders review-ready Shorts.

## Upload and run

1. Open this repository's **Releases** page and choose **Draft a new release**.
2. Enter a tag such as `source-2026-11-14`.
3. Upload exactly one source video file (`.mp4`, `.mov`, `.mkv`, `.webm`, or `.m4v`) as a release asset.
4. Publish the release.
5. Open **Actions → Create YouTube Shorts → Run workflow**.
6. Enter that release tag, choose 5, 7, or 10 Shorts, and use the professional 10–60 second duration ladder. The AI first prefers 10–15 seconds, then 20–25 seconds, then longer complete thoughts up to 60 seconds.
7. Start the workflow. When it finishes, download the review artifact. If both AI daily quotas are exhausted, note the failed run ID from its URL; on the next day, run again with the same Release tag/video asset and enter that ID in `resume_run_id` to continue from the saved checkpoint. The same resume input also restores any completed Whisper chunks if transcription itself was interrupted.

The ZIP contains MP4 clips, SRT captions, and `shorts_metadata.json` with Hindi hooks, titles, descriptions, hashtags, tags, and review status.

## Required repository secrets

Add these under **Settings → Secrets and variables → Actions → New repository secret**:

- `OPENAI_API_KEY` — preferred first provider for speech-to-text and text generation.
- `GEMINI_API_KEY` — automatic fallback for speech-to-text and text generation if OpenAI is unavailable or fails.

Optional repository variables:

- `OPENAI_MODEL` — text generation model override.
- `GEMINI_MODEL` — Gemini text generation model override.
- `TRANSCRIPTION_MODEL` — OpenAI transcription model override (defaults to `whisper-1`).
- `GEMINI_TRANSCRIPTION_MODEL` — Gemini transcription model override (defaults to `gemini-3.5-transcribe`).

## Requirements and limitations

- Upload exactly one completed source video per release.
- Source video must be longer than 10 minutes. Horizontal 16:9 footage is placed fully uncropped in the center of a 9:16 canvas; a blurred, darkened duplicate fills the unused top and bottom. Existing vertical footage is proportionally fitted without cropping.
- Audio is transcribed locally with the multilingual Whisper medium model, explicitly set to Hindi, in configurable 10-minute chunks by default (`TRANSCRIPTION_CHUNK_SECONDS`, allowed 60–600 seconds). A scripture-aware Hindi proofreading pass runs on every transcript batch before captions are rendered. It corrects clear Devanagari spelling and ASR substitutions while preserving the speaker's words, row timestamps and meaning; uncertain words are not guessed. The proofreading result is validated to ensure every source row is returned exactly once and is not expanded into a paraphrase. Text proofreading and clip selection use the existing OpenAI and Gemini API keys and may incur additional API usage. API usage may incur charges; free-tier quotas are not guaranteed or unlimited.
- Whisper transcribes the source locally on the GitHub runner. OpenAI and Gemini then share transcript review in small batches, alternating providers. Every batch requires both an engaging hook and a complete, accurate religious context; neither criterion may be sacrificed for the other. TPM/RPM limits trigger wait-and-retry on the same batch. If one provider reaches a daily quota or fails, its unfinished batch is offered to the other provider. Each completed 10-minute Whisper chunk is atomically saved as its own JSON checkpoint, including timestamped Hindi transcript segments, source identity, provider, and model. Silent chunks are also checkpointed so they are not repeatedly transcribed. If Whisper stops part-way through, already completed chunks are restored and only missing chunks are transcribed on the next run. Every successful AI batch saves its input transcript, full response, validated candidates, provider/model, and reported token usage. The workflow uploads all available Whisper and AI checkpoints in a 30-day artifact even when the processing step fails. On the next manual run, enter the previous run ID in `resume_run_id`; the workflow restores the latest available stage outputs and continues without repeating completed Whisper or AI work. The logs report completed batches and API-reported token usage; when an API does not report usage, the log labels it as unavailable rather than inventing a count. AI chooses only transcript-aligned boundaries for complete thoughts, prefers 10–15 seconds, then 20–25 seconds, and expands only as needed up to 60 seconds. Each accepted Short receives a quality score (minimum 8/10), score breakdown, and improvement notes. Hindi captions are burned into the rendered clips. If fewer than the requested number meet the quality threshold, fewer are returned.
- Review artifact is retained for 7 days.
- Horizontal-source framing preserves the complete original image; only a blurred background duplicate is enlarged and cropped to fill the vertical canvas.
- Large uploads are handled as GitHub Release assets, not committed to the repository's source tree.


## Short-source test (manual, one clip)

For a controlled test using a source video of 10 minutes or less, open **Actions → Test Create One Short → Run workflow** and enter the Release tag. This separate test workflow creates exactly one complete-thought review clip (10–60 seconds) and passes the explicit `--allow-short-test` flag. The regular **Create YouTube Shorts** workflow continues to require a source video longer than 10 minutes. The test workflow uses the same OpenAI-first, Gemini-fallback transcription and text-generation flow and may incur API charges.

## Approval and privacy

- No YouTube URL downloading, YouTube Data API, or OAuth is used.
- The workflow downloads only the video asset you uploaded to this repository's Release.
- It does not upload Shorts, edit channel metadata, or publish anything.
- Every Short remains pending your review and approval.
- Use only videos you own or are authorized to edit.

This repository is intentionally independent. It does not connect to the Channel Manager, Live Scheduler, website, or any other repository/workflow.
