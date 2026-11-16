# Jyotirling YouTube Shorts

A standalone, manual, approval-first workflow. Upload one completed vertical source video as a GitHub Release asset; the workflow then transcribes the uploaded audio, selects compelling moments, and renders review-ready Shorts.

## Upload and run

1. Open this repository's **Releases** page and choose **Draft a new release**.
2. Enter a tag such as `source-2026-11-14`.
3. Upload exactly one source video file (`.mp4`, `.mov`, `.mkv`, `.webm`, or `.m4v`) as a release asset.
4. Publish the release.
5. Open **Actions → Create YouTube Shorts → Run workflow**.
6. Enter that release tag, choose 5, 7, or 10 Shorts, and choose the duration range.
7. Start the workflow. When it finishes, download the `jyotirling-shorts-review` artifact.

The ZIP contains MP4 clips, SRT captions, and `shorts_metadata.json` with Hindi hooks, titles, descriptions, hashtags, tags, and review status.

## Required repository secrets

Add these under **Settings → Secrets and variables → Actions → New repository secret**:

- `OPENAI_API_KEY` — required for speech-to-text and available as the first text-generation provider.
- `GEMINI_API_KEY` — optional fallback for text generation.

Optional repository variables:

- `OPENAI_MODEL` — text generation model override.
- `GEMINI_MODEL` — Gemini text generation model override.
- `TRANSCRIPTION_MODEL` — defaults to `whisper-1`.

## Requirements and limitations

- Upload exactly one completed source video per release.
- Source video must be longer than 10 minutes and already vertical; no crop/reframing is performed.
- Audio is transcribed in 10-minute chunks using OpenAI speech-to-text. Transcription usage may incur API charges.
- AI chooses moments from the generated transcript; Hindi captions are burned into the rendered clips.
- Review artifact is retained for 7 days.
- Large uploads are handled as GitHub Release assets, not committed to the repository's source tree.


## Short-source test (manual, one clip)

For a controlled test using a source video of 10 minutes or less, open **Actions → Test Create One Short → Run workflow** and enter the Release tag. This separate test workflow creates exactly one 30–45 second review clip and passes the explicit `--allow-short-test` flag. The regular **Create YouTube Shorts** workflow continues to require a source video longer than 10 minutes. The test workflow uses the same OpenAI transcription and text-generation APIs and may incur API charges.

## Approval and privacy

- No YouTube URL downloading, YouTube Data API, or OAuth is used.
- The workflow downloads only the video asset you uploaded to this repository's Release.
- It does not upload Shorts, edit channel metadata, or publish anything.
- Every Short remains pending your review and approval.
- Use only videos you own or are authorized to edit.

This repository is intentionally independent. It does not connect to the Channel Manager, Live Scheduler, website, or any other repository/workflow.
