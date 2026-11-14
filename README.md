# Jyotirling YouTube Shorts

A standalone, manual, approval-first workflow for preparing YouTube Shorts from a public URL of a completed **vertical** video that you own or are authorized to download and edit.

## Run it

1. Open **Actions → Create YouTube Shorts → Run workflow**.
2. Enter the public YouTube video URL.
3. Choose 5, 7, or 10 Shorts and the requested duration range.
4. Start the workflow and wait for it to finish.
5. Download the `jyotirling-shorts-review` artifact from the completed run.

The ZIP contains MP4 clips, SRT captions, and `shorts_metadata.json` with Hindi hooks, titles, descriptions, hashtags, tags, and review status.

## Required repository secrets

Add at least one AI key under **Settings → Secrets and variables → Actions → New repository secret**:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`

Optional model overrides can be added as repository variables:

- `OPENAI_MODEL` (default: `gpt-6-luna`)
- `GEMINI_MODEL` (default: `gemini-3.8-flash`)

API keys from any other repository are not copied or shared with this repository.

## Requirements and limitations

- Completed source video longer than 10 minutes.
- Source must already be vertical; the workflow preserves the original framing and does not crop/reframe.
- Public timed captions/transcript must be available in Hindi, Sanskrit, Urdu, Bengali, or English.
- YouTube download access may be blocked or changed; if retrieval fails, the run stops and does not fabricate content.
- FFmpeg renders MP4 clips and burns the available transcript captions.
- Review artifact is retained for 7 days.

## Approval and privacy

- This workflow does not use YouTube Data API or OAuth.
- It does not upload Shorts, edit channel metadata, or publish anything.
- Every Short remains pending your review and approval.
- Use only videos you own or are authorized to download and edit. A public URL alone does not grant reuse rights.

This repository is intentionally independent. It does not connect to the Channel Manager, Live Scheduler, website, or any other repository/workflow.
