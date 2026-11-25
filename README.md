# Jyotirling SBV Research & JSON Creator

This repository does not download videos, transcribe audio, create SBV files, render Shorts, or upload anything to YouTube. You supply the SBV file you downloaded from YouTube. The system researches the complete timestamped transcript and prepares the exact JSON manifest consumed by jyotirlingshivmathlive.

## Workflow

1. In this repository, create a GitHub Release and attach exactly one .sbv file. A tag such as sbv-2026-11-25-gita-3 is suitable.
2. Open Actions → SBV Research and Live JSON Publisher → Run workflow.
3. Enter the SBV Release tag. Set the Live Release tag to Friday (or another existing tag). Choose a maximum of 5, 7, or 10 Shorts.
4. The workflow parses every SBV cue, divides the full transcript into timestamp-based batches, and sends every batch independently to OpenAI, Gemini and Cloudflare Workers AI.
5. The providers propose complete-thought clips. Python rejects invalid durations/timestamps and reconstructs subtitles directly from the original SBV text; AI is never allowed to write or paraphrase subtitles.
6. The workflow validates shorts_manifest.json and replaces only that asset in the selected Live GitHub Release using gh release upload --clobber.
7. Open the Live repository and manually run Actions → Jyotirling Shivmath YouTube Shorts Creator with that same Release tag (for example, Friday). The existing Live workflow continues to use its existing source.mp4 and the newly published manifest.

## Required Actions secrets

Configure these in Settings → Secrets and variables → Actions → Secrets in this repository:

- OPENAI_API_KEY
- GEMINI_API_KEY
- CLOUDFLARE_API_TOKEN
- CLOUDFLARE_ACCOUNT_ID
- LIVE_REPO_TOKEN — a GitHub token with Contents: Read and write access to the private lettermanagementorthosgpgi-coder/jyotirlingshivmathlive repository. It is used only to replace the manifest asset in an existing Live Release.

Optional repository variables:

- OPENAI_MODEL
- GEMINI_MODEL
- CLOUDFLARE_MODEL (defaults to @cf/meta/llama-3.2-1b-instruct)

The SBV Release must contain one .sbv asset. The target Live Release must already exist and contain source.mp4. The workflow does not create or replace the video asset.

## Quality and safety rules

- The entire SBV is parsed and every cue is accounted for; long transcripts are batched without dropping the end of the file.
- All three configured AI providers independently review every batch. Provider failures are written to the report; no candidate is accepted below 8/10.
- Candidate clips must be 10–60 seconds, have valid transcript boundaries, and not overlap with other selected clips.
- Hindi subtitles are reconstructed verbatim from SBV and converted to clip-relative timestamps. No AI-generated subtitle text is used.
- If fewer than the requested number pass quality checks, the system returns fewer; it does not invent weak clips to reach a target.
- A machine-readable sbv_review_report.json and the final manifest are retained as a 14-day Actions artifact.
- This repository does not modify the Live repository's code or workflows, upload/publish to YouTube, or change channel metadata.

