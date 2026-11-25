# Jyotirling SBV Research & JSON Creator

This repository does not download videos, transcribe audio, create SBV files, render Shorts, or upload anything to YouTube. You supply the SBV file you downloaded from YouTube. The system researches the complete timestamped transcript and prepares the exact JSON manifest consumed by jyotirlingshivmathlive.

## Workflow

1. In this repository, create a GitHub Release and attach exactly one `.sbv` file. Do not upload an MP4 to SBV Creator. A tag such as sbv-2026-11-25-gita-3 is suitable.
2. Open Actions → SBV Research and Live JSON Publisher → Run workflow.
3. Enter the SBV input Release tag. Set the Live Release tag to Friday (or another existing tag). Choose a maximum of 5, 7, or 10 Shorts.
4. The workflow parses every SBV cue, divides the full transcript into timestamp-based batches, and sends each larger transcript batch to OpenAI only. Gemini and Cloudflare are not called.
5. OpenAI proposes complete-thought clips. Python rejects invalid durations/timestamps and reconstructs subtitles directly from the original SBV text; AI is never allowed to write or paraphrase subtitles.
6. The workflow validates shorts_manifest.json and deletes every existing asset from the target Live Release, including any old `source.mp4`, old manifest, and previously generated Shorts; then uploads only the newly generated `shorts_manifest.json`.
7. Open the Live repository and manually run Actions → Jyotirling Shivmath YouTube Shorts Creator with that same Release tag (for example, Friday). Only after the SBV Creator workflow finishes and the Release contains only `shorts_manifest.json`, manually upload the new `source.mp4` to that Friday Release. Then run the Live workflow; it reads both files and generates the Shorts.

## Required Actions secrets

Configure these in Settings → Secrets and variables → Actions → Secrets in this repository:

- OPENAI_API_KEY
- LIVE_REPO_TOKEN — a GitHub token with Contents: Read and write access to the private lettermanagementorthosgpgi-coder/jyotirlingshivmathlive repository. It is used to delete every asset from the selected Live Release and publish only the new manifest. You manually upload `source.mp4` afterward.

Optional repository variables:

- OPENAI_MODEL

The input SBV Creator Release must contain exactly one `.sbv` file. The target Live Release (for example, Friday) must already exist. WARNING: the SBV Creator workflow deletes EVERY asset currently attached to that Live Release—including any old `source.mp4`, old JSON, and all previously generated Shorts—and then uploads exactly one asset: the new `shorts_manifest.json`. The Release/tag itself remains. Only AFTER the SBV Creator workflow has completed, manually upload the new `source.mp4` to Friday. Then run the Live workflow, which will generate and upload the new Shorts.

## Quality and safety rules

- The entire SBV is parsed and every cue is accounted for; long transcripts are batched without dropping the end of the file. If adjacent SBV cue intervals overlap, only the earlier cue's end time is trimmed to the next cue's start; all original caption text and cue start times are preserved. Same-start or out-of-order cues that cannot be normalized safely are rejected.
- OpenAI alone reviews every batch. The default batch window is 10 minutes (rather than 2 minutes), reducing the number of API requests; requests are paced at least 120 seconds apart. If OpenAI reports a cooldown longer than the configured safe wait, the run stops immediately, writes an incomplete report, and does not publish a manifest. No candidate is accepted below 8/10.
- Candidate clips must be 10–60 seconds, have valid transcript boundaries, and not overlap with other selected clips.
- Hindi subtitles are reconstructed verbatim from SBV and converted to clip-relative timestamps. No AI-generated subtitle text is used.
- If fewer than the requested number pass quality checks, the system returns fewer; it does not invent weak clips to reach a target.
- A machine-readable sbv_review_report.json and the final manifest are retained as a 14-day Actions artifact.
- This repository does not modify the Live repository's code or workflows, upload/publish to YouTube, or change channel metadata. It intentionally clears the entire selected Live Release and leaves only the newly generated JSON. You then manually upload a fresh `source.mp4`; the Live workflow generates and uploads Shorts packages afterward.

