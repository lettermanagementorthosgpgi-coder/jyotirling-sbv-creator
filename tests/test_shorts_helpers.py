import unittest

from src.shorts_creator import timecode


class ShortsHelpersTest(unittest.TestCase):
    def test_timecode(self):
        self.assertEqual(timecode(65.25), "00:01:05,250")

    def test_timecode_hours(self):
        self.assertEqual(timecode(3661.5), "01:01:01,500")

    def test_provider_quota_classification(self):
        from src.ai_providers import provider_limit_kind
        self.assertEqual(provider_limit_kind("HTTP 429 rate_limit_exceeded: tokens per minute"), "minute")
        self.assertEqual(provider_limit_kind("RESOURCE_EXHAUSTED: RequestsPerDay quota exceeded"), "daily")
        self.assertEqual(provider_limit_kind("insufficient_quota"), "daily")
        self.assertEqual(provider_limit_kind("Read operation timed out"), "other")

    def test_silent_local_whisper_chunk_is_skipped(self):
        from unittest.mock import patch
        from src.ai_providers import transcribe_audio

        with patch.dict("os.environ", {"TRANSCRIPTION_PROVIDER": "local", "TRANSCRIPTION_ALLOW_API_FALLBACK": "false"}):
            with patch("src.ai_providers._transcribe_local", return_value=([], "faster-whisper-small-hi")):
                segments, provider, model = transcribe_audio("silent-part.mp3")

        self.assertEqual(segments, [])
        self.assertEqual(provider, "local")
        self.assertEqual(model, "faster-whisper-small-hi")


if __name__ == "__main__":
    unittest.main()
