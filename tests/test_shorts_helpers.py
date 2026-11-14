import unittest

from src.shorts_creator import get_id, timecode


class ShortsHelpersTest(unittest.TestCase):
    def test_short_url(self):
        self.assertEqual(get_id("https://youtu.be/lP_bnXwptx8"), "lP_bnXwptx8")

    def test_watch_url(self):
        self.assertEqual(
            get_id("https://www.youtube.com/watch?v=lP_bnXwptx8&t=30"),
            "lP_bnXwptx8",
        )

    def test_embed_url(self):
        self.assertEqual(
            get_id("https://www.youtube.com/embed/lP_bnXwptx8"),
            "lP_bnXwptx8",
        )

    def test_reject_invalid_url(self):
        with self.assertRaises(ValueError):
            get_id("https://example.com/not-a-video")

    def test_timecode(self):
        self.assertEqual(timecode(65.25), "00:01:05,250")


if __name__ == "__main__":
    unittest.main()
