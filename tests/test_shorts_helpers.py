import unittest

from src.shorts_creator import timecode


class ShortsHelpersTest(unittest.TestCase):
    def test_timecode(self):
        self.assertEqual(timecode(65.25), "00:01:05,250")

    def test_timecode_hours(self):
        self.assertEqual(timecode(3661.5), "01:01:01,500")


if __name__ == "__main__":
    unittest.main()
