import unittest

from src.repair_shorts import (
    correct_caption_text,
    find_bhakti_clip,
    find_word_clip,
    merge_utterances,
)


class RepairShortsTest(unittest.TestCase):
    def test_corrects_reported_asr_variant(self):
        self.assertEqual(correct_caption_text("वे जैनी हैं।"), "वे ज्ञानी हैं।")

    def test_merges_segments_into_complete_utterance(self):
        rows = [
            {"start": 1.0, "duration": 1.0, "text": "यह पहला"},
            {"start": 2.1, "duration": 1.0, "text": "वाक्य है।"},
        ]
        result = merge_utterances(rows)
        self.assertEqual(len(result), 1)
        self.assertIn("वाक्य है", result[0]["text"])

    def test_finds_corrected_word_in_target_window(self):
        utterances = [
            {"start": 100.0, "end": 112.0, "text": "यह जैनी का वर्णन है।"}
        ]
        clip = find_word_clip(utterances, 104.0, 110.0)
        self.assertEqual(clip["start"], 100.0)
        self.assertEqual(clip["end"], 112.0)

    def test_requires_both_bhakti_sons(self):
        utterances = [
            {"start": 200.0, "end": 207.0, "text": "भक्ति के दो पुत्र हैं।"},
            {"start": 207.1, "end": 215.0, "text": "उनके नाम ज्ञान और वैराग्य हैं।"},
        ]
        clip = find_bhakti_clip(utterances, 202.0, 205.0)
        self.assertEqual(clip["end"], 215.0)

    def test_rejects_missing_son_name(self):
        utterances = [
            {"start": 200.0, "end": 207.0, "text": "भक्ति के दो पुत्र हैं।"},
            {"start": 207.1, "end": 215.0, "text": "एक का नाम ज्ञान है।"},
        ]
        with self.assertRaises(RuntimeError):
            find_bhakti_clip(utterances, 202.0, 205.0)


if __name__ == "__main__":
    unittest.main()
