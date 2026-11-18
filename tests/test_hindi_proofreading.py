import unittest

from src.shorts_creator import validate_corrected_rows


class HindiProofreadingTests(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {"start": 1.0, "duration": 1.2, "text": "भग्मान ने कहा"},
            {"start": 2.2, "duration": 1.4, "text": "ज्ञान और पुन्ने"},
        ]

    def test_accepts_one_corrected_text_for_each_row(self):
        result = validate_corrected_rows(
            self.raw,
            [
                {"id": 0, "text": "भगवान ने कहा"},
                {"id": 1, "text": "ज्ञान और पुण्य"},
            ],
            [0, 1],
        )
        self.assertEqual(result, ["भगवान ने कहा", "ज्ञान और पुण्य"])

    def test_rejects_missing_row(self):
        with self.assertRaises(ValueError):
            validate_corrected_rows(
                self.raw, [{"id": 0, "text": "भगवान ने कहा"}], [0, 1]
            )

    def test_rejects_duplicate_or_unknown_row_id(self):
        with self.assertRaises(ValueError):
            validate_corrected_rows(
                self.raw,
                [
                    {"id": 0, "text": "भगवान ने कहा"},
                    {"id": 0, "text": "भगवान ने कहा"},
                ],
                [0, 1],
            )

    def test_rejects_paraphrase_or_expansion(self):
        with self.assertRaises(ValueError):
            validate_corrected_rows(
                self.raw,
                [
                    {"id": 0, "text": "भगवान ने कहा कि हमें धर्म का पालन करना चाहिए और हमेशा सत्य बोलना चाहिए।"},
                    {"id": 1, "text": "ज्ञान और पुण्य"},
                ],
                [0, 1],
            )

    def test_keeps_timestamp_rows_separate(self):
        result = validate_corrected_rows(
            self.raw,
            [
                {"id": 1, "text": "ज्ञान और पुण्य"},
                {"id": 0, "text": "भगवान ने कहा"},
            ],
            [0, 1],
        )
        self.assertEqual(result[0], "भगवान ने कहा")
        self.assertEqual(result[1], "ज्ञान और पुण्य")


if __name__ == "__main__":
    unittest.main()
