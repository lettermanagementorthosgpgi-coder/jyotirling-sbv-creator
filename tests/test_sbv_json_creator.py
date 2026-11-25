import unittest
from src.sbv_json_creator import parse_sbv, make_batches, _subtitles, _candidate

class SBVTests(unittest.TestCase):
    def test_parse_multiline_and_ms(self):
        cues=parse_sbv("0:00:00.000,0:00:02.500\nपहली पंक्ति\nदूसरी पंक्ति\n\n0:00:03.000,0:00:04.000\nअगला\n")
        self.assertEqual(len(cues),2); self.assertEqual(cues[0]["text"],"पहली पंक्ति\nदूसरी पंक्ति"); self.assertEqual(cues[0]["end"],2.5)
    def test_invalid_timestamp(self):
        with self.assertRaises(ValueError): parse_sbv("गलत समय\nपाठ\n")
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError): parse_sbv("0:00:00.000,0:00:03.000\nएक\n\n0:00:02.000,0:00:04.000\nदो\n")
    def test_batches_keep_every_cue(self):
        cues=[{"start":i*10.0,"end":i*10.0+2,"text":str(i)} for i in range(80)]
        batches=make_batches(cues,100); self.assertEqual(sum(map(len,batches)),80); self.assertEqual([c for b in batches for c in b],cues)
    def test_subtitles_are_verbatim_and_relative(self):
        cues=[{"start":10.0,"end":12.0,"text":"नमस्ते"},{"start":12.0,"end":16.0,"text":"भगवान शिव"}]
        out=_subtitles(cues,11,15); self.assertEqual(out[0]["start_seconds"],0); self.assertEqual(out[0]["end_seconds"],1); self.assertEqual(out[1]["text"],"भगवान शिव")
    def test_candidate_duration_guard(self):
        cues=[{"start":10.0,"end":20.0,"text":"विचार"}]
        item={"start_seconds":10,"end_seconds":20,"quality_score":9,"title":"शीर्षक","description":"विवरण","reason":"पूर्ण"}
        self.assertIsNotNone(_candidate(item,cues,"openai",10,60)); item["end_seconds"]=80
        self.assertIsNone(_candidate(item,cues,"openai",10,60))

if __name__=="__main__": unittest.main()
