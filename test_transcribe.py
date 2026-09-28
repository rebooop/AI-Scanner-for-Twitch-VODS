import unittest

from transcribe import build_chunks
from windows import build_windows, format_timestamp
from rank import normalize_score
from select_results import select_top
from download import DownloadError, validate_video_url


class ChunkPlanningTests(unittest.TestCase):
    def test_chunks_cover_a_three_hour_vod_with_context_overlap(self) -> None:
        chunks = build_chunks(total_seconds=3 * 3600, chunk_minutes=30, overlap_seconds=3)
        self.assertEqual(len(chunks), 6)
        self.assertEqual(chunks[0].start, 0)
        self.assertEqual(chunks[0].core_end, 1800)
        self.assertEqual(chunks[1].start, 1797)
        self.assertEqual(chunks[-1].core_end, 10800)

    def test_last_chunk_is_shortened_to_vod_duration(self) -> None:
        chunks = build_chunks(total_seconds=3700, chunk_minutes=30, overlap_seconds=3)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[-1].core_end, 3700)
        self.assertEqual(chunks[-1].start, 3597)
        self.assertEqual(chunks[-1].duration, 103)

    def test_overlap_must_be_smaller_than_chunk(self) -> None:
        with self.assertRaises(ValueError):
            build_chunks(total_seconds=100, chunk_minutes=1, overlap_seconds=60)

    def test_windows_cover_a_transcript_with_overlap(self) -> None:
        transcript = {"duration": 150, "segments": [{"id": 0, "start": 50, "end": 55, "text": "hello"}]}
        windows = build_windows(transcript, window_seconds=90, overlap_seconds=30)
        self.assertEqual([(item["start"], item["end"]) for item in windows], [(0.0, 90.0), (60.0, 150.0)])
        self.assertEqual(windows[0]["segment_ids"], [0])
        self.assertEqual(windows[1]["segment_ids"], [])

    def test_timestamp_formatting(self) -> None:
        self.assertEqual(format_timestamp(3661.2), "01:01:01")

    def test_llm_score_is_clamped_to_its_window(self) -> None:
        window = {"id": "w_0001", "start": 60, "end": 150}
        score = normalize_score({"candidate": True, "start": 0, "end": 999, "overall_score": 110,
                                 "scores": {"standalone_clarity": 8, "reaction": 8}}, window)
        self.assertEqual((score["start"], score["end"], score["overall_score"]), (60.0, 150.0, 100))

    def test_tactical_only_score_cannot_be_selected(self) -> None:
        window = {"id": "w_0001", "start": 0, "end": 90}
        raw = {"candidate": True, "tactical_only": True, "start": 5, "end": 20, "overall_score": 92,
               "scores": {"standalone_clarity": 9, "reaction": 8}}
        score = normalize_score(raw, window)
        self.assertFalse(score["candidate"])
        self.assertEqual(score["overall_score"], 35)

    def test_selection_removes_near_duplicate_candidates(self) -> None:
        raw = {"scores": [
            {"candidate": True, "overall_score": 90, "start": 100, "end": 130},
            {"candidate": True, "overall_score": 80, "start": 105, "end": 128},
            {"candidate": True, "overall_score": 70, "start": 200, "end": 230},
        ]}
        selected = select_top(raw, top_k=10)
        self.assertEqual([item["overall_score"] for item in selected], [90, 70])

    def test_url_must_include_http_scheme_and_host(self) -> None:
        validate_video_url("https://www.youtube.com/watch?v=test")
        with self.assertRaises(DownloadError):
            validate_video_url("youtube.com/watch?v=test")


if __name__ == "__main__":
    unittest.main()
