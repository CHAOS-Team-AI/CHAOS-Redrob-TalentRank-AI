"""
test_writer.py — Unit tests for submission CSV writer.
Run: python3 -m unittest tests/test_writer.py -v
"""
import sys, csv, unittest, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from writer import write_submission


def _make_ranked(n=100, base_score=0.9):
    return [
        {
            "candidate_id": f"CAND_{i+1:07d}",
            "rank": i + 1,
            "score": max(0.01, base_score - i * 0.003),
            "reasoning": f"Test reasoning for candidate {i+1}.",
        }
        for i in range(n)
    ]


class TestWriter(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _out(self, name="sub.csv"):
        return os.path.join(self.tmp, name)

    def test_creates_file(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        self.assertTrue(Path(out).exists())

    def test_wrong_count_raises(self):
        with self.assertRaises((ValueError, AssertionError)):
            write_submission(_make_ranked(50), self._out())

    def test_header_row_correct(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        self.assertEqual(header, ["candidate_id", "rank", "score", "reasoning"])

    def test_exactly_100_data_rows(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        with open(out, encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 101)  # header + 100

    def test_scores_non_increasing(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        scores = []
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                scores.append(float(row[2]))
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1],
                f"Score not non-increasing at row {i+2}: {scores[i]} < {scores[i+1]}")

    def test_tiebreak_cid_ascending(self):
        """When two candidates have equal scores, lower candidate_id must appear first."""
        ranked = [
            {"candidate_id": "CAND_0000050", "rank": 1, "score": 0.5000, "reasoning": "r1"},
            {"candidate_id": "CAND_0000010", "rank": 2, "score": 0.5000, "reasoning": "r2"},
        ] + [
            {"candidate_id": f"CAND_{i+100:07d}", "rank": i+3, "score": 0.4 - i*0.001, "reasoning": "r"}
            for i in range(98)
        ]
        out = self._out()
        write_submission(ranked, out)
        ids = []
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ids.append(row[0])
        idx_10 = ids.index("CAND_0000010")
        idx_50 = ids.index("CAND_0000050")
        self.assertLess(idx_10, idx_50,
            "Tie-break violated: CAND_0000010 should appear before CAND_0000050")

    def test_ranks_are_1_to_100(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        ranks = []
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ranks.append(int(row[1]))
        self.assertEqual(sorted(ranks), list(range(1, 101)))

    def test_no_duplicate_candidate_ids(self):
        out = self._out()
        write_submission(_make_ranked(100), out)
        ids = []
        with open(out, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                ids.append(row[0])
        self.assertEqual(len(ids), len(set(ids)), "Duplicate candidate_ids found")

    def test_utf8_encoding(self):
        out = self._out()
        ranked = _make_ranked(100)
        ranked[0]["reasoning"] = "Candidate naïve résumé — café style"
        write_submission(ranked, out)
        with open(out, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("naïve", content)


if __name__ == "__main__":
    unittest.main()
