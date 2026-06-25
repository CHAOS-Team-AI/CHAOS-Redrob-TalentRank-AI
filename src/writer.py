"""
writer.py — Write the submission CSV in the exact format required by the validator.

Key insight from validator source:
  - Equal scores (string equality) in the CSV trigger the tie-break check
  - We use 6 decimal places to avoid false ties from rounding
  - Sort by (-raw_score, candidate_id) before writing
  - Rows are written in rank order; the rank field matches the row position
"""

import csv
from pathlib import Path
from typing import List, Dict, Any


def write_submission(ranked: List[Dict[str, Any]], output_path: str) -> None:
    if len(ranked) != 100:
        raise ValueError(f"Expected 100 candidates, got {len(ranked)}")

    # Sort: score DESC (full float precision), then candidate_id ASC for ties
    sorted_ranked = sorted(ranked, key=lambda x: (-x["score"], x["candidate_id"]))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score_str = None
        for i, item in enumerate(sorted_ranked):
            rank = i + 1
            # Use 6dp to avoid false ties from rounding
            score_str = f"{item['score']:.6f}"
            # Safety: if still equal after 6dp and ID order is wrong, nudge score down
            if prev_score_str == score_str:
                # Already sorted by (score, cid) so this only fires if truly equal score
                pass  # keep as-is; candidate_id order was set by sort above
            prev_score_str = score_str
            writer.writerow([
                item["candidate_id"],
                rank,
                score_str,
                item["reasoning"],
            ])

    print(f"✅ Submission written: {out_path}  ({len(sorted_ranked)} rows)")
