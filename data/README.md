# data/

Place the following files here before running the ranker:

| File | Size | Purpose |
|---|---|---|
| `candidates.jsonl` | ~465 MB | Full 100K candidate dataset (from hackathon bundle) |
| `sample_candidates.json` | ~296 KB | 500-candidate sample (included, used by demo) |

## Setup

```bash
# Copy from the hackathon bundle:
cp /path/to/hackathon_bundle/candidates.jsonl data/

# Verify:
wc -l data/candidates.jsonl   # should print 100000
```

`candidates.jsonl` is listed in `.gitignore` (too large for git).
`sample_candidates.json` IS committed (small enough).
