# Makefile — Redrob Hackathon Ranker
# Usage: make <target>

PYTHON     = python3
CANDIDATES = data/candidates.jsonl
SUBMISSION = submission/submission.csv
METADATA   = submission/submission_metadata.yaml
VALIDATE   = validate_submission.py

.PHONY: help install rank validate demo test precompute clean

help:
	@echo ""
	@echo "Redrob Hackathon — Available Commands"
	@echo "────────────────────────────────────────"
	@echo "  make install      Install Python dependencies"
	@echo "  make rank         Run full ranker → submission/submission.csv"
	@echo "  make validate     Validate the submission CSV"
	@echo "  make test         Run all 45 unit tests"
	@echo "  make demo         Launch Streamlit demo app"
	@echo "  make precompute   Pre-compute embeddings (optional, speeds up rank)"
	@echo "  make clean        Remove caches and outputs"
	@echo ""

install:
	pip install -r requirements.txt

rank:
	@echo "Running ranker on $(CANDIDATES)..."
	$(PYTHON) rank.py --candidates $(CANDIDATES) --out $(SUBMISSION)
	@echo ""
	@echo "Validating..."
	$(PYTHON) $(VALIDATE) $(SUBMISSION)

validate:
	$(PYTHON) $(VALIDATE) $(SUBMISSION)

test:
	$(PYTHON) -m unittest tests/test_features.py tests/test_honeypot.py tests/test_writer.py -v

demo:
	streamlit run demo/app.py

precompute:
	$(PYTHON) precompute_embeddings.py \
		--candidates $(CANDIDATES) \
		--output outputs/embeddings.pkl

rank-with-embeddings:
	$(PYTHON) rank.py \
		--candidates $(CANDIDATES) \
		--embeddings-cache outputs/embeddings.pkl \
		--out $(SUBMISSION)
	$(PYTHON) $(VALIDATE) $(SUBMISSION)

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f outputs/embeddings.pkl
	@echo "Clean done."
