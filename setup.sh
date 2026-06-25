#!/usr/bin/env bash
# setup.sh — One-command environment setup for the Redrob Hackathon ranker.
# Usage: bash setup.sh

set -e

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Redrob Hackathon — Environment Setup"
echo "═══════════════════════════════════════════════════"
echo ""

# 1. Python version check
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION"

if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3,9) else 1)'; then
    echo "   Version OK (need ≥ 3.9)"
else
    echo "❌ Python 3.9+ required. Please upgrade."
    exit 1
fi

# 2. Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# 3. Verify core imports
echo ""
echo "Verifying imports..."
python3 -c "import pandas; print(f'  ✅ pandas {pandas.__version__}')"
python3 -c "import numpy; print(f'  ✅ numpy {numpy.__version__}')"
python3 -c "import sklearn; print(f'  ✅ scikit-learn {sklearn.__version__}')"

# Sentence-transformers is optional
python3 -c "
try:
    import sentence_transformers
    print(f'  ✅ sentence-transformers {sentence_transformers.__version__}')
except ImportError:
    print('  ⚠️  sentence-transformers not installed')
    print('      Semantic scoring will be disabled (other 7 dimensions still active).')
    print('      Install with: pip install sentence-transformers')
"

python3 -c "
try:
    import streamlit
    print(f'  ✅ streamlit {streamlit.__version__}')
except ImportError:
    print('  ⚠️  streamlit not installed — demo will not run')
    print('      Install with: pip install streamlit plotly')
"

# 4. Check data
echo ""
echo "Checking data..."
if [ -f "data/candidates.jsonl" ]; then
    LINES=$(wc -l < data/candidates.jsonl)
    echo "  ✅ data/candidates.jsonl ($LINES lines)"
elif [ -f "data/sample_candidates.json" ]; then
    echo "  ✅ data/sample_candidates.json (sample data)"
    echo "  ℹ️  For full ranking, copy candidates.jsonl to data/"
else
    echo "  ⚠️  No candidate data found in data/"
    echo "      Place candidates.jsonl or sample_candidates.json in data/"
fi

# 5. Run tests
echo ""
echo "Running tests..."
python3 -m unittest tests/test_features.py tests/test_honeypot.py tests/test_writer.py -v 2>&1 | tail -5

# 6. Done
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    Rank:      python rank.py --candidates data/candidates.jsonl"
echo "    Validate:  python validate_submission.py submission/submission.csv"
echo "    Demo:      streamlit run demo/app.py"
echo "    Tests:     python3 -m unittest tests/ -v"
echo "═══════════════════════════════════════════════════"
echo ""
