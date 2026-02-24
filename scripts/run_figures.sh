#!/usr/bin/env bash
# =============================================================================
# Generate publication-quality figures from benchmark results
#
# Edit the settings below, then run:
#   bash scripts/run_figures.sh
# =============================================================================

# ── User Settings (edit these) ───────────────────────────────────────────────
RESULTS_DIR="comparison_results/benchmark_latest"  # Benchmark output directory
ROI_IDS="roi2,roi3,roi4,roi5"                     # Comma-separated ROI ids
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Generating Publication Figures ==="
echo "  Results: $RESULTS_DIR"
echo "  ROIs:    $ROI_IDS"
echo ""

cd "$PROJECT_DIR"
python scripts/make_figures.py --results-dir "$RESULTS_DIR" --roi-ids "$ROI_IDS"

echo ""
echo "Figures saved to $RESULTS_DIR/figures/"
