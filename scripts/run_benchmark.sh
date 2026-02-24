#!/usr/bin/env bash
# =============================================================================
# HiMReg Benchmark — Compare HiMReg vs SimpleITK/Elastix
#
# Edit the settings below, then run:
#   bash scripts/run_benchmark.sh
# =============================================================================

# ── User Settings (edit these) ───────────────────────────────────────────────
BENCHMARK_DIR="benchmark"                         # Directory containing roi{N}/ subdirs
ROI_IDS="2,3,4,5,6"                              # Comma-separated ROI ids
FIXED_SUFFIX="he"                                # Fixed image filename suffix (e.g. he.tif)
MOVING_SUFFIX="791"                              # Moving image filename suffix (e.g. 791.tif)
OUTPUT="comparison_results/benchmark"              # Output directory
DEVICE="cuda"                                     # cuda or cpu
HIMREG_MODE="bspline"                            # affine | diff | bspline
RUN_WSIREG=false                                  # true to also run wsireg comparison
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WSIREG_FLAG=""
if [ "$RUN_WSIREG" = true ]; then
    WSIREG_FLAG="--run-wsireg"
fi

# Format ROI IDs for figure generation (add "roi" prefix)
ROI_IDS_PREFIXED=""
IFS=',' read -ra ROI_ARRAY <<< "$ROI_IDS"
for i in "${!ROI_ARRAY[@]}"; do
    if [ "$i" -gt 0 ]; then ROI_IDS_PREFIXED+=","; fi
    ROI_IDS_PREFIXED+="roi${ROI_ARRAY[$i]}"
done

echo "=== HiMReg Benchmark ==="
echo "  Data:   $BENCHMARK_DIR"
echo "  ROIs:   $ROI_IDS"
echo "  Mode:   $HIMREG_MODE"
echo "  Output: $OUTPUT"
echo ""

cd "$PROJECT_DIR"
python src/compare.py \
    --benchmark-dir "$BENCHMARK_DIR" \
    --roi-ids "$ROI_IDS" \
    --fixed-suffix "$FIXED_SUFFIX" \
    --moving-suffix "$MOVING_SUFFIX" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --himreg-mode "$HIMREG_MODE" \
    --affine-scales "10,8,6,4,3,2,1" \
    --affine-iterations "300,280,250,200,150,100,60" \
    --affine-lrs "0.005,0.003,0.002,0.001,0.0007,0.0004,0.0001" \
    --affine-loss-type mi_gradcc \
    --affine-optimizer-type adam \
    --no-affine-constrained \
    --rigid-lrs "1.0,0.5,0.3,0.1,0.05,0.02,0.005" \
    --stage-rigid-iters "150,140,125,100,75,50,30" \
    --stage-affine-iters "300,280,250,200,150,100,60" \
    --patience 40 \
    --min-delta 1e-5 \
    --himreg-mi-num-samples 15000 \
    --himreg-optimizer-scales-from-physical-shift \
    --mask-weighted-loss \
    --affine-center-mode image \
    --affine-required-valid-ratio 0.05 \
    --affine-invalid-sample-strategy lr_decay \
    --affine-invalid-lr-decay 0.5 \
    --affine-oob-penalty-weight 0.0 \
    --affine-oob-penalty-adaptive \
    --seed 42 \
    --deterministic \
    $WSIREG_FLAG

echo ""
echo "Refreshing figure exports under $OUTPUT/figures ..."
rm -rf "$OUTPUT/figures"
python scripts/make_figures.py --results-dir "$OUTPUT" --roi-ids "$ROI_IDS_PREFIXED"

echo ""
echo "Done. Results saved to $OUTPUT/"
