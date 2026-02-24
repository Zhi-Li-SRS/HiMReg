#!/usr/bin/env bash
# =============================================================================
# Apply a saved deformation grid to images
#
# Edit the settings below, then run:
#   bash scripts/run_apply_transform.sh
# =============================================================================

# ── User Settings (edit these) ───────────────────────────────────────────────
COORDS="pred/maldi/redox_rescale_coords.npy"   # Path to saved coords.npy
IMAGES="data/maldi/redox_rescale.tif"           # Space-separated image paths
OUTPUT=""                                        # Output path (empty = auto: <image>_warped.tif)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

OUTPUT_FLAG=""
if [ -n "$OUTPUT" ]; then
    OUTPUT_FLAG="--output $OUTPUT"
fi

echo "=== Applying Transform ==="
echo "  Coords: $COORDS"
echo "  Images: $IMAGES"
echo ""

cd "$PROJECT_DIR"
python apply_transform.py --coords "$COORDS" --images $IMAGES $OUTPUT_FLAG

echo ""
echo "Done."
