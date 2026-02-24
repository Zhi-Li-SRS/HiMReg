#!/usr/bin/env bash
# =============================================================================
# HiMReg Registration
#
# Edit the paths and settings below, then run:
#   bash scripts/run_register.sh
# =============================================================================

# ── User Settings (edit these) ───────────────────────────────────────────────
FIXED="data/maldi/he_rescale.tif"       # Path to fixed/reference image
MOVING="data/maldi/redox_rescale.tif"   # Path to moving image to align
OUTPUT="pred"                            # Output directory
MODE="bspline"                           # Registration mode: affine | diff | bspline
DEVICE="cuda"                            # Device: cuda | cpu
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Generate config with tuned defaults
TMP_CONFIG=$(mktemp /tmp/himreg_config_XXXXXX.yaml)
trap "rm -f $TMP_CONFIG" EXIT

cat > "$TMP_CONFIG" <<EOF
runtime:
  seed: 42
  deterministic: true

io:
  fixed: "$FIXED"
  moving: "$MOVING"
  output: "$OUTPUT"
  device: "$DEVICE"

affine:
  scales: [10, 8, 6, 4, 3, 2, 1]
  iterations: [300, 280, 250, 200, 150, 100, 60]
  scale_dependent_lr: [0.005, 0.003, 0.002, 0.001, 0.0007, 0.0004, 0.0001]
  loss_type: "mi_gradcc"
  optimizer_type: "adam"
  patience: 40
  min_delta: 1.0e-5
  mi_num_samples: 15000
  optimizer_scales_from_physical_shift: true
  stages: ["rigid", "affine"]
  stage_iterations:
    rigid: [150, 140, 125, 100, 75, 50, 30]
    affine: [300, 280, 250, 200, 150, 100, 60]
  stage_lrs:
    rigid: [1.0, 0.5, 0.3, 0.1, 0.05, 0.02, 0.005]
  stage_mi_bins:
    rigid: 16
    affine: 32
  loss_weights:
    mi: 1.0
    gradcc: 0.25
  mask_weighted_loss: true
  gradcc_sigma: 1.0
  scale_dependent_patience: true
  center_mode: "image"
  required_valid_ratio: 0.05
  invalid_sample_strategy: "lr_decay"
  invalid_lr_decay: 0.5
  oob_penalty_weight: 0.0
  oob_penalty_adaptive: true

diff:
  scales: [8, 6, 4, 3, 2, 1]
  iterations: [250, 200, 180, 150, 100, 60]
  loss_type: "mi_gradcc"
  tolerance: 5.0e-4
  max_tolerance_iters: 80
  mask_weighted_loss: false
  regularization_weight: 0.05

bspline:
  num_resolutions: 10
  final_grid_spacing: 100
  grid_spacing_schedule: [512, 392, 256, 128, 64, 32, 16, 4, 2, 1]
  max_step_lengths: [100.0, 90.0, 70.0, 50.0, 40.0, 30.0, 20.0, 10.0, 1.0, 1.0]
  max_iterations: 200
  num_samples: 50000
  num_histogram_bins: 32
  bending_energy_weight: 0.0
  tolerance: 5.0e-4
  max_tolerance_iters: 80

registration:
  register_type: "$MODE"
EOF

echo "=== HiMReg Registration ==="
echo "  Fixed:  $FIXED"
echo "  Moving: $MOVING"
echo "  Output: $OUTPUT"
echo "  Mode:   $MODE"
echo "  Device: $DEVICE"
echo ""

cd "$PROJECT_DIR"
python HiMReg.py --config "$TMP_CONFIG"

echo ""
echo "Done. Results saved to $OUTPUT/"
