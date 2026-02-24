# TODO

- [x] Add a wsireg-aligned preset in benchmark CLI to enforce wsireg-like affine/nonlinear behavior.
- [x] Add bspline valid-sample safeguards (RequiredRatioOfValidSamples-like handling + OOB penalty hook).
- [x] Wire new bspline safeguard parameters through compare.py -> HiMReg -> BSplineRegistration path.
- [x] Update run script defaults to use ROI 2,3,4,5 and wsireg-aligned preset.
- [x] Keep baseline reporting label consistent as `Elastix` while preserving backend provenance.
- [x] Change `case_seed` generation to stable by `case_id` with `roiN -> seed + (N-1)` compatibility.
- [x] Prevent stale baseline warp artifacts (`simpleitk_warped.tif` vs `elastix_warped.tif`) from surviving across reruns.
- [x] Run benchmark and verify outputs/config reflect preset.

## Task A: HiMReg analysis
- [x] Inspect `HiMReg.py` and `config.yaml` to map the registration pipeline entry points and high-level prep steps.
- [x] Trace data handling/rescaling/pyramid/loss/optimizer flow through `src/config.py`, `src/data_load.py`, `src/utils.py`, `src/affinemorph.py`, `src/diffeomorph.py`, and `src/losses.py`.
- [x] Capture the dataflow graph (fixed/moving entry→rescale/normalize→pyramid→loss/optimizer/early stop) and note any potential stability/tuning concerns.

## Review

- Full benchmark completed for `roi2,roi3,roi4,roi5` with no case-level failures.
- Verified in `comparison_results/benchmark/metrics_all_cases.csv`:
  - `method` uses `Elastix` (backend remains `SimpleITK-affine-MI` for provenance).
  - `CaseSeed` is deterministic and compatible for ROI IDs (`43,44,45,46` for base seed 42).
- Verified in `comparison_results/benchmark/run_config.json`:
  - `wsireg_aligned_preset=true`, `init_mode=centroid`, `affine_loss_type=mi`,
    `affine_optimizer_type=asgd`, `affine_constrained=false`, `roi_ids=2,3,4,5`.
- Residual behavior gap remains in nonlinear deformation strength (notably ROI4):
  HiMReg B-spline grid remains much more non-rigid than affine baseline (`Elastix` label).

## Task B: Read-only analysis of wsireg defaults

- [x] Map wsireg default registration pipeline and configuration structure.
- [x] Document default pyramid (resolution) and rescale/preprocess strategies with file references.
- [x] List key parameters (pyramid levels, smoothing/resample, optimizer/metric steps, init strategy) and where defined in code.
- [x] Identify HiMReg alignment opportunities against wsireg defaults.

## Task C – benchmark cases 2-5 audit

- [x] Review `benchmark/`, `comparison_results/`, and `comparison_results/benchmark/run_config.json` to pin down what “benchmark 2-5” refers to and how the cases are configured.
- [x] Trace through `src/compare.py` and related configs to document how those cases are loaded, preprocessed, and evaluated.
- [x] Identify any evaluation differences (resampling, masks, ROI choices, metric ranges, etc.) that could make the reported results look worse.

## Task D: Regression point lookup

- [x] Search `git log`, `git diff`, `git show`, and `git grep` for mentions of rescale, pyramid scales, MI/CC, DiffAdam, early stopping, and match_anything_affine to identify deleted/modified commits tied to prior good performance.
- [x] For each candidate commit, note hash, summary, and key differences that might explain performance regression.
- [x] Summarize findings (or note missing evidence) and update lessons if corrections were needed.

## Task E: Port WSiReg strengths into HiMReg (main pipeline)

### Spec (agree before implementation)
- [x] Define the *mandatory* spatial rescale step in HiMReg main: move/resize moving to fixed grid (same as benchmark `sync_moving_scale_to_fixed`), and clarify the coordinate space that saved `coords.npy` refers to.
- [x] Decide default preprocessing in `config.yaml` to match benchmark: BF inversion + robust percentile normalize for fixed, robust normalize for moving.
- [x] Replace old brute-force `AutoParameterTuning` with a wsireg-style **preset/auto-defaults** strategy (no multi-run search): MI+ASGD affine, unconstrained affine, centroid init, valid-sample safeguards.

### Implementation
- [x] Extract `sync_moving_scale_to_fixed` (and any tiny helpers it needs) into a shared module under `src/` so `HiMReg.py` and `src/compare.py` use the same code path.
- [x] Extend config schema (`src/config.py` + `config.yaml`):
  - preprocessing defaults (`preprocessing.fixed`, `preprocessing.moving`)
  - mandatory scale-sync options (on by default)
  - wsireg-aligned preset flag
  - bspline safeguard params (`required_valid_ratio`, `invalid_sample_strategy`, `invalid_lr_decay`, `oob_penalty_*`)
- [x] Update `HiMReg.py` main to apply preprocessing + mandatory scale-sync before registration (benchmark parity).
- [x] Fix affine->bspline chaining so bspline uses `init_affine` instead of warping the moving image and “forgetting” the affine transform; ensure saved `coords.npy` represents the full transform used to produce the final warped image.
- [x] Make affine `required_valid_ratio` guardrails effective even when `mask_weighted_loss=false` (wsireg-like behavior).

### Verification
- [x] Run a single-case smoke check (`python HiMReg.py --config config.yaml`) and confirm outputs exist and `coords.npy` warps the same preprocessed/rescaled moving input.
- [x] Re-run benchmark ROI `2,3,4,5` (`python src/compare.py ...`) and confirm main pipeline and benchmark pipeline now share preprocessing/rescale/preset behavior.

### Review (2026-02-24)
- Main pipeline now shares benchmark rescale path via `src/spatial_sync.py`; coordinate-space metadata is emitted in `*_coords_meta.json`.
- Default preprocessing and wsireg-aligned defaults are enabled in `config.yaml` and validated in `src/config.py`.
- Affine->nonlinear chaining now preserves affine initialization through `init_affine` (no separate pre-warp stage).
- Benchmark completed successfully for ROI `2,3,4,5`; outputs are under `comparison_results/benchmark/`.
- Reviewer-facing overlays were generated for each ROI and summary figures were produced (`nature_figure.png/.pdf`, `summary_metrics.png`).
- User-reported overlay artifact was reproduced from stale `comparison_results/benchmark/figures/roi*/overlay_*.png` exports (older timestamps than latest benchmark outputs).
- Added automatic figure regeneration in `scripts/run_benchmark.sh` (clears stale `figures/` then runs `scripts/make_figures.py`).
- Tightened wsireg-aligned preset nonlinear guardrails (`required_valid_ratio>=0.2`, `oob_penalty_weight>=0.2`) in both benchmark (`src/compare.py`) and main config preset (`src/config.py`), and aligned `config.yaml` defaults.
