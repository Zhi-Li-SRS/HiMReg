# HiMReg: Hierarchical Multimodal Image Registration Framework

HiMReg is a GPU-enabled hierarchical multimodal registration framework designed for cross-domain image co-registration. The default workflow aligns fluorescence (e.g. NADH/post-AF) or H&E imagery and can be extended to MALDI for metabolic and lipidomic correlation analysis. The pipeline couples global affine alignment with local diffeomorphic refinement, supports several similarity metrics, and exposes utilities for evaluation and visualization.

![Architecture](flow_figure/flowchart.png)

## Repository Overview
- `HiMReg.py`: Entry point that orchestrates image loading, affine registration, and diffeomorphic refinement based on `config.yaml`.
- `src/`: Core registration logic, including affine/diffeomorphic optimizers, loss functions, dataset helpers, and configuration utilities.
- `model/`: Optional Lightweight feature extractors (Stable Diffusion backbones, etc.) for future improvements.
- `cleandift_configs/`, `cleandift_figures/`, `comparison_results/`: Reference configurations and visual assets from CleanDIFT experiments and benchmarking.
- `data/`: Sample image volumes used in development and quick experimentation.

## Installation
```bash
git clone https://github.com/yourusername/HiMReg.git
cd HiMReg
pip install -r requirements.txt
```
Ensure you have a GPU and the corresponding PyTorch build if you intend to run the full pipeline on large images.

## Configure the Pipeline
HiMReg is now configured entirely through `config.yaml` located at the repository root. The configuration validates key parameters before execution, so keep list lengths aligned (e.g. `affine.scales` and `affine.iterations`).

```yaml
runtime:
  seed: 42
  deterministic: true

io:
  fixed: "data/K21/NADH-4.tif"
  moving: "data/K21/postaf-4.tif"
  output: "pred"

affine:
  scales: [6, 4, 2, 1]
  iterations: [400, 200, 100, 50]
  scale_dependent_lr: [0.003, 0.001, 0.0005, 0.0003]
  loss_type: "mi"
  patience: 30
  min_delta: 1.0e-5

diff:
  scales: [8, 6, 4, 2, 1]
  iterations: [800, 600, 400, 100, 50]
  loss_type: "mi"
  tolerance: 1.0e-3
  max_tolerance_iters: 60

registration:
  register_type: "diff"      # "affine" for affine-only runs
```

### Notes for Configuration
- Tune the `runtime` block to set the global seed and toggle deterministic execution.
- Adjust the `affine.patience`/`affine.min_delta` pair to control early stopping for the affine stage.
- `diff.tolerance` and `diff.max_tolerance_iters` govern when diffeomorphic refinement halts early.
- Adjust the `io` block to point to your fixed/moving volumes. Paths are validated before the run starts.
- `scale_dependent_lr` should mirror the number of affine scales to control the learning rate per pyramid level.
- Switch `loss_type` to `cc` or `dice` if you are working with structural similarity or segmentation masks.
- Set `registration.register_type` to `affine` when you only need global affine alignment.

## Run Registration
1. Edit `config.yaml` to reference your data and desired hyperparameters.
2. Launch the pipeline:
   ```bash
   python HiMReg.py
   ```

## Evaluation & Utilities
- `src/compare.py` offers utilities to benchmark HiMReg against Elastix using the same image pairs.
- `comparison_results/` contains example output plots (MI, DICE trends, qualitative comparisons).
- Loss definitions and custom metrics live in `src/losses.py` for further extension.

## License
This project is licensed under the MIT License. See `LICENSE` for details.

## Citation
If you use HiMReg in your research, please cite:

```bibtex
@article{paper,
  title = {},
  author = {},
  journal = {},
  year = {2025}
}
```
