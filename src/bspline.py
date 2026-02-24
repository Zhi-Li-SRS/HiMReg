"""B-Spline nonlinear registration module.

Mirrors the default 'nl' parameter set from WSiReg:
  - 10-level multi-resolution image pyramid (RecursiveImagePyramid)
  - FinalGridSpacingInPhysicalUnits = 100
  - GridSpacingSchedule  [512, 392, 256, 128, 64, 32, 16, 4, 2, 1]
  - MaximumStepLength    [100, 90, 70, 50, 40, 30, 20, 10, 1, 1]
  - AdaptiveStochasticGradientDescent
  - AdvancedMattesMutualInformation, 32 bins, 50 000 random samples
  - No explicit bending-energy penalty
"""

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from src.asgd import AdaptiveStochasticGradientDescent
from src.data_load import Image
from src.losses import MutualInformation
from src.utils import downsample


# ───────────────────────── BSplineTransform ─────────────────────────

class BSplineTransform(nn.Module):
    """B-spline FFD parameterised by a control-point grid.

    Control points store displacements in *normalised* [-1, 1] coords.
    `forward()` returns a dense displacement field via cubic B-spline
    basis function evaluation (C2-smooth, approximating — matching Elastix).
    """

    def __init__(self, image_h: int, image_w: int, grid_spacing: int,
                 device: torch.device):
        super().__init__()
        self.image_h = image_h
        self.image_w = image_w
        self.grid_spacing = grid_spacing

        grid_h = image_h // grid_spacing + 3
        grid_w = image_w // grid_spacing + 3

        self.control_points = nn.Parameter(
            torch.zeros(1, 2, grid_h, grid_w, device=device)
        )

        # Pre-compute pixel → control-point-space mappings (rebuilt on refine).
        self._build_index_cache(image_h, image_w, grid_spacing, grid_h, grid_w, device)

    def _build_index_cache(self, image_h: int, image_w: int,
                           grid_spacing: int, grid_h: int, grid_w: int,
                           device: torch.device) -> None:
        """Pre-compute integer indices and B-spline weights for all pixels."""
        # Map pixel coords to control-point space (+1 for padding cell).
        uy = torch.arange(image_h, device=device, dtype=torch.float32) / grid_spacing + 1.0
        ux = torch.arange(image_w, device=device, dtype=torch.float32) / grid_spacing + 1.0

        iy = uy.long()
        ix = ux.long()
        ty = uy - iy.float()
        tx = ux - ix.float()

        # B-spline weights: 4 weights per pixel along each axis — [4, H] / [4, W]
        wy = self._bspline_weights(ty)  # [4, H]
        wx = self._bspline_weights(tx)  # [4, W]

        # Control point indices for the 4×4 support: [4, H] / [4, W]
        cy = torch.stack([
            (iy - 1).clamp(0, grid_h - 1),
            iy.clamp(0, grid_h - 1),
            (iy + 1).clamp(0, grid_h - 1),
            (iy + 2).clamp(0, grid_h - 1),
        ], dim=0)
        cx = torch.stack([
            (ix - 1).clamp(0, grid_w - 1),
            ix.clamp(0, grid_w - 1),
            (ix + 1).clamp(0, grid_w - 1),
            (ix + 2).clamp(0, grid_w - 1),
        ], dim=0)

        # Register as buffers (non-parameter, move with .to(device)).
        self.register_buffer("_wy", wy)
        self.register_buffer("_wx", wx)
        self.register_buffer("_cy", cy)
        self.register_buffer("_cx", cx)

    @staticmethod
    def _bspline_weights(t: torch.Tensor) -> torch.Tensor:
        """Cubic B-spline basis weights for fractional coordinate t in [0, 1).

        Returns [4, len(t)] tensor — one weight per basis function.
        """
        t2 = t * t
        t3 = t2 * t
        w0 = (1.0 - t) ** 3 / 6.0
        w1 = (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0
        w2 = (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0
        w3 = t3 / 6.0
        return torch.stack([w0, w1, w2, w3], dim=0)

    def forward(self) -> torch.Tensor:
        """Dense displacement field [1, H, W, 2] via cubic B-spline evaluation."""
        cp = self.control_points  # [B, 2, grid_h, grid_w]
        H, W = self.image_h, self.image_w

        # Separable 4×4 accumulation.
        disp = torch.zeros(cp.shape[0], 2, H, W, device=cp.device, dtype=cp.dtype)
        for j in range(4):
            wy_j = self._wy[j]  # [H]
            cy_j = self._cy[j]  # [H]
            for i in range(4):
                wx_i = self._wx[i]  # [W]
                cx_i = self._cx[i]  # [W]
                # Gather control point values: cp[:, :, cy_j, cx_i] → [B, 2, H, W]
                cp_val = cp[:, :, cy_j.unsqueeze(1), cx_i.unsqueeze(0)]
                weight = wy_j.unsqueeze(-1) * wx_i.unsqueeze(0)  # [H, W]
                disp = disp + cp_val * weight.unsqueeze(0).unsqueeze(0)

        return disp.permute(0, 2, 3, 1)  # [B, H, W, 2]

    def refine_grid(self, new_spacing: int, new_h: int, new_w: int) -> None:
        """Upsample control points for a finer grid / larger image.

        Displacement values are rescaled so that the *physical* offset
        they encode stays constant when the image resolution changes.
        """
        old_h, old_w = self.image_h, self.image_w

        new_grid_h = new_h // new_spacing + 3
        new_grid_w = new_w // new_spacing + 3

        with torch.no_grad():
            upsampled = F.interpolate(
                self.control_points.data,
                size=(new_grid_h, new_grid_w),
                mode="bicubic",
                align_corners=True,
            )
            # Scale displacements: normalised-coord space changes with
            # image resolution.  channel 0 = x  ↔ width,  channel 1 = y ↔ height.
            if new_w != old_w:
                upsampled[:, 0] *= old_w / new_w
            if new_h != old_h:
                upsampled[:, 1] *= old_h / new_h

        self.image_h = new_h
        self.image_w = new_w
        self.grid_spacing = new_spacing
        self.control_points = nn.Parameter(upsampled)

        # Rebuild pixel→control-point index cache for the new resolution.
        device = upsampled.device
        self._build_index_cache(new_h, new_w, new_spacing, new_grid_h, new_grid_w, device)


# ───────────────────── BSplineBendingEnergy (optional) ──────────────

class BSplineBendingEnergy(nn.Module):
    """Second-derivative penalty for a displacement field (optional)."""

    def forward(self, disp: torch.Tensor) -> torch.Tensor:
        d = disp.permute(0, 3, 1, 2)
        d2_xx = d[:, :, :, 2:] + d[:, :, :, :-2] - 2 * d[:, :, :, 1:-1]
        d2_yy = d[:, :, 2:, :] + d[:, :, :-2, :] - 2 * d[:, :, 1:-1, :]
        dx = d[:, :, :, 1:] - d[:, :, :, :-1]
        d2_xy = dx[:, :, 1:, :] - dx[:, :, :-1, :]
        h = min(d2_xx.shape[2], d2_yy.shape[2], d2_xy.shape[2])
        w = min(d2_xx.shape[3], d2_yy.shape[3], d2_xy.shape[3])
        return (d2_xx[:, :, :h, :w] ** 2
                + d2_yy[:, :, :h, :w] ** 2
                + 2 * d2_xy[:, :, :h, :w] ** 2).mean()


# ──────────────────── BSplineRegistration (WSiReg-style) ────────────

# WSiReg 'nl' defaults -------------------------------------------------
_DEFAULT_GRID_SCHEDULE = [512, 392, 256, 128, 64, 32, 16, 4, 2, 1]
_DEFAULT_MAX_STEPS     = [100., 90., 70., 50., 40., 30., 20., 10., 1., 1.]


class BSplineRegistration:
    """WSiReg / Elastix-style multi-resolution B-spline registration.

    Parameters
    ----------
    fixed_images, moving_images : Image
        HiMReg Image objects.
    num_resolutions : int
        Number of image-pyramid levels (Elastix *NumberOfResolutions*).
    final_grid_spacing : int
        Control-point spacing in pixels at **full resolution**
        (Elastix *FinalGridSpacingInPhysicalUnits*).
    grid_spacing_schedule : list[int]
        Per-level multiplier on *final_grid_spacing* (one per resolution).
    max_step_lengths : list[float]
        Per-level displacement cap in **pixels at full resolution**
        (Elastix *MaximumStepLength*).
    max_iterations : int
        Max iterations per resolution level.
    num_samples : int
        Random spatial samples per MI evaluation.
    num_histogram_bins : int
        Histogram bins for Mattes MI.
    bending_energy_weight : float
        Weight for optional bending-energy regularisation (0 = off, WSiReg default).
    tolerance : float
        Convergence slope threshold for early stopping.
    max_tolerance_iters : int
        Sliding window size for convergence check.
    init_affine : Tensor or None
        Affine matrix from the previous stage.
    """

    def __init__(
        self,
        fixed_images: Image,
        moving_images: Image,
        num_resolutions: int = 10,
        final_grid_spacing: int = 100,
        grid_spacing_schedule: Optional[List[int]] = None,
        max_step_lengths: Optional[List[float]] = None,
        max_iterations: int = 200,
        num_samples: int = 50_000,
        num_histogram_bins: int = 32,
        bending_energy_weight: float = 0.0,
        tolerance: float = 5e-4,
        max_tolerance_iters: int = 80,
        required_valid_ratio: float = 0.0,
        invalid_sample_strategy: str = "none",
        invalid_lr_decay: float = 0.5,
        oob_penalty_weight: float = 0.0,
        oob_penalty_adaptive: bool = True,
        init_affine: Optional[torch.Tensor] = None,
        align_corners: bool = True,
    ):
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims

        self.num_resolutions = num_resolutions
        self.final_grid_spacing = final_grid_spacing
        self.grid_spacing_schedule = list(grid_spacing_schedule or _DEFAULT_GRID_SCHEDULE)
        self.max_step_lengths = list(max_step_lengths or _DEFAULT_MAX_STEPS)
        self.max_iterations = max_iterations
        self.num_samples = num_samples
        self.num_histogram_bins = num_histogram_bins
        self.bending_energy_weight = float(bending_energy_weight)
        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.required_valid_ratio = float(required_valid_ratio or 0.0)
        self.invalid_sample_strategy = str(invalid_sample_strategy or "none").lower()
        self.invalid_lr_decay = float(invalid_lr_decay or 0.5)
        self.oob_penalty_weight = float(oob_penalty_weight or 0.0)
        self.oob_penalty_adaptive = bool(oob_penalty_adaptive)
        self.align_corners = align_corners

        # Pad schedule / step-length lists to match num_resolutions.
        while len(self.grid_spacing_schedule) < num_resolutions:
            self.grid_spacing_schedule.insert(0, self.grid_spacing_schedule[0])
        while len(self.max_step_lengths) < num_resolutions:
            self.max_step_lengths.insert(0, self.max_step_lengths[0])

        valid_invalid_strategies = {"none", "lr_decay", "early_stop"}
        if self.invalid_sample_strategy not in valid_invalid_strategies:
            raise ValueError(
                f"invalid_sample_strategy must be one of {sorted(valid_invalid_strategies)}, got {self.invalid_sample_strategy}"
            )

        # Loss: pure MI (Elastix default for B-spline).
        self.mi_loss_fn = MutualInformation(kernel_type="b-spline")
        self.mi_loss_fn.set_num_bins(num_histogram_bins)
        self.bending_energy_fn = BSplineBendingEnergy() if bending_energy_weight > 0 else None

        if init_affine is None:
            init_affine = (
                torch.eye(self.dims + 1, device=self.device)
                .unsqueeze(0)
                .repeat(self.fixed_images.size(), 1, 1)
            )
        self.affine = init_affine.detach()
        self.final_coordinates: Optional[torch.Tensor] = None

        # Convergence tracking.
        self._losses: deque = deque(maxlen=max_tolerance_iters)

    # ─── convergence helpers ───

    def _compute_slope(self) -> float:
        if len(self._losses) < 2:
            return 0.0
        x = np.arange(len(self._losses))
        y = np.array(self._losses)
        n = len(self._losses)
        denom = n * (x ** 2).sum() - x.sum() ** 2
        if denom == 0:
            return 0.0
        return float((n * np.dot(x, y) - x.sum() * y.sum()) / denom)

    def _converged(self, loss_val: float) -> bool:
        self._losses.append(loss_val)
        if len(self._losses) < self.max_tolerance_iters:
            return False
        return self._compute_slope() > self.tolerance

    @staticmethod
    def _coord_oob_penalty(coords_xy: torch.Tensor) -> torch.Tensor:
        oob = torch.relu(coords_xy.abs() - 1.0)
        return (oob * oob).mean()

    # ─── random MI sampling (new samples every iteration) ───

    def _sampled_mi(self, moved: torch.Tensor, fixed: torch.Tensor) -> torch.Tensor:
        """MI via random pixel sampling (Elastix ``NewSamplesEveryIteration``)."""
        b = moved.shape[0]
        pred_flat = moved.reshape(b, -1)
        tgt_flat = fixed.reshape(b, -1)
        n = pred_flat.shape[1]
        k = min(self.num_samples, n)
        if k <= 0 or k >= n:
            return self.mi_loss_fn(moved, fixed)
        # Uniform random indices (new every call → new every iteration).
        idx = torch.randint(0, n, (b, k), device=moved.device)
        return self.mi_loss_fn(pred_flat.gather(1, idx), tgt_flat.gather(1, idx))

    # ─── image pyramid ───

    @staticmethod
    def _downsample_factor(num_resolutions: int, level: int) -> int:
        """Elastix RecursiveImagePyramid: downsample by 2^(N-1-level)."""
        return 2 ** max(num_resolutions - 1 - level, 0)

    def _prepare_images(
        self,
        arrays: torch.Tensor,
        image_obj: Image,
        size_down: List[int],
        ds_factor: int,
    ) -> torch.Tensor:
        """Gaussian-blur + downsample to *size_down*."""
        if ds_factor > 1:
            sigmas = 0.5 * torch.tensor(
                [s / sd for s, sd in zip(arrays.shape[2:], size_down)],
                device=self.device,
            )
            return downsample(
                arrays, size=size_down,
                mode=image_obj.interpolate_mode, sigma=sigmas,
            )
        return F.interpolate(
            arrays, size=size_down,
            mode=image_obj.interpolate_mode, align_corners=self.align_corners,
        )

    # ─── main optimisation loop ───

    def optimize(self, save_transformed: bool = False):
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_pixel_to_physical()
        moving_p2t = self.moving_images.get_physical_to_pixel()
        full_h, full_w = fixed_arrays.shape[2:]
        affine_map = torch.matmul(
            moving_p2t, torch.matmul(self.affine, fixed_t2p)
        )[:, :-1]

        transformed_images = [] if save_transformed else None
        transform: Optional[BSplineTransform] = None

        for level in range(self.num_resolutions):
            # 1. Image pyramid ------------------------------------------------
            ds = self._downsample_factor(self.num_resolutions, level)
            h_down = max(full_h // ds, 4)
            w_down = max(full_w // ds, 4)

            # 2. Grid spacing at full resolution for this level ----------------
            gs_full = self.final_grid_spacing * self.grid_spacing_schedule[level]

            # Skip if grid is too coarse to provide any interior CPs.
            if gs_full >= max(full_h, full_w):
                continue

            # Grid spacing mapped to downsampled resolution.
            gs_down = max(int(round(gs_full / ds)), 1)
            # Need at least 1 interior control point.
            if h_down // gs_down < 1 and w_down // gs_down < 1:
                continue

            # 3. Downsample images ---------------------------------------------
            fixed_down = self._prepare_images(
                fixed_arrays, self.fixed_images, [h_down, w_down], ds,
            )
            moving_down = self._prepare_images(
                moving_arrays, self.moving_images,
                [max(moving_arrays.shape[2] // ds, 4),
                 max(moving_arrays.shape[3] // ds, 4)],
                ds,
            )

            # 4. Create / refine B-spline transform ----------------------------
            if transform is None:
                transform = BSplineTransform(h_down, w_down, gs_down, self.device)
            else:
                transform.refine_grid(gs_down, h_down, w_down)

            # 5. ASGD optimiser (auto-estimated `a`, reset per level) ----------
            optimizer = AdaptiveStochasticGradientDescent(
                [transform.control_points],
                a=None, alpha=1.0, max_iter=self.max_iterations,
            )

            # 6. Affine grid at this resolution --------------------------------
            affine_coords = F.affine_grid(
                affine_map, fixed_down.shape, align_corners=self.align_corners,
            )

            # MaximumStepLength in normalised [-1, 1] coords.
            max_step_norm = self.max_step_lengths[level] * 2.0 / max(full_h, full_w)

            self._losses.clear()
            n_cp = transform.control_points.numel() // 2
            pbar = tqdm(range(self.max_iterations))
            for i in pbar:
                optimizer.zero_grad()

                disp = transform()                       # [1, H, W, 2]
                coords = affine_coords + disp
                moved = F.grid_sample(
                    moving_down, coords,
                    mode="bilinear", align_corners=self.align_corners,
                )

                loss = self._sampled_mi(moved, fixed_down)

                moved_valid = F.grid_sample(
                    torch.ones_like(moving_down),
                    coords,
                    mode="nearest",
                    align_corners=self.align_corners,
                )
                valid_ratio = float(moved_valid.mean().item())

                if self.required_valid_ratio > 0.0 and valid_ratio < self.required_valid_ratio:
                    if self.invalid_sample_strategy == "early_stop":
                        print(
                            f"[WARN] bspline valid_ratio={valid_ratio:.4f} < required_valid_ratio={self.required_valid_ratio:.4f}; "
                            f"early-stop level {level} at iter {i+1}/{self.max_iterations}"
                        )
                        break
                    if self.invalid_sample_strategy == "lr_decay":
                        optimizer.state["a_value"] *= float(self.invalid_lr_decay)

                if self.oob_penalty_weight > 0.0:
                    penalty = self._coord_oob_penalty(coords)
                    weight = float(self.oob_penalty_weight)
                    if self.oob_penalty_adaptive and self.required_valid_ratio > 0.0:
                        overlap_scale = max(
                            (self.required_valid_ratio - valid_ratio) / max(self.required_valid_ratio, 1e-12), 0.0
                        )
                        weight *= (1.0 + 4.0 * overlap_scale)
                    loss = loss + (weight * penalty)

                if self.bending_energy_fn is not None:
                    loss = loss + self.bending_energy_weight * self.bending_energy_fn(disp)

                loss.backward()

                loss_val = loss.item()
                if not np.isfinite(loss_val):
                    optimizer.zero_grad()
                    pbar.set_description(
                        f"bspline L{level} gs={gs_full} NaN (skip)"
                    )
                    continue

                # --- apply MaximumStepLength clamp ---
                old_cp = transform.control_points.data.clone()
                optimizer.step()
                with torch.no_grad():
                    delta = transform.control_points.data - old_cp
                    delta.clamp_(-max_step_norm, max_step_norm)
                    transform.control_points.data = old_cp + delta

                pbar.set_description(
                    f"bspline L{level} gs={gs_full} cp={n_cp} "
                    f"loss={loss_val:.4f}"
                )
                if self._converged(loss_val):
                    print(
                        f"  B-spline converged at level {level} "
                        f"(gs_full={gs_full}) after {i+1} iters"
                    )
                    break

            # Cache final coordinates at this level.
            with torch.no_grad():
                self.final_coordinates = affine_coords + transform().detach()

            if save_transformed:
                with torch.no_grad():
                    out = F.grid_sample(
                        moving_down, self.final_coordinates,
                        mode="bilinear", align_corners=self.align_corners,
                    )
                transformed_images.append(out.cpu())

        return transformed_images if save_transformed else None

    # ─── public interface (same as DiffRegistration) ───

    def get_final_coordinates(self) -> torch.Tensor:
        if self.final_coordinates is None:
            raise RuntimeError("Run optimize() first.")
        return self.final_coordinates

    def apply_transform(self, moving_image: torch.Tensor) -> torch.Tensor:
        if self.final_coordinates is None:
            raise RuntimeError("Run optimize() first.")
        return F.grid_sample(
            moving_image, self.final_coordinates,
            mode="bilinear", align_corners=True,
        )
