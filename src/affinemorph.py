from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import tqdm

from src.asgd import AdaptiveStochasticGradientDescent
from src.data_load import Image
from src.losses import DICELoss, LNCC, MutualInformation
from src.preprocess import compute_tissue_mask, torch_gradient_magnitude
from src.utils import downsample


class AffineRegistration:
    def __init__(
        self,
        scales=[8, 6, 4, 2, 1],
        iterations=[800, 600, 400, 200, 100],
        fixed_images=Image,
        moving_images=Image,
        loss_type="mi",
        optimizer_params={},
        loss_params={},
        optimizer_lr=1e-3,
        scale_dependent_lr=None,
        patience=20,
        min_delta=1e-5,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=7,
        dice_kernel_size=7,
        dice_smooth=1e-6,
        dice_stride=3,
        tolerance=1e-4,
        max_tolerance_iters=500,
        init_rigid=None,
        init_rigid_params: Optional[torch.Tensor] = None,
        blur=True,
        align_corners=True,
        moved_mask=False,
        stages=None,
        stage_iterations=None,
        loss_weights=None,
        mask_weighted_loss=False,
        gradcc_sigma=1.0,
        mi_num_samples: Optional[int] = None,
        optimizer_scales_from_physical_shift: bool = True,
        stage_mi_bins: Optional[Dict[str, int]] = None,
        scale_dependent_patience: bool = True,
        center_mode: str = "none",
        required_valid_ratio: float = 0.0,
        invalid_sample_strategy: str = "none",
        oob_penalty_weight: float = 0.0,
        oob_penalty_adaptive: bool = True,
        invalid_lr_decay: float = 0.5,
        optimizer_type: str = "adam",
        asgd_a: Optional[float] = None,
        asgd_alpha: float = 1.0,
        constrained_affine: bool = False,
        scale_mi_bins: Optional[Dict[str, int]] = None,
        scale_loss_schedule: Optional[Dict[str, str]] = None,
        max_step_lengths: Optional[List[float]] = None,
        stage_lrs: Optional[Dict[str, List[float]]] = None,
    ):
        self.scales = scales
        self.iterations = iterations
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims
        self.blur = blur
        self.align_corners = align_corners
        self.moved_mask = moved_mask
        self.loss_type = loss_type
        self.optimizer_params = optimizer_params
        self.mask_weighted_loss = mask_weighted_loss
        self.gradcc_sigma = gradcc_sigma
        self.mi_num_samples = int(mi_num_samples) if mi_num_samples is not None else None
        self.optimizer_scales_from_physical_shift = bool(optimizer_scales_from_physical_shift)
        self.stage_mi_bins = dict(stage_mi_bins) if stage_mi_bins else {}
        self.scale_dependent_patience = bool(scale_dependent_patience)
        self.center_mode = str(center_mode or "none").lower()
        self.required_valid_ratio = float(required_valid_ratio or 0.0)
        self.invalid_sample_strategy = str(invalid_sample_strategy or "none").lower()
        self.oob_penalty_weight = float(oob_penalty_weight or 0.0)
        self.oob_penalty_adaptive = bool(oob_penalty_adaptive)
        self.invalid_lr_decay = float(invalid_lr_decay or 0.5)
        self.optimizer_type = str(optimizer_type or "adam").lower()
        self.asgd_a = asgd_a
        self.asgd_alpha = float(asgd_alpha)
        self.constrained_affine = bool(constrained_affine)
        self.scale_mi_bins = dict(scale_mi_bins) if scale_mi_bins else {}
        self.scale_loss_schedule = dict(scale_loss_schedule) if scale_loss_schedule else {}
        self.max_step_lengths = list(max_step_lengths) if max_step_lengths else None
        self.stage_lrs = dict(stage_lrs) if stage_lrs else None

        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.losses = deque(maxlen=max_tolerance_iters)

        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.patience_counter = 0

        self.scale_dependent_lr = scale_dependent_lr
        self.default_lr = optimizer_lr

        self.stages = stages or ["rigid", "affine"]
        self.stage_iterations = dict(stage_iterations or {})
        self.rigid_iterations = self._resolve_stage_iterations("rigid")
        self.affine_iterations = self._resolve_stage_iterations("affine")

        self.loss_weights = {"mi": 1.0, "gradcc": 0.25}
        if loss_weights:
            self.loss_weights.update(loss_weights)

        self._init_loss_function(
            loss_type,
            mi_kernel_type,
            cc_kernel_type,
            cc_kernel_size,
            dice_smooth,
            dice_kernel_size,
            dice_stride,
            loss_params,
        )
        self.init_affine_params(init_rigid, init_rigid_params=init_rigid_params)

        self.final_affine_matrix = None

        assert fixed_images.dims == moving_images.dims, (
            f"Dimension mismatch: fixed={fixed_images.dims}D, moving={moving_images.dims}D"
        )

        if self.dims != 2:
            raise ValueError("Rigid->Affine staged optimization currently supports 2D images only.")

        valid_center_modes = {"none", "image", "tissue"}
        if self.center_mode not in valid_center_modes:
            raise ValueError(f"center_mode must be one of {sorted(valid_center_modes)}, got {self.center_mode}")
        valid_invalid_strategies = {"none", "lr_decay", "early_stop"}
        if self.invalid_sample_strategy not in valid_invalid_strategies:
            raise ValueError(
                f"invalid_sample_strategy must be one of {sorted(valid_invalid_strategies)}, got {self.invalid_sample_strategy}"
            )

    def _resolve_stage_iterations(self, stage_name: str) -> List[int]:
        stage_iters = self.stage_iterations.get(stage_name)
        if stage_iters is None:
            if stage_name == "rigid":
                return [max(int(i // 2), 20) for i in self.iterations]
            return list(self.iterations)
        if len(stage_iters) != len(self.scales):
            raise ValueError(f"{stage_name} stage iterations must match scales length.")
        return stage_iters

    def validate_inputs(self, scales, iterations):
        if len(iterations) != len(scales):
            raise ValueError("Number of iterations must match number of scales")

    def _init_loss_function(
        self,
        loss_type,
        mi_kernel_type,
        cc_kernel_type,
        cc_kernel_size,
        dice_smooth,
        dice_kernel_size,
        dice_stride,
        loss_params,
    ):
        # Always create MI and CC/GradCC so they are available for per-scale switching.
        self.mi_loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        self.gradcc_loss_fn = LNCC(
            kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
        )
        self.cc_loss_fn = LNCC(
            kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
        )

        if loss_type == "mi":
            self.loss_fn = self.mi_loss_fn
        elif loss_type == "cc":
            self.loss_fn = self.cc_loss_fn
        elif loss_type == "dice":
            self.loss_fn = DICELoss(
                spatial_dims=self.dims,
                smooth=dice_smooth,
                kernel_size=dice_kernel_size,
                stride=dice_stride,
                **loss_params,
            )
        elif loss_type == "mi_gradcc":
            self.loss_fn = None
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

    def init_affine_params(self, init_rigid: Optional[torch.Tensor], init_rigid_params: Optional[torch.Tensor]) -> None:
        if init_rigid is not None:
            affine = init_rigid
        else:
            affine = torch.eye(self.dims, self.dims + 1).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)

        self.affine = nn.Parameter(affine.to(self.device))
        self.rigid_params = nn.Parameter(torch.zeros((self.fixed_images.size(), 3), device=self.device))
        self.row = torch.zeros((self.fixed_images.size(), 1, self.dims + 1), device=self.device)
        self.row[:, 0, -1] = 1
        if init_rigid_params is not None:
            init_rigid_params = torch.as_tensor(init_rigid_params, dtype=self.rigid_params.dtype, device=self.device)
            if init_rigid_params.shape != self.rigid_params.shape:
                raise ValueError(
                    f"init_rigid_params must have shape {tuple(self.rigid_params.shape)}, got {tuple(init_rigid_params.shape)}"
                )
            self.rigid_params.data.copy_(init_rigid_params)
        else:
            self._initialize_auto_rigid_translation()

    def _initialize_auto_rigid_translation(self):
        fixed = self.fixed_images().detach().cpu().numpy()
        moving = self.moving_images().detach().cpu().numpy()
        for batch_idx in range(fixed.shape[0]):
            fixed_mask = compute_tissue_mask(np.asarray(fixed[batch_idx, 0], dtype=np.float32), min_area=128)
            moving_mask = compute_tissue_mask(np.asarray(moving[batch_idx, 0], dtype=np.float32), min_area=128)
            fy, fx = np.nonzero(fixed_mask > 0)
            my, mx = np.nonzero(moving_mask > 0)
            if fy.size == 0 or my.size == 0:
                continue
            fixed_center = np.array([fx.mean(), fy.mean()], dtype=np.float32)
            moving_center = np.array([mx.mean(), my.mean()], dtype=np.float32)
            # Note: rigid/affine matrices operate in the same physical/pixel coordinate system
            # as fixed_t2p/moving_p2t. Use pixel-unit translation here (not normalized [-1, 1]).
            tx = float(fixed_center[0] - moving_center[0])
            ty = float(fixed_center[1] - moving_center[1])
            self.rigid_params.data[batch_idx, 1] = float(tx)
            self.rigid_params.data[batch_idx, 2] = float(ty)

    def get_affine_matrix(self):
        return torch.cat([self.affine, self.row], dim=1)

    def get_rigid_matrix(self):
        theta = self.rigid_params[:, 0]
        tx = self.rigid_params[:, 1]
        ty = self.rigid_params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        matrix = torch.zeros((self.fixed_images.size(), 2, 3), device=self.device)
        matrix[:, 0, 0] = cos_theta
        matrix[:, 0, 1] = -sin_theta
        matrix[:, 1, 0] = sin_theta
        matrix[:, 1, 1] = cos_theta
        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        return matrix

    def _compute_slope(self):
        if len(self.losses) < 2:
            return 0
        x = np.arange(len(self.losses))
        y = np.array(self.losses)
        xy_sum = np.dot(x, y)
        x_sum = x.sum()
        y_sum = y.sum()
        x_squared_sum = (x**2).sum()
        n_samples = len(self.losses)
        numerator = n_samples * xy_sum - x_sum * y_sum
        denominator = n_samples * x_squared_sum - x_sum**2
        if denominator == 0:
            return 0
        return numerator / denominator

    def converged(self, loss):
        self.losses.append(loss)
        if len(self.losses) < self.max_tolerance_iters:
            return False
        slope = self._compute_slope()
        return slope > self.tolerance

    def prepare_images_for_scale(
        self, fixed_arrays: torch.Tensor, moving_arrays: torch.Tensor, size_down: List[int], scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_size_down = [max(int(s / scale), 1) for s in moving_arrays.shape[2:]]

        if self.blur and scale > 1:
            fixed_sigmas = 0.5 * torch.tensor(
                [sz / szdown for sz, szdown in zip(fixed_arrays.shape[2:], size_down)],
                device=fixed_arrays.device,
            )
            moving_sigmas = 0.5 * torch.tensor(
                [sz / szdown for sz, szdown in zip(moving_arrays.shape[2:], moving_size_down)],
                device=moving_arrays.device,
            )
            fixed_image_down = downsample(
                fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, sigma=fixed_sigmas
            )
            # Downsample moving as well (not just blur) to mimic elastix-style pyramids.
            moving_image_down = downsample(
                moving_arrays,
                size=moving_size_down,
                mode=self.moving_images.interpolate_mode,
                sigma=moving_sigmas,
            )
        else:
            fixed_image_down = fixed_arrays if list(fixed_arrays.shape[2:]) == list(size_down) else F.interpolate(
                fixed_arrays,
                size=size_down,
                mode=self.fixed_images.interpolate_mode,
                align_corners=self.align_corners,
            )
            moving_image_down = moving_arrays if list(moving_arrays.shape[2:]) == list(moving_size_down) else F.interpolate(
                moving_arrays,
                size=moving_size_down,
                mode=self.moving_images.interpolate_mode,
                align_corners=self.align_corners,
            )
        return fixed_image_down, moving_image_down

    def get_lr_for_scale(self, scale_index):
        if self.scale_dependent_lr is not None and scale_index < len(self.scale_dependent_lr):
            return self.scale_dependent_lr[scale_index]
        return self.default_lr

    def early_stopping_check(self, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.patience

    def _early_stopping_check_with_patience(self, current_loss, patience):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= patience

    def _get_fixed_tissue_mask(self, fixed_arrays: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.mask_weighted_loss and not self.moved_mask:
            return None
        fixed_np = fixed_arrays.detach().cpu().numpy()
        masks = []
        for batch_idx in range(fixed_np.shape[0]):
            mask = compute_tissue_mask(np.asarray(fixed_np[batch_idx, 0], dtype=np.float32), min_area=128)
            masks.append(mask)
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(self.device).float().unsqueeze(1)
        return mask_tensor

    def _get_moving_tissue_mask(self, moving_arrays: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.mask_weighted_loss and not self.moved_mask:
            return None
        moving_np = moving_arrays.detach().cpu().numpy()
        masks = []
        for batch_idx in range(moving_np.shape[0]):
            mask = compute_tissue_mask(np.asarray(moving_np[batch_idx, 0], dtype=np.float32), min_area=128)
            masks.append(mask)
        mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).to(self.device).float().unsqueeze(1)
        return mask_tensor

    def _get_center_of_rotation(
        self,
        fixed_t2p: torch.Tensor,
        fixed_mask_full: Optional[torch.Tensor],
        fixed_size_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if self.center_mode == "none":
            return None

        b = fixed_t2p.shape[0]
        # Normalized center (0,0) maps to image center with align_corners=True.
        center_norm = torch.zeros((b, 3), device=fixed_t2p.device, dtype=fixed_t2p.dtype)
        center_norm[:, 2] = 1.0
        center_phys_h = torch.einsum("ntd,nd->nt", fixed_t2p, center_norm)
        center_phys = center_phys_h[:, :2]

        if self.center_mode != "tissue" or fixed_mask_full is None:
            return center_phys

        h, w = int(fixed_size_hw[0]), int(fixed_size_hw[1])
        centers = []
        for batch_idx in range(b):
            mask = fixed_mask_full[batch_idx, 0]
            idx = torch.nonzero(mask > 0.5, as_tuple=False)
            if idx.numel() == 0:
                centers.append(center_phys[batch_idx])
                continue
            y = idx[:, 0].float()
            x = idx[:, 1].float()
            x_norm = 2.0 * x / max(float(w - 1), 1.0) - 1.0
            y_norm = 2.0 * y / max(float(h - 1), 1.0) - 1.0
            pt = torch.stack(
                [x_norm.mean(), y_norm.mean(), torch.ones((), device=mask.device, dtype=x_norm.dtype)], dim=0
            ).to(fixed_t2p.dtype)
            c_phys_h = torch.einsum("td,d->t", fixed_t2p[batch_idx], pt)
            centers.append(c_phys_h[:2])
        return torch.stack(centers, dim=0)

    # --- Constrained affine parameterization ---

    @staticmethod
    def _build_constrained_affine_matrix(params: torch.Tensor) -> torch.Tensor:
        """Build a 2x3 affine matrix from 6 constrained parameters.

        params: [B, 6] — (raw_sx, raw_sy, theta, shear, tx, ty)
        Returns: [B, 2, 3]

        The decomposition is A = R(theta) @ diag(sx, sy) @ Shear(shear)
        where Shear = [[1, shear], [0, 1]], giving:
          a11 = sx * cos, a12 = sx*shear*cos - sy*sin
          a21 = sx * sin, a22 = sx*shear*sin + sy*cos
        """
        raw_sx = params[:, 0]
        raw_sy = params[:, 1]
        theta = params[:, 2]
        shear = params[:, 3]
        tx = params[:, 4]
        ty = params[:, 5]

        sx = F.softplus(raw_sx)
        sy = F.softplus(raw_sy)

        c = torch.cos(theta)
        s = torch.sin(theta)

        a11 = sx * c
        a12 = sx * shear * c - sy * s
        a21 = sx * s
        a22 = sx * shear * s + sy * c

        b = params.shape[0]
        matrix = torch.zeros(b, 2, 3, device=params.device, dtype=params.dtype)
        matrix[:, 0, 0] = a11
        matrix[:, 0, 1] = a12
        matrix[:, 0, 2] = tx
        matrix[:, 1, 0] = a21
        matrix[:, 1, 1] = a22
        matrix[:, 1, 2] = ty
        return matrix

    @staticmethod
    def _decompose_affine_matrix(M: torch.Tensor) -> torch.Tensor:
        """Decompose a [B, 2, 3] affine into 6 constrained params.

        Returns [B, 6]: (raw_sx, raw_sy, theta, shear, tx, ty)
        where raw_sx = inverse_softplus(sx), raw_sy = inverse_softplus(sy).

        Uses direct analytical decomposition of A = R @ diag(sx, sy) @ Shear:
          theta = atan2(a21, a11)
          sx = sqrt(a11^2 + a21^2)
          shear = (c*a12 + s*a22) / sx
          sy = c*a22 - s*a12
        """
        a11 = M[:, 0, 0]
        a12 = M[:, 0, 1]
        a21 = M[:, 1, 0]
        a22 = M[:, 1, 1]
        tx = M[:, 0, 2]
        ty = M[:, 1, 2]

        theta = torch.atan2(a21, a11)
        sx = torch.sqrt(a11 ** 2 + a21 ** 2).clamp_min(1e-6)
        c = torch.cos(theta)
        s = torch.sin(theta)

        shear = (c * a12 + s * a22) / sx
        sy = (c * a22 - s * a12).clamp_min(1e-6)

        # Inverse softplus: raw = log(exp(x) - 1)
        raw_sx = torch.log(torch.exp(sx) - 1.0 + 1e-8)
        raw_sy = torch.log(torch.exp(sy) - 1.0 + 1e-8)

        return torch.stack([raw_sx, raw_sy, theta, shear, tx, ty], dim=1)

    # --- Per-scale MI bins ---

    def _get_mi_bins_for_scale(self, scale: float) -> Optional[int]:
        """Return MI bin count for the given pyramid scale, or None to keep current."""
        if not self.scale_mi_bins:
            return None
        if scale >= 8:
            return self.scale_mi_bins.get("coarse", 10)
        if scale >= 3:
            return self.scale_mi_bins.get("mid", 20)
        return self.scale_mi_bins.get("fine", 48)

    # --- Multi-metric switching ---

    def _get_loss_type_for_scale(self, scale: float) -> str:
        """Return loss type for the given pyramid scale."""
        if not self.scale_loss_schedule:
            return self.loss_type
        if scale >= 6:
            return self.scale_loss_schedule.get("coarse", self.loss_type)
        if scale >= 3:
            return self.scale_loss_schedule.get("mid", self.loss_type)
        return self.scale_loss_schedule.get("fine", self.loss_type)

    def _compute_loss_for_scale(
        self,
        moved_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moved_mask: Optional[torch.Tensor],
        scale: float,
    ) -> torch.Tensor:
        """Compute loss with per-scale metric switching."""
        original_loss_type = self.loss_type
        self.loss_type = self._get_loss_type_for_scale(scale)
        loss = self._compute_loss(moved_image, fixed_image, moved_mask)
        self.loss_type = original_loss_type
        return loss

    @staticmethod
    def _apply_center_of_rotation(matrix_2x3: torch.Tensor, center_phys: Optional[torch.Tensor]) -> torch.Tensor:
        if center_phys is None:
            return matrix_2x3
        a = matrix_2x3[:, :, :2]
        t = matrix_2x3[:, :, 2]
        ac = torch.bmm(a, center_phys.unsqueeze(-1)).squeeze(-1)
        t_adj = t + center_phys - ac
        out = matrix_2x3.clone()
        out[:, :, 2] = t_adj
        return out

    @staticmethod
    def _coord_oob_penalty(coords_xy: torch.Tensor) -> torch.Tensor:
        oob = torch.relu(coords_xy.abs() - 1.0)
        return (oob * oob).mean()

    def _compute_loss(
        self,
        moved_image: torch.Tensor,
        fixed_image: torch.Tensor,
        moved_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        def mi_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
            if self.mi_num_samples is None or self.mi_num_samples <= 0:
                return self.mi_loss_fn(pred, target, mask=mask)

            b = pred.shape[0]
            pred_flat = pred.reshape(b, -1)
            target_flat = target.reshape(b, -1)
            n = pred_flat.shape[1]
            k = min(int(self.mi_num_samples), int(n))
            if k <= 0:
                return self.mi_loss_fn(pred, target, mask=mask)

            if mask is not None:
                # Mask-aware sampling (elastix-like): sample only from valid/tissue pixels.
                mask_flat = (mask.reshape(b, -1) > 0.05)
                idx_batches = []
                for batch_idx in range(b):
                    pos = torch.nonzero(mask_flat[batch_idx], as_tuple=False).squeeze(1)
                    if pos.numel() == 0:
                        pos = torch.arange(n, device=pred.device)
                    if pos.numel() >= k:
                        perm = torch.randperm(pos.numel(), device=pred.device)[:k]
                        idx_b = pos[perm]
                    else:
                        # Sample with replacement when the mask is tiny.
                        idx_b = pos[torch.randint(0, pos.numel(), (k,), device=pred.device)]
                    idx_batches.append(idx_b)
                idx = torch.stack(idx_batches, dim=0)
            else:
                # Stratified spatial sampling — divide pixels into k strata and
                # sample one random pixel per stratum.  This gives uniform spatial
                # coverage with lower variance than pure random (multinomial).
                stratum_size = n / k  # float division
                offsets = torch.rand(b, k, device=pred.device)  # [0, 1) per stratum
                base = torch.arange(k, device=pred.device, dtype=torch.float32).unsqueeze(0)  # (1, k)
                idx = ((base + offsets) * stratum_size).long().clamp(max=n - 1)  # (b, k)

            pred_s = pred_flat.gather(1, idx)
            target_s = target_flat.gather(1, idx)
            return self.mi_loss_fn(pred_s, target_s, mask=None)

        if self.loss_type == "mi_gradcc":
            mask = moved_mask if self.mask_weighted_loss else None
            mi_loss_val = mi_loss(moved_image, fixed_image, mask=mask)
            moved_grad = torch_gradient_magnitude(moved_image, sigma=self.gradcc_sigma)
            fixed_grad = torch_gradient_magnitude(fixed_image, sigma=self.gradcc_sigma)
            gradcc_loss = self.gradcc_loss_fn(moved_grad, fixed_grad, mask=mask)
            return self.loss_weights["mi"] * mi_loss_val + self.loss_weights["gradcc"] * gradcc_loss

        if self.loss_type == "mi":
            mask = moved_mask if self.mask_weighted_loss else None
            return mi_loss(moved_image, fixed_image, mask=mask)

        if self.loss_type == "cc":
            mask = moved_mask if self.mask_weighted_loss else None
            return self.cc_loss_fn(moved_image, fixed_image, mask=mask)
        return self.loss_fn(moved_image, fixed_image)

    def _compute_param_scales(
        self,
        stage_name: str,
        fixed_t2p: torch.Tensor,
        constrained_6: bool = False,
    ) -> torch.Tensor:
        """Compute parameter reparameterization scales (ITK-style PhysicalShift).

        Returns a tensor of per-parameter scale factors.  The optimizer works on
        ``raw_params`` and the effective parameter is ``raw_params * param_scales``.
        This avoids the momentum/variance mismatch that occurs when directly
        dividing gradients with Adam.

        When constrained_6=True, returns [B, 6] scales for the 6-param
        constrained affine: (raw_sx, raw_sy, theta, shear, tx, ty).
        """
        sx = fixed_t2p[:, 0, 0].abs().clamp_min(1e-6)
        sy = fixed_t2p[:, 1, 1].abs().clamp_min(1e-6)

        if not self.optimizer_scales_from_physical_shift:
            if stage_name == "rigid":
                return torch.ones(self.rigid_params.shape, device=self.device)
            if constrained_6:
                b = fixed_t2p.shape[0]
                return torch.ones(b, 6, device=self.device)
            return torch.ones(self.affine.shape, device=self.device)

        if stage_name == "rigid":
            radius = torch.maximum(sx, sy)
            scales = torch.ones_like(self.rigid_params)
            scales[:, 0] = 1.0 / radius
            return scales

        if stage_name == "affine" and constrained_6:
            # [raw_sx, raw_sy, theta, shear, tx, ty]
            radius = torch.maximum(sx, sy)
            b = fixed_t2p.shape[0]
            scales = torch.ones(b, 6, device=self.device)
            scales[:, 2] = 1.0 / radius  # theta
            return scales

        if stage_name == "affine":
            scales = torch.ones_like(self.affine)
            scales[:, 0, 0] = 1.0 / sx
            scales[:, 0, 1] = 1.0 / sy
            scales[:, 1, 0] = 1.0 / sx
            scales[:, 1, 1] = 1.0 / sy
            return scales

        return torch.ones(1, device=self.device)

    def _get_scale_patience(self, scale: float) -> int:
        """Return patience adjusted for the current pyramid scale."""
        if not self.scale_dependent_patience:
            return self.patience
        if scale >= 10:
            return self.patience * 4
        if scale >= 6:
            return self.patience * 3
        if scale >= 3:
            return self.patience * 2
        return self.patience

    def _build_rigid_matrix_from(self, params: torch.Tensor) -> torch.Tensor:
        """Build a 2x3 rigid matrix from a [B, 3] parameter tensor (theta, tx, ty)."""
        theta = params[:, 0]
        tx = params[:, 1]
        ty = params[:, 2]
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        matrix = torch.zeros((params.shape[0], 2, 3), device=params.device)
        matrix[:, 0, 0] = cos_theta
        matrix[:, 0, 1] = -sin_theta
        matrix[:, 1, 0] = sin_theta
        matrix[:, 1, 1] = cos_theta
        matrix[:, 0, 2] = tx
        matrix[:, 1, 2] = ty
        return matrix

    def _run_stage(
        self,
        stage_name: str,
        stage_iterations: List[int],
        fixed_arrays: torch.Tensor,
        moving_arrays: torch.Tensor,
        fixed_t2p: torch.Tensor,
        moving_p2t: torch.Tensor,
        fixed_mask_full: Optional[torch.Tensor],
        moving_mask_full: Optional[torch.Tensor],
        save_transformed: bool,
        transformed_images: Optional[List[torch.Tensor]],
    ):
        # --- Decide whether to use constrained 6-param affine ---
        use_constrained = (self.constrained_affine and stage_name == "affine")

        # --- Reparameterized optimisation ---
        param_scales = self._compute_param_scales(
            stage_name, fixed_t2p, constrained_6=use_constrained
        ).detach()

        if stage_name == "rigid":
            raw_params = nn.Parameter(self.rigid_params.data / param_scales.clamp_min(1e-12))
            def get_effective_matrix():
                effective = raw_params * param_scales
                return self._build_rigid_matrix_from(effective)
        elif stage_name == "affine" and use_constrained:
            # Decompose current affine into 6 constrained params.
            decomposed = self._decompose_affine_matrix(self.affine.data[:, :2, :])
            raw_params = nn.Parameter(decomposed / param_scales.clamp_min(1e-12))
            def get_effective_matrix():
                effective = raw_params * param_scales
                return self._build_constrained_affine_matrix(effective)
        elif stage_name == "affine":
            raw_params = nn.Parameter(self.affine.data / param_scales.clamp_min(1e-12))
            def get_effective_matrix():
                effective = raw_params * param_scales
                return torch.cat([effective, self.row], dim=1)[:, :2, :]
        else:
            raise ValueError(f"Unknown affine stage: {stage_name}")

        # --- Create optimizer (Adam or ASGD) ---
        if self.stage_lrs and stage_name in self.stage_lrs:
            initial_lr = self.stage_lrs[stage_name][0]
        else:
            initial_lr = self.get_lr_for_scale(0)
        if self.optimizer_type == "asgd":
            optimizer = AdaptiveStochasticGradientDescent(
                [raw_params],
                a=self.asgd_a,
                alpha=self.asgd_alpha,
                max_iter=stage_iterations[0] if stage_iterations else 200,
            )
        else:
            optimizer = Adam([raw_params], lr=initial_lr, **self.optimizer_params)

        fixed_size = fixed_arrays.shape[2:]
        center_phys = self._get_center_of_rotation(
            fixed_t2p=fixed_t2p,
            fixed_mask_full=fixed_mask_full,
            fixed_size_hw=(int(fixed_size[0]), int(fixed_size[1])),
        )
        init_grid = (
            torch.eye(self.dims, self.dims + 1)
            .to(self.fixed_images.device)
            .unsqueeze(0)
            .repeat(self.fixed_images.size(), 1, 1)
        )

        for scale_idx, (scale, iters) in enumerate(zip(self.scales, stage_iterations)):
            self.best_loss = float("inf")
            self.patience_counter = 0
            self.losses.clear()
            effective_patience = self._get_scale_patience(scale)

            # --- Per-scale MI bins ---
            mi_bins = self._get_mi_bins_for_scale(scale)
            if mi_bins is not None and hasattr(self, "mi_loss_fn"):
                self.mi_loss_fn.set_num_bins(int(mi_bins))

            # --- Per-scale LR / optimizer reset ---
            if self.stage_lrs and stage_name in self.stage_lrs:
                scale_lr = self.stage_lrs[stage_name][scale_idx]
            else:
                scale_lr = self.get_lr_for_scale(scale_idx)
            if self.optimizer_type == "asgd":
                optimizer.reset(max_iter=iters)
            else:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = scale_lr

            print(f"Stage: {stage_name}, Scale: {scale}, LR: {scale_lr}, Patience: {effective_patience}")
            size_down = [max(int(s / scale), 1) for s in fixed_size]
            fixed_image_down, moving_image_blur = self.prepare_images_for_scale(
                fixed_arrays, moving_arrays, size_down, scale
            )

            if fixed_mask_full is not None:
                fixed_mask_down = F.interpolate(
                    fixed_mask_full, size=size_down, mode="nearest"
                )
            else:
                fixed_mask_down = None

            if moving_mask_full is not None:
                moving_mask_down = F.interpolate(
                    moving_mask_full, size=moving_image_blur.shape[2:], mode="nearest"
                )
            else:
                moving_mask_down = None

            fixed_image_coords = F.affine_grid(
                init_grid, fixed_image_down.shape, align_corners=self.align_corners
            )
            fixed_image_coords_homo = torch.cat(
                [
                    fixed_image_coords,
                    torch.ones(list(fixed_image_coords.shape[:-1]) + [1], device=fixed_image_coords.device),
                ],
                dim=-1,
            )
            fixed_image_coords_homo = torch.einsum("ntd,n...d->n...t", fixed_t2p, fixed_image_coords_homo)

            pbar = tqdm(range(iters))
            for i in pbar:
                optimizer.zero_grad()
                affine_stage_matrix = self._apply_center_of_rotation(get_effective_matrix(), center_phys)
                affine_h = torch.cat([affine_stage_matrix, self.row], dim=1)
                coords = torch.einsum("ntd,n...d->n...t", affine_h, fixed_image_coords_homo)
                coords = torch.einsum("ntd,n...d->n...t", moving_p2t, coords)
                moved_image = F.grid_sample(
                    moving_image_blur, coords[..., :-1], mode="bilinear", align_corners=self.align_corners
                )

                moved_mask = None
                moved_valid = None
                if (
                    self.required_valid_ratio > 0.0
                    or fixed_mask_down is not None
                    or moving_mask_down is not None
                ):
                    moved_valid = F.grid_sample(
                        torch.ones_like(moving_image_blur),
                        coords[..., :-1],
                        mode="nearest",
                        align_corners=self.align_corners,
                    )
                    valid_ratio = float(moved_valid.mean().item())
                else:
                    valid_ratio = 1.0

                if fixed_mask_down is not None:
                    fixed_valid = fixed_mask_down * moved_valid
                    denom = fixed_mask_down.sum(dim=(1, 2, 3)).clamp_min(1.0)
                    numer = fixed_valid.sum(dim=(1, 2, 3))
                    valid_ratio = float((numer / denom).mean().item())
                    if moving_mask_down is not None:
                        warped_moving_mask = F.grid_sample(
                            moving_mask_down,
                            coords[..., :-1],
                            mode="bilinear",
                            align_corners=self.align_corners,
                        )
                        moved_mask = fixed_valid * torch.clamp(warped_moving_mask, 0.0, 1.0)
                    else:
                        moved_mask = fixed_valid

                # Use per-scale metric switching.
                loss = self._compute_loss_for_scale(moved_image, fixed_image_down, moved_mask, scale)

                if self.required_valid_ratio > 0.0 and valid_ratio < self.required_valid_ratio:
                    if self.invalid_sample_strategy == "early_stop":
                        print(
                            f"[WARN] valid_ratio={valid_ratio:.4f} < required_valid_ratio={self.required_valid_ratio:.4f}; "
                            f"early-stop stage {stage_name} at iter {i+1}/{iters}"
                        )
                        break
                    if self.invalid_sample_strategy == "lr_decay":
                        if self.optimizer_type == "asgd":
                            # ASGD uses its own decaying schedule; scale `a` instead of `lr`
                            optimizer.state["a_value"] *= float(self.invalid_lr_decay)
                        else:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] *= float(self.invalid_lr_decay)

                if self.oob_penalty_weight > 0.0:
                    penalty = self._coord_oob_penalty(coords[..., :-1])
                    weight = float(self.oob_penalty_weight)
                    if self.oob_penalty_adaptive and self.required_valid_ratio > 0.0:
                        overlap_scale = max(
                            (self.required_valid_ratio - valid_ratio) / max(self.required_valid_ratio, 1e-12), 0.0
                        )
                        weight *= (1.0 + 4.0 * overlap_scale)
                    loss = loss + (weight * penalty)
                loss.backward()

                # --- MaximumStepLength clamping (per pyramid level) ---
                if self.max_step_lengths and scale_idx < len(self.max_step_lengths):
                    old_params_snap = raw_params.data.clone()
                    optimizer.step()
                    with torch.no_grad():
                        delta = raw_params.data - old_params_snap
                        max_step = self.max_step_lengths[scale_idx]
                        delta.clamp_(-max_step, max_step)
                        raw_params.data.copy_(old_params_snap + delta)
                else:
                    optimizer.step()

                current_loss = loss.item()
                if self._early_stopping_check_with_patience(current_loss, effective_patience):
                    print(
                        f"Early stopping at stage {stage_name} iteration {i+1}/{iters} "
                        f"(patience {effective_patience})"
                    )
                    break
                pbar.set_description(
                    f"stage: {stage_name}, scale: {scale}, iter: {i+1}/{iters}, loss: {current_loss:.4f}"
                )

            # Sync the canonical parameters so external code (get_rigid_matrix,
            # get_affine_matrix, optimize()) sees the final values.
            with torch.no_grad():
                effective = raw_params * param_scales
                if stage_name == "rigid":
                    self.rigid_params.data.copy_(effective)
                elif use_constrained:
                    # Rebuild the full 2x3 matrix and store it.
                    self.affine.data.copy_(
                        self._build_constrained_affine_matrix(effective).detach()
                    )
                else:
                    self.affine.data.copy_(effective)

            if save_transformed and transformed_images is not None:
                transformed_images.append(moved_image.detach().cpu())

            if stage_name == "rigid":
                stage_matrix = self._apply_center_of_rotation(self.get_rigid_matrix(), center_phys)
                current_full = torch.cat([stage_matrix, self.row], dim=1)
            else:
                stage_matrix = self._apply_center_of_rotation(self.affine[:, :2, :], center_phys)
                current_full = torch.cat([stage_matrix, self.row], dim=1)
            self.final_affine_matrix = torch.matmul(moving_p2t, torch.matmul(current_full, fixed_t2p))

    def optimize(self, save_transformed=False):
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_pixel_to_physical()
        moving_p2t = self.moving_images.get_physical_to_pixel()

        transformed_images = [] if save_transformed else None
        fixed_mask_full = self._get_fixed_tissue_mask(fixed_arrays)
        moving_mask_full = self._get_moving_tissue_mask(moving_arrays)

        self.final_affine_matrix = torch.matmul(moving_p2t, torch.matmul(self.get_affine_matrix(), fixed_t2p))

        for stage_name in self.stages:
            if stage_name == "rigid":
                stage_iterations = self.rigid_iterations
            elif stage_name == "affine":
                if stage_name in self.stages and "rigid" in self.stages:
                    self.affine.data.copy_(self.get_rigid_matrix().detach())
                stage_iterations = self.affine_iterations
            else:
                raise ValueError(f"Unknown stage name: {stage_name}")

            self._run_stage(
                stage_name=stage_name,
                stage_iterations=stage_iterations,
                fixed_arrays=fixed_arrays,
                moving_arrays=moving_arrays,
                fixed_t2p=fixed_t2p,
                moving_p2t=moving_p2t,
                fixed_mask_full=fixed_mask_full,
                moving_mask_full=moving_mask_full,
                save_transformed=save_transformed,
                transformed_images=transformed_images,
            )

        return transformed_images if save_transformed else None

    def get_final_transform(self):
        if self.final_affine_matrix is None:
            raise RuntimeError("Final transformation matrix not computed. Run optimize() first.")
        return self.final_affine_matrix

    def apply_transform(self, moving_image):
        if self.final_affine_matrix is None:
            raise RuntimeError("Final transformation matrix not computed. Run optimize() first.")

        grid = F.affine_grid(self.final_affine_matrix[:, :-1], moving_image.shape, align_corners=True)
        transformed = F.grid_sample(moving_image, grid, mode="bilinear", align_corners=True)
        return transformed
