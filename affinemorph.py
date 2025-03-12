from collections import deque
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm

from data_load import Image
from losses import LNCC, MutualInformation
from utils import *


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
        scale_dependent_lr=None,  # Optional list of learning rates for each scale
        patience=10,  # Patience for early stopping
        min_delta=1e-5,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=7,
        tolerance=1e-4,
        max_tolerance_iters=500,
        init_rigid=None,
        blur=True,
        align_corners=True,
        moved_mask=False,
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

        # Set convergence params
        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.losses = deque(maxlen=max_tolerance_iters)

        # Eearly stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.patience_counter = 0

        # Scale-dependent learning rates
        self.scale_dependent_lr = scale_dependent_lr
        self.default_lr = optimizer_lr

        self._init_loss_function(loss_type, mi_kernel_type, cc_kernel_type, cc_kernel_size, loss_params)
        self.init_affine_params(init_rigid)

        # Initialize optimizer
        initial_lr = self.get_lr_for_scale(0)
        self.optimizer = Adam([self.affine], lr=initial_lr, **optimizer_params)

        # Initialize final transformation matrix
        self.final_affine_matrix = None

    def validate_inputs(self, scales, iterations):
        """Validate input parameters."""
        if len(iterations) != len(scales):
            raise ValueError("Number of iterations must match number of scales")

    def _init_loss_function(self, loss_type, mi_kernel_type, cc_kernel_type, cc_kernel_size, loss_params):
        """Initialize loss function."""
        if loss_type == "mi":
            self.loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        elif loss_type == "cc":
            self.loss_fn = LNCC(
                kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
            )
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

    def init_affine_params(self, init_rigid: Optional[torch.Tensor]) -> None:
        """Initialize affine transformation parameters."""
        if init_rigid is not None:
            affine = init_rigid
        else:
            affine = torch.eye(self.dims, self.dims + 1).unsqueeze(0).repeat(self.fixed_images.size(), 1, 1)

        self.affine = nn.Parameter(affine.to(self.device))
        self.row = torch.zeros((self.fixed_images.size(), 1, self.dims + 1), device=self.device)
        self.row[:, 0, -1] = 1

    def get_affine_matrix(self):
        return torch.cat([self.affine, self.row], dim=1)

    def _compute_slope(self):
        """Compute the slope of the best-fit line using simple linear regression."""
        if len(self.losses) < 2:
            return 0

        x = np.arange(len(self.losses))
        y = np.array(self.losses)

        xy_sum = np.dot(x, y)
        x_sum = x.sum()
        y_sum = y.sum()
        x_squared_sum = (x**2).sum()
        N = len(self.losses)

        numerator = N * xy_sum - x_sum * y_sum
        denominator = N * x_squared_sum - x_sum**2
        if denominator == 0:
            return 0
        slope = numerator / denominator
        return slope

    def converged(self, loss):
        """Check if the loss has increased (i.e., slope > threshold)."""
        self.losses.append(loss)
        if len(self.losses) < self.max_tolerance_iters:
            return False
        else:
            slope = self._compute_slope()
            return slope > self.tolerance

    def prepare_images_for_scale(
        self, fixed_arrays: torch.Tensor, moving_arrays: torch.Tensor, size_down: List[int], scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare images for current scale level."""
        if self.blur and scale > 1:
            sigmas = 0.5 * torch.tensor(
                [sz / szdown for sz, szdown in zip(fixed_arrays.shape[2:], size_down)],
                device=fixed_arrays.device,
            )
            fixed_image_down = downsample(
                fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, sigma=sigmas
            )
            moving_image_blur = downsample(
                moving_arrays,
                size=moving_arrays.shape[2:],
                mode=self.moving_images.interpolate_mode,
                sigma=sigmas,
            )
        else:
            fixed_image_down = F.interpolate(
                fixed_arrays,
                size=size_down,
                mode=self.fixed_images.interpolate_mode,
                align_corners=self.align_corners,
            )
            moving_image_blur = moving_arrays

        return fixed_image_down, moving_image_blur

    def get_lr_for_scale(self, scale_index):
        """Get learning rate for the current scale"""
        if self.scale_dependent_lr is not None and scale_index < len(self.scale_dependent_lr):
            return self.scale_dependent_lr[scale_index]
        return self.default_lr

    def early_stopping_check(self, current_loss):
        """Check if early stopping criteria is met"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.patience

    def optimize(self, save_transformed=False):
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_pixel_to_physical()
        moving_p2t = self.moving_images.get_physical_to_pixel()
        fixed_size = fixed_arrays.shape[2:]
        init_grid = (
            torch.eye(self.dims, self.dims + 1)
            .to(self.fixed_images.device)
            .unsqueeze(0)
            .repeat(self.fixed_images.size(), 1, 1)  # (B, 2, 3)
        )

        transformed_images = [] if save_transformed else None

        self.final_affine_matrix = torch.matmul(moving_p2t, torch.matmul(self.get_affine_matrix(), fixed_t2p))

        for scale_idx, (scale, iters) in enumerate(zip(self.scales, self.iterations)):
            # Reset early stopping counters for this scale
            self.best_loss = float("inf")
            self.patience_counter = 0
            self.losses.clear()

            scale_lr = self.get_lr_for_scale(scale_idx)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = scale_lr

            print(f"Scale: {scale}, Learning rate: {scale_lr}")

            size_down = [max(int(s / scale), 1) for s in fixed_size]

            fixed_image_down, moving_image_blur = self.prepare_images_for_scale(
                fixed_arrays, moving_arrays, size_down, scale
            )

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
                self.optimizer.zero_grad()
                affinemat = self.get_affine_matrix()
                coords = torch.einsum("ntd,n...d->n...t", affinemat, fixed_image_coords_homo)
                coords = torch.einsum("ntd,n...d->n...t", moving_p2t, coords)
                moved_image = F.grid_sample(
                    moving_image_blur, coords[..., :-1], mode="bilinear", align_corners=self.align_corners
                )

                if self.moved_mask:
                    moved_mask = F.grid_sample(
                        torch.ones_like(moving_image_blur),
                        coords[..., :-1],
                        mode="nearest",
                        align_corners=self.align_corners,
                    )
                else:
                    moved_mask = None

                loss = self.loss_fn(moved_image, fixed_image_down)
                loss.backward()
                self.optimizer.step()

                current_loss = loss.item()
                if self.early_stopping_check(current_loss):
                    print(
                        f"Early stopping at iteration {i+1}/{iters} - Loss hasn't improved for {self.patience} iterations"
                    )
                    break

                pbar.set_description(f"scale: {scale}, iter: {i+1}/{iters}, loss: {current_loss:.4f}")

            if save_transformed:
                transformed_images.append(moved_image.detach().cpu())

            # Update cumulative transformation after each scale
            current_matrix = torch.matmul(moving_p2t, torch.matmul(self.get_affine_matrix(), fixed_t2p))
            self.final_affine_matrix = current_matrix

        return transformed_images if save_transformed else None

    def get_final_transform(self):
        """Get the final affine transformation matrix."""
        if self.final_affine_matrix is None:
            raise RuntimeError("Final transformation matrix not computed. Run optimize() first.")
        return self.final_affine_matrix

    def apply_transform(self, moving_image):
        """Apply the final transformation to a new image."""
        if self.final_affine_matrix is None:
            raise RuntimeError("Final transformation matrix not computed. Run optimize() first.")

        grid = F.affine_grid(self.final_affine_matrix[:, :-1], moving_image.shape, align_corners=True)
        transformed = F.grid_sample(moving_image, grid, mode="bilinear", align_corners=True)
        return transformed
