from collections import deque
from typing import List, Optional

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
        optimizer="Adam",
        optimizer_params={},
        loss_params={},
        optimizer_lr=3e-3,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=7,
        tolerance=1e-6,
        max_tolerance_iters=1000,
        init_rigid=None,
        blur=True,
        align_corners=True,
        moved_mask=False,
    ):

        self.scales = scales
        self.iterations = iterations
        assert len(self.iterations) == len(self.scales), "Number of iterations must match number of scales"

        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims
        self.blur = blur
        self.align_corners = align_corners
        self.moved_mask = moved_mask

        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.losses = deque(maxlen=max_tolerance_iters)

        # Initialize loss function
        if loss_type == "mi":
            self.loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        elif loss_type == "cc":
            self.loss_fn = LNCC(
                kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
            )

        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        # Initialize affine parameters
        if init_rigid is not None:
            affine = init_rigid
        else:
            affine = torch.eye(self.dims, self.dims + 1).unsqueeze(0).repeat(fixed_images.size(), 1, 1)
        self.affine = nn.Parameter(affine.to(self.device))
        self.row = torch.zeros((fixed_images.size(), 1, self.dims + 1), device=self.device)
        self.row[:, 0, -1] = 1  # （batch, 1, self.dims + 1）but last element is 1s

        # Initialize optimizer
        if optimizer == "SGD":
            self.optimizer = SGD([self.affine], lr=optimizer_lr, **optimizer_params)
        elif optimizer == "Adam":
            self.optimizer = Adam([self.affine], lr=optimizer_lr, **optimizer_params)
        else:
            raise ValueError(f"Optimizer {optimizer} not supported")

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
            .repeat(self.fixed_images.size(), 1, 1)  # (B, self.dims, self.dims + 1)
        )

        transformed_images = [] if save_transformed else None

        for scale, iters in zip(self.scales, self.iterations):
            self.losses.clear()
            size_down = [max(int(s / scale), 1) for s in fixed_size]

            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor(
                    [sz / szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device
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

                if self.converged(loss.item()):
                    break

                pbar.set_description(f"scale: {scale}, iter: {i+1}/{iters}, loss: {loss.item():.4f}")

            if save_transformed:
                transformed_images.append(moved_image)

        if save_transformed:
            return transformed_images
