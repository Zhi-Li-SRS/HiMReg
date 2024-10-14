from functools import partial
from typing import List, Optional, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm

from losses import LNCC, MutualInformation
from utils import *


class DiffRegistration:
    def __init__(
        self,
        scales,
        iterations,
        fixed_images,
        moving_images,
        loss_type="mi",
        deformation_type="geodesic",
        optimizer="Adam",
        optimizer_params={},
        optimizer_lr=1e-4,
        integrator_n=6,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=3,
        smooth_warp_sigma=0.4,
        smooth_grad_sigma=1,
        loss_params={},
        tolerance=1e-6,
        max_tolerance_iters=1000,
        init_affine=None,
        custom_loss=None,
        blur=True,
        loss_device=None,
        progress_bar=True,
    ):
        self.scales = scales
        self.iterations = iterations
        assert len(self.iterations) == len(
            self.scales
        ), "Number of iterations must match number of scales"

        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims
        self.progress_bar = progress_bar
        self.blur = blur
        self.loss_device = loss_device
        self.convergence_monitor = ConvergenceMonitor(max_tolerance_iters, tolerance)

        # Initialize loss function
        if loss_type == "mi":
            self.loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        elif loss_type == "cc":
            self.loss_fn = LNCC(
                kernel_type=cc_kernel_type,
                spatial_dims=self.dims,
                kernel_size=cc_kernel_size**loss_params,
            )

        elif loss_type == "custom":
            self.loss_fn = custom_loss
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        # Initialize deformation
        if deformation_type == "geodesic":
            geodesic_params = {
                "fixed_images": fixed_images,
                "integrator_n": integrator_n,
                "optimizer": optimizer,
                "optimizer_lr": optimizer_lr,
                "optimizer_params": optimizer_params,
                "smoothing_grad_sigma": smooth_grad_sigma,
                "init_scale": scales[0],
            }
            self.warp = GeodesicShooting(**geodesic_params)
        else:
            raise ValueError(f"Invalid deformation type: {deformation_type}")

        self.smooth_warp_sigma = smooth_warp_sigma

        # Initialize affine
        if init_affine is None:
            init_affine = (
                torch.eye(self.dims + 1, device=fixed_images.device)
                .unsqueeze(0)
                .repeat(fixed_images.size(), 1, 1)
            )
        self.affine = init_affine.detach()

    def get_warped_coordinates(self, fixed_images, moving_images, shape=None):
        """Get the warped coordinates of the moving images."""
        fixed_arrays = fixed_images()
        if shape is None:
            shape = fixed_images.shape
        else:
            shape = [fixed_arrays.shape[0], 1] + list(shape)

        fixed_t2p = fixed_images.get_torch2phy()
        moving_p2t = moving_images.get_phy2torch()
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[
            :, :-1
        ]
        fixed_image_affinecoords = F.affine_grid(
            affine_map_init, shape, align_corners=True
        )
        warp_field = self.warp.get_warp().clone()

        if tuple(warp_field.shape[1:-1]) != tuple(shape[2:]):
            warp_field = F.interpolate(
                warp_field.permute(*self.warp.permute_vtoimg),
                size=shape[2:],
                mode="trilinear",
                align_corners=True,
            ).permute(*self.warp.permute_imgtov)

        if self.smooth_warp_sigma > 0:
            warp_gaussian = [
                gaussian_1d(s, truncated=2)
                for s in (
                    torch.zeros(self.dims, device=fixed_arrays.device)
                    + self.smooth_warp_sigma
                )
            ]
            warp_field = seperate_filter(
                warp_field.permute(*self.warp.permute_vtoimg), warp_gaussian
            ).permute(*self.warp.permute_imgtov)

        moved_coords = fixed_image_affinecoords + warp_field
        return moved_coords

    def evaluate(self, fixed_images, moving_images, shape=None):
        moving_arrays = moving_images()
        moved_coords = self.get_warped_coordinates(
            fixed_images, moving_images, shape=shape
        )
        moved_image = F.grid_sample(
            moving_arrays, moved_coords, mode="bilinear", align_corners=True
        )
        return moved_image

    def optimize(self, save_transformed=False):
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[
            :, :-1
        ]

        transformed_images = [] if save_transformed else None
        warp_gaussian = [
            gaussian_1d(s, truncated=2)
            for s in (
                torch.zeros(self.dims, device=fixed_arrays.device)
                + self.smooth_warp_sigma
            )
        ]

        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            size_down = [max(int(s / scale), 1) for s in fixed_size]

            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor(
                    [sz / szdown for sz, szdown in zip(fixed_size, size_down)],
                    device=fixed_arrays.device,
                )
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(
                    fixed_arrays,
                    size=size_down,
                    mode=self.fixed_images.interpolate_mode,
                    gaussians=gaussians,
                )
                moving_image_blur = seperate_filter(moving_arrays, gaussians)
            else:
                fixed_image_down = F.interpolate(
                    fixed_arrays,
                    size=size_down,
                    mode=self.fixed_images.interpolate_mode,
                    align_corners=True,
                )
                moving_image_blur = moving_arrays

            self.warp.set_size(size_down)
            fixed_image_affinecoords = F.affine_grid(
                affine_map_init, fixed_image_down.shape, align_corners=True
            )

            pbar = tqdm(range(iters)) if self.progress_bar else range(iters)
            for i in pbar:
                self.warp.set_zero_grad()
                warp_field = self.warp.get_warp()

                if self.smooth_warp_sigma > 0:
                    warp_field = seperate_filter(
                        warp_field.permute(*self.warp.permute_vtoimg), warp_gaussian
                    ).permute(*self.warp.permute_imgtov)

                moved_coords = fixed_image_affinecoords + warp_field
                moved_image = F.grid_sample(
                    moving_image_blur, moved_coords, mode="bilinear", align_corners=True
                )

                if self.loss_device is not None:
                    moved_image = moved_image.to(self.loss_device)
                    fixed_image_down = fixed_image_down.to(self.loss_device)

                loss = self.loss_fn(moved_image, fixed_image_down)
                loss.backward(retain_graph=True)
                self.warp.step()

                if self.convergence_monitor.converged(loss.item()):
                    break

                if self.progress_bar:
                    pbar.set_description(
                        f"scale: {scale}, iter: {i+1}/{iters}, loss: {loss.item():.4f}"
                    )

            if save_transformed:
                transformed_images.append(moved_image.detach().cpu())

        if save_transformed:
            return transformed_images


class GeodesicShooting(nn.Module):
    def __init__(
        self,
        fixed_images,
        integrator_n=6,
        optimizer="Adam",
        optimizer_lr=1e-2,
        optimizer_params={},
        smoothing_grad_sigma=0.5,
        init_scale=1,
    ) -> None:
        super().__init__()
        self.num_images = num_images = fixed_images.size()
        spatial_dims = fixed_images.shape[2:]
        if init_scale > 1:
            spatial_dims = [max(int(s / init_scale), 1) for s in spatial_dims]
        self.n_dims = len(spatial_dims)
        self.device = fixed_images.device

        self.permute_imgtov = (0, *range(2, self.n_dims + 2), 1)
        self.permute_vtoimg = (0, self.n_dims + 1, *range(1, self.n_dims + 1))

        velocity_field = torch.zeros(
            [num_images, *spatial_dims, self.n_dims],
            dtype=torch.float32,
            device=fixed_images.device,
        )

        self.smoothing_grad_sigma = smoothing_grad_sigma
        if smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [
                gaussian_1d(s, truncated=2)
                for s in (
                    torch.zeros(self.n_dims, device=fixed_images.device)
                    + smoothing_grad_sigma
                )
            ]

        self.initialize_grid(spatial_dims)
        self.register_parameter("velocity_field", nn.Parameter(velocity_field))
        self.attach_grad_hook()

        self.integrator_n = integrator_n

        self.optimizer = getattr(torch.optim, optimizer)(
            [self.velocity_field], lr=optimizer_lr, **optimizer_params
        )
        self.optimizer_lr = optimizer_lr
        self.optimizer_name = optimizer

    def attach_grad_hook(self):
        if self.smoothing_grad_sigma > 0:
            hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
            self.velocity_field.register_hook(hook)

    def initialize_grid(self, size):
        grid = F.affine_grid(
            torch.eye(self.n_dims, self.n_dims + 1, device=self.device)[None].expand(
                self.num_images, -1, -1
            ),
            [self.num_images, self.n_dims, *size],
            align_corners=True,
        )
        self.register_buffer("grid", grid)

    def set_zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def get_warp(self):
        if self.integrator_n == "auto":
            raise NotImplementedError("Automatic integrator_n not implemented yet")
        else:
            n = self.integrator_n
        warp = scaling_and_squaring(self.velocity_field, self.grid, n=n)
        return warp

    def get_inverse_warp(self, *args, **kwargs):
        return compute_inverse_warp_exp(self.get_warp().detach(), self.grid)

    def set_size(self, size):
        mode = "bilinear" if self.n_dims == 2 else "trilinear"
        old_shape = self.velocity_field.shape
        old_optimizer_state = self.optimizer.state_dict()

        velocity_field = F.interpolate(
            self.velocity_field.detach().permute(*self.permute_vtoimg),
            size=size,
            mode=mode,
            align_corners=True,
        ).permute(*self.permute_imgtov)
        velocity_field = nn.Parameter(velocity_field)
        self.register_parameter("velocity_field", velocity_field)
        self.attach_grad_hook()

        self.initialize_grid(size)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            [self.velocity_field], lr=self.optimizer_lr
        )

        state_dict = old_optimizer_state["state"]
        old_optimizer_state["param_groups"] = self.optimizer.state_dict()["param_groups"]
        for g in state_dict.keys():
            for k, v in state_dict[g].items():
                if isinstance(v, torch.Tensor) and v.shape == old_shape:
                    state_dict[g][k] = F.interpolate(
                        v.permute(*self.permute_vtoimg),
                        size=size,
                        mode=mode,
                        align_corners=True,
                    ).permute(*self.permute_imgtov)

        self.optimizer.load_state_dict(old_optimizer_state)
