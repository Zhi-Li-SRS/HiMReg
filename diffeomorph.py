from copy import deepcopy
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
        deformation_type="compositive",
        optimizer="Adam",
        optimizer_params={},
        optimizer_lr=0.25,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=7,
        smooth_warp_sigma=0.4,
        smooth_grad_sigma=1,
        loss_params={},
        tolerance=1e-6,
        max_tolerance_iters=1000,
        init_affine=None,
        custom_loss=None,
        blur=True,
        align_corners=True,
        loss_device=None,
        progress_bar=True,
    ):
        self.scales = scales
        self.iterations = iterations
        assert len(self.iterations) == len(self.scales), "Number of iterations must match number of scales"

        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims
        self.progress_bar = progress_bar
        self.blur = blur
        self.align_corners = align_corners
        self.loss_device = loss_device
        self.convergence_monitor = ConvergenceMonitor(max_tolerance_iters, tolerance)

        # Initialize loss function
        if loss_type == "mi":
            self.loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        elif loss_type == "cc":
            self.loss_fn = LNCC(
                kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
            )

        elif loss_type == "custom":
            self.loss_fn = custom_loss
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

        compositive_params = {
            "fixed_images": fixed_images,
            "moving_images": moving_images,
            "optimizer": optimizer,
            "optimizer_lr": optimizer_lr,
            "optimizer_params": optimizer_params,
            "init_scale": scales[0],
            "smoothing_grad_sigma": smooth_grad_sigma,
            "smoothing_warp_sigma": smooth_warp_sigma,
        }
        self.warp = CompositiveWarp(**compositive_params)
        smooth_warp_sigma = 0

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
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]
        fixed_image_affinecoords = F.affine_grid(affine_map_init, shape, align_corners=self.align_corners)
        warp_field = self.warp.get_warp().clone()

        if tuple(warp_field.shape[1:-1]) != tuple(shape[2:]):
            warp_field = F.interpolate(
                warp_field.permute(*self.warp.permute_vtoimg),
                size=shape[2:],
                mode="trilinear",
                align_corners=self.align_corners,
            ).permute(*self.warp.permute_imgtov)

        if self.smooth_warp_sigma > 0:
            warp_gaussian = [
                gaussian_1d(s, truncated=2)
                for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)
            ]
            warp_field = seperate_filter(
                warp_field.permute(*self.warp.permute_vtoimg), warp_gaussian
            ).permute(*self.warp.permute_imgtov)

        moved_coords = fixed_image_affinecoords + warp_field
        return moved_coords

    def evaluate(self, fixed_images, moving_images, shape=None):
        moving_arrays = moving_images()
        moved_coords = self.get_warped_coordinates(fixed_images, moving_images, shape=shape)
        moved_image = F.grid_sample(
            moving_arrays, moved_coords, mode="bilinear", align_corners=self.align_corners
        )
        return moved_image

    def compute_regularization_loss(self, warp_field):
        """Compute regularization loss."""
        grad = torch.gradient(warp_field, dim=(1, 2, 3))
        grad_norm = sum(torch.sum(g**2) for g in grad)
        return grad_norm

    def optimize(self, save_transformed=False):
        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_torch2phy()
        moving_p2t = self.moving_images.get_phy2torch()
        fixed_size = fixed_arrays.shape[2:]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]

        transformed_images = [] if save_transformed else None
        warp_gaussian = [
            gaussian_1d(s, truncated=2)
            for s in (torch.zeros(self.dims, device=fixed_arrays.device) + self.smooth_warp_sigma)
        ]

        for scale, iters in zip(self.scales, self.iterations):
            self.convergence_monitor.reset()
            size_down = [max(int(s / scale), 1) for s in fixed_size]

            if self.blur and scale > 1:
                sigmas = 0.5 * torch.tensor(
                    [sz / szdown for sz, szdown in zip(fixed_size, size_down)], device=fixed_arrays.device
                )
                gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
                fixed_image_down = downsample(
                    fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, gaussians=gaussians
                )
                moving_image_blur = seperate_filter(moving_arrays, gaussians)
            else:
                fixed_image_down = F.interpolate(
                    fixed_arrays,
                    size=size_down,
                    mode=self.fixed_images.interpolate_mode,
                    align_corners=self.align_corners,
                )
                moving_image_blur = moving_arrays

            self.warp.set_size(size_down)
            fixed_image_affinecoords = F.affine_grid(
                affine_map_init, fixed_image_down.shape, align_corners=self.align_corners
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
                    moving_image_blur, moved_coords, mode="bilinear", align_corners=self.align_corners
                )

                if self.loss_device is not None:
                    moved_image = moved_image.to(self.loss_device)
                    fixed_image_down = fixed_image_down.to(self.loss_device)

                sim_loss = self.loss_fn(moved_image, fixed_image_down)
                reg_loss = self.compute_regularization_loss(warp_field)
                loss = sim_loss + 0.05 * reg_loss
                loss.backward(retain_graph=True)
                self.warp.step()

                if self.convergence_monitor.converged(loss.item()):
                    break

                if self.progress_bar:
                    pbar.set_description(f"scale: {scale}, iter: {i+1}/{iters}, loss: {loss.item():.4f}")

            if save_transformed:
                transformed_images.append(moved_image.detach().cpu())

        if save_transformed:
            return transformed_images


class CompositiveWarp(nn.Module):
    """
    Class for compositive warp function (collects gradients of dL/dp)
    The image is computed as M \circ (\phi + u)
    """

    def __init__(
        self,
        fixed_images,
        moving_images,
        optimizer="Adam",
        optimizer_lr=1e-2,
        optimizer_params={},
        init_scale=1,
        smoothing_grad_sigma=0.5,
        smoothing_warp_sigma=0.5,
        optimize_inverse_warp=False,
    ) -> None:
        super().__init__()
        self.num_images = num_images = fixed_images.size()
        spatial_dims = fixed_images.shape[2:]
        self.n_dims = len(spatial_dims)
        # permute indices
        self.permute_imgtov = (0, *range(2, self.n_dims + 2), 1)
        self.permute_vtoimg = (0, self.n_dims + 1, *range(1, self.n_dims + 1))
        self.device = fixed_images.device

        # define warp and register it as a parameter
        # define inverse warp and register it as a buffer
        self.optimize_inverse_warp = optimize_inverse_warp
        # set size
        if init_scale > 1:
            spatial_dims = [max(int(s / init_scale), 1) for s in spatial_dims]
        warp = torch.zeros(
            [num_images, *spatial_dims, self.n_dims], dtype=torch.float32, device=fixed_images.device
        )  # [N, HWD, dims]
        self.register_parameter("warp", nn.Parameter(warp))
        if self.optimize_inverse_warp:
            inv = torch.zeros(
                [num_images, *spatial_dims, self.n_dims], dtype=torch.float32, device=fixed_images.device
            )  # [N, HWD, dims]
        else:
            inv = torch.zeros([1], dtype=torch.float32, device=fixed_images.device)  # dummy
        self.register_buffer("inv", inv)

        # attach grad hook if smooothing of the gradient is required
        self.smoothing_grad_sigma = smoothing_grad_sigma
        if smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [
                gaussian_1d(s, truncated=2)
                for s in (torch.zeros(self.n_dims, device=fixed_images.device) + smoothing_grad_sigma)
            ]
        self.attach_grad_hook()

        oparams = deepcopy(optimizer_params)
        self.smoothing_warp_sigma = smoothing_warp_sigma
        if self.smoothing_warp_sigma > 0:
            smoothing_warp_gaussians = [
                gaussian_1d(s, truncated=2)
                for s in (torch.zeros(self.n_dims, device=fixed_images.device) + smoothing_warp_sigma)
            ]
            oparams["smoothing_gaussians"] = smoothing_warp_gaussians

        oparams["optimize_inverse_warp"] = optimize_inverse_warp
        if optimize_inverse_warp:
            oparams["warpinv"] = self.inv
        # add optimizer
        optimizer = optimizer.lower()
        self.optimizer = WarpAdam(self.warp, lr=optimizer_lr, **oparams)

    def attach_grad_hook(self):
        """attack the grad hook to the velocity field if needed"""
        if self.smoothing_grad_sigma > 0:
            hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
            self.warp.register_hook(hook)

    def initialize_grid(self):
        """initialize grid to a size
        Simply use the grid from the optimizer, which should be initialized to the correct size
        """
        self.grid = self.optimizer.grid

    def set_zero_grad(self):
        """set the gradient to zero (or None)"""
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def get_warp(self):
        """return warp function"""
        warp = self.warp
        return warp

    def get_inverse_warp(self, n_iters: int = 50, debug: bool = False, lr=0.1):
        """run an optimization procedure to get the inverse warp"""
        if self.optimize_inverse_warp:
            invwarp = self.inv
            invwarp = compute_inverse_warp_displacement(self.warp.data, self.grid, invwarp, iters=20)
        else:
            # no invwarp is defined, start from scratch
            invwarp = compute_inverse_warp_displacement(self.warp.data, self.grid, -self.warp.data, iters=200)
        return invwarp

    def set_size(self, size):
        # print(f"Setting size to {size}")
        """size: [H, W, D] or [H, W]"""
        mode = "bilinear" if self.n_dims == 2 else "trilinear"
        # get new displacement field
        warp = F.interpolate(
            self.warp.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True
        ).permute(*self.permute_imgtov)
        self.register_parameter("warp", nn.Parameter(warp))
        # set new inverse displacement field
        if len(self.inv.shape) > 1:
            self.inv = F.interpolate(
                self.inv.permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True
            ).permute(*self.permute_imgtov)
        self.attach_grad_hook()
        self.optimizer.set_data_and_size(
            self.warp, size, warpinv=self.inv if self.optimize_inverse_warp else None
        )
        # interpolate inverse warp if it exists
        self.initialize_grid()


class WarpAdam:

    def __init__(
        self,
        warp,
        lr,
        warpinv=None,
        beta1=0.9,
        beta2=0.99,
        weight_decay=0,
        eps=1e-8,
        scaledown=False,
        multiply_jacobian=False,
        smoothing_gaussians=None,
        optimize_inverse_warp=False,
    ):
        # init
        if beta1 < 0.0 or beta1 >= 1.0:
            raise ValueError("Invalid beta1 value: {}".format(beta1))
        if beta2 < 0.0 or beta2 >= 1.0:
            raise ValueError("Invalid beta2 value: {}".format(beta2))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid lr value: {}".format(lr))
        self.n_dims = len(warp.shape) - 2
        # get half resolutions
        self.half_resolution = 1.0 / (max(warp.shape[1:-1]) - 1)
        self.warp = warp
        self.warpinv = warpinv
        self.optimize_inverse_warp = optimize_inverse_warp
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.step_t = 0  # initialize step to 0
        self.weight_decay = weight_decay
        self.multiply_jacobian = multiply_jacobian
        self.scaledown = scaledown  # if true, the scale the gradient even if norm is below 1
        self.exp_avg = torch.zeros_like(warp)
        self.exp_avg_sq = torch.zeros_like(warp)
        self.permute_imgtov = (
            0,
            *range(2, self.n_dims + 2),
            1,
        )  # [N, HWD, dims] -> [N, HWD, dims] -> [N, dims, HWD]
        self.permute_vtoimg = (
            0,
            self.n_dims + 1,
            *range(1, self.n_dims + 1),
        )  # [N, dims, HWD] -> [N, HWD, dims]
        # set grid
        self.batch_size = batch_size = warp.shape[0]
        # init grid
        self.affine_init = torch.eye(self.n_dims, self.n_dims + 1, device=warp.device)[None].expand(
            batch_size, -1, -1
        )
        self.initialize_grid(warp.shape[1:-1])
        # gaussian smoothing parameters (if any)
        self.smoothing_gaussians = smoothing_gaussians

    def set_data_and_size(self, warp, size, grid_copy=None, warpinv=None):
        """change the optimization variables sizes"""
        self.warp = warp
        mode = "bilinear" if self.n_dims == 2 else "trilinear"
        self.exp_avg = F.interpolate(
            self.exp_avg.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True
        ).permute(*self.permute_imgtov)
        self.exp_avg_sq = F.interpolate(
            self.exp_avg_sq.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True
        ).permute(*self.permute_imgtov)
        self.half_resolution = 1.0 / (max(warp.shape[1:-1]) - 1)
        self.initialize_grid(size, grid_copy=grid_copy)
        # print(self.warp.shape, warpinv)
        if self.optimize_inverse_warp and warpinv is not None:
            self.warpinv = warpinv

    def initialize_grid(self, size, grid_copy=None):
        """initialize the grid (so that we can use it independent of the grid elsewhere)"""
        if grid_copy is None:
            self.grid = F.affine_grid(
                self.affine_init, [self.batch_size, 1, *size], align_corners=True
            ).detach()
        else:
            self.grid = grid_copy

    def zero_grad(self):
        """set the gradient to none"""
        self.warp.grad = None

    def augment_jacobian(self, u):
        # Multiply u (which represents dL/dphi most likely) with Jacobian indexed by J[..., xyz, ..., phi]
        jac = jacobian(self.warp.data + self.grid, normalize=True)  # [B, dims, H, W, [D], dims]
        if self.n_dims == 2:
            ujac = torch.einsum("bxhwp,bhwp->bhwx", jac, u)
        else:
            ujac = torch.einsum("bxhwdp,bhwdp->bhwdx", jac, u)
        return ujac

    def step(self):
        """check for momentum, and other things"""
        grad = torch.clone(self.warp.grad.data).detach()
        if self.multiply_jacobian:
            grad = self.augment_jacobian(grad)
        # add weight decay term
        if self.weight_decay > 0:
            grad.add_(self.warp.data, alpha=self.weight_decay)
        # compute moments
        self.step_t += 1
        self.exp_avg.mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
        self.exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad.conj(), value=1 - self.beta2)
        # bias correction
        beta_correction1 = 1 - self.beta1**self.step_t
        beta_correction2 = 1 - self.beta2**self.step_t
        denom = (self.exp_avg_sq / beta_correction2).sqrt().add_(self.eps)
        # get updated gradient (this will be normalized and passed in)
        grad = self.exp_avg / beta_correction1 / denom
        # renormalize and update warp
        # gradmax = self.eps + grad.reshape(grad.shape[0], -1).abs().max(1).values  # [B,]
        gradmax = self.eps + grad.norm(p=2, dim=-1, keepdim=True).flatten(1).max(1).values
        gradmax = gradmax.reshape(-1, *([1]) * (self.n_dims + 1))
        if not self.scaledown:  # if scaledown is "True", then we scale down even if the norm is below 1
            gradmax = torch.clamp(gradmax, min=1)
        # print(gradmax.abs().min(), gradmax.abs().max())
        grad = grad / gradmax * self.half_resolution  # norm is now 0.5r
        # multiply by learning rate
        grad.mul_(-self.lr)
        # print(grad.abs().max().item(), self.half_resolution, self.warp.shape)
        # compositional update
        w = grad + F.grid_sample(
            self.warp.data.permute(*self.permute_vtoimg),
            self.grid + grad,
            mode="bilinear",
            align_corners=True,
        ).permute(*self.permute_imgtov)
        # w = grad + self.warp.data
        # smooth result if asked for
        if self.smoothing_gaussians is not None:
            w = seperate_filter(w.permute(*self.permute_vtoimg), self.smoothing_gaussians).permute(
                *self.permute_imgtov
            )
        self.warp.data.copy_(w)
        # add to inverse if exists
        if self.optimize_inverse_warp and self.warpinv is not None:
            invwarp = compute_inverse_warp_displacement(self.warp.data, self.grid, self.warpinv.data, iters=5)
            warp_new = compute_inverse_warp_displacement(invwarp, self.grid, self.warp.data, iters=5)
            self.warp.data.copy_(warp_new)
            self.warpinv.data.copy_(invwarp)
