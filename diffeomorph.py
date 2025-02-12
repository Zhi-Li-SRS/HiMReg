from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, Adam
from tqdm import tqdm

from losses import LNCC, MutualInformation
from utils import *


class DiffRegistration:
    """
    Implements diffeomorphic registration between fi    xed and moving images.

    Args:
        scales: List of scales for multi-resolution optimization
        iterations: List of iterations per scale
        fixed_images: Fixed image for registration
        moving_images: Moving image to be registered
        loss_type: Loss function type ('mi' or 'cc')
        optimizer: Optimizer type ('Adam' or 'SGD')
        optimizer_params: Additional optimizer parameters
        optimizer_lr: Learning rate
        mi_kernel_type: Kernel type for mutual information loss
        cc_kernel_type: Kernel type for cross correlation loss
        cc_kernel_size: Kernel size for cross correlation
        smooth_warp_sigma: Sigma for warp field smoothing
        smooth_grad_sigma: Sigma for gradient smoothing
        loss_params: Additional loss function parameters
        tolerance: Convergence tolerance threshold
        max_tolerance_iters: Maximum iterations for convergence check
        init_affine: Initial affine transform
        blur: Whether to apply Gaussian blur at each scale
        align_corners: Grid sample align corners parameter
        loss_device: Device to compute loss on
    """

    def __init__(
        self,
        scales,
        iterations,
        fixed_images,
        moving_images,
        loss_type="mi",
        optimizer="Adam",
        optimizer_params={},
        optimizer_lr=0.5,
        mi_kernel_type="b-spline",
        cc_kernel_type="rectangular",
        cc_kernel_size=7,
        smooth_warp_sigma=0.3,
        smooth_grad_sigma=0.8,
        loss_params={},
        tolerance=1e-3,
        max_tolerance_iters=1000,
        init_affine=None,
        blur=True,
        align_corners=True,
        loss_device=None,
    ):
        # Validate inputs
        self.validate_inputs(scales, iterations)

        # Set basic parameters
        self.scales = scales
        self.iterations = iterations
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.dims = fixed_images.dims
        self.tolerance = tolerance
        self.blur = blur
        self.align_corners = align_corners
        self.loss_device = loss_device

        # Convergence params
        self.tolerance = tolerance
        self.max_tolerance_iters = max_tolerance_iters
        self.losses = deque(maxlen=max_tolerance_iters)

        # Initialize loss function
        self.init_loss_function(loss_type, mi_kernel_type, cc_kernel_type, cc_kernel_size, loss_params)

        # Initialize optimizer
        self.init_optimizer(
            optimizer, optimizer_lr, optimizer_params, smooth_grad_sigma, smooth_warp_sigma, scales[0]
        )

        # Initialize affine transformation
        self._init_affine(init_affine)
        self.smooth_warp_sigma = 0  # Reset after initialization
        self.final_coordinates = None

    def validate_inputs(self, scales, iterations):
        """Validate input parameters."""
        if len(iterations) != len(scales):
            raise ValueError("Number of iterations must match number of scales")

    def init_loss_function(self, loss_type, mi_kernel_type, cc_kernel_type, cc_kernel_size, loss_params):
        """Initialize loss function."""
        if loss_type == "mi":
            self.loss_fn = MutualInformation(kernel_type=mi_kernel_type, **loss_params)
        elif loss_type == "cc":
            self.loss_fn = LNCC(
                kernel_type=cc_kernel_type, spatial_dims=self.dims, kernel_size=cc_kernel_size, **loss_params
            )
        else:
            raise ValueError(f"Loss type {loss_type} not supported")

    def init_optimizer(
        self, optimizer, optimizer_lr, optimizer_params, smooth_grad_sigma, smooth_warp_sigma, init_scale
    ):
        """Initialize optimizer."""
        compositive_params = {
            "fixed_images": self.fixed_images,
            "moving_images": self.moving_images,
            "optimizer": optimizer,
            "optimizer_lr": optimizer_lr,
            "optimizer_params": optimizer_params,
            "init_scale": init_scale,
            "smoothing_grad_sigma": smooth_grad_sigma,
            "smoothing_warp_sigma": smooth_warp_sigma,
        }
        self.warp = DiffOptimizer(**compositive_params)

    def _init_affine(self, init_affine: Optional[torch.Tensor]):
        """Initialize affine transformation."""
        if init_affine is None:
            init_affine = (
                torch.eye(self.dims + 1, device=self.device)
                .unsqueeze(0)
                .repeat(self.fixed_images.size(), 1, 1)
            )
        self.affine = init_affine.detach()

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

    def compute_regularization_loss(self, warp_field: torch.Tensor):
        """Compute regularization loss."""
        grad = torch.gradient(warp_field, dim=(1, 2, 3))
        grad_norm = sum(torch.sum(g**2) for g in grad)
        return grad_norm

    def optimize(self, save_transformed=False):
        """
        Optimize the registration.
        Args:
            save_transformed: Whether to save transformed images during optimization

        Returns:
            List of transformed images if save_transformed=True, else None
        """

        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()
        fixed_t2p = self.fixed_images.get_pixel_to_physical()
        moving_p2t = self.moving_images.get_physical_to_pixel()
        fixed_size = fixed_arrays.shape[2:]
        affine_map_init = torch.matmul(moving_p2t, torch.matmul(self.affine, fixed_t2p))[:, :-1]

        transformed_images = [] if save_transformed else None
        warp_gaussian = self.get_warp_gaussian()

        # Hierarchical optimization
        for scale, iters in zip(self.scales, self.iterations):
            self.losses.clear()

            size_down = [max(int(s / scale), 1) for s in fixed_size]
            fixed_image_down, moving_image_blur = self.prepare_images_for_scale(
                fixed_arrays, moving_arrays, size_down, scale
            )

            # Setup registration at current scale
            self.warp.update_field_size(size_down)
            fixed_image_affinecoords = F.affine_grid(
                affine_map_init, fixed_image_down.shape, align_corners=self.align_corners
            )

            pbar = tqdm(range(iters))
            for i in pbar:
                # Forward pass
                warp_field = self.warp.get_displacement_field()
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

                # Compute loss and optimize
                sim_loss = self.loss_fn(moved_image, fixed_image_down)
                reg_loss = self.compute_regularization_loss(warp_field)
                loss = sim_loss + 0.05 * reg_loss

                self.warp.reset_gradients()
                loss.backward()
                self.warp.optimization_step()

                pbar.set_description(f"scale: {scale}, iter: {i+1}/{iters}, loss: {loss.item():.4f}")

                if self.converged(loss.item()):
                    break

            self.final_coordinates = fixed_image_affinecoords + warp_field

            if save_transformed:
                transformed_images.append(moved_image.detach().cpu())

        return transformed_images if save_transformed else None

    def get_warp_gaussian(self) -> List[torch.Tensor]:
        """Get Gaussian kernels for warp smoothing."""
        if self.smooth_warp_sigma <= 0:
            return []
        return [
            gaussian_1d(s, truncated=2)
            for s in (torch.zeros(self.dims, device=self.device) + self.smooth_warp_sigma)
        ]

    def prepare_images_for_scale(
        self, fixed_arrays: torch.Tensor, moving_arrays: torch.Tensor, size_down: List[int], scale: int
    ) -> tuple:
        """Prepare fixed and moving images for current scale."""
        if self.blur and scale > 1:
            sigmas = 0.5 * torch.tensor(
                [sz / szdown for sz, szdown in zip(fixed_arrays.shape[2:], size_down)], device=self.device
            )
            gaussians = [gaussian_1d(s, truncated=2) for s in sigmas]
            fixed_image_down = downsample(
                fixed_arrays, size=size_down, mode=self.fixed_images.interpolate_mode, sigma=sigmas
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

        return fixed_image_down, moving_image_blur

    def get_final_coordinates(self):
        """Get the final affine coordinates."""
        if self.final_coordinates is None:
            raise RuntimeError("Final coordinates not computed. Run optimize() first.")
        return self.final_coordinates

    def apply_transform(self, moving_image):
        """Apply the final transformation to a new image."""
        if self.final_coordinates is None:
            raise RuntimeError("Final coordinates not computed. Run optimize() first.")

        transformed = F.grid_sample(moving_image, self.final_coordinates, mode="bilinear", align_corners=True)
        return transformed


class DiffOptimizer(nn.Module):
    """
    Optimizer for diffeomorphic registration that computes and optimizes displacement field

    Args:
        fixed_images (torch.Tensor): Fixed images
        moving_images (torch.Tensor): Moving images
        optimizer (str): Optimizer type (Adam or SGD)
        optimizer_lr (float): Learning rate
        optimizer_params (dict): Additional parameters for optimizer
        init_scale (float): Initial scale
        smoothing_grad_sigma (float): Sigma for smoothing gradient
        smoothing_warp_sigma (float): Sigma for smoothing warp field
        optimize_inverse_warp (bool): Whether to optimize inverse warp field
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
        self.num_images = fixed_images.size()
        self.spatial_dims = fixed_images.shape[2:]
        self.n_dims = len(self.spatial_dims)
        # permute indices
        self.permute_imgtov = (0, *range(2, self.n_dims + 2), 1)
        self.permute_vtoimg = (0, self.n_dims + 1, *range(1, self.n_dims + 1))
        self.device = fixed_images.device
        self.optimizer = optimizer
        self.optimizer_lr = optimizer_lr
        self.optimizer_params = optimizer_params
        self.init_scale = init_scale
        self.smoothing_grad_sigma = smoothing_grad_sigma
        self.smoothing_warp_sigma = smoothing_warp_sigma
        self.optimize_inverse_warp = optimize_inverse_warp

        self.setup_transformation_indices()
        self.initialize_displacement_fields()
        self.setup_optimizer()

    def setup_transformation_indices(self):
        "Initialize permutation indices for image and vectors transformation"
        self.permute_imgtov = (0, *range(2, self.n_dims + 2), 1)  # [B, C, H, W] -> [B, H, W, C]
        self.permute_vtoimg = (0, self.n_dims + 1, *range(1, self.n_dims + 1))  # [B, H, W, C] -> [B, C, H, W]

    def initialize_displacement_fields(self):
        """Initialize the forward and inverse displacement fields."""
        init_dims = self.get_initial_dims()

        self.forward_field = self.create_displacement_field(init_dims)
        self.register_parameter("warp", nn.Parameter(self.forward_field))

        self.inverse_field = self.create_inverse_field(init_dims)
        self.register_parameter("inv", nn.Parameter(self.inverse_field))

    def get_initial_dims(self):
        """Calculate the intial dimensions based on the scale."""
        if self.init_scale > 1:
            return [max(int(s / self.init_scale), 1) for s in self.spatial_dims]

        return list(self.spatial_dims)

    def create_displacement_field(self, dimensions: List[int]):
        """Create a zero-initialized displacement field."""
        d = torch.zeros([self.num_images, *dimensions, self.n_dims], dtype=torch.float32, device=self.device)
        return d

    def create_inverse_field(self, dimensions: List[int]):
        """Create a inverse displacement field if needed."""
        if self.optimize_inverse_warp:
            return self.create_displacement_field(dimensions)
        return torch.zeros([1], dtype=torch.float32, device=self.device)

    def setup_optimizer(self):
        """Setup the optimizer."""
        self.setup_smoothing()

        optimizer_params = self.prepare_optimizer_params()

        self.optimizer = DiffAdam(self.warp, lr=self.optimizer_lr, **optimizer_params)

    def setup_smoothing(self):
        if self.smoothing_grad_sigma > 0:
            self.smoothing_grad_gaussians = [
                gaussian_1d(s, truncated=2)
                for s in torch.zeros(self.n_dims, device=self.device) + self.smoothing_grad_sigma
            ]
            self.attach_gradient_hooks()

    def prepare_optimizer_params(self):
        params = deepcopy(self.optimizer_params)
        if self.smoothing_warp_sigma > 0:
            params["smoothing_gaussians"] = [
                gaussian_1d(s, truncated=2)
                for s in torch.zeros(self.n_dims, device=self.device) + self.smoothing_warp_sigma
            ]
        params["optimize_inverse_warp"] = self.optimize_inverse_warp
        if self.optimize_inverse_warp:
            params["warpinv"] = self.inv

        return params

    def attach_gradient_hooks(self):
        """Attach gradient smoothing hooks."""
        hook = partial(grad_smoothing_hook, gaussians=self.smoothing_grad_gaussians)
        self.warp.register_hook(hook)

    def initialize_grid(self):
        self.grid = self.optimizer.grid

    def reset_gradients(self):
        self.optimizer.zero_grad()

    def optimization_step(self):
        self.optimizer.step()

    def get_displacement_field(self):
        return self.warp

    def get_inverse_field(self):
        if self.optimize_inverse_warp:
            invfield = compute_inverse_warp_displacement(self.warp.data, self.grid, self.inv, iters=20)
        else:
            invfield = compute_inverse_warp_displacement(
                self.warp.data, self.grid, -self.warp.data, iters=200
            )
        return invfield

    def update_field_size(self, new_size: Tuple[int, ...]):
        """
        Update the size of displacement fields

        Args:
            new_size: New spatial dimensions
        """
        mode = "bilinear" if self.n_dims == 2 else "trilinear"
        # Interpolate and wrap as Parameter
        interpolated_warp = self.interpolate_field(self.warp, new_size, mode)
        self.warp = nn.Parameter(interpolated_warp)

        # Update inverse field if needed
        if len(self.inv.shape) > 1:
            interpolated_inv = self.interpolate_field(self.inv, new_size, mode)
            self.inv = nn.Parameter(interpolated_inv)

        self.attach_gradient_hooks()
        optimizer_params = self.prepare_optimizer_params()
        self.optimizer = DiffAdam(self.warp, lr=self.optimizer_lr, **optimizer_params)

        self.initialize_grid()

    def interpolate_field(self, field: torch.Tensor, size: Tuple[int, ...], mode: str):
        """Helper method to interpolate displacement fields"""
        return F.interpolate(
            field.detach().permute(*self.permute_vtoimg), size=size, mode=mode, align_corners=True
        ).permute(*self.permute_imgtov)


class DiffAdam:
    """
    Adam like optimizer specialized for diffeomorphic registration

    Args:
        Warp (Tensor): The warp field to optimize
        lr (float): learning rate
        warpinv (Tensor, optional): The inverse warp field to optimize. Defaults to None
        beta1 (float): Exponential decay rate for first moment. Defaults to 0.9
        beta2 (float): Exponential decay rate for second moment. Defaults to 0.99
        weight_decay (float): Weight decay factor. Defaults to 0
        eps (float): Term added for numerical stability. Defaults to 1e-8
        scaledown (bool): Whether to scale gradients even when norm is below 1
        multiply_jacobian (bool): Whether to multiply gradients with Jacobian
        smoothing_gaussians (Tensor, optional): Gaussian kernels for smoothing
        optimize_inverse_warp (bool): Whether to optimize inverse warping field
    """

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

        # initialize basic parameters
        self.n_dims = len(warp.shape) - 2
        self.batch_size = warp.shape[0]
        self.half_resolution = 1.0 / (max(warp.shape[1:-1]) - 1)

        # initialize optimizer parameters
        self.warp = warp
        self.warpinv = warpinv
        self.lr = lr
        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay

        # initialize optimizer states
        self.step_t = 0  # initialize step to 0
        self.exp_avg = torch.zeros_like(warp)  # initialize first moment
        self.exp_avg_sq = torch.zeros_like(warp)  # initialize second moment

        # initialize additional parameters
        self.multiply_jacobian = multiply_jacobian
        self.scaledown = scaledown
        self.optimize_inverse_warp = optimize_inverse_warp
        self.smoothing_gaussians = smoothing_gaussians

        self.setup_permutation_indices()
        self.initialize_transformation_grid()

    def setup_permutation_indices(self):
        # [N, HWD, dims] -> [N, dims, HWD]
        self.permute_imgtov = (0, *range(2, self.n_dims + 2), 1)
        # [N, dims, HWD] -> [N, HWD, dims]
        self.permute_vtoimg = (0, self.n_dims + 1, *range(1, self.n_dims + 1))

    def initialize_transformation_grid(self):
        """Initialize the transformation grid"""
        self.affine_init = torch.eye(self.n_dims, self.n_dims + 1, device=self.warp.device)[None].expand(
            self.batch_size, -1, -1
        )
        self.grid = F.affine_grid(
            self.affine_init, [self.batch_size, 1, *self.warp.shape[1:-1]], align_corners=True
        ).detach()  # grid is fixed and do not need any gradient

    def compute_moments(self, grad):
        """Compute the moments of the gradient"""
        self.step_t += 1

        # Update biased first moment estimate
        self.exp_avg = self.beta1 * self.exp_avg + (1 - self.beta1) * grad
        # Update biased second raw moment estimate
        self.exp_avg_sq = self.beta2 * self.exp_avg_sq + (1 - self.beta2) * (grad**2)

        bias_correction1 = 1 - self.beta1**self.step_t
        bias_correction2 = 1 - self.beta2**self.step_t
        denom = (self.exp_avg_sq / bias_correction2).sqrt() + self.eps

        return self.exp_avg / bias_correction1 / denom

    def update_warp_field(self, grad):
        """Update warp field using compositional update."""

        w = grad + F.grid_sample(
            self.warp.data.permute(*self.permute_vtoimg),
            self.grid + grad,
            mode="bilinear",
            align_corners=True,
        ).permute(*self.permute_imgtov)

        if self.smoothing_gaussians is not None:
            w = seperate_filter(w.permute(*self.permute_vtoimg), self.smoothing_gaussians).permute(
                *self.permute_imgtov
            )
        return w

    def _optimize_inverse_warp(self, w):
        """Optimize the inverse warp field."""
        if self.optimize_inverse_warp and self.warpinv is not None:
            invwarp = compute_inverse_warp_displacement(self.warp.data, self.grid, self.warpinv.data, iters=5)
            warp_new = compute_inverse_warp_displacement(invwarp, self.grid, self.warp.data, iters=5)
            return warp_new, invwarp
        return w, None

    def zero_grad(self):
        """Clear gradients."""
        self.warp.grad = None

    def augment_jacobian(self, u):
        """
        Augment the gradient with the Jacobian

        Args:
            u (Tensor): The gradient

        Returns:
            ujac (Tensor): The gradient augmented with the Jacobian
        """
        jac = jacobian(self.warp.data + self.grid, normalize=True)  # [B, dims, H, W, [D], dims]
        if self.n_dims == 2:
            jac_reshape = jac.permute(0, 2, 3, 1, 4)
            u_reshape = u.unsqueeze(-2)
            ujac = torch.matmul(u_reshape, jac_reshape.transpose(-2, -1))
            ujac = ujac.squeeze(-2)  # [B,H,W,2]
        else:
            jac_reshape = jac.permute(0, 2, 3, 4, 1, 5)
            u_reshape = u.unsqueeze(-2)
            ujac = torch.matmul(u_reshape, jac_reshape.transpose(-2, -1))
            ujac = ujac.squeeze(-2)  # [B,H,W,D,3]
        return ujac

    def step(self):
        """Perform a single optimization step.

        Updates the warp field using a compositional update rule while
        maintaining momentum and adaptive learning rates from Adam.
        """
        if self.warp.grad is None:
            return
        grad = self.warp.grad.data.detach()

        if self.multiply_jacobian:
            grad = self.augment_jacobian(grad)

        if self.weight_decay > 0:
            grad = grad + self.weight_decay * self.warp.data

        grad = self.compute_moments(grad)

        # Normalize the gradient using L2 norm
        grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).view(-1, *([1] * (grad.dim() - 1)))
        grad_norm = torch.clamp(grad_norm, min=1e-8)
        normalized_grad = grad / grad_norm

        update = -self.lr * normalized_grad

        w = self.update_warp_field(update)
        w, invwarp = self._optimize_inverse_warp(w)

        # Update parameters
        self.warp.data.copy_(w)
        if invwarp is not None:
            self.warpinv.data.copy_(invwarp)
