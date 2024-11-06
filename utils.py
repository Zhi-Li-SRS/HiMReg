from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision import transforms


def jacobian_2d(u: torch.Tensor, normalize: bool):
    """u: displacement vector of size [N, H, W, 2]"""
    B, H, W, _ = u.shape
    newshape = [B, 2, H, W, 2]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    # Compute Jacobian of u and v using image_gradient_singlechannel function
    for i in range(2):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W), normalize)
    return J


def jacobian_3d(u: torch.Tensor, normalize: bool):
    """u: displacement vector of size [N, H, W, D, 3]"""
    B, H, W, D, _ = u.shape
    newshape = [B, 3, H, W, D, 3]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    for i in range(3):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W, D), normalize)
    return J


def jacobian(u: torch.Tensor, normalize=True):
    """
    u: displacement vector of size [N, H, W, [D], dims]
    """
    if len(u.shape) == 4:
        return jacobian_2d(u, normalize)
    elif len(u.shape) == 5:
        return jacobian_3d(u, normalize)
    else:
        raise ValueError(f"jacobian not implemented for tensors of shape {u.shape}")


def image_gradient_singlechannel(image, normalize=False):
    """
    Compute the gradient of an image using central difference approximation.
    Args:
        image: Image tensor of size [N, C, H, W] or [N, C, D, H, W]
        normalize: default is False, if True, normalize the gradient by the image size
    """
    dims = len(image.shape) - 2
    device = image.device
    grad = None
    if dims == 2:
        B, C, H, W = image.shape
        if normalize:
            facx, facy = (W - 1) / 2, (H - 1) / 2
        else:
            facx, facy = 1, 1
        k = torch.cuda.FloatTensor([[-1.0, 0.0, 1.0]], device=device)[None, None] / 2
        gradx = F.conv2d(image, facx * k, padding=(0, 1))
        grady = F.conv2d(image, facy * k.permute(0, 1, 3, 2), padding=(1, 0))
        grad = torch.cat([gradx, grady], dim=1)
    elif dims == 3:
        B, C, D, H, W = image.shape
        if normalize:
            facx, facy, facz = (W - 1) / 2, (H - 1) / 2, (D - 1) / 2
        else:
            facx, facy, facz = 1, 1, 1
        k = torch.cuda.FloatTensor([[[-1.0, 0.0, 1.0]]], device=device)[None, None] / 2
        gradx = F.conv3d(image, facx * k, padding=(0, 0, 1))
        grady = F.conv3d(image, facy * k.permute(0, 1, 2, 4, 3), padding=(0, 1, 0))
        gradz = F.conv3d(image, facz * k.permute(0, 1, 4, 2, 3), padding=(1, 0, 0))
        grad = torch.cat([gradx, grady, gradz], dim=1)
    else:
        raise ValueError("Invalid dimension: {}".format(dims))
    return grad


def gaussian_smooth(x, sigma) -> torch.Tensor:
    """
    Apply Gaussian smoothing using scipy's gaussian filter
    Args:
        x: Input tensor
        sigma: Standard deviation for Gaussian kernel
    """
    x_np = x.cpu().numpy()
    x_smooth = ndimage.gaussian_filter(x_np, sigma)
    return torch.from_numpy(x_smooth).to(x.device)


def downsample(image, size, mode, sigma=None, gaussians=None) -> torch.Tensor:
    """
    Downsample image using torchvision transforms
    Args:
        image: Input tensor [B, C, H, W] or [B, C, D, H, W]
        size: Target size
        mode: Interpolation mode
        sigma: Optional sigma for Gaussian smoothing
    """
    if sigma is not None:
        image = gaussian_smooth(image, sigma)

    if len(image.shape) == 4:  # (B, C, H, W)
        resize = transforms.Resize(
            size=size, interpolation=getattr(transforms.InterpolationMode, mode.upper())
        )
        return resize(image)
    else:  # 3D case
        # Handle 3D case using F.interpolate since torchvision doesn't support 3D
        return F.interpolate(image, size=size, mode=mode, align_corners=True)


def scaling_and_squaring(u, grid, n=6):
    """
    Apply scaling and squaring to a displacement field.
    """
    dims = u.shape[-1]
    v = (1.0 / 2**n) * u
    if dims == 3:
        for i in range(n):
            vimg = v.permute(0, 4, 1, 2, 3)
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 4, 1)
    elif dims == 2:
        for i in range(n):
            vimg = v.permute(0, 3, 1, 2)
            v = v + F.grid_sample(vimg, v + grid, align_corners=True).permute(0, 2, 3, 1)
    else:
        raise ValueError("Invalid dimension: {}".format(dims))
    return v


def integer_to_onehot(image: torch.Tensor, background_label: int = 0, max_label=None):
    """
    Convert integer labels to one-hot vectors.

    Args:
        image: Integer labels.
        background_label: Label value to be considered as background, and this is the label we need to ignore.
        max_label: Maximum value of the label.

    Returns:
        One-hot representation of the input image.
    """
    if max_label is None:
        max_label = int(image.max())

    # check if the background_label is valid
    if background_label >= 0 and background_label <= max_label:
        num_labels = max_label
    else:
        num_labels = max_label + 1

    onehot = torch.zeros((num_labels, *image.shape), dtype=torch.float32, device=image.device)
    count = 0
    for i in range(num_labels + 1):
        if i == background_label:
            continue
        onehot[count, ...] = image == i
        count += 1
    return onehot


def grad_smoothing_hook(grad: torch.Tensor, gaussians: List[torch.Tensor]):
    """this backward hook will smooth out the gradient using the gaussians
    has to be called with a partial function
    """
    # grad is of shape [B, H, W, D, dims]
    if len(grad.shape) == 5:
        permute_vtoimg = (0, 4, 1, 2, 3)
        permute_imgtov = (0, 2, 3, 4, 1)
    elif len(grad.shape) == 4:
        permute_vtoimg = (0, 3, 1, 2)
        permute_imgtov = (0, 2, 3, 1)
    return gaussian_smooth(grad.permute(*permute_vtoimg), gaussians).permute(*permute_imgtov)


def compute_inverse_warp_exp(warp, grid, lr=5e-3, iters=200, n=10):
    """compute warp inverse using exponential map"""
    with torch.set_grad_enabled(True):
        vel = nn.Parameter(torch.zeros_like(warp))
        optim = torch.optim.Adam([vel], lr=lr)
        permute_vtoimg = (0, 4, 1, 2, 3) if len(warp.shape) == 5 else (0, 3, 1, 2)
        permute_imgtov = (0, 2, 3, 4, 1) if len(warp.shape) == 5 else (0, 2, 3, 1)
        pbar = range(iters)
        for i in pbar:
            optim.zero_grad()
            invwarp = scaling_and_squaring(vel, grid, n=n)
            loss = invwarp + F.grid_sample(
                warp.permute(*permute_vtoimg), grid + invwarp, mode="bilinear", align_corners=True
            ).permute(*permute_imgtov)
            loss2 = warp + F.grid_sample(
                invwarp.permute(*permute_vtoimg), grid + warp, mode="bilinear", align_corners=True
            ).permute(*permute_imgtov)
            loss = (loss**2).sum() + (loss2**2).sum()
            loss.backward()
            optim.step()
    return scaling_and_squaring(vel.data, grid, n=n)


def compute_inverse_warp_displacement(warp, grid, initial_inverse=None, iters=20, lr=1e-2):
    """
    Compute the inverse warp using a given warp, grid and optional initialization
    """
    permute_vtoimg = (0, 4, 1, 2, 3) if len(warp.shape) == 5 else (0, 3, 1, 2)
    permute_imgtov = (0, 2, 3, 4, 1) if len(warp.shape) == 5 else (0, 2, 3, 1)
    # in case this block is being called within a no_grad block
    with torch.set_grad_enabled(True):
        if initial_inverse is None:
            invwarp = nn.Parameter(torch.zeros_like(warp.detach()))
        else:
            invwarp = nn.Parameter(initial_inverse.detach())
        optim = torch.optim.SGD([invwarp], lr=lr)
        for i in range(iters):
            optim.zero_grad()
            loss = invwarp + F.grid_sample(
                warp.permute(*permute_vtoimg), grid + invwarp, mode="bilinear", align_corners=True
            ).permute(*permute_imgtov)
            loss2 = warp + F.grid_sample(
                invwarp.permute(*permute_vtoimg), grid + warp, mode="bilinear", align_corners=True
            ).permute(*permute_imgtov)
            loss = (loss**2).sum() + (loss2**2).sum()
            loss.backward()
            optim.step()
    return invwarp.data
