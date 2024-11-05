from collections import deque
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


def gaussian_1d(sigma: torch.Tensor, truncated=4.0, approx="erf", normalize=True):
    """
    Compute 1D Gaussian kernel.

    Args:
        sigma: Standard deviation of the Gaussian kernel.
        truncated: Truncate the Gaussian kernel at this many standard deviations.
        approx: Approximation method. Either "erf" or "exp".
        normalize: Normalize the Gaussian kernel.

    returns:
        1D Gaussian kernel: torch.Tensor.
    """
    device = sigma.device
    sigma = torch.as_tensor(
        sigma, dtype=torch.float, device=device if isinstance(sigma, torch.Tensor) else None
    )
    if truncated < 0.0:
        raise ValueError("truncated must be positve")
    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    if approx.lower() == "erf":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        t = 0.70710678 / torch.abs(sigma)
        out = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
        out = out.clamp(min=0)
    elif approx.lower() == "sampled":
        x = torch.arange(-tail, tail + 1, dtype=torch.float, device=device)
        out = torch.exp(-0.5 / (sigma**2) * x**2)
        if not normalize:
            out = out / (2.5066282 * sigma)
    else:
        raise ValueError(f"Unknown approximation method: {approx}")

    return out / out.sum() if normalize else out


def make_regtangular_kernel(kernel_size):
    """Create a rectangular kernel of size kernel_size."""
    return torch.ones(kernel_size)


def make_triangular_kernel(kernel_size):
    """Create a triangular kernel of size kernel_size."""

    size = (kernel_size + 1) // 2
    if size % 2 == 0:
        size -= 1
    f = torch.ones((1, 1, size), dtype=torch.float).div(size)
    padding = (kernel_size - size) // 2 + size // 2
    return F.conv1d(f, f, padding=padding).reshape(-1)


def make_gaussian_kernel(kernel_size, sigma):
    """Create a Gaussian kernel of size kernel_size."""
    sigma = torch.tensor(kernel_size / 3.0)
    kernel = gaussian_1d(sigma, truncated=4.0, approx="erf", normalize=True) * (2.5066282 * sigma)

    return kernel[:kernel_size]


kernel_dict = {
    "rectangular": make_regtangular_kernel,
    "triangular": make_triangular_kernel,
    "gaussian": make_gaussian_kernel,
}


def seperate_filter(x, kernels, mode=None):
    """Seperate the filter into two 1D filters."""
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a torch.Tensor")

    spatial_dims = x.dim() - 2
    if isinstance(kernels, torch.Tensor):
        kernels = [kernels] * spatial_dims

    _kernels = [s.to(x) for s in kernels]  # Convert to tensor
    paddings = [(k.shape[0] - 1) // 2 for k in _kernels]
    c = x.shape[1]
    pad_mode = "constant" if mode is None else mode

    for d in range(spatial_dims):
        s = [1] * len(x.shape)
        s[d + 2] = -1
        _kernel = _kernels[d].reshape(s)
        if _kernel.numel() == 1 and _kernel[0] == 1:
            continue

        _kernel = _kernel.repeat([c, 1] + [1] * spatial_dims)
        padding = [0] * spatial_dims
        padding[d] = paddings[d]
        reversed_padding = padding[::-1]
        reversed_padding_repeated_twice = [[p, p] for p in reversed_padding]
        sum_reversed_padding_repeated_twice = []
        for p in reversed_padding_repeated_twice:
            sum_reversed_padding_repeated_twice.extend(p)

        padded_input = F.pad(x, sum_reversed_padding_repeated_twice, mode=pad_mode)
        if spatial_dims == 1:
            x = F.conv1d(input=padded_input, weight=_kernel, groups=c)
        elif spatial_dims == 2:
            x = F.conv2d(input=padded_input, weight=_kernel, groups=c)
        elif spatial_dims == 3:
            x = F.conv3d(input=padded_input, weight=_kernel, groups=c)
        else:
            raise NotImplementedError(f"Unsupported spatial_dims: {spatial_dims}.")
    return x


def downsample(image, size, mode, sigma=None, gaussians=None) -> torch.Tensor:
    """
    this function is to downsample the image to the given size. If sigma is provided, then use this sigma for downsampling, otherwise infer sigma
    Args:
        image (tensor): input image
        size (list): target size
        mode (str): interpolation mode
        sigma (list): sigma for gaussian filter
    """
    if gaussians is None:
        if sigma is None:
            orig_size = list(image.shape[2:])
            sigma = [
                0.5 * orig_size[i] / size[i] for i in range(len(orig_size))
            ]  # use sigma as the downsampling factor
        if isinstance(sigma, torch.Tensor):
            sigma = sigma.clone().detach().to(dtype=torch.float32, device=image.device)
        else:
            sigma = torch.tensor(sigma, dtype=torch.float32, device=image.device)
        gaussians = [gaussian_1d(s, truncated=2) for s in sigma]
    # otherwise gaussians is given, just downsample
    image_smooth = seperate_filter(image, gaussians)
    image_down = F.interpolate(image_smooth, size=size, mode=mode, align_corners=True)
    return image_down


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
    return seperate_filter(grad.permute(*permute_vtoimg), gaussians).permute(*permute_imgtov)


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
