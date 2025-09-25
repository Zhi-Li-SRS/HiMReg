from collections import deque

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torchvision import transforms


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible runs."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _tensor_to_numpy(image) -> np.ndarray:
    """Convert a tensor/array to 2D numpy array."""

    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3:
            tensor = tensor[0]
        array = tensor.numpy()
    elif isinstance(image, np.ndarray):
        array = image
    else:
        raise TypeError("Expected torch.Tensor or np.ndarray for image data")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Overlay images must be 2D arrays after channel/batch selection")
    return array


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array_min = float(array.min())
    array_max = float(array.max())
    if array_max > array_min:
        return (array - array_min) / (array_max - array_min)
    return np.zeros_like(array, dtype=np.float32)


def _build_overlay(fixed, moving, alpha: float = 0.7) -> np.ndarray:
    fixed_norm = _normalize_array(fixed)
    moving_norm = _normalize_array(moving)

    base = ((fixed_norm + moving_norm) / 2.0) * (1.0 - alpha)
    overlay = np.stack(
        [
            np.clip(base + moving_norm * alpha, 0.0, 1.0),
            np.clip(base, 0.0, 1.0),
            np.clip(base + fixed_norm * alpha, 0.0, 1.0),
        ],
        axis=-1,
    )
    return overlay


def save_registration_overlay(
    fixed_image, moving_before, moving_after, output_path: str, titles=("Before", "After"), alpha: float = 0.7
) -> None:
    """Save side-by-side overlays of fixed vs moving images before/after registration."""

    fixed_np = _tensor_to_numpy(fixed_image)
    before_np = _tensor_to_numpy(moving_before)
    after_np = _tensor_to_numpy(moving_after)

    if fixed_np.shape != before_np.shape:
        raise ValueError("Fixed and moving-before images must share the same spatial shape")
    if fixed_np.shape != after_np.shape:
        raise ValueError("Fixed and moving-after images must share the same spatial shape")

    overlays = [
        _build_overlay(fixed_np, before_np, alpha=alpha),
        _build_overlay(fixed_np, after_np, alpha=alpha),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    for ax, img, title in zip(axes, overlays, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def image_gradient_singlechannel(image, normalize=False):
    """
    Compute the gradient of an image using central difference approximation.
    Args:
        image: Image tensor of size [N, C, H, W] or [N, C, D, H, W]
        normalize: default is False, if True, normalize the gradient by the image size
    """
    dims = len(image.shape) - 2  # Remove batch and channel dimensions
    device = image.device
    grad = None
    if dims == 2:
        B, C, H, W = image.shape
        if normalize:
            # Scale factors to normalize the gradient by image size
            facx, facy = (W - 1) / 2, (H - 1) / 2
        else:
            facx, facy = 1, 1
        kernel = (
            torch.cuda.FloatTensor([[-1.0, 0.0, 1.0]], device=device).unsqueeze(0).unsqueeze(0) / 2
        )  # (1,1,1,3)
        gradx = F.conv2d(image, facx * kernel, padding=(0, 1))
        grady = F.conv2d(image, facy * kernel.permute(0, 1, 3, 2), padding=(1, 0))
        grad = torch.cat([gradx, grady], dim=1)
    elif dims == 3:
        B, C, D, H, W = image.shape
        if normalize:
            facx, facy, facz = (W - 1) / 2, (H - 1) / 2, (D - 1) / 2
        else:
            facx, facy, facz = 1, 1, 1
        kernel = torch.cuda.FloatTensor([[[-1.0, 0.0, 1.0]]], device=device).unsqueeze(0).unsqueeze(0) / 2
        gradx = F.conv3d(image, facx * kernel, padding=(0, 0, 1))
        grady = F.conv3d(image, facy * kernel.permute(0, 1, 2, 4, 3), padding=(0, 1, 0))
        gradz = F.conv3d(image, facz * kernel.permute(0, 1, 4, 2, 3), padding=(1, 0, 0))
        grad = torch.cat([gradx, grady, gradz], dim=1)
    else:
        raise ValueError(f"Invalid dimension: {dims}")
    return grad


def jacobian_2d(u, normalize):
    """u: displacement vector of size [N, H, W, 2]"""
    B, H, W, _ = u.shape
    newshape = [B, 2, H, W, 2]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    for i in range(2):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W), normalize)
    return J


def jacobian_3d(u, normalize):
    """u: displacement vector of size [N, H, W, D, 3]"""
    B, H, W, D, _ = u.shape
    newshape = [B, 3, H, W, D, 3]
    J = torch.empty(newshape, dtype=u.dtype, device=u.device)
    for i in range(3):
        J[..., i] = image_gradient_singlechannel(u[..., i].reshape(B, 1, H, W, D), normalize)
    return J


def jacobian(u, normalize=True):
    """
    u: displacement vector of size [N, H, W, [D], dims]
    """
    if len(u.shape) == 4:
        return jacobian_2d(u, normalize)
    elif len(u.shape) == 5:
        return jacobian_3d(u, normalize)
    else:
        raise ValueError(f"Invalid tensor shape: {u.shape}")


def gaussian_1d(sigma, truncated=4.0, normalize=True):
    """
    Compute 1D Gaussian kernel optimized for GPU processing
    """
    device = sigma.device if isinstance(sigma, torch.Tensor) else None
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.clone().detach().to(dtype=torch.float32)
    else:
        sigma = torch.as_tensor(sigma, dtype=torch.float32, device=device)

    tail = int(max(float(sigma) * truncated, 0.5) + 0.5)
    x = torch.arange(-tail, tail + 1, dtype=torch.float32, device=device)
    t = 0.70710678 / torch.abs(sigma)  # 1/√2
    kernel = 0.5 * ((t * (x + 0.5)).erf() - (t * (x - 0.5)).erf())
    kernel = kernel.clamp(min=0)

    return kernel / kernel.sum() if normalize else kernel


def seperate_filter(x, kernels, mode=None):
    """
    Apply separable filtering efficiently on GPUx.

    Args:
        x: Input tensor [B, C, *spatial_dims]
        kernels: Single kernel or list of kernels for each dimension
        mode: Optional padding mode, default is None.
    Returns:
        torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"x must be a torch.Tensor")

    spatial_dims = x.dim() - 2  # [B, C, H, W] -> 2
    kernels = [kernels] * spatial_dims if isinstance(kernels, torch.Tensor) else kernels
    kernels = [k.to(x.device) for k in kernels]  # Convert to tensor

    for dim in range(spatial_dims):
        kernel_shape = [1] * len(x.shape)  # [B, C, H, W]
        kernel_shape[dim + 2] = -1  # Set current dimenstion to -1 for reshape
        kernel = kernels[dim].reshape(kernel_shape)  # [1, 1, H, 1] or [1, 1, 1, W]
        if kernel.numel() == 1 and kernel[0] == 1:
            continue

        channels = x.shape[1]
        kernel = kernel.repeat([channels, 1] + [1] * spatial_dims)

        # Calculate padding
        padding = [0] * (2 * spatial_dims)
        padding[2 * (spatial_dims - 1 - dim)] = (kernel.shape[dim + 2] - 1) // 2
        padding[2 * (spatial_dims - 1 - dim) + 1] = (kernel.shape[dim + 2] - 1) // 2

        # Apply convolution
        x = F.pad(x, padding, mode=mode or "constant")
        conv_fn = getattr(F, f"conv{spatial_dims}d")
        x = conv_fn(x, weight=kernel, groups=channels)

    return x


def downsample(image, size, mode, sigma=None):
    """
    this function is to downsample the image to the given size. If sigma is provided, then use this sigma for downsampling, otherwise infer sigma
    Args:
        image (tensor): input image
        size (list): target size
        mode (str): interpolation mode
        sigma (list): sigma for gaussian filter
    Returns:
        torch.Tensor: downsampled image
    """

    if sigma is None:
        orig_size = list(image.shape[2:])
        sigma = [
            0.5 * orig_size[i] / size[i] for i in range(len(orig_size))
        ]  # use sigma as the downsampling factor

    sigma = torch.as_tensor(sigma, dtype=torch.float32, device=image.device)
    kernels = [gaussian_1d(s, truncated=2) for s in sigma]
    smoothed = seperate_filter(image, kernels)
    image_down = F.interpolate(smoothed, size=size, mode=mode, align_corners=True)
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
        raise ValueError(f"Invalid dimension: {dims}")
    return v


def integer_to_onehot(image, background_label=0, max_label=None):
    """
    Convert integer labels to one-hot vectors.

    Args:
        image (tensor): Integer labels.
        background_label: Label value to be considered as background.
        max_label: Maximum value of the label.

    Returns:
        One-hot representation of the input image.
    """
    if max_label is None:
        max_label = int(image.max())

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


def grad_smoothing_hook(grad, gaussians):
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


def compute_inverse_warp_exp(warp, grid, lr=4e-3, iterations=300, n=10):
    """
    compute warp inverse using exponential map

    Args:
        warp (torch.Tensor): The original deformation field to be inverted
        grid (torch.Tensor): The grid for the original deformation field
        lr (float): learning rate
        iters (int): number of iterations
        n (int): number of scaling and squaring iterations
    """
    with torch.set_grad_enabled(True):
        vel = nn.Parameter(torch.zeros_like(warp))
        optim = torch.optim.Adam([vel], lr=lr)

        is_2d = len(warp.shape) == 4
        permute_vtoimg = (0, 3, 1, 2) if is_2d else (0, 4, 1, 2, 3)
        permute_imgtov = (0, 2, 3, 1) if is_2d else (0, 2, 3, 4, 1)

        pbar = range(iterations)
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
    is_2d = len(warp.shape) == 4
    permute_vtoimg = (0, 3, 1, 2) if is_2d else (0, 4, 1, 2, 3)
    permute_imgtov = (0, 2, 3, 1) if is_2d else (0, 2, 3, 4, 1)
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


def regtangular_kernel(size: int):
    """Create a rectangular kernel."""
    return torch.ones(size)


def triangular_kernel(size: int):
    """Create a triangular kernel"""
    kernel_size = (size + 1) // 2
    if kernel_size % 2 == 0:
        kernel_size -= 1
    base = torch.ones((1, 1, kernel_size), dtype=torch.float32).div(kernel_size)
    padding = (size - kernel_size) // 2 + kernel_size // 2
    return F.conv1d(base, base, padding=padding).reshape(-1)


def gaussian_kernel(size: int):
    """Create a Gaussian kernel of size kernel_size."""
    sigma = torch.tensor(size / 3.0)
    kernel = gaussian_1d(sigma) * (2.5066282 * sigma)

    return kernel[:size]


kernel_dict = {
    "rectangular": regtangular_kernel,
    "triangular": triangular_kernel,
    "gaussian": gaussian_kernel,
}
