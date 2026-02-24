from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(array, [p_low, p_high])
    if hi > lo:
        array = (array - lo) / (hi - lo)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def he_rgb_to_gray(image: np.ndarray, mode: str = "luma", invert_bf: bool = True) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image with shape [H, W, 3/4], got {array.shape}")

    rgb = array[..., :3].astype(np.float32)
    if mode == "luma":
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif mode == "mean":
        gray = rgb.mean(axis=-1)
    else:
        raise ValueError(f"Unsupported gray conversion mode: {mode}")

    if invert_bf:
        gray = gray.max() - gray
    return gray.astype(np.float32)


def to_uint8(image01: np.ndarray) -> np.ndarray:
    array = np.asarray(image01, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    return np.rint(array * 255.0).astype(np.uint8)


def compute_tissue_mask(
    image01: np.ndarray,
    method: str = "otsu",
    min_area: int = 2048,
    morph_radius: int = 3,
) -> np.ndarray:
    image = np.asarray(image01, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Tissue mask expects a 2D image, got shape {image.shape}")

    image_u8 = to_uint8(robust_normalize(image))
    if method != "otsu":
        raise ValueError(f"Unsupported mask method: {method}")

    _, mask = cv2.threshold(image_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = 2 * int(morph_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep[labels == label_idx] = 1

    if keep.sum() == 0:
        keep = (mask > 0).astype(np.uint8)
    return keep.astype(np.float32)


def build_registration_view(image: np.ndarray, preprocessing: Dict) -> np.ndarray:
    prepro = dict(preprocessing or {})
    array = np.asarray(image)

    if array.ndim == 3 and array.shape[-1] in (3, 4):
        image_type = str(prepro.get("image_type", "BF")).upper()
        invert_bf = bool(prepro.get("invert_bf", image_type == "BF"))
        gray_mode = str(prepro.get("gray_mode", "luma"))
        array = he_rgb_to_gray(array, mode=gray_mode, invert_bf=invert_bf)
    elif array.ndim == 3 and (prepro.get("channel_axis") == 0 or array.shape[0] <= 4):
        reduce_mode = str(prepro.get("channel_reduce", "index"))
        if reduce_mode == "max":
            array = np.max(array, axis=0)
        else:
            channel_index = int(prepro.get("channel_index", 0))
            channel_index = max(0, min(channel_index, array.shape[0] - 1))
            array = array[channel_index]
    elif array.ndim != 2:
        raise ValueError(
            f"Registration view expects 2D image or channelized 3D image, got shape {array.shape}"
        )

    array = np.asarray(array, dtype=np.float32)
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    if bool(prepro.get("robust_normalize", False)):
        p = prepro.get("robust_percentiles", (1.0, 99.0))
        p_low, p_high = float(p[0]), float(p[1])
        array = robust_normalize(array, p_low=p_low, p_high=p_high)

    return array.astype(np.float32)


def torch_gradient_magnitude(image: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape {image.shape}")

    kernel_1d = _gaussian_kernel1d(sigma, image.device, image.dtype)
    smoothed = _separable_conv2d(image, kernel_1d)

    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.view(1, 1, 3, 3).repeat(image.shape[1], 1, 1, 1)
    sobel_y = sobel_y.view(1, 1, 3, 3).repeat(image.shape[1], 1, 1, 1)

    gx = F.conv2d(smoothed, sobel_x, padding=1, groups=image.shape[1])
    gy = F.conv2d(smoothed, sobel_y, padding=1, groups=image.shape[1])
    return torch.sqrt(gx * gx + gy * gy + 1.0e-8)


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sigma = max(float(sigma), 0.1)
    radius = max(int(round(2.0 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / torch.clamp(kernel.sum(), min=1.0e-8)
    return kernel


def _separable_conv2d(image: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kx = kernel_1d.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    ky = kernel_1d.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    pad = kernel_1d.numel() // 2
    out = F.conv2d(image, kx, padding=(0, pad), groups=channels)
    out = F.conv2d(out, ky, padding=(pad, 0), groups=channels)
    return out
