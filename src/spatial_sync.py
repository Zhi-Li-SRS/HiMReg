from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def _tensor_to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().float().cpu()
    while array.ndim > 2:
        array = array[0]
    return np.asarray(array.numpy(), dtype=np.float32)


def _as_torch_bchw(array_2d: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array_2d, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def sync_moving_scale_to_fixed(
    moving_sitk: sitk.Image,
    fixed_sitk: sitk.Image,
    enabled: bool,
    mode: str,
) -> Tuple[sitk.Image, Dict[str, float]]:
    """Resize/crop/pad moving image into fixed grid for registration.

    Matches benchmark behavior (`src/compare.py`) so main/benchmark pipelines
    share the same rescale semantics.
    """
    moving_arr = np.asarray(sitk.GetArrayFromImage(sitk.Cast(moving_sitk, sitk.sitkFloat32)), dtype=np.float32)
    fixed_arr = np.asarray(sitk.GetArrayFromImage(sitk.Cast(fixed_sitk, sitk.sitkFloat32)), dtype=np.float32)

    if moving_arr.ndim != 2 or fixed_arr.ndim != 2:
        raise ValueError(
            f"scale sync expects 2D arrays, got moving={moving_arr.shape}, fixed={fixed_arr.shape}"
        )

    moving_h, moving_w = moving_arr.shape
    fixed_h, fixed_w = fixed_arr.shape

    scale_x = float(fixed_w) / float(max(moving_w, 1))
    scale_y = float(fixed_h) / float(max(moving_h, 1))

    if (not enabled) or mode == "none":
        return moving_sitk, {
            "enabled": 0.0,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "isotropic_scale": 1.0,
            "resized_h": float(moving_h),
            "resized_w": float(moving_w),
            "fixed_h": float(fixed_h),
            "fixed_w": float(fixed_w),
        }

    if mode != "isotropic_fit":
        raise ValueError(f"Unsupported scale-sync mode: {mode}")

    iso_scale = min(scale_x, scale_y)
    resized_h = max(1, int(round(moving_h * iso_scale)))
    resized_w = max(1, int(round(moving_w * iso_scale)))

    moving_t = _as_torch_bchw(moving_arr, torch.device("cpu"))
    resized_t = F.interpolate(moving_t, size=(resized_h, resized_w), mode="bilinear", align_corners=True)
    resized_arr = _tensor_to_numpy_2d(resized_t)

    canvas = np.zeros((fixed_h, fixed_w), dtype=np.float32)
    copy_h = min(fixed_h, resized_h)
    copy_w = min(fixed_w, resized_w)

    src_y = max((resized_h - fixed_h) // 2, 0)
    src_x = max((resized_w - fixed_w) // 2, 0)
    dst_y = max((fixed_h - resized_h) // 2, 0)
    dst_x = max((fixed_w - resized_w) // 2, 0)

    canvas[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = resized_arr[
        src_y : src_y + copy_h,
        src_x : src_x + copy_w,
    ]

    synced = sitk.GetImageFromArray(canvas.astype(np.float32))
    synced.CopyInformation(fixed_sitk)

    return synced, {
        "enabled": 1.0,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "isotropic_scale": float(iso_scale),
        "resized_h": float(resized_h),
        "resized_w": float(resized_w),
        "fixed_h": float(fixed_h),
        "fixed_w": float(fixed_w),
        "src_y": float(src_y),
        "src_x": float(src_x),
        "dst_y": float(dst_y),
        "dst_x": float(dst_x),
        "copied_h": float(copy_h),
        "copied_w": float(copy_w),
    }
