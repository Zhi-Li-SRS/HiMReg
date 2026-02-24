import argparse
import csv
import hashlib
import json
import sys
import shutil
import subprocess
import re
import time
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import tifffile
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt
from skimage.metrics import normalized_mutual_information
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from HiMReg import HiMReg
from src.data_load import Image
from src.losses import LNCC, MutualInformation
from src.preprocess import compute_tissue_mask, robust_normalize, torch_gradient_magnitude
from src.spatial_sync import sync_moving_scale_to_fixed as shared_sync_moving_scale_to_fixed
from src.utils import set_global_seed


matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType for Nature-style figures
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["axes.titleweight"] = "bold"
matplotlib.rcParams["axes.labelweight"] = "bold"

# Prefer Arial (per manuscript/figure conventions); fall back gracefully.
try:
    matplotlib.rcParams["font.family"] = "Arial"
    font_manager.findfont("Arial", fallback_to_default=False)
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


@dataclass
class CaseSpec:
    case_id: str
    fixed_path: Path
    moving_path: Path


@dataclass
class MethodOutput:
    method: str
    warped: np.ndarray
    runtime_s: float
    sitk_transform: Optional[sitk.Transform] = None
    himreg_coords: Optional[torch.Tensor] = None
    himreg_affine_matrix: Optional[np.ndarray] = None
    backend: Optional[str] = None
    elastix_transform_fp: Optional[Path] = None
    transformix_bin: Optional[str] = None
    elastix_lib_dirs: Optional[List[str]] = None
    himreg_affine_warped: Optional[np.ndarray] = None
    himreg_diff_warped: Optional[np.ndarray] = None
    himreg_mode: Optional[str] = None


@dataclass
class CaseArtifacts:
    case_id: str
    case_output_dir: Path
    fixed_np: np.ndarray
    moving_synced_np: np.ndarray
    before_np: np.ndarray
    himreg_np: np.ndarray
    baseline_np: np.ndarray
    himreg_grid: Optional[np.ndarray]
    baseline_grid: Optional[np.ndarray]
    baseline_label: str
    himreg_label: str
    wsireg_np: Optional[np.ndarray] = None


def parse_csv_ints(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def parse_csv_floats(text: str) -> List[float]:
    values = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_roi_ids(text: str) -> List[str]:
    roi_ids = [x.strip() for x in text.split(",") if x.strip()]
    if not roi_ids:
        raise ValueError("roi-ids cannot be empty")
    normalized = []
    for roi in roi_ids:
        roi = roi.lower()
        if not roi.startswith("roi"):
            roi = f"roi{roi}"
        normalized.append(roi)
    return normalized


def build_stable_case_seed(base_seed: int, case_id: str) -> int:
    """Build a deterministic per-case seed independent of ROI ordering.

    For ``roiN`` style case ids, preserve historical seed progression:
    ``seed = base_seed + (N - 1)``. This keeps sampling behavior close to
    prior runs while still making seeds stable by case id (not loop order).
    """
    case_id_l = str(case_id).strip().lower()
    match = re.search(r"(\d+)$", case_id_l)
    if match:
        roi_num = int(match.group(1))
        return int((int(base_seed) + max(roi_num - 1, 0)) % (2**31 - 1))

    # Fallback for non-roi ids.
    digest = hashlib.sha256(case_id_l.encode("utf-8")).digest()
    case_hash = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return int((int(base_seed) + case_hash) % (2**31 - 1))


def tensor_to_numpy_2d(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu()
    while arr.ndim > 2:
        arr = arr[0]
    return np.asarray(arr.numpy(), dtype=np.float32)


def as_torch_bchw(array_2d: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array_2d, dtype=np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def save_csv(rows: List[Dict], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mask_overlap_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = float(np.logical_and(a, b).sum())
    a_sum = float(a.sum())
    b_sum = float(b.sum())
    union = float(np.logical_or(a, b).sum())

    if a_sum + b_sum == 0:
        return 1.0, 1.0

    dice = (2.0 * inter) / max(a_sum + b_sum, 1.0)
    iou = inter / max(union, 1.0)
    return float(dice), float(iou)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        return mask_bool
    eroded = binary_erosion(mask_bool)
    return np.logical_xor(mask_bool, eroded)


def surface_distance_metrics(mask_a: np.ndarray, mask_b: np.ndarray) -> Tuple[float, float]:
    surf_a = _surface(mask_a)
    surf_b = _surface(mask_b)

    if surf_a.sum() == 0 and surf_b.sum() == 0:
        return 0.0, 0.0
    if surf_a.sum() == 0 or surf_b.sum() == 0:
        return float("nan"), float("nan")

    dt_a = distance_transform_edt(~surf_a)
    dt_b = distance_transform_edt(~surf_b)
    d_ab = dt_b[surf_a]
    d_ba = dt_a[surf_b]
    all_d = np.concatenate([d_ab, d_ba], axis=0)

    hd95 = float(np.percentile(all_d, 95)) if all_d.size > 0 else float("nan")
    assd = float((d_ab.mean() + d_ba.mean()) / 2.0) if d_ab.size > 0 and d_ba.size > 0 else float("nan")
    return hd95, assd


def _safe_float(value: float) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except Exception:
        return float("nan")
    if np.isfinite(v):
        return v
    return float("nan")


def compute_registration_metrics(
    fixed_np: np.ndarray,
    warped_np: np.ndarray,
    device: torch.device,
    mi_metric: MutualInformation,
    lncc_metric: LNCC,
    grad_sigma: float = 1.0,
) -> Dict[str, float]:
    fixed = robust_normalize(fixed_np)
    warped = robust_normalize(warped_np)

    fixed_t = as_torch_bchw(fixed, device)
    warped_t = as_torch_bchw(warped, device)

    mi = -mi_metric(warped_t, fixed_t).item()

    fixed_grad_t = torch_gradient_magnitude(fixed_t, sigma=grad_sigma)
    warped_grad_t = torch_gradient_magnitude(warped_t, sigma=grad_sigma)
    gradcc = -lncc_metric(warped_grad_t, fixed_grad_t).item()

    fixed_grad = robust_normalize(tensor_to_numpy_2d(fixed_grad_t))
    warped_grad = robust_normalize(tensor_to_numpy_2d(warped_grad_t))

    grad_ssim = ssim(fixed_grad, warped_grad, data_range=1.0)
    nmi = normalized_mutual_information(fixed, warped, bins=64)

    fixed_mask = compute_tissue_mask(fixed, min_area=256) > 0.5
    warped_mask = compute_tissue_mask(warped, min_area=256) > 0.5

    dice, iou = mask_overlap_metrics(fixed_mask, warped_mask)
    hd95, assd = surface_distance_metrics(fixed_mask, warped_mask)

    return {
        "MI": _safe_float(mi),
        "NMI": _safe_float(nmi),
        "GradNCC": _safe_float(gradcc),
        "GradSSIM": _safe_float(grad_ssim),
        "TissueDice": _safe_float(dice),
        "TissueIoU": _safe_float(iou),
        "HD95": _safe_float(hd95),
        "ASSD": _safe_float(assd),
    }


def make_grid_image(shape_hw: Tuple[int, int], step: int = 64, thickness: int = 1) -> np.ndarray:
    h, w = shape_hw
    grid = np.zeros((h, w), dtype=np.float32)
    for y in range(0, h, max(step, 2)):
        y2 = min(y + thickness, h)
        grid[y:y2, :] = 1.0
    for x in range(0, w, max(step, 2)):
        x2 = min(x + thickness, w)
        grid[:, x:x2] = 1.0
    return grid


def build_overlay(fixed_np: np.ndarray, moving_np: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """Default overlay: fixed=blue, moving=yellow. Alignment becomes closer to white/gray."""
    fixed = robust_normalize(fixed_np)
    moving = robust_normalize(moving_np)

    base = 0.18 * fixed  # retain some fixed context in all channels
    r = np.clip(base + alpha * moving, 0.0, 1.0)             # moving -> red
    g = np.clip(base + alpha * moving, 0.0, 1.0)             # moving -> green
    b = np.clip(base + fixed, 0.0, 1.0)                      # fixed -> blue
    return np.stack([r, g, b], axis=-1)


def checkerboard(a: np.ndarray, b: np.ndarray, block: int = 48) -> np.ndarray:
    a01 = robust_normalize(a)
    b01 = robust_normalize(b)
    h, w = a01.shape
    yy, xx = np.indices((h, w))
    pattern = ((yy // block) + (xx // block)) % 2
    out = np.where(pattern == 0, a01, b01)
    return out


def resample_moving_to_fixed_grid(moving_sitk: sitk.Image, fixed_sitk: sitk.Image) -> np.ndarray:
    moved = sitk.Resample(
        moving_sitk,
        fixed_sitk,
        sitk.Transform(2, sitk.sitkIdentity),
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return np.asarray(sitk.GetArrayFromImage(moved), dtype=np.float32)


def _build_elastix_env(extra_lib_dirs: Sequence[str]) -> Dict[str, str]:
    """Return an env dict with LD_LIBRARY_PATH extended for local elastix .deb extractions."""
    env = dict(os.environ)
    extra = [d for d in extra_lib_dirs if d]
    if not extra:
        return env
    current = env.get("LD_LIBRARY_PATH", "")
    prefix = ":".join(extra)
    env["LD_LIBRARY_PATH"] = prefix + (":" + current if current else "")
    return env


def _resolve_elastix_binaries(
    elastix_bin: Optional[str],
    transformix_bin: Optional[str],
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Resolve elastix/transformix paths and required library dirs.

    Priority:
    1) CLI args
    2) PATH
    3) local extracted Ubuntu .deb under third_party/
    """
    lib_dirs: List[str] = []

    resolved_elastix = elastix_bin or shutil.which("elastix")
    resolved_transformix = transformix_bin or shutil.which("transformix")

    # Fall back to local extracted packages.
    local_elastix = ROOT / "third_party" / "elastix" / "usr" / "bin" / "elastix"
    local_transformix = ROOT / "third_party" / "elastix" / "usr" / "bin" / "transformix"
    if resolved_elastix is None and local_elastix.exists():
        resolved_elastix = str(local_elastix)
    if resolved_transformix is None and local_transformix.exists():
        resolved_transformix = str(local_transformix)

    if resolved_elastix and str(local_elastix) == str(resolved_elastix):
        # Local .deb extractions need LD_LIBRARY_PATH set for bundled deps.
        candidates = [
            ROOT / "third_party" / "elastix" / "usr" / "lib" / "x86_64-linux-gnu",
            ROOT / "third_party" / "elastix_deps" / "usr" / "lib",
            ROOT / "third_party" / "elastix_deps" / "usr" / "lib" / "x86_64-linux-gnu",
        ]
        for p in candidates:
            if p.exists():
                lib_dirs.append(str(p))

    return resolved_elastix, resolved_transformix, lib_dirs


def sync_moving_scale_to_fixed(
    moving_sitk: sitk.Image,
    fixed_sitk: sitk.Image,
    enabled: bool,
    mode: str,
) -> Tuple[sitk.Image, Dict[str, float]]:
    return shared_sync_moving_scale_to_fixed(
        moving_sitk=moving_sitk,
        fixed_sitk=fixed_sitk,
        enabled=enabled,
        mode=mode,
    )


def _gradient_mag_u8(image01: np.ndarray) -> np.ndarray:
    image01 = np.asarray(image01, dtype=np.float32)
    img = robust_normalize(image01)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return robust_normalize(mag)


def compute_rigid_init_phasecorr_sweep(
    fixed_np: np.ndarray,
    moving_np: np.ndarray,
    max_angle_deg: float,
    coarse_step_deg: float,
    fine_step_deg: float,
    refine_half_range_deg: float,
    use_fixed_mask: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    fixed = np.asarray(fixed_np, dtype=np.float32)
    moving = np.asarray(moving_np, dtype=np.float32)
    if fixed.shape != moving.shape:
        raise ValueError(f"Init expects same shape, got fixed={fixed.shape}, moving={moving.shape}")

    fixed_grad = _gradient_mag_u8(fixed)
    moving_grad = _gradient_mag_u8(moving)

    mask = None
    if use_fixed_mask:
        mask = compute_tissue_mask(robust_normalize(fixed), min_area=256) > 0.5
        fixed_grad = fixed_grad * mask.astype(np.float32)
        moving_grad = moving_grad * mask.astype(np.float32)

    h, w = fixed_grad.shape
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)

    center = (w * 0.5, h * 0.5)

    def eval_angle(angle_deg: float) -> Tuple[float, float, float]:
        m_rot = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        rotated = cv2.warpAffine(moving_grad, m_rot, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
        (dx, dy), response = cv2.phaseCorrelate(fixed_grad, rotated, window=window)
        return float(response), float(dx), float(dy)

    def build_candidate_matrix(angle_deg: float, dx: float, dy: float) -> np.ndarray:
        m_rot = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
        # cv2.phaseCorrelate(fixed, moved) returns a shift `s` such that shifting `fixed` by `s` aligns to `moved`.
        # Here `moved` is `moving` after applying `m_rot`, so `dx,dy` are in fixed-space pixels.
        #
        # Build a moving->fixed transform: rotate (m_rot), then translate by (-dx,-dy) to align to fixed.
        # HiMReg, however, needs fixed->moving (output->input) parameters for grid_sample, so we invert.
        m_trans = np.array([[1.0, 0.0, -float(dx)], [0.0, 1.0, -float(dy)]], dtype=np.float32)
        m_rot_h = np.vstack([m_rot, [0.0, 0.0, 1.0]]).astype(np.float32)
        m_trans_h = np.vstack([m_trans, [0.0, 0.0, 1.0]]).astype(np.float32)
        m_m2f = (m_trans_h @ m_rot_h).astype(np.float32)
        m_f2m = np.linalg.inv(m_m2f).astype(np.float32)
        return m_f2m[:2]

    angles_coarse = np.arange(-max_angle_deg, max_angle_deg + 1e-6, coarse_step_deg, dtype=np.float32)
    best = {"response": -1.0, "angle_deg": 0.0, "dx": 0.0, "dy": 0.0}
    for a in angles_coarse:
        response, dx, dy = eval_angle(float(a))
        if response > best["response"]:
            best = {"response": response, "angle_deg": float(a), "dx": dx, "dy": dy}

    fine_start = best["angle_deg"] - refine_half_range_deg
    fine_end = best["angle_deg"] + refine_half_range_deg
    angles_fine = np.arange(fine_start, fine_end + 1e-6, fine_step_deg, dtype=np.float32)
    for a in angles_fine:
        response, dx, dy = eval_angle(float(a))
        if response > best["response"]:
            best = {"response": response, "angle_deg": float(a), "dx": dx, "dy": dy}

    m_total = build_candidate_matrix(best["angle_deg"], best["dx"], best["dy"])
    theta = float(np.arctan2(m_total[1, 0], m_total[0, 0]))
    tx = float(m_total[0, 2])
    ty = float(m_total[1, 2])

    meta = {
        "init_mode": "phasecorr_sweep",
        "response": float(best["response"]),
        "angle_deg": float(best["angle_deg"]),
        "phasecorr_dx": float(best["dx"]),
        "phasecorr_dy": float(best["dy"]),
        "theta_rad": float(theta),
        "tx": float(tx),
        "ty": float(ty),
        "use_fixed_mask": 1.0 if use_fixed_mask else 0.0,
    }
    init_params = np.asarray([[theta, tx, ty]], dtype=np.float32)
    return init_params, meta


def run_himreg(
    fixed_image: Image,
    moving_image: Image,
    device: torch.device,
    himreg_mode: str,
    affine_scales: Sequence[int],
    affine_iters: Sequence[int],
    affine_lrs: Sequence[float],
    affine_loss_type: str,
    stage_rigid_iters: Optional[Sequence[int]],
    stage_affine_iters: Optional[Sequence[int]],
    affine_mask_weighted_loss: bool,
    affine_center_mode: str,
    affine_required_valid_ratio: float,
    affine_invalid_sample_strategy: str,
    affine_invalid_lr_decay: float,
    affine_oob_penalty_weight: float,
    affine_oob_penalty_adaptive: bool,
    gradcc_sigma: float,
    init_rigid_params: Optional[torch.Tensor],
    mi_weight: float,
    gradcc_weight: float,
    patience: int,
    min_delta: float,
    mi_num_samples: int,
    optimizer_scales_from_physical_shift: bool,
    diff_scales: Sequence[int],
    diff_iters: Sequence[int],
    diff_loss_type: str,
    diff_optimizer_lr: float,
    diff_smooth_warp_sigma: float,
    diff_smooth_grad_sigma: float,
    diff_tolerance: float,
    diff_max_tolerance_iters: int,
    diff_mi_num_samples: int,
    diff_regularization_weight: float,
    diff_mask_weighted_loss: bool,
    diff_gradcc_sigma: float = 1.0,
    diff_mi_weight: float = 1.0,
    diff_gradcc_weight: float = 0.25,
    # New params for upgraded pipeline
    affine_optimizer_type: str = "adam",
    affine_constrained: bool = False,
    asgd_a: Optional[float] = None,
    asgd_alpha: float = 1.0,
    affine_scale_mi_bins: Optional[Dict[str, int]] = None,
    affine_scale_loss_schedule: Optional[Dict[str, str]] = None,
    affine_max_step_lengths: Optional[List[float]] = None,
    rigid_lrs: Optional[Sequence[float]] = None,
    bspline_num_resolutions: int = 10,
    bspline_final_grid_spacing: int = 100,
    bspline_max_iterations: int = 200,
    bspline_num_samples: int = 50000,
    bspline_num_histogram_bins: int = 32,
    bspline_bending_energy_weight: float = 0.0,
    bspline_tolerance: float = 5e-4,
    bspline_max_tolerance_iters: int = 80,
    bspline_required_valid_ratio: float = 0.0,
    bspline_invalid_sample_strategy: str = "none",
    bspline_invalid_lr_decay: float = 0.5,
    bspline_oob_penalty_weight: float = 0.0,
    bspline_oob_penalty_adaptive: bool = True,
) -> MethodOutput:
    start = time.time()

    stage_iterations = {}
    if stage_rigid_iters is not None:
        stage_iterations["rigid"] = list(stage_rigid_iters)
    if stage_affine_iters is not None:
        stage_iterations["affine"] = list(stage_affine_iters)

    affine_kwargs = {
        "loss_type": affine_loss_type,
        "patience": int(patience),
        "min_delta": float(min_delta),
        "stages": ["rigid", "affine"],
        "stage_iterations": stage_iterations,
        "loss_weights": {"mi": float(mi_weight), "gradcc": float(gradcc_weight)},
        "mask_weighted_loss": bool(affine_mask_weighted_loss),
        "gradcc_sigma": float(gradcc_sigma),
        "init_rigid_params": init_rigid_params,
        "mi_num_samples": int(mi_num_samples),
        "optimizer_scales_from_physical_shift": bool(optimizer_scales_from_physical_shift),
        "stage_mi_bins": {"rigid": 16, "affine": 32},
        "scale_dependent_patience": True,
        "center_mode": str(affine_center_mode),
        "required_valid_ratio": float(affine_required_valid_ratio),
        "invalid_sample_strategy": str(affine_invalid_sample_strategy),
        "invalid_lr_decay": float(affine_invalid_lr_decay),
        "oob_penalty_weight": float(affine_oob_penalty_weight),
        "oob_penalty_adaptive": bool(affine_oob_penalty_adaptive),
        "optimizer_type": str(affine_optimizer_type),
        "constrained_affine": bool(affine_constrained),
        "asgd_a": asgd_a,
        "asgd_alpha": float(asgd_alpha),
    }
    if affine_scale_mi_bins:
        affine_kwargs["scale_mi_bins"] = affine_scale_mi_bins
    if affine_scale_loss_schedule:
        affine_kwargs["scale_loss_schedule"] = affine_scale_loss_schedule
    if affine_max_step_lengths:
        affine_kwargs["max_step_lengths"] = affine_max_step_lengths
    if rigid_lrs is not None:
        affine_kwargs["stage_lrs"] = {"rigid": list(rigid_lrs)}

    bspline_kwargs = {}
    if himreg_mode == "bspline":
        bspline_kwargs = {
            "num_resolutions": int(bspline_num_resolutions),
            "final_grid_spacing": int(bspline_final_grid_spacing),
            "max_iterations": int(bspline_max_iterations),
            "num_samples": int(bspline_num_samples),
            "num_histogram_bins": int(bspline_num_histogram_bins),
            "bending_energy_weight": float(bspline_bending_energy_weight),
            "tolerance": float(bspline_tolerance),
            "max_tolerance_iters": int(bspline_max_tolerance_iters),
            "required_valid_ratio": float(bspline_required_valid_ratio),
            "invalid_sample_strategy": str(bspline_invalid_sample_strategy),
            "invalid_lr_decay": float(bspline_invalid_lr_decay),
            "oob_penalty_weight": float(bspline_oob_penalty_weight),
            "oob_penalty_adaptive": bool(bspline_oob_penalty_adaptive),
        }

    registration = HiMReg(
        fixed_images=fixed_image,
        moving_images=moving_image,
        affine_scales=list(affine_scales),
        affine_iterations=list(affine_iters),
        scale_dependent_lr=list(affine_lrs),
        diff_scales=list(diff_scales),
        diff_iterations=list(diff_iters),
        affine_kwargs=affine_kwargs,
        diff_kwargs={
            "loss_type": diff_loss_type,
            "optimizer_lr": float(diff_optimizer_lr),
            "smooth_warp_sigma": float(diff_smooth_warp_sigma),
            "smooth_grad_sigma": float(diff_smooth_grad_sigma),
            "tolerance": float(diff_tolerance),
            "max_tolerance_iters": int(diff_max_tolerance_iters),
            "mi_num_samples": int(diff_mi_num_samples),
            "gradcc_sigma": float(diff_gradcc_sigma),
            "loss_weights": {"mi": float(diff_mi_weight), "gradcc": float(diff_gradcc_weight)},
            "mask_weighted_loss": bool(diff_mask_weighted_loss),
            "regularization_weight": float(diff_regularization_weight),
        },
        register_type=himreg_mode,
        bspline_kwargs=bspline_kwargs,
    )

    affine_transformed, nl_transformed, coords = registration.register(register_type=himreg_mode, save_transformed=True)
    if not affine_transformed:
        raise RuntimeError("HiMReg returned no affine output. Please check optimization settings.")

    affine_matrix = registration.affine_registration.get_final_transform().detach().float().cpu().numpy()

    runtime = time.time() - start
    affine_np = tensor_to_numpy_2d(affine_transformed[-1])
    if himreg_mode in ("diff", "bspline"):
        if not nl_transformed:
            raise RuntimeError(f"HiMReg {himreg_mode} mode returned no nonlinear output.")
        warped_np = tensor_to_numpy_2d(nl_transformed[-1])
        backend = f"HiMReg-{himreg_mode}"
    else:
        warped_np = affine_np
        backend = "HiMReg-affine"

    return MethodOutput(
        method="HiMReg",
        warped=warped_np,
        runtime_s=runtime,
        sitk_transform=None,
        himreg_coords=coords.detach().to(device),
        himreg_affine_matrix=affine_matrix,
        backend=backend,
        himreg_affine_warped=affine_np,
        himreg_diff_warped=warped_np if himreg_mode in ("diff", "bspline") else None,
        himreg_mode=himreg_mode,
    )


def _run_sitk_affine_baseline(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    iterations: int,
    hist_bins: int,
    sampling: float,
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[float],
) -> Tuple[np.ndarray, float, sitk.Transform]:
    start = time.time()

    fixed = sitk.Cast(fixed_sitk, sitk.sitkFloat32)
    moving = sitk.Cast(moving_sitk, sitk.sitkFloat32)

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetInitialTransform(initial, inPlace=False)
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=int(hist_bins))
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(max(min(sampling, 1.0), 1e-3)), sitk.sitkWallClock)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = reg.Execute(fixed, moving)

    warped = sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    runtime = time.time() - start

    warped_np = np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)
    return warped_np, runtime, final_transform


def _write_elastix_affine_param_file(
    output_path: Path,
    iterations: int,
    hist_bins: int,
    shrink_factors: Sequence[int],
) -> None:
    if len(shrink_factors) < 1:
        raise ValueError("shrink_factors must be non-empty")

    pyramid_schedule = " ".join([f"{s} {s}" for s in shrink_factors])
    lines = [
        '(Registration "MultiResolutionRegistration")',
        '(FixedImageDimension 2)',
        '(MovingImageDimension 2)',
        '(UseDirectionCosines "true")',
        '(Interpolator "LinearInterpolator")',
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(FinalBSplineInterpolationOrder 1)',
        '(Resampler "DefaultResampler")',
        '(DefaultPixelValue 0.0)',
        '(ResultImagePixelType "float")',
        '(ResultImageFormat "mhd")',
        '(WriteResultImage "true")',
        '(CompressResultImage "true")',
        '(Transform "AffineTransform")',
        '(AutomaticTransformInitialization "true")',
        '(AutomaticScalesEstimation "true")',
        '(Metric "AdvancedMattesMutualInformation")',
        f'(NumberOfHistogramBins {int(hist_bins)})',
        '(ImageSampler "Random")',
        '(NumberOfSpatialSamples 10000)',
        '(NewSamplesEveryIteration "true")',
        '(Optimizer "AdaptiveStochasticGradientDescent")',
        f'(MaximumNumberOfIterations {int(iterations)})',
        f'(NumberOfResolutions {len(shrink_factors)})',
        f'(ImagePyramidSchedule {pyramid_schedule})',
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_elastix_cli_affine(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    case_output_dir: Path,
    elastix_bin: str,
    transformix_bin: Optional[str],
    iterations: int,
    hist_bins: int,
    shrink_factors: Sequence[int],
    env: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, float, Optional[Path], Optional[str]]:
    case_output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = case_output_dir / "elastix_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    fixed_fp = work_dir / "fixed.mha"
    moving_fp = work_dir / "moving.mha"
    param_fp = work_dir / "affine.txt"

    sitk.WriteImage(sitk.Cast(fixed_sitk, sitk.sitkFloat32), str(fixed_fp))
    sitk.WriteImage(sitk.Cast(moving_sitk, sitk.sitkFloat32), str(moving_fp))
    _write_elastix_affine_param_file(param_fp, iterations=iterations, hist_bins=hist_bins, shrink_factors=shrink_factors)

    cmd = [
        elastix_bin,
        "-f",
        str(fixed_fp),
        "-m",
        str(moving_fp),
        "-out",
        str(work_dir),
        "-p",
        str(param_fp),
    ]

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    runtime = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"elastix CLI failed: {proc.stderr[-4000:]}")

    result_fp = work_dir / "result.0.mhd"
    if not result_fp.exists():
        raise RuntimeError("elastix did not produce result.0.mhd")

    warped = sitk.ReadImage(str(result_fp), sitk.sitkFloat32)
    warped_np = np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)

    transform_fp = work_dir / "TransformParameters.0.txt"
    if not transform_fp.exists():
        transform_fp = None

    transformix = transformix_bin if transformix_bin else shutil.which("transformix")
    return warped_np, runtime, transform_fp, transformix


def run_baseline(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    case_output_dir: Path,
    baseline_mode: str,
    elastix_bin: Optional[str],
    transformix_bin: Optional[str],
    iterations: int,
    hist_bins: int,
    sampling: float,
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[float],
    require_elastix: bool = False,
) -> MethodOutput:
    baseline_backend = "SimpleITK-affine-MI"
    # Keep result tables/figures aligned with the project's "Elastix" naming,
    # even when baseline execution path is the SimpleITK affine MI backend.
    sitk_method_label = "Elastix"

    if baseline_mode == "elastix":
        binary, resolved_transformix, lib_dirs = _resolve_elastix_binaries(elastix_bin, transformix_bin)
        if binary:
            try:
                env = _build_elastix_env(lib_dirs)
                warped_np, runtime, transform_fp, resolved_transformix = _run_elastix_cli_affine(
                    fixed_sitk=fixed_sitk,
                    moving_sitk=moving_sitk,
                    case_output_dir=case_output_dir,
                    elastix_bin=binary,
                    transformix_bin=resolved_transformix,
                    iterations=iterations,
                    hist_bins=hist_bins,
                    shrink_factors=shrink_factors,
                    env=env,
                )
                return MethodOutput(
                    method="Elastix",
                    warped=warped_np,
                    runtime_s=runtime,
                    sitk_transform=None,
                    himreg_coords=None,
                    backend="elastix-cli-affine",
                    elastix_transform_fp=transform_fp,
                    transformix_bin=resolved_transformix,
                    elastix_lib_dirs=lib_dirs,
                )
            except Exception as exc:
                if require_elastix:
                    raise
                print(f"[WARN] elastix CLI failed for {case_output_dir.name}, fallback to SimpleITK affine: {exc}")
        elif require_elastix:
            raise RuntimeError(
                "baseline-mode=elastix but elastix binary not found. "
                "Provide --elastix-bin/--transformix-bin or install elastix."
            )

    warped_np, runtime, sitk_tfm = _run_sitk_affine_baseline(
        fixed_sitk=fixed_sitk,
        moving_sitk=moving_sitk,
        iterations=iterations,
        hist_bins=hist_bins,
        sampling=sampling,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
    )
    return MethodOutput(
        method=sitk_method_label,
        warped=warped_np,
        runtime_s=runtime,
        sitk_transform=sitk_tfm,
        himreg_coords=None,
        backend=baseline_backend,
    )


def run_wsireg(
    fixed_sitk: sitk.Image,
    moving_sitk: sitk.Image,
    case_output_dir: Path,
    reg_params: List[str],
) -> MethodOutput:
    """Run wsireg (elastix-backed) registration and return warped image."""
    import tempfile
    from wsireg.wsireg2d import WsiReg2D

    fixed_np = sitk.GetArrayFromImage(fixed_sitk).astype(np.float32)
    moving_np = sitk.GetArrayFromImage(moving_sitk).astype(np.float32)

    fixed_spacing = fixed_sitk.GetSpacing()
    moving_spacing = moving_sitk.GetSpacing()
    fixed_res = float(fixed_spacing[0]) if fixed_spacing[0] > 0 else 1.0
    moving_res = float(moving_spacing[0]) if moving_spacing[0] > 0 else 1.0

    wsireg_dir = case_output_dir / "wsireg_work"
    wsireg_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()

    wsi_reg = WsiReg2D("wsireg_bench", str(wsireg_dir))
    wsi_reg.add_modality(
        "fixed",
        fixed_np,
        image_res=fixed_res,
        preprocessing={"image_type": "FL", "as_uint8": False},
    )
    wsi_reg.add_modality(
        "moving",
        moving_np,
        image_res=moving_res,
        preprocessing={"image_type": "FL", "as_uint8": False},
    )
    wsi_reg.add_reg_path(
        src_modality_name="moving",
        tgt_modality_name="fixed",
        reg_params=list(reg_params),
    )

    wsi_reg.register_images()
    output_fps = wsi_reg.transform_images(
        file_writer="ome.tiff",
        transform_non_reg=False,
        to_original_size=True,
    )

    runtime = time.time() - start

    # Read back the warped image
    warped_np = None
    if output_fps:
        warped_raw = tifffile.imread(output_fps[0])
        warped_np = np.squeeze(warped_raw).astype(np.float32)
    else:
        raise RuntimeError("wsireg produced no output files")

    # Resample to fixed grid if size mismatch
    if warped_np.shape != fixed_np.shape:
        warped_sitk = sitk.GetImageFromArray(warped_np)
        warped_sitk.CopyInformation(moving_sitk)
        warped_resampled = sitk.Resample(
            warped_sitk,
            fixed_sitk,
            sitk.Transform(2, sitk.sitkIdentity),
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
        warped_np = np.asarray(sitk.GetArrayFromImage(warped_resampled), dtype=np.float32)

    reg_label = "+".join(reg_params)
    return MethodOutput(
        method="WSiReg",
        warped=warped_np,
        runtime_s=runtime,
        sitk_transform=None,
        himreg_coords=None,
        backend=f"wsireg-{reg_label}",
    )


def warp_grid_himreg(
    moving_shape: Tuple[int, int],
    coords: torch.Tensor,
    device: torch.device,
    step: int,
) -> np.ndarray:
    moving_grid = make_grid_image(moving_shape, step=step, thickness=1)
    moving_grid_t = as_torch_bchw(moving_grid, device)
    warped_grid = F.grid_sample(moving_grid_t, coords, mode="bilinear", align_corners=True)
    return tensor_to_numpy_2d(warped_grid)


def warp_grid_sitk(
    moving_sitk: sitk.Image,
    fixed_sitk: sitk.Image,
    transform: sitk.Transform,
    step: int,
) -> np.ndarray:
    grid = make_grid_image((moving_sitk.GetSize()[1], moving_sitk.GetSize()[0]), step=step, thickness=1)
    grid_sitk = sitk.GetImageFromArray(grid.astype(np.float32))
    grid_sitk.CopyInformation(moving_sitk)

    warped = sitk.Resample(
        grid_sitk,
        fixed_sitk,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    return np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)


def warp_grid_elastix(
    moving_sitk: sitk.Image,
    fixed_sitk: sitk.Image,
    transform_fp: Path,
    transformix_bin: str,
    lib_dirs: Sequence[str],
    step: int,
) -> np.ndarray:
    """Warp a reference grid with transformix using an elastix parameter file."""
    grid = make_grid_image((moving_sitk.GetSize()[1], moving_sitk.GetSize()[0]), step=step, thickness=1)
    grid_sitk = sitk.GetImageFromArray(grid.astype(np.float32))
    grid_sitk.CopyInformation(moving_sitk)

    work_dir = transform_fp.parent / "transformix_grid"
    work_dir.mkdir(parents=True, exist_ok=True)
    grid_fp = work_dir / "grid.mha"
    sitk.WriteImage(grid_sitk, str(grid_fp))

    cmd = [
        transformix_bin,
        "-in",
        str(grid_fp),
        "-out",
        str(work_dir),
        "-tp",
        str(transform_fp),
    ]
    env = _build_elastix_env(list(lib_dirs))
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"transformix failed: {proc.stderr[-4000:]}")

    for name in ("result.mhd", "result.mha", "result.tif", "result.tiff"):
        result_fp = work_dir / name
        if result_fp.exists():
            warped = sitk.ReadImage(str(result_fp), sitk.sitkFloat32)
            return np.asarray(sitk.GetArrayFromImage(warped), dtype=np.float32)

    raise RuntimeError("transformix did not produce a result image in expected formats.")


def visualize_case(
    case_output_dir: Path,
    case_id: str,
    fixed_np: np.ndarray,
    before_np: np.ndarray,
    himreg_np: np.ndarray,
    baseline_np: np.ndarray,
    himreg_grid: Optional[np.ndarray],
    baseline_grid: Optional[np.ndarray],
    metric_rows: List[Dict],
    himreg_label: str,
    baseline_label: str,
) -> None:
    case_output_dir.mkdir(parents=True, exist_ok=True)

    # Panel
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    panel_images = [
        (fixed_np, "Fixed"),
        (before_np, "Before (resampled)"),
        (himreg_np, f"HiMReg {himreg_label}"),
        (baseline_np, f"{baseline_label} baseline"),
    ]
    for ax, (img, title) in zip(axes[0], panel_images):
        ax.imshow(robust_normalize(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    overlay_before = build_overlay(fixed_np, before_np)
    overlay_himreg = build_overlay(fixed_np, himreg_np)
    overlay_baseline = build_overlay(fixed_np, baseline_np)
    checker = checkerboard(himreg_np, baseline_np)

    axes[1, 0].imshow(overlay_before)
    axes[1, 0].set_title("Overlay Before")
    axes[1, 1].imshow(overlay_himreg)
    axes[1, 1].set_title("Overlay HiMReg")
    axes[1, 2].imshow(overlay_baseline)
    axes[1, 2].set_title(f"Overlay {baseline_label}")
    axes[1, 3].imshow(checker, cmap="gray")
    axes[1, 3].set_title(f"Checkerboard (HiMReg/{baseline_label})")
    for ax in axes[1]:
        ax.axis("off")

    summary_line = []
    for row in metric_rows:
        if row["method"] in {"HiMReg", baseline_label}:
            summary_line.append(f"{row['method']} NMI={row['NMI']:.3f}, Dice={row['TissueDice']:.3f}")
    fig.suptitle(f"{case_id} | " + " | ".join(summary_line), fontsize=11)

    fig.tight_layout()
    fig.savefig(case_output_dir / "comparison_panel.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Grid warps
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(make_grid_image(fixed_np.shape, step=64, thickness=1), cmap="gray")
    axes[0].set_title("Reference Grid")

    if himreg_grid is not None:
        axes[1].imshow(robust_normalize(himreg_grid), cmap="gray")
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("HiMReg Warped Grid")

    if baseline_grid is not None:
        axes[2].imshow(robust_normalize(baseline_grid), cmap="gray")
    else:
        axes[2].text(0.5, 0.5, "N/A", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title(f"{baseline_label} Warped Grid")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(case_output_dir / "grid_warp_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_roi_montage(
    artifacts: List[CaseArtifacts],
    output_path: Path,
    dpi: int,
) -> None:
    if not artifacts:
        return

    n_rows = len(artifacts)
    n_cols = 7
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows), squeeze=False)
    column_titles = [
        "SRS (791 moving)",
        "H&E (fixed)",
        "Before registration",
        f"HiMReg ({artifacts[0].himreg_label})",
        artifacts[0].baseline_label,
        "HiMReg warped grid",
        f"{artifacts[0].baseline_label} warped grid",
    ]

    for r, art in enumerate(artifacts):
        images = [
            art.moving_synced_np,
            art.fixed_np,
            art.before_np,
            art.himreg_np,
            art.baseline_np,
            art.himreg_grid if art.himreg_grid is not None else np.zeros_like(art.fixed_np),
            art.baseline_grid if art.baseline_grid is not None else np.zeros_like(art.fixed_np),
        ]
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(robust_normalize(images[c]), cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(column_titles[c], fontsize=10)
        axes[r, 0].set_ylabel(art.case_id, fontsize=10, rotation=90, labelpad=10, va="center")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def magenta_green_overlay(fixed_np: np.ndarray, warped_np: np.ndarray) -> np.ndarray:
    """Create a magenta/green overlay: fixed=green, warped=magenta. Alignment→gray."""
    f = robust_normalize(fixed_np)
    w = robust_normalize(warped_np)
    r = np.clip(w, 0.0, 1.0)
    g = np.clip(f, 0.0, 1.0)
    b = np.clip(w, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def blue_yellow_overlay(fixed_np: np.ndarray, warped_np: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    """High-contrast overlay: fixed=blue, warped=moving=yellow. Alignment pops as near-white."""
    return build_overlay(fixed_np, warped_np, alpha=alpha)


def plot_elastix_vs_wsireg_figure(
    artifacts: List[CaseArtifacts],
    all_rows: List[Dict],
    output_dir: Path,
) -> None:
    """Focused figure: overlays + metrics for Before vs Elastix vs WSiReg.

    Produces: elastix_vs_wsireg_figure.png/.pdf
    """
    if not artifacts:
        return

    has_wsireg = any(art.wsireg_np is not None for art in artifacts)
    if not has_wsireg:
        print("[INFO] WSiReg outputs not present; skipping elastix_vs_wsireg_figure.")
        return

    baseline_label = artifacts[0].baseline_label
    # If baseline isn't elastix, still plot as "baseline" but keep the label truthful.
    methods_overlay = ["Before", baseline_label, "WSiReg"]

    # Figure layout: top montage (overlays), bottom row of bar charts (metrics).
    FONTSIZE = 8
    AXES_LW = 0.6
    FIG_WIDTH_IN = 180 / 25.4

    n_rows = len(artifacts)
    n_cols = len(methods_overlay)
    panel_a_height = 1.15 * n_rows

    metrics = ["NMI", "GradNCC", "GradSSIM", "TissueDice", "TissueIoU", "HD95", "ASSD", "Runtime_s"]
    panel_b_height = 1.8
    total_height = panel_a_height + panel_b_height + 0.55

    fig = plt.figure(figsize=(FIG_WIDTH_IN, total_height))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 1,
        height_ratios=[panel_a_height, panel_b_height],
        hspace=0.35,
        figure=fig,
    )

    # ---- Panel (a): overlays ----
    gs_a = gs[0].subgridspec(n_rows, n_cols, wspace=0.04, hspace=0.08)
    col_titles = ["Before registration", baseline_label, "WSiReg"]
    for r, art in enumerate(artifacts):
        overlays = [
            blue_yellow_overlay(art.fixed_np, art.before_np),
            blue_yellow_overlay(art.fixed_np, art.baseline_np),
            blue_yellow_overlay(art.fixed_np, art.wsireg_np if art.wsireg_np is not None else art.before_np),
        ]
        for c in range(n_cols):
            ax = fig.add_subplot(gs_a[r, c])
            ax.imshow(overlays[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(AXES_LW)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=FONTSIZE, pad=3)
            if c == 0:
                ax.set_ylabel(art.case_id, fontsize=FONTSIZE, rotation=90, labelpad=6)

    fig.text(0.01, 0.98, "a", fontsize=FONTSIZE + 3, fontweight="bold", va="top")

    # ---- Panel (b): summary metrics ----
    from collections import OrderedDict
    colors = OrderedDict(
        [
            ("Before", "#9A9A9A"),
            (baseline_label, "#1F77B4"),  # blue
            ("WSiReg", "#FF7F0E"),        # orange
        ]
    )
    gs_b = gs[1].subgridspec(2, 4, wspace=0.55, hspace=0.55)
    rng = np.random.default_rng(42)

    for idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs_b[idx // 4, idx % 4])
        labels = []
        means = []
        stds = []
        scatters = []
        for method in colors.keys():
            vals = np.asarray([float(r[metric]) for r in all_rows if r["method"] == method], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            labels.append(method)
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
            scatters.append(vals)

        x = np.arange(len(labels))
        bar_colors = [colors.get(l, "#777777") for l in labels]
        ax.bar(
            x, means, yerr=stds, capsize=2,
            color=bar_colors, edgecolor="black",
            linewidth=AXES_LW, alpha=0.88, width=0.68,
        )
        for xi, vals in zip(x, scatters):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                xi + jitter, vals,
                s=14, color="black", zorder=5, alpha=0.7,
                linewidths=0.3, edgecolors="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONTSIZE - 1, rotation=20, ha="right")
        ax.set_title(metric, fontsize=FONTSIZE, pad=3)
        ax.tick_params(axis="both", labelsize=FONTSIZE - 1, width=AXES_LW, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_linewidth(AXES_LW)
        ax.grid(axis="y", alpha=0.2, linewidth=AXES_LW)

    fig.text(0.01, panel_b_height / total_height + 0.02, "b", fontsize=FONTSIZE + 3, fontweight="bold", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"elastix_vs_wsireg_figure.{ext}",
            dpi=300,
            bbox_inches="tight",
            transparent=False,
        )
    plt.close(fig)
    print(f"Elastix vs WSiReg figure saved to {output_dir / 'elastix_vs_wsireg_figure.png'} and .pdf")


def plot_nature_figure(
    artifacts: List[CaseArtifacts],
    all_rows: List[Dict],
    output_dir: Path,
) -> None:
    """Generate a Nature-style comparison figure (panels a+b).

    Panel a: ROI overlay montage (Before / HiMReg / SimpleITK).
    Panel b: Grouped bar charts for NMI, TissueDice, GradNCC (+ HD95).
    """
    import matplotlib
    import matplotlib.font_manager as font_manager
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType for Nature submission
    matplotlib.rcParams["ps.fonttype"] = 42

    # Try to use Arial; fall back gracefully.
    try:
        matplotlib.rcParams["font.family"] = "Arial"
        font_manager.findfont("Arial", fallback_to_default=False)
    except Exception:
        matplotlib.rcParams["font.family"] = "DejaVu Sans"

    FONTSIZE = 7
    AXES_LW = 0.5
    FIG_WIDTH_MM = 180
    FIG_WIDTH_IN = FIG_WIDTH_MM / 25.4

    # NPG-inspired palette (colorblind-safe).
    COLOR_BEFORE = "#999999"
    COLOR_HIMREG = "#E64B35"
    COLOR_BASELINE = "#4DBBD5"

    n_rows = len(artifacts)
    if n_rows == 0:
        return

    baseline_label = artifacts[0].baseline_label
    has_wsireg = any(art.wsireg_np is not None for art in artifacts)

    # ---- Panel (a): Overlay montage ----
    n_overlay_cols = 4 if has_wsireg else 3
    panel_a_height = 1.1 * n_rows  # inches per row

    # ---- Panel (b): Bar charts ----
    bar_metrics = ["NMI", "TissueDice", "GradNCC", "HD95"]
    n_bar_cols = len(bar_metrics)
    panel_b_height = 1.6

    total_height = panel_a_height + panel_b_height + 0.5  # gap

    fig = plt.figure(figsize=(FIG_WIDTH_IN, total_height))

    # GridSpec: top section for overlays, bottom for bars.
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 1,
        height_ratios=[panel_a_height, panel_b_height],
        hspace=0.35,
        figure=fig,
    )

    # ---- Panel (a) ----
    gs_a = gs[0].subgridspec(n_rows, n_overlay_cols, wspace=0.04, hspace=0.08)
    col_titles = ["Before", "HiMReg", baseline_label]
    if has_wsireg:
        col_titles.append("WSiReg")

    for r, art in enumerate(artifacts):
        overlays = [
            blue_yellow_overlay(art.fixed_np, art.before_np),
            blue_yellow_overlay(art.fixed_np, art.himreg_np),
            blue_yellow_overlay(art.fixed_np, art.baseline_np),
        ]
        if has_wsireg:
            wsireg_img = art.wsireg_np if art.wsireg_np is not None else art.before_np
            overlays.append(blue_yellow_overlay(art.fixed_np, wsireg_img))
        for c in range(n_overlay_cols):
            ax = fig.add_subplot(gs_a[r, c])
            ax.imshow(overlays[c])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(AXES_LW)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=FONTSIZE, pad=3)
            if c == 0:
                ax.set_ylabel(art.case_id, fontsize=FONTSIZE, rotation=90, labelpad=6)

    # Panel label.
    fig.text(0.01, 0.98, "a", fontsize=FONTSIZE + 3, fontweight="bold", va="top")

    # ---- Panel (b): Grouped bar charts ----
    COLOR_WSIREG = "#59A14F"
    gs_b = gs[1].subgridspec(1, n_bar_cols, wspace=0.45)
    preferred_order = ["Before", "HiMReg", baseline_label, "WSiReg"]
    methods = [m for m in preferred_order if any(row["method"] == m for row in all_rows)]
    method_colors = {
        "Before": COLOR_BEFORE,
        "HiMReg": COLOR_HIMREG,
        baseline_label: COLOR_BASELINE,
        "WSiReg": COLOR_WSIREG,
    }

    for idx, metric in enumerate(bar_metrics):
        ax = fig.add_subplot(gs_b[0, idx])
        means, stds, colors, labels = [], [], [], []
        scatter_data = []
        for method in methods:
            vals = np.asarray(
                [float(r[metric]) for r in all_rows if r["method"] == method],
                dtype=np.float64,
            )
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
            colors.append(method_colors.get(method, "#777777"))
            labels.append(method)
            scatter_data.append(vals)

        x = np.arange(len(labels))
        bars = ax.bar(
            x, means, yerr=stds, capsize=2, color=colors,
            edgecolor="black", linewidth=AXES_LW, width=0.65, alpha=0.85,
        )
        # Overlay individual ROI data points.
        for xi, vals in zip(x, scatter_data):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                xi + jitter, vals,
                s=12, color="black", zorder=5, alpha=0.7, linewidths=0.3, edgecolors="white",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=FONTSIZE - 1, rotation=25, ha="right")
        ax.set_title(metric, fontsize=FONTSIZE, pad=3)
        ax.tick_params(axis="both", labelsize=FONTSIZE - 1, width=AXES_LW, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_linewidth(AXES_LW)
        ax.grid(axis="y", alpha=0.2, linewidth=AXES_LW)

    fig.text(0.01, panel_b_height / total_height + 0.02, "b", fontsize=FONTSIZE + 3, fontweight="bold", va="top")

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"nature_figure.{ext}",
            dpi=300,
            bbox_inches="tight",
            transparent=False,
        )
    plt.close(fig)
    print(f"Nature figure saved to {output_dir / 'nature_figure.png'} and .pdf")


def build_cases(args: argparse.Namespace) -> List[CaseSpec]:
    if args.fixed and args.moving:
        fixed = Path(args.fixed)
        moving = Path(args.moving)
        if not fixed.exists():
            raise FileNotFoundError(f"fixed not found: {fixed}")
        if not moving.exists():
            raise FileNotFoundError(f"moving not found: {moving}")
        case_id = args.case_name.strip() if args.case_name else "single_case"
        return [CaseSpec(case_id=case_id, fixed_path=fixed, moving_path=moving)]

    bench_dir = Path(args.benchmark_dir)
    if not bench_dir.exists():
        raise FileNotFoundError(f"benchmark dir not found: {bench_dir}")

    cases: List[CaseSpec] = []
    roi_ids = parse_roi_ids(args.roi_ids)
    for roi in roi_ids:
        fixed = bench_dir / f"{roi}_{args.fixed_suffix}.tif"
        moving = bench_dir / f"{roi}_{args.moving_suffix}.tif"
        if not fixed.exists():
            raise FileNotFoundError(f"Missing fixed image for {roi}: {fixed}")
        if not moving.exists():
            raise FileNotFoundError(f"Missing moving image for {roi}: {moving}")
        cases.append(CaseSpec(case_id=roi, fixed_path=fixed, moving_path=moving))
    return cases


def summarize_metrics(rows: List[Dict]) -> List[Dict]:
    metrics = ["MI", "NMI", "GradNCC", "GradSSIM", "TissueDice", "TissueIoU", "HD95", "ASSD", "Runtime_s"]
    methods = sorted({row["method"] for row in rows})
    out = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        for metric in metrics:
            vals = np.asarray([float(r[metric]) for r in method_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean": float(vals.mean()) if vals.size else float("nan"),
                    "std": float(vals.std(ddof=0)) if vals.size else float("nan"),
                    "n": int(vals.size),
                }
            )
    return out


def plot_summary(rows: List[Dict], output_dir: Path) -> None:
    metrics = ["NMI", "GradNCC", "GradSSIM", "TissueDice", "TissueIoU", "HD95", "ASSD", "Runtime_s"]
    preferred_order = ["Before", "HiMReg", "SimpleITK", "Elastix", "WSiReg"]
    methods = [m for m in preferred_order if any(row["method"] == m for row in rows)]
    colors = {
        "Before": "gray",
        "HiMReg": "#e66f51",
        "SimpleITK": "#4f83cc",
        "Elastix": "#4f83cc",
        "WSiReg": "#59a14f",
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        means = []
        stds = []
        labels = []
        for method in methods:
            vals = np.asarray([float(r[metric]) for r in rows if r["method"] == method], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
            labels.append(method)

        x = np.arange(len(labels))
        bar_colors = [colors.get(label, "#777777") for label in labels]
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, color=bar_colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "summary_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def get_default_preprocessing(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    fixed_default = {
        "image_type": "BF",
        "invert_bf": True,
        "gray_mode": "luma",
        "robust_normalize": True,
        "robust_percentiles": [1.0, 99.0],
    }
    moving_default = {
        "image_type": "FL",
        "robust_normalize": True,
        "robust_percentiles": [1.0, 99.0],
    }

    fixed_pre = fixed_default
    moving_pre = moving_default

    if args.fixed_preprocessing_json:
        fixed_pre = json.loads(args.fixed_preprocessing_json)
    if args.moving_preprocessing_json:
        moving_pre = json.loads(args.moving_preprocessing_json)

    return fixed_pre, moving_pre


def run_case(
    case: CaseSpec,
    args: argparse.Namespace,
    device: torch.device,
    fixed_pre: Dict,
    moving_pre: Dict,
    mi_metric: MutualInformation,
    lncc_metric: LNCC,
    case_seed: int,
) -> Tuple[List[Dict], CaseArtifacts]:
    case_output = Path(args.output) / case.case_id
    case_output.mkdir(parents=True, exist_ok=True)

    # Keep per-case randomness stable so ROI order does not change outcomes.
    set_global_seed(case_seed, args.deterministic)

    fixed_image = Image.load_file(str(case.fixed_path), device=device, preprocessing=fixed_pre)
    moving_image_raw = Image.load_file(str(case.moving_path), device=device, preprocessing=moving_pre)

    fixed_sitk = fixed_image.images[0].itk_image
    moving_sitk_raw = moving_image_raw.images[0].itk_image

    moving_sitk, scale_sync_meta = sync_moving_scale_to_fixed(
        moving_sitk=moving_sitk_raw,
        fixed_sitk=fixed_sitk,
        enabled=bool(args.moving_scale_sync),
        mode=str(args.scale_sync_mode),
    )
    moving_image = Image(moving_sitk, device=device, preprocessing={})

    fixed_np = tensor_to_numpy_2d(fixed_image())
    moving_np = tensor_to_numpy_2d(moving_image())
    moving_before = resample_moving_to_fixed_grid(moving_sitk, fixed_sitk)

    init_meta = {"init_mode": "centroid"}
    init_rigid_params_t = None
    if bool(getattr(args, "wsireg_aligned_preset", False)):
        init_meta = {"init_mode": "wsireg_centroid_auto"}
    elif args.init_mode == "phasecorr_sweep":
        init_params_np, init_meta = compute_rigid_init_phasecorr_sweep(
            fixed_np=fixed_np,
            moving_np=moving_np,
            max_angle_deg=args.init_max_angle_deg,
            coarse_step_deg=args.init_coarse_step_deg,
            fine_step_deg=args.init_fine_step_deg,
            refine_half_range_deg=args.init_refine_half_range_deg,
            use_fixed_mask=bool(args.init_use_fixed_mask),
        )
        init_rigid_params_t = torch.from_numpy(init_params_np).to(device=device)

    # Parse JSON strings for scale_mi_bins and scale_loss_schedule if provided.
    affine_scale_mi_bins = None
    if getattr(args, "affine_scale_mi_bins", None):
        affine_scale_mi_bins = json.loads(args.affine_scale_mi_bins)
    affine_scale_loss_schedule = None
    if getattr(args, "affine_scale_loss_schedule", None):
        affine_scale_loss_schedule = json.loads(args.affine_scale_loss_schedule)

    himreg_out = run_himreg(
        fixed_image=fixed_image,
        moving_image=moving_image,
        device=device,
        himreg_mode=str(args.himreg_mode),
        affine_scales=args.affine_scales,
        affine_iters=args.affine_iterations,
        affine_lrs=args.affine_lrs,
        affine_loss_type=args.affine_loss_type,
        stage_rigid_iters=args.stage_rigid_iters,
        stage_affine_iters=args.stage_affine_iters,
        affine_mask_weighted_loss=args.affine_mask_weighted_loss,
        affine_center_mode=args.affine_center_mode,
        affine_required_valid_ratio=args.affine_required_valid_ratio,
        affine_invalid_sample_strategy=args.affine_invalid_sample_strategy,
        affine_invalid_lr_decay=args.affine_invalid_lr_decay,
        affine_oob_penalty_weight=args.affine_oob_penalty_weight,
        affine_oob_penalty_adaptive=args.affine_oob_penalty_adaptive,
        gradcc_sigma=args.gradcc_sigma,
        init_rigid_params=init_rigid_params_t,
        mi_weight=args.mi_weight,
        gradcc_weight=args.gradcc_weight,
        patience=args.patience,
        min_delta=args.min_delta,
        mi_num_samples=args.himreg_mi_num_samples,
        optimizer_scales_from_physical_shift=args.himreg_optimizer_scales_from_physical_shift,
        diff_scales=args.diff_scales,
        diff_iters=args.diff_iterations,
        diff_loss_type=args.diff_loss_type,
        diff_optimizer_lr=args.diff_optimizer_lr,
        diff_smooth_warp_sigma=args.diff_smooth_warp_sigma,
        diff_smooth_grad_sigma=args.diff_smooth_grad_sigma,
        diff_tolerance=args.diff_tolerance,
        diff_max_tolerance_iters=args.diff_max_tolerance_iters,
        diff_mi_num_samples=args.diff_mi_num_samples,
        diff_regularization_weight=args.diff_regularization_weight,
        diff_mask_weighted_loss=args.diff_mask_weighted_loss,
        diff_gradcc_sigma=args.diff_gradcc_sigma,
        diff_mi_weight=args.diff_mi_weight,
        diff_gradcc_weight=args.diff_gradcc_weight,
        # New params
        affine_optimizer_type=args.affine_optimizer_type,
        affine_constrained=args.affine_constrained,
        asgd_a=args.asgd_a,
        asgd_alpha=args.asgd_alpha,
        affine_scale_mi_bins=affine_scale_mi_bins,
        affine_scale_loss_schedule=affine_scale_loss_schedule,
        affine_max_step_lengths=getattr(args, "affine_max_step_lengths", None),
        rigid_lrs=getattr(args, "rigid_lrs", None),
        bspline_num_resolutions=args.bspline_num_resolutions,
        bspline_final_grid_spacing=args.bspline_final_grid_spacing,
        bspline_max_iterations=args.bspline_max_iterations,
        bspline_num_samples=args.bspline_num_samples,
        bspline_num_histogram_bins=args.bspline_num_histogram_bins,
        bspline_bending_energy_weight=args.bspline_bending_energy_weight,
        bspline_tolerance=args.bspline_tolerance,
        bspline_max_tolerance_iters=args.bspline_max_tolerance_iters,
        bspline_required_valid_ratio=args.bspline_required_valid_ratio,
        bspline_invalid_sample_strategy=args.bspline_invalid_sample_strategy,
        bspline_invalid_lr_decay=args.bspline_invalid_lr_decay,
        bspline_oob_penalty_weight=args.bspline_oob_penalty_weight,
        bspline_oob_penalty_adaptive=args.bspline_oob_penalty_adaptive,
    )

    # Default behavior: when comparing against elastix, do not silently fall back to SimpleITK.
    # This keeps the benchmark honest and avoids "baseline" meaning different things across runs.
    require_elastix = False
    if args.baseline_mode == "elastix":
        require_elastix = bool(args.require_elastix) or (not bool(getattr(args, "allow_sitk_fallback", False)))

    baseline_out = run_baseline(
        fixed_sitk=fixed_sitk,
        moving_sitk=moving_sitk,
        case_output_dir=case_output,
        baseline_mode=args.baseline_mode,
        elastix_bin=args.elastix_bin,
        transformix_bin=args.transformix_bin,
        iterations=args.baseline_iterations,
        hist_bins=args.baseline_hist_bins,
        sampling=args.baseline_sampling,
        shrink_factors=args.baseline_shrink_factors,
        smoothing_sigmas=args.baseline_smoothing_sigmas,
        require_elastix=require_elastix,
    )

    baseline_label = str(baseline_out.method)
    himreg_label = "diff" if args.himreg_mode == "diff" else "affine"

    # Run wsireg if requested
    wsireg_out = None
    if getattr(args, "run_wsireg", False):
        try:
            wsireg_out = run_wsireg(
                fixed_sitk=fixed_sitk,
                moving_sitk=moving_sitk,
                case_output_dir=case_output,
                reg_params=args.wsireg_reg_params,
            )
            print(f"  WSiReg completed in {wsireg_out.runtime_s:.1f}s ({wsireg_out.backend})")
        except Exception as exc:
            print(f"  [WARN] WSiReg failed for {case.case_id}: {exc}")

    # Save warped images
    tifffile.imwrite(case_output / "fixed_processed.tif", fixed_np.astype(np.float32))
    moving_synced_np = np.asarray(sitk.GetArrayFromImage(moving_sitk), dtype=np.float32)
    tifffile.imwrite(case_output / "moving_scale_synced.tif", moving_synced_np)
    tifffile.imwrite(case_output / "moving_before_resampled.tif", moving_before.astype(np.float32))
    tifffile.imwrite(case_output / "himreg_warped.tif", himreg_out.warped.astype(np.float32))
    if himreg_out.himreg_affine_warped is not None:
        tifffile.imwrite(case_output / "himreg_affine_warped.tif", himreg_out.himreg_affine_warped.astype(np.float32))
    if himreg_out.himreg_diff_warped is not None:
        tifffile.imwrite(case_output / "himreg_diff_warped.tif", himreg_out.himreg_diff_warped.astype(np.float32))
    tifffile.imwrite(case_output / "baseline_warped.tif", baseline_out.warped.astype(np.float32))
    elastix_warp_fp = case_output / "elastix_warped.tif"
    simpleitk_warp_fp = case_output / "simpleitk_warped.tif"
    if baseline_label == "Elastix":
        tifffile.imwrite(elastix_warp_fp, baseline_out.warped.astype(np.float32))
        if simpleitk_warp_fp.exists():
            simpleitk_warp_fp.unlink()
    if baseline_label == "SimpleITK":
        tifffile.imwrite(simpleitk_warp_fp, baseline_out.warped.astype(np.float32))
        if elastix_warp_fp.exists():
            elastix_warp_fp.unlink()
    if wsireg_out is not None:
        tifffile.imwrite(case_output / "wsireg_warped.tif", wsireg_out.warped.astype(np.float32))
    (case_output / "scale_sync.json").write_text(json.dumps(scale_sync_meta, indent=2), encoding="utf-8")
    (case_output / "init_rigid.json").write_text(json.dumps(init_meta, indent=2), encoding="utf-8")
    if himreg_out.himreg_affine_matrix is not None:
        np.save(case_output / "himreg_affine_matrix.npy", himreg_out.himreg_affine_matrix)

    rows: List[Dict] = []
    outputs = [
        ("Before", moving_before, 0.0, "resample-only"),
        ("HiMReg", himreg_out.warped, himreg_out.runtime_s, himreg_out.backend),
        (baseline_label, baseline_out.warped, baseline_out.runtime_s, baseline_out.backend),
    ]
    if wsireg_out is not None:
        outputs.append(("WSiReg", wsireg_out.warped, wsireg_out.runtime_s, wsireg_out.backend))

    for method, warped, runtime_s, backend in outputs:
        metric_dict = compute_registration_metrics(
            fixed_np=fixed_np,
            warped_np=warped,
            device=device,
            mi_metric=mi_metric,
            lncc_metric=lncc_metric,
            grad_sigma=args.gradcc_sigma,
        )
        row = {
            "case_id": case.case_id,
            "CaseSeed": int(case_seed),
            "fixed": str(case.fixed_path),
            "moving": str(case.moving_path),
            "method": method,
            "backend": backend,
            "HiMRegMode": str(args.himreg_mode),
            "BaselineMode": str(args.baseline_mode),
            "ScaleSyncEnabled": float(scale_sync_meta.get("enabled", 0.0)),
            "ScaleSyncMode": str(args.scale_sync_mode),
            "ScaleX_FixedOverMoving": float(scale_sync_meta.get("scale_x", float("nan"))),
            "ScaleY_FixedOverMoving": float(scale_sync_meta.get("scale_y", float("nan"))),
            "IsotropicScaleApplied": float(scale_sync_meta.get("isotropic_scale", float("nan"))),
            "InitMode": str(init_meta.get("init_mode", "centroid")),
            "InitResponse": float(init_meta.get("response", float("nan"))),
            "InitAngleDeg": float(init_meta.get("angle_deg", float("nan"))),
            "InitTx": float(init_meta.get("tx", float("nan"))),
            "InitTy": float(init_meta.get("ty", float("nan"))),
            **metric_dict,
            "Runtime_s": float(runtime_s),
        }
        rows.append(row)

    himreg_grid = None
    if himreg_out.himreg_coords is not None:
        moving_shape = tuple(moving_image().shape[2:])
        himreg_grid = warp_grid_himreg(
            moving_shape=moving_shape,
            coords=himreg_out.himreg_coords,
            device=device,
            step=args.grid_step,
        )

    baseline_grid = None
    if baseline_out.sitk_transform is not None:
        baseline_grid = warp_grid_sitk(
            moving_sitk=moving_sitk,
            fixed_sitk=fixed_sitk,
            transform=baseline_out.sitk_transform,
            step=args.grid_step,
        )
    elif baseline_out.elastix_transform_fp is not None and baseline_out.transformix_bin:
        baseline_grid = warp_grid_elastix(
            moving_sitk=moving_sitk,
            fixed_sitk=fixed_sitk,
            transform_fp=baseline_out.elastix_transform_fp,
            transformix_bin=str(baseline_out.transformix_bin),
            lib_dirs=baseline_out.elastix_lib_dirs or [],
            step=args.grid_step,
        )

    if himreg_grid is not None:
        tifffile.imwrite(case_output / "himreg_grid_warped.tif", himreg_grid.astype(np.float32))
    if baseline_grid is not None:
        tifffile.imwrite(case_output / "baseline_grid_warped.tif", baseline_grid.astype(np.float32))

    visualize_case(
        case_output_dir=case_output,
        case_id=case.case_id,
        fixed_np=fixed_np,
        before_np=moving_before,
        himreg_np=himreg_out.warped,
        baseline_np=baseline_out.warped,
        himreg_grid=himreg_grid,
        baseline_grid=baseline_grid,
        metric_rows=rows,
        himreg_label=himreg_label,
        baseline_label=baseline_label,
    )

    save_csv(rows, case_output / "metrics.csv")
    artifacts = CaseArtifacts(
        case_id=case.case_id,
        case_output_dir=case_output,
        fixed_np=fixed_np,
        moving_synced_np=moving_synced_np,
        before_np=moving_before,
        himreg_np=himreg_out.warped,
        baseline_np=baseline_out.warped,
        himreg_grid=himreg_grid,
        baseline_grid=baseline_grid,
        baseline_label=baseline_label,
        himreg_label=himreg_label,
        wsireg_np=wsireg_out.warped if wsireg_out is not None else None,
    )
    return rows, artifacts


def validate_lengths(args: argparse.Namespace) -> None:
    n_scales = len(args.affine_scales)
    if len(args.affine_iterations) != n_scales:
        raise ValueError("affine-iterations must match affine-scales length")
    if len(args.affine_lrs) != n_scales:
        raise ValueError("affine-lrs must match affine-scales length")

    if args.stage_rigid_iters is not None and len(args.stage_rigid_iters) != n_scales:
        raise ValueError("stage-rigid-iters must match affine-scales length")
    if args.stage_affine_iters is not None and len(args.stage_affine_iters) != n_scales:
        raise ValueError("stage-affine-iters must match affine-scales length")

    if len(args.baseline_shrink_factors) != len(args.baseline_smoothing_sigmas):
        raise ValueError("baseline-shrink-factors must match baseline-smoothing-sigmas length")
    if len(args.diff_scales) != len(args.diff_iterations):
        raise ValueError("diff-iterations must match diff-scales length")

    # B-spline validation (WSiReg-style — no per-scale lists to validate)


def apply_wsireg_aligned_preset(args: argparse.Namespace) -> None:
    """Force a wsireg/elastix-like optimization preset for HiMReg.

    This preset intentionally removes custom affine/nonlinear behavior that
    diverges from wsireg defaults, so comparisons are less confounded.
    """
    if not bool(getattr(args, "wsireg_aligned_preset", False)):
        return

    # Affine: wsireg defaults are MI + ASGD + unconstrained affine.
    args.affine_loss_type = "mi"
    args.affine_optimizer_type = "asgd"
    args.affine_constrained = False
    args.asgd_a = None
    args.asgd_alpha = 1.0
    args.affine_scale_loss_schedule = None
    args.affine_scale_mi_bins = None
    args.affine_center_mode = "none"

    # Initialization: avoid custom phase-correlation sweep.
    args.init_mode = "centroid"
    args.init_use_fixed_mask = False

    # Nonlinear safety guardrails:
    # The nominal elastix RequiredRatioOfValidSamples is 0.05, but in this
    # pure-PyTorch implementation we need stricter defaults to avoid local
    # collapse artifacts (e.g., thin yellow-band failures in overlays).
    args.bspline_required_valid_ratio = max(float(args.bspline_required_valid_ratio), 0.2)
    if str(args.bspline_invalid_sample_strategy).lower() == "none":
        args.bspline_invalid_sample_strategy = "lr_decay"
    args.bspline_oob_penalty_weight = max(float(args.bspline_oob_penalty_weight), 0.2)
    args.bspline_oob_penalty_adaptive = True

    print(
        "[INFO] Applied wsireg-aligned preset: MI+ASGD affine, unconstrained affine, "
        "centroid init, stronger bspline valid-sample/OOB safeguards."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-ROI benchmark: HiMReg (affine/diff) vs SimpleITK/Elastix baseline (moving 791 -> fixed HE)"
    )

    parser.add_argument("--fixed", type=str, default=None, help="Single-case fixed path")
    parser.add_argument("--moving", type=str, default=None, help="Single-case moving path")
    parser.add_argument("--case-name", type=str, default=None, help="Single-case id")

    parser.add_argument("--benchmark-dir", type=str, default="review_response_data/benchmark")
    parser.add_argument("--roi-ids", type=str, default="2,3,4,5,6", help="Comma separated ROI ids")
    parser.add_argument("--fixed-suffix", type=str, default="he", help="Suffix for fixed image file")
    parser.add_argument("--moving-suffix", type=str, default="791", help="Suffix for moving image file")

    parser.add_argument("--output", type=str, default="comparison_results/benchmark_diff_vs_sitk_roi2to6")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--himreg-mode", type=str, default="diff", choices=["affine", "diff", "bspline"])

    parser.add_argument("--affine-scales", type=parse_csv_floats, default=parse_csv_floats("10,8,6,4,3,2,1"))
    parser.add_argument("--affine-iterations", type=parse_csv_ints, default=parse_csv_ints("300,280,250,200,150,100,60"))
    parser.add_argument("--affine-lrs", type=parse_csv_floats, default=parse_csv_floats("0.005,0.003,0.002,0.001,0.0007,0.0004,0.0001"))
    parser.add_argument("--affine-loss-type", type=str, default="mi_gradcc", choices=["mi", "cc", "dice", "mi_gradcc"])

    parser.add_argument("--affine-optimizer-type", type=str, default="adam", choices=["adam", "asgd"])
    parser.add_argument("--affine-constrained", dest="affine_constrained", action="store_true")
    parser.add_argument("--no-affine-constrained", dest="affine_constrained", action="store_false")
    parser.set_defaults(affine_constrained=False)
    parser.add_argument("--asgd-a", type=float, default=None)
    parser.add_argument("--asgd-alpha", type=float, default=1.0)
    parser.add_argument("--affine-scale-mi-bins", type=str, default=None,
                        help="Per-scale MI bins as JSON, e.g. '{\"coarse\":10,\"mid\":20,\"fine\":48}'")
    parser.add_argument("--affine-scale-loss-schedule", type=str, default=None,
                        help="Per-scale loss schedule as JSON, e.g. '{\"coarse\":\"mi\",\"mid\":\"mi_gradcc\",\"fine\":\"cc\"}'")
    parser.add_argument("--affine-max-step-lengths", type=parse_csv_floats, default=None,
                        help="Per-pyramid-level max parameter step (comma-separated floats)")
    parser.add_argument("--rigid-lrs", type=parse_csv_floats,
                        default=parse_csv_floats("1.0,0.5,0.3,0.1,0.05,0.02,0.005"),
                        help="Per-scale learning rates for the rigid stage (comma-separated floats)")

    parser.add_argument("--bspline-num-resolutions", type=int, default=10)
    parser.add_argument("--bspline-final-grid-spacing", type=int, default=100)
    parser.add_argument("--bspline-max-iterations", type=int, default=200)
    parser.add_argument("--bspline-num-samples", type=int, default=50000)
    parser.add_argument("--bspline-num-histogram-bins", type=int, default=32)
    parser.add_argument("--bspline-bending-energy-weight", type=float, default=0.0)
    parser.add_argument("--bspline-tolerance", type=float, default=5e-4)
    parser.add_argument("--bspline-max-tolerance-iters", type=int, default=80)
    parser.add_argument(
        "--bspline-required-valid-ratio",
        type=float,
        default=0.0,
        help="Minimum in-bounds ratio required each bspline iter (0 disables).",
    )
    parser.add_argument(
        "--bspline-invalid-sample-strategy",
        type=str,
        default="none",
        choices=["none", "lr_decay", "early_stop"],
        help="How bspline handles low valid ratio.",
    )
    parser.add_argument("--bspline-invalid-lr-decay", type=float, default=0.5)
    parser.add_argument("--bspline-oob-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--bspline-oob-penalty-adaptive",
        dest="bspline_oob_penalty_adaptive",
        action="store_true",
    )
    parser.add_argument(
        "--no-bspline-oob-penalty-adaptive",
        dest="bspline_oob_penalty_adaptive",
        action="store_false",
        help="Disable adaptive OOB penalty scaling in bspline stage.",
    )
    parser.set_defaults(bspline_oob_penalty_adaptive=True)

    parser.add_argument("--stage-rigid-iters", type=parse_csv_ints, default=parse_csv_ints("150,140,125,100,75,50,30"))
    parser.add_argument("--stage-affine-iters", type=parse_csv_ints, default=parse_csv_ints("300,280,250,200,150,100,60"))
    parser.add_argument("--diff-scales", type=parse_csv_ints, default=parse_csv_ints("8,6,4,3,2,1"))
    parser.add_argument("--diff-iterations", type=parse_csv_ints, default=parse_csv_ints("250,200,180,150,100,60"))
    parser.add_argument("--diff-loss-type", type=str, default="mi_gradcc", choices=["mi", "cc", "mi_gradcc"])
    parser.add_argument("--diff-optimizer-lr", type=float, default=0.5)
    parser.add_argument("--diff-smooth-warp-sigma", type=float, default=0.4)
    parser.add_argument("--diff-smooth-grad-sigma", type=float, default=1.0)
    parser.add_argument("--diff-tolerance", type=float, default=5e-4)
    parser.add_argument("--diff-max-tolerance-iters", type=int, default=80)
    parser.add_argument("--diff-mi-num-samples", type=int, default=25000)
    parser.add_argument("--diff-gradcc-sigma", type=float, default=1.0)
    parser.add_argument("--diff-mi-weight", type=float, default=1.0)
    parser.add_argument("--diff-gradcc-weight", type=float, default=0.25)
    parser.add_argument("--diff-regularization-weight", type=float, default=0.05)

    parser.add_argument("--mi-weight", type=float, default=1.0)
    parser.add_argument("--gradcc-weight", type=float, default=0.25)
    parser.add_argument("--gradcc-sigma", type=float, default=1.0)

    parser.add_argument("--mask-weighted-loss", dest="affine_mask_weighted_loss", action="store_true")
    parser.add_argument("--no-mask-weighted-loss", dest="affine_mask_weighted_loss", action="store_false")
    parser.set_defaults(affine_mask_weighted_loss=True)

    parser.add_argument("--diff-mask-weighted-loss", dest="diff_mask_weighted_loss", action="store_true")
    parser.add_argument("--no-diff-mask-weighted-loss", dest="diff_mask_weighted_loss", action="store_false")
    parser.set_defaults(diff_mask_weighted_loss=False)

    parser.add_argument(
        "--affine-center-mode",
        type=str,
        default="image",
        choices=["none", "image", "tissue"],
        help="Center of rotation mode for HiMReg rigid/affine stages (elastix-like).",
    )
    parser.add_argument(
        "--affine-required-valid-ratio",
        type=float,
        default=0.05,
        help="Minimum fraction of fixed tissue pixels that must map in-bounds each iter (0 disables).",
    )
    parser.add_argument(
        "--affine-invalid-sample-strategy",
        type=str,
        default="lr_decay",
        choices=["none", "lr_decay", "early_stop"],
        help="What to do when valid ratio falls below the threshold.",
    )
    parser.add_argument("--affine-invalid-lr-decay", type=float, default=0.5)
    parser.add_argument("--affine-oob-penalty-weight", type=float, default=0.0)
    parser.add_argument(
        "--affine-oob-penalty-adaptive",
        dest="affine_oob_penalty_adaptive",
        action="store_true",
    )
    parser.add_argument(
        "--no-affine-oob-penalty-adaptive",
        dest="affine_oob_penalty_adaptive",
        action="store_false",
        help="Disable adaptive scaling of OOB penalty when overlap is poor.",
    )
    parser.set_defaults(affine_oob_penalty_adaptive=True)

    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--himreg-mi-num-samples",
        type=int,
        default=15000,
        help="Number of stratified spatial samples for MI each iteration (0 disables sampling).",
    )
    parser.add_argument(
        "--himreg-optimizer-scales-from-physical-shift",
        dest="himreg_optimizer_scales_from_physical_shift",
        action="store_true",
        help="Apply ITK-style parameter scaling to approximate elastix AutomaticScalesEstimation.",
    )
    parser.add_argument(
        "--no-himreg-optimizer-scales-from-physical-shift",
        dest="himreg_optimizer_scales_from_physical_shift",
        action="store_false",
        help="Disable ITK-style parameter scaling.",
    )
    parser.set_defaults(himreg_optimizer_scales_from_physical_shift=True)

    parser.add_argument("--baseline-mode", type=str, default="sitk", choices=["elastix", "sitk"])
    parser.add_argument("--elastix-bin", type=str, default=None)
    parser.add_argument("--transformix-bin", type=str, default=None)
    parser.add_argument(
        "--require-elastix",
        action="store_true",
        help="Fail if elastix CLI is unavailable or fails. (Deprecated: baseline-mode=elastix is strict by default.)",
    )
    parser.add_argument(
        "--allow-sitk-fallback",
        action="store_true",
        help="When baseline-mode=elastix, allow fallback to SimpleITK if elastix is unavailable or fails.",
    )
    parser.add_argument("--baseline-iterations", type=int, default=300)
    parser.add_argument("--baseline-hist-bins", type=int, default=32)
    parser.add_argument("--baseline-sampling", type=float, default=0.2)
    parser.add_argument("--baseline-shrink-factors", type=parse_csv_ints, default=parse_csv_ints("8,4,2,1"))
    parser.add_argument("--baseline-smoothing-sigmas", type=parse_csv_floats, default=parse_csv_floats("3,2,1,0"))

    parser.add_argument("--grid-step", type=int, default=64)

    parser.add_argument("--moving-scale-sync", dest="moving_scale_sync", action="store_true")
    parser.add_argument("--no-moving-scale-sync", dest="moving_scale_sync", action="store_false")
    parser.set_defaults(moving_scale_sync=True)
    parser.add_argument("--scale-sync-mode", type=str, default="isotropic_fit", choices=["none", "isotropic_fit"])

    parser.add_argument("--init-mode", type=str, default="phasecorr_sweep", choices=["centroid", "phasecorr_sweep"])
    parser.add_argument("--init-max-angle-deg", type=float, default=30.0)
    parser.add_argument("--init-coarse-step-deg", type=float, default=5.0)
    parser.add_argument("--init-fine-step-deg", type=float, default=1.0)
    parser.add_argument("--init-refine-half-range-deg", type=float, default=6.0)
    parser.add_argument("--init-use-fixed-mask", dest="init_use_fixed_mask", action="store_true")
    parser.add_argument("--no-init-use-fixed-mask", dest="init_use_fixed_mask", action="store_false")
    parser.set_defaults(init_use_fixed_mask=True)

    parser.add_argument(
        "--wsireg-aligned-preset",
        dest="wsireg_aligned_preset",
        action="store_true",
        help="Apply wsireg-like registration preset (MI+ASGD affine, unconstrained affine, no phasecorr init).",
    )
    parser.add_argument(
        "--no-wsireg-aligned-preset",
        dest="wsireg_aligned_preset",
        action="store_false",
    )
    parser.set_defaults(wsireg_aligned_preset=False)

    parser.add_argument("--fixed-preprocessing-json", type=str, default=None)
    parser.add_argument("--moving-preprocessing-json", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true")
    parser.add_argument("--no-deterministic", dest="deterministic", action="store_false")
    parser.set_defaults(deterministic=True)

    parser.add_argument("--make-roi-montage", dest="make_roi_montage", action="store_true")
    parser.add_argument("--no-make-roi-montage", dest="make_roi_montage", action="store_false")
    parser.set_defaults(make_roi_montage=True)
    parser.add_argument("--roi-montage-name", type=str, default="roi2to6_panel_7col.png")
    parser.add_argument("--roi-montage-dpi", type=int, default=220)

    parser.add_argument("--make-nature-figure", dest="make_nature_figure", action="store_true",
                        help="Generate a Nature-quality comparison figure (PNG + PDF).")
    parser.add_argument("--no-make-nature-figure", dest="make_nature_figure", action="store_false")
    parser.set_defaults(make_nature_figure=False)

    parser.add_argument(
        "--make-elastix-wsireg-figure",
        dest="make_elastix_wsireg_figure",
        action="store_true",
        help="Generate a focused figure comparing Before vs Elastix vs WSiReg (high-contrast overlay, Arial).",
    )
    parser.add_argument(
        "--no-make-elastix-wsireg-figure",
        dest="make_elastix_wsireg_figure",
        action="store_false",
    )
    parser.set_defaults(make_elastix_wsireg_figure=True)

    parser.add_argument("--run-wsireg", dest="run_wsireg", action="store_true",
                        help="Also run wsireg (elastix-backed) as a third comparison method.")
    parser.add_argument("--no-run-wsireg", dest="run_wsireg", action="store_false")
    parser.set_defaults(run_wsireg=False)
    parser.add_argument(
        "--wsireg-reg-params",
        type=str,
        default="rigid,affine,nl",
        help="Comma-separated wsireg registration pipeline (e.g. 'rigid,affine,nl'). "
             "Available: rigid, affine, similarity, nl, nl-reduced, nl-mid, nl2, nl3.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Parse wsireg reg params from comma-separated string to list
    if hasattr(args, "wsireg_reg_params") and isinstance(args.wsireg_reg_params, str):
        args.wsireg_reg_params = [p.strip() for p in args.wsireg_reg_params.split(",") if p.strip()]
    apply_wsireg_aligned_preset(args)
    validate_lengths(args)
    set_global_seed(args.seed, args.deterministic)

    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fixed_pre, moving_pre = get_default_preprocessing(args)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "fixed_preprocessing": fixed_pre,
                "moving_preprocessing": moving_pre,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cases = build_cases(args)

    print(f"Running benchmark on {len(cases)} case(s) using device={device}.")
    print(f"Fixed preprocessing: {fixed_pre}")
    print(f"Moving preprocessing: {moving_pre}")

    mi_metric = MutualInformation().to(device)
    lncc_metric = LNCC(spatial_dims=2).to(device)

    all_rows: List[Dict] = []
    case_artifacts: List[CaseArtifacts] = []
    failed_cases: List[Tuple[str, str]] = []

    for idx, case in enumerate(cases, start=1):
        print(f"\n[{idx}/{len(cases)}] Case: {case.case_id}")
        try:
            case_seed = build_stable_case_seed(int(args.seed), case.case_id)
            case_rows, artifacts = run_case(
                case=case,
                args=args,
                device=device,
                fixed_pre=fixed_pre,
                moving_pre=moving_pre,
                mi_metric=mi_metric,
                lncc_metric=lncc_metric,
                case_seed=case_seed,
            )
            all_rows.extend(case_rows)
            case_artifacts.append(artifacts)
            print(f"Completed {case.case_id}. Saved outputs under {(Path(args.output) / case.case_id)}")
        except Exception as exc:
            msg = str(exc)
            failed_cases.append((case.case_id, msg))
            print(f"[ERROR] Case {case.case_id} failed: {msg}")

    if all_rows:
        save_csv(all_rows, output_dir / "metrics_all_cases.csv")

        summary_rows = summarize_metrics(all_rows)
        save_csv(summary_rows, output_dir / "metrics_summary.csv")
        plot_summary(all_rows, output_dir)
        if args.make_roi_montage and case_artifacts:
            plot_roi_montage(
                artifacts=case_artifacts,
                output_path=output_dir / str(args.roi_montage_name),
                dpi=int(args.roi_montage_dpi),
            )
        if args.make_nature_figure and case_artifacts:
            plot_nature_figure(
                artifacts=case_artifacts,
                all_rows=all_rows,
                output_dir=output_dir,
            )
        if getattr(args, "make_elastix_wsireg_figure", False) and case_artifacts:
            plot_elastix_vs_wsireg_figure(
                artifacts=case_artifacts,
                all_rows=all_rows,
                output_dir=output_dir,
            )

    if failed_cases:
        fail_rows = [{"case_id": cid, "error": err} for cid, err in failed_cases]
        save_csv(fail_rows, output_dir / "failed_cases.csv")
        print(f"\nBenchmark finished with {len(failed_cases)} failed case(s). See failed_cases.csv")
    else:
        print("\nBenchmark finished without case-level failures.")

    print(f"Results written to: {output_dir}")


if __name__ == "__main__":
    main()
