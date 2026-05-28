import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import tifffile
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from src.utils import save_registration_overlay


MODEL_ID = "zju-community/matchanything_eloftr"

_MODEL_CACHE: Dict[Tuple[str, str], Tuple[AutoImageProcessor, AutoModelForKeypointMatching]] = {}


def get_matchanything_model(
    device: torch.device, model_id: str = MODEL_ID
) -> Tuple[AutoImageProcessor, AutoModelForKeypointMatching]:
    """Load (and cache) the MatchAnything-ELoFTR processor + model on `device`."""
    key = (model_id, str(device))
    if key not in _MODEL_CACHE:
        processor = AutoImageProcessor.from_pretrained(model_id, use_fast=False)
        model = AutoModelForKeypointMatching.from_pretrained(model_id).to(device).eval()
        _MODEL_CACHE[key] = (processor, model)
    return _MODEL_CACHE[key]


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    arr = array.astype(np.float32, copy=False)
    min_val = float(arr.min())
    max_val = float(arr.max())
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)


def array_to_pil(image_array: np.ndarray) -> Image.Image:
    array = np.asarray(image_array)
    if array.ndim == 0:
        raise ValueError("The image is empty")
    while array.ndim > 3:
        array = array[0]
    if array.ndim == 3:
        if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
            array = np.moveaxis(array, 0, -1)
        elif array.shape[-1] not in {1, 3, 4}:
            array = array[0]
            return array_to_pil(array)
    array = _normalize_to_uint8(array)
    if array.ndim == 2:
        return Image.fromarray(array, mode="L").convert("RGB")
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return Image.fromarray(array[..., 0], mode="L").convert("RGB")
        if channels >= 3:
            return Image.fromarray(array[..., :3], mode="RGB")
    raise ValueError(f"Unparsable image shape: {array.shape}")


_array_to_pil = array_to_pil  # backward-compat alias


def _load_image(path: Path) -> Tuple[Image.Image, np.ndarray]:
    lower = path.name.lower()
    if lower.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        array = tifffile.imread(path)
    else:
        with Image.open(path) as pil_img:
            array = np.asarray(pil_img)
    pil_image = _array_to_pil(array)
    return pil_image, np.asarray(array)


def _prepare_array_for_warp(arr: np.ndarray):
    if arr.ndim < 2:
        raise ValueError("Input array must have at least two dimensions (H, W)")
    metadata: Dict[str, Any] = {"orig_shape": arr.shape, "dtype": arr.dtype}
    H, W = arr.shape[-2:]
    leading_shape = arr.shape[:-2]
    if not leading_shape:
        prepared = arr
    else:
        channel_dim = int(np.prod(leading_shape))
        metadata["leading_shape"] = leading_shape
        prepared = np.moveaxis(arr.reshape(channel_dim, H, W), 0, -1)
    prepared = np.ascontiguousarray(prepared)
    return prepared, metadata


def _restore_array_from_warp(warped: np.ndarray, metadata: Dict[str, Any]):
    dtype = metadata["dtype"]
    orig_shape = metadata["orig_shape"]
    if "leading_shape" not in metadata:
        return warped.astype(dtype, copy=False).reshape(orig_shape)
    leading_shape = metadata["leading_shape"]
    H, W = orig_shape[-2:]
    restored = np.moveaxis(warped, -1, 0).reshape(leading_shape + (H, W))
    return restored.astype(dtype, copy=False)


def _warp_array_affine(array: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.warpAffine(
        array, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def _compute_matches_from_outputs(
    processor: AutoImageProcessor,
    model: AutoModelForKeypointMatching,
    fixed_img: Image.Image,
    moving_img: Image.Image,
    device: torch.device,
    match_threshold: float,
):
    images = [fixed_img, moving_img]
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    size_tensor = torch.tensor(
        [[(img.height, img.width) for img in images]],
        device=outputs.keypoints.device,
        dtype=outputs.keypoints.dtype,
    )
    keypoints = outputs.keypoints.clone()
    keypoints = keypoints * size_tensor.flip(-1).reshape(-1, 2, 1, 2)

    matches = outputs.matches
    scores = outputs.matching_scores

    keypoints_pair = keypoints[0]
    matches_pair = matches[0]
    scores_pair = scores[0]

    valid_matches = torch.logical_and(scores_pair > match_threshold, matches_pair > -1)
    valid0 = valid_matches[0]
    valid1 = valid_matches[1]

    if not torch.any(valid0) or not torch.any(valid1):
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    matched_keypoints0 = keypoints_pair[0][valid0]
    matched_keypoints1 = keypoints_pair[1][valid1]
    matched_scores = scores_pair[0][valid0]

    k0 = matched_keypoints0.detach().cpu().numpy().astype(np.float32)
    k1 = matched_keypoints1.detach().cpu().numpy().astype(np.float32)
    s = matched_scores.detach().cpu().numpy().astype(np.float32)
    return k0, k1, s


def select_spatially_diverse(
    keypoints_fixed: np.ndarray,
    keypoints_moving: np.ndarray,
    scores: np.ndarray,
    img_hw: Tuple[int, int],
    target_k: int = 6000,
    grid: int = 8,
    per_cell_cap: int = 192,
):
    H, W = img_hw
    order = np.argsort(-scores)
    xf, xm, sc = keypoints_fixed[order], keypoints_moving[order], scores[order]

    cell_h = max(1, H // grid)
    cell_w = max(1, W // grid)
    taken = np.zeros((grid, grid), dtype=np.int32)

    sel_f, sel_m, sel_s = [], [], []
    for p, q, s in zip(xf, xm, sc):
        r = min(grid - 1, int(p[1] // cell_h))
        c = min(grid - 1, int(p[0] // cell_w))
        if taken[r, c] < per_cell_cap:
            sel_f.append(p)
            sel_m.append(q)
            sel_s.append(s)
            taken[r, c] += 1
            if len(sel_f) >= target_k:
                break

    if len(sel_f) < 4:
        k = min(target_k, len(scores))
        idx = np.argsort(-scores)[:k]
        return keypoints_fixed[idx], keypoints_moving[idx], scores[idx]

    return np.stack(sel_f), np.stack(sel_m), np.array(sel_s)


def _estimate_affine(
    keypoints_fixed: np.ndarray,
    keypoints_moving: np.ndarray,
    ransac_threshold: float,
    ransac_confidence: float,
):
    """Full 6-DoF RANSAC affine, matching the MatchAnything official `ransac_affine` default."""
    if keypoints_fixed.shape[0] < 3:
        raise RuntimeError("Affine estimation needs at least 3 correspondences")
    matrix, inliers = cv2.estimateAffine2D(
        keypoints_moving.astype(np.float32),
        keypoints_fixed.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        confidence=ransac_confidence,
        maxIters=5000,
    )
    if matrix is None or inliers is None:
        raise RuntimeError("cv2.estimateAffine2D failed to find a valid transform")
    if inliers.sum() < 3:
        raise RuntimeError("Too few inliers for affine transform")
    return matrix.astype(np.float32), inliers.astype(bool).ravel()


def _warp_rgb_preview(moving_img: Image.Image, matrix: np.ndarray, width: int, height: int):
    moving_np = np.asarray(moving_img, dtype=np.uint8)
    return cv2.warpAffine(
        moving_np,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def register_pair_matchanything(
    fixed_array: np.ndarray,
    moving_array: np.ndarray,
    *,
    device: torch.device,
    model_id: str = MODEL_ID,
    match_threshold: float = 0.1,
    ransac_threshold: float = 3.0,
    ransac_confidence: float = 0.999,
    sample_target_k: int = 6000,
) -> Dict[str, Any]:
    """Register `moving_array` onto `fixed_array` with MatchAnything-ELoFTR + RANSAC affine.

    This mirrors the authors' official `ransac_affine` post-processing default
    (full 6-DoF affine, RANSAC threshold 3.0 px, match threshold 0.1).

    Returns a dict with `warped`, `affine_matrix`, `runtime_s`, `n_matches`, `n_inliers`.
    The `warped` array is on the fixed grid, with leading channels (if any) preserved
    from `moving_array`.
    """
    processor, model = get_matchanything_model(device, model_id)

    fixed_pil = array_to_pil(fixed_array)
    moving_pil = array_to_pil(moving_array)
    H, W = fixed_pil.height, fixed_pil.width

    start = time.time()
    kf, km, scores = _compute_matches_from_outputs(
        processor, model, fixed_pil, moving_pil, device, match_threshold
    )
    if kf.shape[0] < 3:
        raise RuntimeError(f"Too few matches after thresholding: {kf.shape[0]}")

    kf_sel, km_sel, sc_sel = select_spatially_diverse(
        kf, km, scores, img_hw=(H, W), target_k=sample_target_k, grid=8, per_cell_cap=192
    )
    matrix, inlier_mask = _estimate_affine(kf_sel, km_sel, ransac_threshold, ransac_confidence)

    moving_prepared, moving_meta = _prepare_array_for_warp(moving_array)
    warped_prepared = _warp_array_affine(moving_prepared, matrix, W, H)
    warped = _restore_array_from_warp(warped_prepared, moving_meta)
    runtime_s = time.time() - start

    return {
        "warped": warped,
        "affine_matrix": matrix,
        "runtime_s": float(runtime_s),
        "n_matches_all": int(kf.shape[0]),
        "n_matches_used": int(kf_sel.shape[0]),
        "n_inliers": int(inlier_mask.sum()),
        "mean_inlier_score": float(sc_sel[inlier_mask].mean()) if inlier_mask.any() else 0.0,
        "keypoints_fixed_inliers": kf_sel[inlier_mask],
        "keypoints_moving_inliers": km_sel[inlier_mask],
        "keypoints_fixed": kf_sel,
        "keypoints_moving": km_sel,
        "matching_scores": sc_sel,
        "inlier_mask": inlier_mask,
    }


def run_registration_affine(
    fixed_path: Path,
    moving_path: Path,
    output_dir: Path,
    device_str: str,
    match_threshold: float,
    ransac_threshold: float,
    ransac_confidence: float,
    sample_target_k: int,
):
    device = torch.device(device_str)

    fixed_img, fixed_raw = _load_image(fixed_path)
    moving_img, moving_raw = _load_image(moving_path)
    H, W = fixed_img.height, fixed_img.width

    result = register_pair_matchanything(
        fixed_array=fixed_raw,
        moving_array=moving_raw,
        device=device,
        match_threshold=match_threshold,
        ransac_threshold=ransac_threshold,
        ransac_confidence=ransac_confidence,
        sample_target_k=sample_target_k,
    )
    matrix = result["affine_matrix"]
    inlier_mask = result["inlier_mask"]
    kf_sel = result["keypoints_fixed"]
    km_sel = result["keypoints_moving"]
    sc_sel = result["matching_scores"]
    warped_raw = result["warped"]

    warped_rgb = _warp_rgb_preview(moving_img, matrix, W, H)

    output_dir.mkdir(parents=True, exist_ok=True)
    warped_path = output_dir / "moving_transformed_affine.tiff"
    tifffile.imwrite(warped_path, warped_raw)

    keypoints_fixed_inliers = result["keypoints_fixed_inliers"]
    keypoints_moving_inliers = result["keypoints_moving_inliers"]
    scores_inliers = sc_sel[inlier_mask]

    transform_npz_path = output_dir / "transform_affine_data.npz"
    np.savez(
        transform_npz_path,
        affine_matrix=matrix,
        keypoints_fixed=kf_sel,
        keypoints_moving=km_sel,
        inliers_fixed=keypoints_fixed_inliers,
        inliers_moving=keypoints_moving_inliers,
        matching_scores=sc_sel,
        inlier_mask=inlier_mask.astype(np.uint8),
    )

    fixed_gray = np.asarray(fixed_img.convert("L"), dtype=np.float32)
    moving_gray = np.asarray(moving_img.convert("L"), dtype=np.float32)
    warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    overlay_path = output_dir / "overlay_compare_affine.png"
    save_registration_overlay(
        fixed_gray,
        moving_gray,
        warped_gray,
        output_path=str(overlay_path),
        titles=("Before", "After"),
        alpha=0.7,
    )

    summary = {
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "model_id": MODEL_ID,
        "device": device_str,
        "match_threshold": match_threshold,
        "ransac_threshold": ransac_threshold,
        "ransac_confidence": ransac_confidence,
        "num_matches_all": int(result["n_matches_all"]),
        "num_matches_used": int(result["n_matches_used"]),
        "num_inliers": int(result["n_inliers"]),
        "mean_inlier_score": float(result["mean_inlier_score"]),
        "runtime_s": float(result["runtime_s"]),
        "affine_matrix": matrix.tolist(),
        "output_files": {
            "warped_tiff": str(warped_path),
            "transform_npz": str(transform_npz_path),
            "overlay": str(overlay_path),
        },
    }

    json_path = output_dir / "transform_affine_summary.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    return {
        "summary": summary,
        "warped_path": warped_path,
        "transform_npz": transform_npz_path,
        "transform_json": json_path,
        "overlay_path": overlay_path,
        "affine_matrix": matrix,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MatchAnything-ELOFTR affine registration")
    parser.add_argument(
        "--fixed",
        type=Path,
        default="data/maldi/he_rescale.png",
        help="Fixed image path (PNG/JPEG/TIFF/OME-TIFF)",
    )
    parser.add_argument(
        "--moving",
        type=Path,
        default="data/maldi/redox_rescale.png",
        help="Moving image path (PNG/JPEG/TIFF/OME-TIFF)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="matchanything_outputs", help="Directory for outputs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: cpu | cuda (default: auto)",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.1,
        help="ELoFTR matching confidence threshold (MatchAnything paper default: 0.1)",
    )
    parser.add_argument(
        "--ransac-threshold", type=float, default=3.0, help="RANSAC reprojection threshold (pixels)"
    )
    parser.add_argument("--ransac-confidence", type=float, default=0.999, help="RANSAC confidence")
    parser.add_argument(
        "--sample-target-k", type=int, default=6000, help="Target number of spatially diverse matches"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; use --device cpu")

    results = run_registration_affine(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_dir=args.output_dir,
        device_str=args.device,
        match_threshold=args.match_threshold,
        ransac_threshold=args.ransac_threshold,
        ransac_confidence=args.ransac_confidence,
        sample_target_k=args.sample_target_k,
    )

    summary = results["summary"]
    print("Affine registration completed.")
    print(f"  Matches (all / used): {summary['num_matches_all']} / {summary['num_matches_used']}")
    print(f"  Inliers: {summary['num_inliers']} (mean score {summary['mean_inlier_score']:.4f})")
    print("  Result files:")
    print(f"    Warped image: {summary['output_files']['warped_tiff']}")
    print(f"    Transform params (npz): {summary['output_files']['transform_npz']}")
    print(f"    Overlay: {summary['output_files']['overlay']}")
    print(f"    Summary (json): {results['transform_json']}")


if __name__ == "__main__":
    main()
