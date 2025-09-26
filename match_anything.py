import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np
import torch
import tifffile
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
from src.utils import save_registration_overlay

MODEL_ID = "zju-community/matchanything_eloftr"


def _normalize_to_uint8(array: np.ndarray):
    """Normalize an array to uint8."""
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


def _array_to_pil(image_array: np.ndarray):
    """Convert an array to a PIL image."""
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
            return _array_to_pil(array)

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


def _load_image(path: Path):
    """Load an image from a path and return both PIL RGB proxy and original array."""
    lower = path.name.lower()
    if lower.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff")):
        array = tifffile.imread(path)
    else:
        with Image.open(path) as pil_img:
            array = np.asarray(pil_img)
    pil_image = _array_to_pil(array)
    return pil_image, np.asarray(array)


def _prepare_array_for_warp(arr: np.ndarray):
    """Rearrange array so spatial axes are last two dims (H,W) with optional channel dim."""
    if arr.ndim < 2:
        raise ValueError("Input array must have at least two dimensions (H, W)")
    metadata = {"orig_shape": arr.shape, "dtype": arr.dtype}
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
    """Restore warped array back to original shape using stored metadata."""
    dtype = metadata["dtype"]
    orig_shape = metadata["orig_shape"]
    if "leading_shape" not in metadata:
        return warped.astype(dtype, copy=False).reshape(orig_shape)
    leading_shape = metadata["leading_shape"]
    H, W = orig_shape[-2:]
    restored = np.moveaxis(warped, -1, 0).reshape(leading_shape + (H, W))
    return restored.astype(dtype, copy=False)


def _warp_array_homography(
    array: np.ndarray,
    H: np.ndarray,
    fixed_width: int,
    fixed_height: int,
    interpolation: int = cv2.INTER_LINEAR,
):
    """Warp an array (H,W[,C]) with a homography, preserving dtype."""
    return cv2.warpPerspective(
        array, H, (fixed_width, fixed_height), flags=interpolation, borderMode=cv2.BORDER_CONSTANT
    )


def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a numpy array."""
    return tensor.detach().cpu().numpy().astype(np.float32)


def _compute_matches(
    processor: AutoImageProcessor,
    model: AutoModelForKeypointMatching,
    fixed_img: Image.Image,
    moving_img: Image.Image,
    device: torch.device,
    match_threshold: float,
):
    """Compute matches between two images."""

    images = [fixed_img, moving_img]
    inputs = processor(images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = [tuple((img.height, img.width) for img in images)]
    processed = processor.post_process_keypoint_matching(
        outputs, target_sizes=target_sizes, threshold=match_threshold
    )[0]
    keypoints_fixed = _tensor_to_numpy(processed["keypoints0"])
    keypoints_moving = _tensor_to_numpy(processed["keypoints1"])
    scores = _tensor_to_numpy(processed["matching_scores"])
    return keypoints_fixed, keypoints_moving, scores


def _estimate_homography(
    keypoints_fixed: np.ndarray,
    keypoints_moving: np.ndarray,
    ransac_threshold: float,
    ransac_confidence: float,
):
    """Estimate homography between two images."""

    if keypoints_fixed.shape[0] < 4:
        raise RuntimeError("Less than 4 matches, cannot estimate homography")
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask = cv2.findHomography(
        keypoints_moving.reshape(-1, 1, 2),
        keypoints_fixed.reshape(-1, 1, 2),
        method,
        ransac_threshold,
        confidence=ransac_confidence,
    )
    if H is None or mask is None:
        raise RuntimeError("RANSAC did not return a valid homography")
    inlier_mask = mask.ravel().astype(bool)
    if inlier_mask.sum() < 4:
        raise RuntimeError("RANSAC inliers insufficient, registration failed")
    return H.astype(np.float32), inlier_mask


def _warp_rgb_preview(moving_img: Image.Image, H: np.ndarray, fixed_width: int, fixed_height: int):
    """Warp an RGB proxy image for visualization outputs."""

    moving_np = np.asarray(moving_img, dtype=np.uint8)
    return cv2.warpPerspective(
        moving_np,
        H,
        (fixed_width, fixed_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _prepare_grayscale(image: Image.Image, target_shape: Tuple[int, int]):
    """Prepare a grayscale image."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = target_shape
    if gray.shape != (h, w):
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_LINEAR)
    return gray


def run_registration(
    fixed_path: Path,
    moving_path: Path,
    output_dir: Path,
    device_str: str,
    match_threshold: float,
    ransac_threshold: float,
    ransac_confidence: float,
):
    """Run registration."""

    device = torch.device(device_str)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=False)
    model = AutoModelForKeypointMatching.from_pretrained(MODEL_ID).to(device).eval()

    fixed_img, _ = _load_image(fixed_path)
    moving_img, moving_raw = _load_image(moving_path)
    moving_prepared, moving_meta = _prepare_array_for_warp(moving_raw)

    keypoints_fixed, keypoints_moving, scores = _compute_matches(
        processor, model, fixed_img, moving_img, device, match_threshold
    )

    H, inlier_mask = _estimate_homography(
        keypoints_fixed, keypoints_moving, ransac_threshold, ransac_confidence
    )

    fixed_width, fixed_height = fixed_img.width, fixed_img.height
    warped_rgb = _warp_rgb_preview(moving_img, H, fixed_width, fixed_height)
    warped_prepared = _warp_array_homography(
        moving_prepared, H, fixed_width, fixed_height, interpolation=cv2.INTER_LINEAR
    )
    warped_raw = _restore_array_from_warp(warped_prepared, moving_meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    warped_path = output_dir / "moving_transformed.tiff"
    tifffile.imwrite(warped_path, warped_raw)

    keypoints_fixed_inliers = keypoints_fixed[inlier_mask]
    keypoints_moving_inliers = keypoints_moving[inlier_mask]
    scores_inliers = scores[inlier_mask]

    transform_npz_path = output_dir / "transform_data.npz"
    np.savez(
        transform_npz_path,
        homography=H,
        keypoints_fixed=keypoints_fixed,
        keypoints_moving=keypoints_moving,
        inliers_fixed=keypoints_fixed_inliers,
        inliers_moving=keypoints_moving_inliers,
        matching_scores=scores,
        inlier_mask=inlier_mask.astype(np.uint8),
    )

    summary = {
        "fixed_path": str(fixed_path),
        "moving_path": str(moving_path),
        "model_id": MODEL_ID,
        "device": device_str,
        "match_threshold": match_threshold,
        "ransac_threshold": ransac_threshold,
        "ransac_confidence": ransac_confidence,
        "num_matches": int(keypoints_fixed.shape[0]),
        "num_inliers": int(inlier_mask.sum()),
        "mean_inlier_score": float(scores_inliers.mean()) if scores_inliers.size else 0.0,
        "output_files": {"warped_tiff": str(warped_path), "transform_npz": str(transform_npz_path)},
    }

    transform_json_path = output_dir / "transform_summary.json"
    with open(transform_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    fixed_gray = _prepare_grayscale(fixed_img, (fixed_height, fixed_width))
    moving_gray = _prepare_grayscale(moving_img, (fixed_height, fixed_width))
    warped_gray = cv2.cvtColor(warped_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    overlay_path = output_dir / "overlay_before_after.png"
    save_registration_overlay(
        fixed_gray,
        moving_gray,
        warped_gray,
        output_path=str(overlay_path),
        titles=("Before", "After"),
        alpha=0.7,
    )

    summary["output_files"]["overlay"] = str(overlay_path)
    summary["homography"] = H.tolist()

    return {
        "summary": summary,
        "warped_path": warped_path,
        "transform_npz": transform_npz_path,
        "transform_json": transform_json_path,
        "overlay_path": overlay_path,
        "homography": H,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use MatchAnything-ELOFTR for image registration")
    parser.add_argument(
        "--fixed",
        type=Path,
        default="data/maldi/he_rescale.png",
        help="Fixed image path (supports PNG/JPEG/TIFF/OME-TIFF)",
    )
    parser.add_argument(
        "--moving", type=Path, default="data/maldi/redox_rescale.png", help="Moving image path"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="matchanything_outputs",
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device, optional cpu/cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--match-threshold", type=float, default=0.2, help="Matching confidence threshold (default: 0.2)"
    )
    parser.add_argument(
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="RANSAC re-projection threshold (pixels, default: 3.0)",
    )
    parser.add_argument(
        "--ransac-confidence", type=float, default=0.999, help="RANSAC confidence (default: 0.999)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available, please use --device cpu")
    results = run_registration(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_dir=args.output_dir,
        device_str=args.device,
        match_threshold=args.match_threshold,
        ransac_threshold=args.ransac_threshold,
        ransac_confidence=args.ransac_confidence,
    )
    summary = results["summary"]
    print("Registration completed:")
    print(f"  Number of matches: {summary['num_matches']}, RANSAC inliers: {summary['num_inliers']}")
    print(f"  Number of matches: {summary['num_matches']}, RANSAC inliers: {summary['num_inliers']}")
    print(f"  Homography: {results['homography']}")
    print("  Result files:")
    print(f"    Transformed image: {summary['output_files']['warped_tiff']}")
    print(f"    Transform parameters (npz): {summary['output_files']['transform_npz']}")
    print(f"    Overlay image: {summary['output_files']['overlay']}")
    print(f"    Summary (json): {results['transform_json']}")


if __name__ == "__main__":
    main()
