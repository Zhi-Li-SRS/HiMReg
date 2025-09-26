"""Apply saved deformation coordinates to one or more 2D images."""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warp image(s) using pre-computed deformation coordinates (coords.npy)."
    )
    parser.add_argument(
        "--images",
        "-i",
        nargs="+",
        default=["data/maldi/redox_rescale.tif"],
        help="Path(s) to image files to be warped. Channels can be separate files or within one file.",
    )
    parser.add_argument(
        "--coords",
        "-c",
        default="pred/maldi/redox_rescale_coords.npy",
        help="Path to coords.npy containing the deformation grid (shape [H,W,2] or [1,H,W,2]).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=(
            "Output file path. Extension controls format (e.g. .tif, .png, .npy). "
            "Defaults to '<first_image>_warped.tif'."
        ),
    )
    parser.add_argument(
        "--channel-layout",
        choices=["auto", "first", "last"],
        default="auto",
        help="Interpretation of channels for each input image (auto-detect, channel-first, or channel-last).",
    )
    parser.add_argument(
        "--mode",
        choices=["bilinear", "nearest"],
        default="bilinear",
        help="Sampling mode passed to torch.nn.functional.grid_sample.",
    )
    parser.add_argument(
        "--padding-mode",
        choices=["zeros", "border", "reflection"],
        default="zeros",
        help=("Padding mode when采样落在图像外边界时的策略，默认保持与训练时一致的'zeros'"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for computation (e.g. 'cpu', 'cuda', 'cuda:0').",
    )
    parser.add_argument(
        "--no-align-corners",
        action="store_true",
        help="If set, use align_corners=False when sampling (default is True).",
    )
    return parser.parse_args()


def _read_image(path: Path) -> np.ndarray:
    """Read an image from a file."""
    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        return tifffile.imread(path)
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        with np.load(path) as data:
            if "arr_0" not in data:
                raise ValueError(f"NPZ file '{path}' does not contain key 'arr_0'.")
            return data["arr_0"]
    return io.imread(path)


def _to_channel_first(array: np.ndarray, layout: str) -> np.ndarray:
    if array.ndim == 2:
        return array[np.newaxis, ...]
    if array.ndim == 3:
        if layout == "first":
            return array
        if layout == "last":
            return np.moveaxis(array, -1, 0)
        if array.shape[0] <= 4 and array.shape[0] <= array.shape[-1]:
            return array
        return np.moveaxis(array, -1, 0)
    raise ValueError("Only 2D images are supported. Received array with shape " f"{array.shape}.")


def load_images(image_paths: Iterable[str], layout: str, device: torch.device):
    tensors = []
    height = width = None

    for path_str in image_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Image file '{path}' not found.")
        array = _read_image(path)
        array = np.asarray(array)
        assert array.ndim in {
            2,
            3,
        }, f"Unsupported image rank for '{path}'. Expected 2D or 3D array, got shape {array.shape}."
        channel_first = _to_channel_first(array, layout)
        if height is None:
            height, width = channel_first.shape[-2:]
        elif channel_first.shape[-2:] != (height, width):
            raise ValueError(
                "All input images must share the same spatial dimensions. "
                f"First image size was {(height, width)}, but '{path}' has {channel_first.shape[-2:]}."
            )
        tensor = torch.from_numpy(channel_first).float().to(device)
        tensors.append(tensor)

    stacked = torch.cat(tensors, dim=0)
    return stacked.unsqueeze(0)  # add batch dimension


def load_coords(coords_path: str, device: torch.device):
    """Load coordinates from a file."""
    path = Path(coords_path)
    if not path.exists():
        raise FileNotFoundError(f"Coords file '{path}' not found.")
    coords = np.load(path)
    coords = np.asarray(coords, dtype=np.float32)

    if coords.ndim == 3 and coords.shape[-1] == 2:
        coords = coords[np.newaxis, ...]
    if coords.ndim != 4 or coords.shape[-1] != 2:
        raise ValueError(
            "Expected coords array with shape [H,W,2] or [B,H,W,2]. "
            f"Received array with shape {coords.shape}."
        )

    return torch.from_numpy(coords).to(device)


def _resize_coords(coords: torch.Tensor, spatial_size: torch.Size, align_corners: bool):
    """Resize coordinates to match the spatial size of the image."""
    if coords.shape[1:3] == tuple(spatial_size):
        return coords

    coords_permuted = coords.permute(0, 3, 1, 2)
    resized = F.interpolate(coords_permuted, size=spatial_size, mode="bilinear", align_corners=align_corners)
    return resized.permute(0, 2, 3, 1)


def apply_transform(
    image: torch.Tensor, coords: torch.Tensor, mode: str, padding_mode: str, align_corners: bool
):
    """Apply the transformation to the image or images."""
    if coords.size(0) == 1 and image.size(0) > 1:
        coords = coords.expand(image.size(0), -1, -1, -1)
    elif coords.size(0) != image.size(0):
        raise ValueError(
            "Batch size mismatch between image and coords. "
            f"Image batch: {image.size(0)}, Coords batch: {coords.size(0)}."
        )

    # Resize coordinates to match the spatial size of the image if needed
    coords = _resize_coords(coords, image.shape[2:], align_corners)
    warped = F.grid_sample(image, coords, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    return warped


def save_tensor(tensor: torch.Tensor, output_path: str):
    """Save a tensor to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array = tensor.squeeze(0).detach().cpu().numpy()
    if array.ndim == 3:
        # grid_sample keeps channel-first; move to channel-last for common image formats
        array_to_save = np.moveaxis(array, 0, -1)
    elif array.ndim == 2:
        array_to_save = array
    else:
        raise ValueError(f"Unexpected tensor shape after squeezing: {array.shape}.")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, array)
        return
    if suffix == ".npz":
        np.savez_compressed(path, array)
        return
    if suffix in {".tif", ".tiff"}:
        tifffile.imwrite(path, array_to_save.astype(np.float32))
        return

    if array_to_save.ndim == 3 and array_to_save.shape[-1] > 4:
        raise ValueError(
            "Image formats such as PNG/JPEG support at most 4 channels. "
            "Please use a TIFF or NPY/NPZ output for higher channel counts."
        )
    io.imsave(path, array_to_save.astype(np.float32))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    image = load_images(args.images, args.channel_layout, device)
    coords = load_coords(args.coords, device)

    align_corners = not args.no_align_corners

    warped = apply_transform(image, coords, args.mode, args.padding_mode, align_corners)

    if args.output is None:
        base = Path(args.images[0])
        default_name = f"{base.stem}_warped.tif"
        output_path = base.with_name(default_name)
    else:
        output_path = args.output

    save_tensor(warped, output_path)
    print(f"Saved warped image to {output_path}")


if __name__ == "__main__":
    main()
