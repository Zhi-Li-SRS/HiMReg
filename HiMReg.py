import os
import json

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F

from src.affinemorph import AffineRegistration
from src.bspline import BSplineRegistration
from src.data_load import Image
from src.diffeomorph import DiffRegistration
from src.config import load_config
from src.spatial_sync import sync_moving_scale_to_fixed
from src.utils import save_registration_overlay, set_global_seed


class HiMReg:
    """Class for performing affine and diffeomorphic registration on a pair of images.
    Args:
        fixed_images: Image object of fixed image.
        moving_images: Image object of moving image.
        affine_scales (list): List of scales for affine registration.
        affine_iterations (list): List of iterations for affine registration.
        diff_scales (list): List of scales for diffeomorphic registration.
        diff_iterations (list): List of iterations for diffeomorphic registration.
        affine_kwargs (dict): Additional keyword arguments for affine registration. Defaults to {}.
        diff_kwargs (dict): Additional keyword arguments for diffeomorphic registration. Defaults to {}.
    """

    def __init__(
        self,
        fixed_images,
        moving_images,
        affine_scales,
        affine_iterations,
        scale_dependent_lr,
        diff_scales,
        diff_iterations,
        affine_kwargs=None,
        diff_kwargs=None,
        register_type="diff",
        bspline_kwargs=None,
    ):
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device

        affine_kwargs = dict(affine_kwargs or {})
        diff_kwargs = dict(diff_kwargs or {})
        bspline_kwargs = dict(bspline_kwargs or {})

        patience = affine_kwargs.pop("patience", 50)
        min_delta = affine_kwargs.pop("min_delta", 1e-5)
        self.diff_scales = list(diff_scales)
        self.diff_iterations = list(diff_iterations)
        self._diff_kwargs = diff_kwargs
        self._bspline_kwargs = bspline_kwargs

        self.affine_registration = AffineRegistration(
            scales=affine_scales,
            iterations=affine_iterations,
            fixed_images=fixed_images,
            moving_images=moving_images,
            patience=patience,
            min_delta=min_delta,
            scale_dependent_lr=scale_dependent_lr,
            **affine_kwargs,
        )

        self._register_type_hint = register_type

    def _build_nonlinear_registration(self, register_type: str, init_affine: torch.Tensor):
        if register_type == "bspline":
            return BSplineRegistration(
                fixed_images=self.fixed_images,
                moving_images=self.moving_images,
                init_affine=init_affine,
                **self._bspline_kwargs,
            )
        if register_type == "diff":
            return DiffRegistration(
                scales=self.diff_scales,
                iterations=self.diff_iterations,
                fixed_images=self.fixed_images,
                moving_images=self.moving_images,
                init_affine=init_affine,
                **self._diff_kwargs,
            )
        raise ValueError(f"Unsupported nonlinear register_type: {register_type}")

    def register(self, register_type="affine", save_transformed=True):
        """
        Perform registration of the moving image to the fixed image.

        Args:
            register_type (str): Type of registration ("affine", "diff", or "bspline").
                                 "affine" performs only affine registration.
                                 "diff" performs affine followed by diffeomorphic registration.
                                 "bspline" performs affine followed by B-spline registration.

            save_transformed (bool): Whether to save transformed images during optimization.

        Returns:
            tuple: (affine_transformed, nonlinear_transformed, final_coordinates)
                  If register_type='affine': nonlinear_transformed will be None.
        """

        # Affine registration
        original_size = self.fixed_images().shape[2:]  # spatial dimensions
        batch_size = self.fixed_images.size()
        input_shape = [batch_size, 1, *original_size]  # [B, C, H, W]

        affine_transformed = self.affine_registration.optimize(save_transformed=save_transformed)
        affine_matrix = self.affine_registration.get_final_transform()

        if register_type == "affine":
            affine_coords = F.affine_grid(affine_matrix[:, :-1], input_shape, align_corners=True)
            affine_transformed = self._upsample_transformed_images(affine_transformed, original_size)
            return affine_transformed, None, affine_coords

        # Nonlinear registration (diff or bspline), chained from affine init.
        fixed_p2t = self.fixed_images.get_physical_to_pixel()
        moving_t2p = self.moving_images.get_pixel_to_physical()
        affine_init_phys = torch.matmul(moving_t2p, torch.matmul(affine_matrix, fixed_p2t))
        nl_reg = self._build_nonlinear_registration(register_type, init_affine=affine_init_phys.detach())
        nl_transformed = nl_reg.optimize(save_transformed=save_transformed)
        final_coordinates = nl_reg.get_final_coordinates()
        if final_coordinates is not None:
            current_size = final_coordinates.shape[1:-1]
            if current_size != original_size:
                final_coordinates = F.interpolate(
                    final_coordinates.permute(0, 3, 1, 2),
                    size=original_size,
                    mode="bilinear",
                    align_corners=True,
                ).permute(0, 2, 3, 1)

            nl_transformed = self._upsample_transformed_images(nl_transformed, original_size)
        return affine_transformed, nl_transformed, final_coordinates

    def _upsample_transformed_images(self, transformed_images, target_size):
        """Helper method to upsample transformed images to target size."""
        if not transformed_images:
            return transformed_images

        last_transformed = transformed_images[-1]
        upsampled_transformed = F.interpolate(
            last_transformed, size=target_size, mode="bilinear", align_corners=True
        )
        return transformed_images[:-1] + [upsampled_transformed]

    @staticmethod
    def apply_transformation(image: torch.Tensor, coords_path: str, device: str):
        coords = torch.from_numpy(np.load(coords_path)).to(device).float()
        if coords.shape[1:-1] != image.shape[2:]:
            coords = F.interpolate(
                coords.permute(0, 3, 1, 2),  # [B, 2, H, W]
                size=image.shape[2:],
                mode="bilinear",
                align_corners=True,
            ).permute(
                0, 2, 3, 1
            )  # Back to [B, H, W, 2]

        transformed = F.grid_sample(image, coords, mode="bilinear", align_corners=True)
        return transformed


def main(config_path: str = "config.yaml"):
    """Main function for HiMReg image registration."""
    # Load configuration
    config_manager = load_config(config_path)
    affine_cfg = config_manager.affine_config
    diff_cfg = config_manager.diff_config
    prepro_cfg = config_manager.preprocessing_config

    # Ensure reproducibility
    set_global_seed(config_manager.seed, config_manager.deterministic)

    io_cfg = config_manager.config.get("io", {})
    device_str = io_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Load images
    try:
        fixed_image = Image.load_file(
            config_manager.fixed_image_path,
            device=device,
            preprocessing=prepro_cfg.get("fixed", {}),
        )
        moving_image_raw = Image.load_file(
            config_manager.moving_image_path,
            device=device,
            preprocessing=prepro_cfg.get("moving", {}),
        )
        scale_sync_cfg = config_manager.scale_sync_config
        moving_sitk_synced, scale_sync_meta = sync_moving_scale_to_fixed(
            moving_sitk=moving_image_raw.images[0].itk_image,
            fixed_sitk=fixed_image.images[0].itk_image,
            enabled=bool(scale_sync_cfg.get("enabled", True)),
            mode=str(scale_sync_cfg.get("mode", "isotropic_fit")),
        )
        moving_image = Image(moving_sitk_synced, device=device, preprocessing={})
        print(f"Loaded fixed image: {config_manager.fixed_image_path}")
        print(f"Loaded moving image: {config_manager.moving_image_path}")
        print(
            f"Applied scale-sync: enabled={bool(scale_sync_cfg.get('enabled', True))}, "
            f"mode={str(scale_sync_cfg.get('mode', 'isotropic_fit'))}, "
            f"isotropic_scale={scale_sync_meta.get('isotropic_scale', 1.0):.4f}"
        )
    except Exception as exc:
        print(f"Error loading images: {exc}")
        return

    # Create HiMReg instance
    registration = HiMReg(
        fixed_images=fixed_image,
        moving_images=moving_image,
        affine_scales=affine_cfg["scales"],
        affine_iterations=affine_cfg["iterations"],
        scale_dependent_lr=affine_cfg["scale_dependent_lr"],
        diff_scales=diff_cfg["scales"],
        diff_iterations=diff_cfg["iterations"],
        affine_kwargs=config_manager.get_affine_kwargs(),
        diff_kwargs=config_manager.get_diff_kwargs(),
        register_type=config_manager.register_type,
        bspline_kwargs=config_manager.get_bspline_kwargs(),
    )

    # Perform registration
    print(f"Starting {config_manager.register_type} registration...")
    affine_transformed, diff_transformed, final_coordinates = registration.register(
        register_type=config_manager.register_type, save_transformed=True
    )

    # Setup output paths
    moving_dir, moving_filename = os.path.split(config_manager.moving_image_path)
    moving_basename = os.path.splitext(moving_filename)[0]
    relative_path = os.path.relpath(moving_dir, start=os.path.dirname(moving_dir))
    output_dir = os.path.join(config_manager.output_dir, relative_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save results
    reg_type = config_manager.register_type

    if affine_transformed:
        affine_pred_path = os.path.join(output_dir, f"{moving_basename}_affine.tif")
        last_affine_image = affine_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(affine_pred_path, last_affine_image)
        print(f"Saved affine result: {affine_pred_path}")

    if diff_transformed:
        nl_pred_path = os.path.join(output_dir, f"{moving_basename}_{reg_type}.tif")
        last_nl_image = diff_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(nl_pred_path, last_nl_image)
        print(f"Saved {reg_type} result: {nl_pred_path}")

    if final_coordinates is not None:
        coords_pred_path = os.path.join(output_dir, f"{moving_basename}_coords.npy")
        final_coords_np = final_coordinates.detach().cpu().numpy()
        np.save(coords_pred_path, final_coords_np)
        print(f"Saved coordinates: {coords_pred_path}")
        coords_meta_path = os.path.join(output_dir, f"{moving_basename}_coords_meta.json")
        with open(coords_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "moving_image_original": config_manager.moving_image_path,
                    "moving_preprocessing": prepro_cfg.get("moving", {}),
                    "fixed_preprocessing": prepro_cfg.get("fixed", {}),
                    "scale_sync": scale_sync_meta,
                    "coords_reference": "fixed-grid output -> preprocessed+scale-synced moving input",
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
        print(f"Saved coordinates metadata: {coords_meta_path}")

    # Save before/after overlay comparison
    fixed_tensor = fixed_image()
    moving_tensor = moving_image()

    warped_tensor = None
    if diff_transformed:
        warped_tensor = diff_transformed[-1]
    elif affine_transformed:
        warped_tensor = affine_transformed[-1]

    if warped_tensor is not None:
        if moving_tensor.shape[2:] != fixed_tensor.shape[2:]:
            moving_tensor = F.interpolate(
                moving_tensor,
                size=fixed_tensor.shape[2:],
                mode=fixed_image.interpolate_mode,
                align_corners=True,
            )

        if warped_tensor.shape[2:] != fixed_tensor.shape[2:]:
            warped_tensor = F.interpolate(
                warped_tensor,
                size=fixed_tensor.shape[2:],
                mode=fixed_image.interpolate_mode,
                align_corners=True,
            )

        overlay_path = os.path.join(output_dir, f"{moving_basename}_{reg_type}_overlay.png")

        save_registration_overlay(
            fixed_image=fixed_tensor,
            moving_before=moving_tensor,
            moving_after=warped_tensor,
            output_path=overlay_path,
        )
        print(f"Saved overlay comparison: {overlay_path}")

    print(f"Registration complete. Results saved in {config_manager.output_dir}")


if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="HiMReg image registration")
    _parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    _args = _parser.parse_args()
    main(config_path=_args.config)
