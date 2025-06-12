import argparse
import copy
import os

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F
from tqdm import tqdm

from affinemorph import AffineRegistration
from data_load import Image
from diffeomorph import DiffRegistration
from losses import MutualInformation


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
        affine_kwargs={},
        diff_kwargs={},
    ):
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device

        self.affine_registration = AffineRegistration(
            scales=affine_scales,
            iterations=affine_iterations,
            fixed_images=fixed_images,
            moving_images=moving_images,
            patience=50,
            min_delta=1e-5,
            scale_dependent_lr=scale_dependent_lr,  # Scale-specific learning rates
            **affine_kwargs,
        )

        self.diff_registration = DiffRegistration(
            scales=diff_scales,
            iterations=diff_iterations,
            fixed_images=fixed_images,
            moving_images=moving_images,
            **diff_kwargs,
        )

    def register(self, register_type="affine", save_transformed=True):
        """
        Perform registration of the moving image to the fixed image.

        Args:
            register_type (str): Type of registration to perform ("affine" or "diff").
                                     "affine" performs only affine registration.
                                    "diff" performs affine followed by diffeomorphic registration.

            save_transformed (bool): Whether to save transformed images during optimization.
        Returns:
            tuple: (affine_transformed, diff_transformed, final_coordinates)
                  If register_type='affine': diff_transformed will be None

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

        # Diffeomorphic registration
        affine_result = affine_transformed[-1].squeeze().detach().cpu().numpy()
        self.diff_registration.moving_images = Image(affine_result, device=self.device)
        self.diff_registration.init_affine = affine_matrix
        diff_transformed = self.diff_registration.optimize(save_transformed=save_transformed)
        final_coordinates = self.diff_registration.get_final_coordinates()
        if final_coordinates is not None:
            current_size = final_coordinates.shape[1:-1]
            if current_size != original_size:
                final_coordinates = F.interpolate(
                    final_coordinates.permute(0, 3, 1, 2),  # [B, H, W, 2] -> [B, 2, H, W]
                    size=original_size,
                    mode="bilinear",
                    align_corners=True,
                ).permute(
                    0, 2, 3, 1
                )  # [B, 2, H, W] -> [B, H, W, 2]

            diff_transformed = self._upsample_transformed_images(diff_transformed, original_size)
        return affine_transformed, diff_transformed, final_coordinates

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


def get_args():
    parser = argparse.ArgumentParser(description="Multiscale cross modal image registration")
    parser.add_argument("--fixed", type=str, default="data/K21/NADH-4.tif", help="Path to fixed image")
    parser.add_argument("--moving", type=str, default="data/k21/postaf-4.tif", help="Path to moving image")
    parser.add_argument("--output", type=str, default="pred", help="Output directory")
    parser.add_argument("--affine_scales", nargs="+", type=int, default=[6, 4, 2, 1])
    parser.add_argument("--affine_iterations", nargs="+", type=int, default=[400, 400, 200, 200])
    parser.add_argument("--scale_dependent_lr", nargs="+", type=float, default=[3e-3, 1e-3, 5e-4, 3e-4])
    parser.add_argument("--diff_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument("--diff_iterations", nargs="+", type=int, default=[800, 600, 400, 100, 50])
    parser.add_argument(
        "--loss_type", choices=["mi", "cc", "dice"], default="mi", help="Loss type for registration"
    )

    parser.add_argument(
        "--register_type", choices=["affine", "diff"], default="affine", help="Type of registration"
    )
    parser.add_argument("--auto_tune", default=True, help="Enable automatic parameter tuning")
    parser.add_argument("--target_loss", type=float, default=-0.1, help="Target loss value for auto-tuning")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    fixed_image = Image.load_file(args.fixed, device=device)
    moving_image = Image.load_file(args.moving, device=device)

    registration = HiMReg(
        fixed_images=fixed_image,
        moving_images=moving_image,
        affine_scales=args.affine_scales,
        affine_iterations=args.affine_iterations,
        scale_dependent_lr=args.scale_dependent_lr,
        diff_scales=args.diff_scales,
        diff_iterations=args.diff_iterations,
        init_affine_matrix=initial_affine_torch,
        affine_kwargs={"loss_type": args.loss_type},
        diff_kwargs={"loss_type": args.loss_type},
    )

    affine_transformed, diff_transformed, final_coordinates = registration.register(
        register_type=args.register_type, save_transformed=True
    )

    # Setup output paths
    moving_dir, moving_filename = os.path.split(args.moving)
    moving_basename = os.path.splitext(moving_filename)[0]
    relative_path = os.path.relpath(moving_dir, start=os.path.dirname(moving_dir))
    os.makedirs(os.path.join(args.output, relative_path), exist_ok=True)

    if affine_transformed:
        affine_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_affine.tif")
        last_affine_image = affine_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(affine_pred_path, last_affine_image)

    if diff_transformed:
        diff_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_diff.tif")
        last_diff_image = diff_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(diff_pred_path, last_diff_image)

    if final_coordinates is not None:
        coords_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_coords.npy")
        final_coords_np = final_coordinates.detach().cpu().numpy()
        np.save(coords_pred_path, final_coords_np)

    print(f"Registration complete. Results saved in {args.output}")


if __name__ == "__main__":
    main()
