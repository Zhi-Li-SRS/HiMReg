import argparse
import os

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F

from affinemorph import AffineRegistration
from data_load import Image
from diffeomorph import DiffRegistration


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
            **affine_kwargs,
        )

        self.diff_registration = DiffRegistration(
            scales=diff_scales,
            iterations=diff_iterations,
            fixed_images=fixed_images,
            moving_images=moving_images,
            **diff_kwargs,
        )

    def register(self, save_transformed=True):
        """
        Perform registration of the moving image to the fixed image.

        Args:
            save_transformed (bool): Whether to save transformed images during optimization.
        Returns:
            tuple: (transformed_images, final_coordinates)
        """

        affine_transformed = self.affine_registration.optimize(save_transformed=save_transformed)

        affine_matrix = self.affine_registration.get_final_transform()

        self.diff_registration.init_affine = affine_matrix
        diff_transformed = self.diff_registration.optimize(save_transformed=save_transformed)

        final_coordinates = None
        final_coordinates = self.diff_registration.get_final_coordinates()

        return diff_transformed, final_coordinates

    @staticmethod
    def apply_transformation(image: torch.Tensor, coords_path: str, device: str):
        coords = torch.from_numpy(np.load(coords_path)).to(device).float()
        transformed = F.grid_sample(image, coords, mode="bilinear", align_corners=True)
        return transformed


def get_args():
    parser = argparse.ArgumentParser(description="Multiscale cross modal image registration")
    parser.add_argument("--fixed", type=str, default="data/D385_7A2/he_roi1.tif", help="Path to fixed image")
    parser.add_argument(
        "--moving", type=str, default="data/D385_7A2/lipid_unsat_roi1.tif", help="Path to moving image"
    )
    parser.add_argument("--output", type=str, default="pred", help="Output directory")
    parser.add_argument("--affine_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument("--affine_iterations", nargs="+", type=int, default=[800, 600, 400, 100, 50])
    parser.add_argument("--diff_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument("--diff_iterations", nargs="+", type=int, default=[800, 600, 400, 100, 50])
    parser.add_argument("--loss_type", choices=["mi", "cc"], default="mi", help="Loss type for registration")
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
        diff_scales=args.diff_scales,
        diff_iterations=args.diff_iterations,
        affine_kwargs={"loss_type": args.loss_type},
        diff_kwargs={"loss_type": args.loss_type},
    )

    transformed_images = registration.register(save_transformed=True)

    if transformed_images:
        diff_transformed, final_coordinates = transformed_images

        # Create the output directory
        moving_dir, moving_filename = os.path.split(args.moving)
        moving_basename = os.path.splitext(moving_filename)[0]
        relative_path = os.path.relpath(moving_dir, start=os.path.dirname(moving_dir))

        # affine_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_affine.tif")
        diff_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_diff.tif")
        coords_pred_path = os.path.join(args.output, relative_path, f"{moving_basename}_coords.npy")

        os.makedirs(os.path.dirname(diff_pred_path), exist_ok=True)

        # Save diffeomorphic transformed image
        last_diff_image = diff_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(diff_pred_path, last_diff_image)

        final_coords_np = final_coordinates.detach().cpu().numpy()
        np.save(coords_pred_path, final_coords_np)

    print("Registration complete. Results saved in output directory.")


if __name__ == "__main__":
    main()
