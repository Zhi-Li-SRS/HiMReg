import argparse
import os

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F

from affine import AffineRegistration
from data_load import BatchedImages, Image
from diffeomorph import DiffRegistration


class Registration:
    """Class for performing affine and diffeomorphic registration on a pair of images.
    Args:
        fixed_images (BatchedImages): BatchedImages object of fixed images.
        moving_images (BatchedImages): BatchedImages object of moving images.
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

    def register(self, save_transformed=False):

        fixed_arrays = self.fixed_images()
        moving_arrays = self.moving_images()

        affine_transformed = self.affine_registration.optimize(
            save_transformed=save_transformed
        )

        affine_matrix = self.affine_registration.get_affine_matrix()

        self.diff_registration.affine = affine_matrix
        diff_transformed = self.diff_registration.optimize(
            save_transformed=save_transformed
        )

        if save_transformed:
            return affine_transformed, diff_transformed
        else:
            return None

    def get_final_transformation(self):
        final_warp = self.diff_registration.get_warped_coordinates(
            self.fixed_images, self.moving_images
        )
        return final_warp

    def apply_transformation(self, image: torch.Tensor):
        moved_coords = self.get_final_transformation()
        transformed_image = torch.nn.functional.grid_sample(
            image, moved_coords, mode="bilinear", align_corners=True
        )
        return transformed_image

    def save_affine_matrix(self, output_path):
        affine_matrix = (
            self.affine_registration.get_affine_matrix().detach().cpu().numpy()
        )
        np.save(output_path, affine_matrix)

    def apply_affine(self, image_path, matrix_path, device="cuda"):
        """Apply affine transformation to an image."""
        affine_matrix = np.load(matrix_path)
        affine_matrix = torch.from_numpy(affine_matrix).to(device)

        image = Image.load_file(image_path, device=device)
        image_batch = BatchedImages(image)
        transformed_image = F.affine_grid(
            affine_matrix[:, :image], image_batch().shape, aline_corners=True
        )
        transformed_image = F.grid_sample(
            image_batch(), transformed_image, mode="bilinear", align_corners=True
        )
        return transformed_image


def get_args():
    parser = argparse.ArgumentParser(
        description="Multiscale cross modal image registration"
    )
    parser.add_argument(
        "--fixed",
        type=str,
        default="data/codex/codex_roi1.tif",
        help="Path to fixed image",
    )
    parser.add_argument(
        "--moving",
        type=str,
        default="data/codex/791_roi1.tif",
        help="Path to moving image",
    )
    parser.add_argument("--output", type=str, default="pred", help="Output directory")
    parser.add_argument("--affine_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument(
        "--affine_iterations", nargs="+", type=int, default=[800, 600, 400, 200, 100]
    )
    parser.add_argument("--diff_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument(
        "--diff_iterations", nargs="+", type=int, default=[800, 600, 400, 200, 100]
    )
    parser.add_argument(
        "--loss_type",
        choices=["mi", "cc"],
        default="mi",
        help="Loss type for registration",
    )
    parser.add_argument(
        "--integrator_n", type=int, default=6, help="Integrator n for GeoDiscShooting"
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed_image = Image.load_file(args.fixed, device=device)
    moving_image = Image.load_file(args.moving, device=device)
    fixed_image = BatchedImages(fixed_image)
    moving_image = BatchedImages(moving_image)

    registration = Registration(
        fixed_images=fixed_image,
        moving_images=moving_image,
        affine_scales=args.affine_scales,
        affine_iterations=args.affine_iterations,
        diff_scales=args.diff_scales,
        diff_iterations=args.diff_iterations,
        affine_kwargs={"loss_type": args.loss_type},
        diff_kwargs={
            "loss_type": args.loss_type,
            "deformation_type": "compositive",
            "integrator_n": args.integrator_n,
        },
    )

    transformed_images = registration.register(save_transformed=True)

    if transformed_images:
        affine_transformed, diff_transformed = transformed_images

        # Create the output directory
        moving_dir, moving_filename = os.path.split(args.moving)
        moving_basename = os.path.splitext(moving_filename)[0]
        relative_path = os.path.relpath(moving_dir, start=os.path.dirname(moving_dir))

        affine_pred_path = os.path.join(
            args.output, relative_path, f"{moving_basename}_affine.tif"
        )
        diff_pred_path = os.path.join(
            args.output, relative_path, f"{moving_basename}_diff.tif"
        )
        matrix_pred_path = os.path.join(
            args.output, relative_path, f"{moving_basename}_affine_matrix.npy"
        )
        # Create necessary directories
        os.makedirs(os.path.dirname(affine_pred_path), exist_ok=True)
        os.makedirs(os.path.dirname(diff_pred_path), exist_ok=True)
        os.makedirs(os.path.dirname(matrix_pred_path), exist_ok=True)

        # Save affine transformed image
        last_affine_image = affine_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(affine_pred_path, last_affine_image)

        # Save diffeomorphic transformed image
        last_diff_image = diff_transformed[-1].squeeze().detach().cpu().numpy()
        io.imsave(diff_pred_path, last_diff_image)

        # Save affine matrix
        registration.save_affine_matrix(matrix_pred_path)

    print("Registration complete. Results saved in output directory.")


if __name__ == "__main__":
    main()
