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


class AutoParameterTuning:
    """Automatically tuning registration parameters to achieve target loss.


    Args:
        fixed_images: Image object of fixed image
        moving_images: Image object of moving image
        scales: List of scales to use (will be preserved)
        target_loss: Target loss value to achieve (default: -0.1)
        max_iter_per_scale: Maximum iterations allowed per scale
        min_iter_per_scale: Minimum iterations allowed per scale
        patience: Number of consecutive iterations without improvement before early stopping
        tolerance: How close to target loss is considered acceptable
        register_type: Type of registration ("affine" or "diff")
        lr_candidates: List of learning rate values to try
    """

    def __init__(
        self,
        fixed_images,
        moving_images,
        scales=[8, 4, 2, 1],
        target_loss=-0.1,
        max_iter_per_scale=1000,
        min_iter_per_scale=50,
        patience=5,
        tolerance=0.01,
        register_type="affine",
        lr_candidates=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    ):
        self.fixed_images = fixed_images
        self.moving_images = moving_images
        self.device = fixed_images.device
        self.scales = scales
        self.target_loss = target_loss
        self.max_iter_per_scale = max_iter_per_scale
        self.min_iter_per_scale = min_iter_per_scale
        self.patience = patience
        self.tolerance = tolerance
        self.register_type = register_type
        self.lr_candidates = lr_candidates

        # Initialize MI loss for evaluation
        self.loss_fn = MutualInformation()

    def evaluate_registration(self, transformed_image):
        """Evaluate registration quality by computing MI loss."""
        with torch.no_grad():
            loss = self.loss_fn(transformed_image, self.fixed_images()).item()
        return loss

    def find_best_learning_rate(self, iterations):
        """Find the best learning rate while keeping iterations fixed."""
        from HiMReg import HiMReg

        print("Finding optimal learning rate...")
        best_lr = self.lr_candidates[0]  # Default to first candidate
        best_loss_diff = float("inf")

        for lr in self.lr_candidates:
            print(f"Testing learning rate: {lr}")

            # Create registration with current learning rate
            registration = HiMReg(
                fixed_images=self.fixed_images,
                moving_images=self.moving_images,
                affine_scales=self.scales,
                affine_iterations=iterations,
                diff_scales=self.scales,
                diff_iterations=iterations,
                affine_kwargs={"loss_type": "mi", "optimizer_lr": lr},
                diff_kwargs={"loss_type": "mi", "optimizer_lr": lr},
            )

            # Run registration
            affine_transformed, diff_transformed, _ = registration.register(
                register_type=self.register_type, save_transformed=True
            )

            # Evaluate result
            if self.register_type == "affine":
                final_transformed = affine_transformed[-1]
            else:
                final_transformed = diff_transformed[-1]

            current_loss = self.evaluate_registration(final_transformed)
            loss_diff = abs(current_loss - self.target_loss)

            print(f"Learning rate {lr} achieved loss: {current_loss}, diff from target: {loss_diff}")

            if loss_diff < best_loss_diff:
                best_loss_diff = loss_diff
                best_lr = lr

                # If we're already close enough, stop early
                if loss_diff < self.tolerance:
                    break

        print(f"Best learning rate found: {best_lr}")
        return best_lr

    def optimize_parameters(self):
        """Find optimal iteration counts and learning rate to achieve target loss."""
        print("Starting automated parameter tuning...")

        # Start with default iterations
        iterations = [self.max_iter_per_scale // 2] * len(self.scales)
        best_iterations = copy.deepcopy(iterations)
        best_lr = self.lr_candidates[0]  # Start with first learning rate
        best_loss_diff = float("inf")

        best_lr = self.find_best_learning_rate(iterations)

        # Track which scale we're currently optimizing
        current_scale_idx = 0
        no_improvement_count = 0

        for tuning_cycle in range(10):  # Maximum 10 overall tuning cycles
            print(f"Tuning cycle {tuning_cycle+1} with iterations: {iterations}, lr: {best_lr}")

            # Create HiMReg with current parameters
            registration = HiMReg(
                fixed_images=self.fixed_images,
                moving_images=self.moving_images,
                affine_scales=self.scales,
                affine_iterations=iterations,
                diff_scales=self.scales,
                diff_iterations=iterations,
                affine_kwargs={"loss_type": "mi", "optimizer_lr": best_lr},
                diff_kwargs={"loss_type": "mi", "optimizer_lr": best_lr},
            )

            # Run registration and get results
            affine_transformed, diff_transformed, _ = registration.register(
                register_type=self.register_type, save_transformed=True
            )

            # Evaluate the registration result
            if self.register_type == "affine":
                final_transformed = affine_transformed[-1]
            else:
                final_transformed = diff_transformed[-1]

            current_loss = self.evaluate_registration(final_transformed)
            loss_diff = abs(current_loss - self.target_loss)

            print(f"Current loss: {current_loss}, Target: {self.target_loss}, Difference: {loss_diff}")

            # Check if we've improved
            if loss_diff < best_loss_diff:
                best_loss_diff = loss_diff
                best_iterations = copy.deepcopy(iterations)
                no_improvement_count = 0

                # If we're close enough to target, we're done
                if loss_diff < self.tolerance:
                    print(
                        f"Target loss achieved within tolerance! Final iterations: {best_iterations}, lr: {best_lr}"
                    )
                    break
            else:
                no_improvement_count += 1

            if no_improvement_count >= self.patience:
                # Every few cycles, try optimizing the learning rate again
                if tuning_cycle % 3 == 2:
                    best_lr = self.find_best_learning_rate(iterations)
                    no_improvement_count = 0
                    continue

                # Otherwise move to next scale
                current_scale_idx = (current_scale_idx + 1) % len(self.scales)
                no_improvement_count = 0
                print(f"Moving to tuning scale {self.scales[current_scale_idx]}")

            # Adjust iterations for the current scale based on loss difference
            if current_loss > self.target_loss:  # Loss is too high (less negative)
                iterations[current_scale_idx] = min(
                    int(iterations[current_scale_idx] * 1.5), self.max_iter_per_scale
                )
            else:  # Loss is too low (more negative)
                iterations[current_scale_idx] = max(
                    int(iterations[current_scale_idx] * 0.7), self.min_iter_per_scale
                )

        return best_iterations, best_lr

    def get_optimal_himreg(self):
        """Return a HiMReg instance with optimized parameters."""
        # Get optimal iteration counts and learning rate
        optimal_iterations, optimal_lr = self.optimize_parameters()

        print(f"Optimal configuration found:")
        print(f"  - Iterations: {optimal_iterations}")
        print(f"  - Learning rate: {optimal_lr}")

        return HiMReg(
            fixed_images=self.fixed_images,
            moving_images=self.moving_images,
            affine_scales=self.scales,
            affine_iterations=optimal_iterations,
            diff_scales=self.scales,
            diff_iterations=optimal_iterations,
            affine_kwargs={"loss_type": "mi", "optimizer_lr": optimal_lr},
            diff_kwargs={"loss_type": "mi", "optimizer_lr": optimal_lr},
        )


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
    parser.add_argument(
        "--fixed", type=str, default="data/vander/DAPI-3_diff.tif", help="Path to fixed image"
    )
    parser.add_argument(
        "--moving", type=str, default="data/vander/proteins-3_rescale.tif", help="Path to moving image"
    )
    parser.add_argument("--output", type=str, default="pred", help="Output directory")
    parser.add_argument("--affine_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument("--affine_iterations", nargs="+", type=int, default=[800, 600, 400, 100, 50])
    parser.add_argument("--diff_scales", nargs="+", type=int, default=[8, 6, 4, 2, 1])
    parser.add_argument("--diff_iterations", nargs="+", type=int, default=[800, 600, 400, 100, 50])
    parser.add_argument("--loss_type", choices=["mi", "cc"], default="mi", help="Loss type for registration")

    parser.add_argument(
        "--register_type", choices=["affine", "diff"], default="affine", help="Type of registration"
    )
    parser.add_argument("--auto_tune", action="store_true", help="Enable automatic parameter tuning")
    parser.add_argument("--target_loss", type=float, default=-0.1, help="Target loss value for auto-tuning")
    parser.add_argument(
        "--lr_candidates",
        type=float,
        nargs="+",
        default=[1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        help="Learning rate candidates to try during auto-tuning",
    )
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    fixed_image = Image.load_file(args.fixed, device=device)
    moving_image = Image.load_file(args.moving, device=device)

    if args.auto_tune:
        tuner = AutoParameterTuning(
            fixed_images=fixed_image,
            moving_images=moving_image,
            scales=args.affine_scales,  # Keep the same scales
            target_loss=args.target_loss,
            register_type=args.register_type,
            lr_candidates=args.lr_candidates,
        )

        registration = tuner.get_optimal_himreg()
        print("Using automatically tuned parameters")
    else:
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
