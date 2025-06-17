import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_gradient_magnitude
from skimage import feature
from skimage.metrics import structural_similarity as ssim

from src.data_load import Image
from HiMReg import HiMReg
from src.losses import LNCC, MutualInformation

# Set global font settings
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


class BatchImage:
    """Wrapper class for batch tensor processing"""

    def __init__(self, image_tensor, device):
        self.array = image_tensor.to(device)
        self.device = device
        self._shape = image_tensor.shape
        self.dims = 2
        self.interpolate_mode = "bilinear"

    def __call__(self):
        return self.array

    @property
    def shape(self):
        return self._shape

    def size(self):
        return self.array.shape[0]

    def get_pixel_to_physical(self):
        batch_size = self.array.shape[0]
        return torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)

    def get_physical_to_pixel(self):
        batch_size = self.array.shape[0]
        return torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)


class RegistrationComparison:
    """Class for comparing registration methods (HiMReg vs Elastix)

    Args:
        fixed_path (str): Path to fixed image
        moving_path (str): Path to moving image
        output_dir (str): Directory to save results
        device (str): Device to run computations on
        batch_size (int): Batch size for registration
        register_type (str): Type of registration to perform (affine or diff)
    """

    def __init__(
        self, fixed_path, moving_path, output_dir, device="cuda", batch_size=10, register_type="affine"
    ):
        self.device = torch.device(device)
        self.fixed_path = fixed_path
        self.moving_path = moving_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.register_type = register_type
        os.makedirs(output_dir, exist_ok=True)

        # Load images
        self.fixed_img = Image.load_file(fixed_path, device=self.device)
        self.moving_img = Image.load_file(moving_path, device=self.device)

        self.fixed_batch = self.fixed_img().repeat(batch_size, 1, 1, 1)
        self.moving_batch = self.moving_img().repeat(batch_size, 1, 1, 1)
        self.original_size = self.fixed_batch.shape[2:]

        # Initialize metrics
        self.mi_metric = MutualInformation()
        self.cc_metric = LNCC(spatial_dims=2)

    @staticmethod
    def dice_coefficient(pred, target):
        """Calculate DICE coefficient between predicted and target binary masks"""
        smooth = 1e-6
        # Threshold to create binary masks
        pred = (pred - pred.min()) / (pred.max() - pred.min())
        target = (target - target.min()) / (target.max() - target.min())
        target = 1 - target

        pred = (pred > 0.5).float()
        target = (target > 0.5).float()

        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))

        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.mean()

    def run_himreg_batch(self, save_transformed=True):
        """Run HiMReg registration on batch"""
        start_time = time.time()

        fixed_images = BatchImage(self.fixed_batch, self.device)
        moving_images = BatchImage(self.moving_batch, self.device)

        registration = HiMReg(
            fixed_images=fixed_images,
            moving_images=moving_images,
            affine_scales=[8, 4, 2],
            affine_iterations=[400, 200, 50],
            diff_scales=[8, 4, 2],
            diff_iterations=[400, 200, 50],
        )

        affine_transformed, diff_transformed, final_coords = registration.register(
            register_type=self.register_type, save_transformed=save_transformed
        )

        end_time = time.time()
        runtime = end_time - start_time
        # Return appropriate result based on registration type
        if self.register_type == "affine":
            return affine_transformed[-1], final_coords, runtime
        else:
            return diff_transformed[-1], final_coords, runtime

    # def run_himreg(self, save_transformed=True):
    #     """Run HiMReg registration"""
    #     start_time = time.time()

    #     registration = HiMReg(
    #         fixed_images=self.fixed_img,
    #         moving_images=self.moving_img,
    #         affine_scales=[8, 4, 2],
    #         affine_iterations=[400, 200, 50],
    #         diff_scales=[2],
    #         diff_iterations=[50],
    #     )

    #     affine_transfo, diff_transformed, final_coords = registration.register(save_transformed=save_transformed)

    #     end_time = time.time()
    #     runtime = end_time - start_time
    #     return diff_transformed[-1], final_coords, runtime

    def run_elastix_batch(self):
        """Run Elastix registration on batch"""
        start_time = time.time()
        results = []
        params = []

        for i in range(self.batch_size):
            result, param, _ = self.run_single_elastix()
            results.append(result)
            params.append(param)

        end_time = time.time()
        runtime = end_time - start_time
        # Stack results
        results_tensor = torch.stack(results, dim=0)
        return results_tensor, params, runtime

    def run_single_elastix(self):
        """Run SimpleITK registration with affine initialization followed by B-spline refinement"""
        start_time = time.time()

        fixed_sitk = sitk.ReadImage(self.fixed_path)
        moving_sitk = sitk.ReadImage(self.moving_path)

        # Cast images to float32
        cast_filter = sitk.CastImageFilter()
        cast_filter.SetOutputPixelType(sitk.sitkFloat32)
        fixed_float = cast_filter.Execute(fixed_sitk)
        moving_float = cast_filter.Execute(moving_sitk)
        fixed_normalized = sitk.RescaleIntensity(fixed_float, 0.0, 1.0)
        moving_normalized = sitk.RescaleIntensity(moving_float, 0.0, 1.0)

        try:
            # Stage 1: Affine Registration
            affine_transform = sitk.AffineTransform(fixed_sitk.GetDimension())
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetInitialTransform(affine_transform)
            registration_method.SetMetricAsMattesMutualInformation()
            registration_method.SetInterpolator(sitk.sitkLinear)
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=0.1,
                numberOfIterations=200,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.Execute(fixed_normalized, moving_normalized)

            # Stage 2: B-spline Registration - building upon the affine result
            transform_domain_mesh_size = [8] * fixed_sitk.GetDimension()
            bspline_transform = sitk.BSplineTransformInitializer(fixed_normalized, transform_domain_mesh_size)
            registration_method = sitk.ImageRegistrationMethod()
            registration_method.SetInitialTransform(bspline_transform)
            registration_method.SetInitialTransform(affine_transform)

            registration_method.SetMetricAsMattesMutualInformation()
            registration_method.SetOptimizerAsGradientDescent(
                learningRate=3e-3,
                numberOfIterations=400,
                convergenceMinimumValue=1e-6,
                convergenceWindowSize=10,
            )
            registration_method.SetOptimizerScalesFromPhysicalShift()
            registration_method.SetInterpolator(sitk.sitkLinear)

            # Execute B-spline registration
            final_transform = registration_method.Execute(fixed_normalized, moving_normalized)

            # Apply final transform
            result_image = sitk.Resample(
                moving_normalized,
                fixed_normalized,
                final_transform,
                sitk.sitkLinear,
                0.0,
                moving_normalized.GetPixelID(),
            )
            result_image = sitk.RescaleIntensity(
                result_image, 0, 255 if fixed_sitk.GetPixelID() == sitk.sitkUInt8 else 65535
            )
            result_image = sitk.Cast(result_image, fixed_sitk.GetPixelID())

            end_time = time.time()
            runtime = end_time - start_time

            result_np = sitk.GetArrayFromImage(result_image)
            result_tensor = torch.from_numpy(result_np).unsqueeze(0).unsqueeze(0).float().to(self.device)

            return result_tensor, final_transform.GetParameters(), runtime

        except Exception as e:
            print(f"Registration failed: {str(e)}")
            raise e

    def evaluate_methods(self):
        """Run and evaluate both registration methods on batch"""
        himreg_result, himreg_coords, himreg_time = self.run_himreg_batch()
        elastix_result, elastix_params, elastix_time = self.run_elastix_batch()

        self.final_coords = himreg_coords
        # Compute average metrics
        fixed_batch = self.fixed_batch.to(self.device).float()
        himreg_result = himreg_result.to(self.device).float()
        elastix_result = elastix_result.to(self.device).float()

        if himreg_result.ndim == 5:
            himreg_result = himreg_result.squeeze(2)
        if elastix_result.ndim == 5:
            elastix_result = elastix_result.squeeze(2)
        if fixed_batch.ndim == 5:
            fixed_batch = fixed_batch.squeeze(2)

        # Average MI over batch
        himreg_mi = (-self.mi_metric(himreg_result, fixed_batch)).item() / self.batch_size
        elastix_mi = (-self.mi_metric(elastix_result, fixed_batch)).item() / self.batch_size

        himreg_dice = self.dice_coefficient(himreg_result, fixed_batch).item()
        elastix_dice = self.dice_coefficient(elastix_result, fixed_batch).item()

        results = {
            "MI": {"HiMReg": himreg_mi, "Elastix": elastix_mi},
            "DICE": {"HiMReg": himreg_dice, "Elastix": elastix_dice},
        }

        return results, himreg_result, elastix_result

    def plot_performance_comparison(self, results):
        """Create improved bar plots for performance metrics"""
        metrics = ["MI", "DICE"]
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        plt.style.use("ggplot")

        for i, metric in enumerate(metrics):
            ax = axes[i]
            values = [results[metric]["Elastix"], results[metric]["HiMReg"]]
            colors = ["royalblue", "salmon"]

            # Calculate improvement percentage
            improvement = ((values[1] - values[0]) / values[0]) * 100

            bars = ax.bar(["Elastix", "HiMReg"], values, color=colors, width=0.4, alpha=0.7)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

            # Style improvements
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylabel(metric, fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_comparison.svg"), dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_results(self, himreg_result, elastix_result, results):
        """Create improved visualization plots"""
        # Single row image comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        plt.style.use("ggplot")

        images = [
            (self.fixed_img(), "Fixed"),
            (self.moving_img(), "Moving"),
            (elastix_result[0:1], "Elastix(Affine+B-spline)"),
            (himreg_result[0:1], "HiMReg"),
        ]

        for i, (img, title) in enumerate(images):
            if img.ndim == 5:
                img = img.squeeze(2)

            axes[i].imshow(img.squeeze().cpu(), cmap="gray")
            axes[i].set_title(title, fontsize=12, fontweight="bold")
            axes[i].axis("off")

        # # Plot displacement field
        # if hasattr(self, "final_coords"):
        #     self.plot_displacement_field(self.final_coords, axes[4], "HiMReg")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "Image_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

        self.plot_performance_comparison(results)


def get_args():
    parser = argparse.ArgumentParser(description="Compare HiMReg and Elastix registration methods")
    parser.add_argument("--fixed", type=str, default="data/maldi/he_rescale.png", help="Path to fixed image")
    parser.add_argument(
        "--moving", type=str, default="data/maldi/redox_rescale.png", help="Path to moving image"
    )
    parser.add_argument("--output", type=str, default="comparison_results", help="Output directory")
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to run on"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Device to run on")
    parser.add_argument(
        "--register_type", type=str, default="affine", choices=["affine", "diff"], help="Type of registration"
    )
    return parser.parse_args()


def main():
    args = get_args()

    comparison = RegistrationComparison(
        fixed_path=args.fixed,
        moving_path=args.moving,
        output_dir=args.output,
        device=args.device,
        batch_size=args.batch_size,
        register_type=args.register_type,
    )

    results, himreg_result, elastix_result = comparison.evaluate_methods()

    comparison.visualize_results(himreg_result, elastix_result, results)


if __name__ == "__main__":
    main()
