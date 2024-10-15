import torch
import torch.nn.functional as F

from utils import *


class MutualInformation(nn.Module):
    """Computes the mutual information between two images using Parzen windowing.
    Args:
        kernel_type: the type of kernel used for Parzen windowing. Default is "b-spline".
        num_bins: the number of bins used to discretize the intensity values. Default is 32.
        sigma_ratio: the ratio used to calculate sigma for the Gaussian kernel. Default is 0.5.
        reduction: the method to reduce the loss. Default is "mean".
        smooth_nr: the value added to the numerator to avoid division by zero. Default is 1e-7.
        smooth_dr: the value added to the denominator to avoid division by zero. Default is 1e-7.
    """

    def __init__(
        self,
        kernel_type="b-spline",
        num_bins=32,
        sigma_ratio=0.5,
        reduction="mean",
        smooth_nr=1e-7,
        smooth_dr=1e-7,
    ) -> None:

        super().__init__()
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")

        self.kernel_type = kernel_type
        self.num_bins = num_bins
        self.reduction = reduction
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

        bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
        sigma = (
            torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        )  # Calculate sigma for Gaussian kernel

        self.preterm = 1 / (2 * sigma**2)  # Preterm for Gaussian kernel
        self.register_buffer("bin_centers", bin_centers[None, None, ...])  # (1, 1, num_bins)

    def estimate_prob_distribution(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        if self.kernel_type == "b-spline":
            return self.estimate_bspline_distribution(img, order=3)
        elif self.kernel_type == "gaussian":
            return self.parzen_windowing_gaussian(img)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def estimate_bspline_distribution(
        self, img: torch.Tensor, order: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        max, min = torch.max(img), torch.min(img)
        padding = 2
        bin_size = (max - min) / (self.num_bins - 2 * padding)
        norm_min = torch.div(min, bin_size) - padding

        window_term = torch.div(img, bin_size) - norm_min
        window_term = torch.clamp(window_term, padding, self.num_bins - padding - 1)
        window_term = window_term.reshape(window_term.shape[0], -1, 1)  # (batch, num_sample, 1)
        bins = torch.arange(self.num_bins, device=window_term.device).reshape(1, 1, -1)
        sample_bin_matrix = torch.abs(bins - window_term)  # (batch, num_sample, num_bins)

        weight = torch.zeros_like(sample_bin_matrix, dtype=torch.float)  # (batch, num_sample, num_bins)
        if order == 0:
            weight = weight + (sample_bin_matrix < 0.5) + (sample_bin_matrix == 0.5) * 0.5
        elif order == 3:
            weight = (
                weight
                + (4 - 6 * sample_bin_matrix**2 + 3 * sample_bin_matrix**3) * (sample_bin_matrix < 1) / 6
            )
            weight = (
                weight + (2 - sample_bin_matrix) ** 3 * (sample_bin_matrix >= 1) * (sample_bin_matrix < 2) / 6
            )
        else:
            raise ValueError(f"Do not support b-spline {order}")

        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bins)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bins)
        return weight, probability

    def estimate_gussian_distribution(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(self, pred, target) -> torch.Tensor:
        if target.shape != pred.shape:
            raise ValueError(f"Target shape {target.shape} differs from pred shape {pred.shape}")

        maxval = max(pred.max(), target.max())
        pred = pred / maxval
        target = target / maxval

        pred_weight, pred_prob = self.estimate_prob_distribution(pred)
        target_weight, target_prob = self.estimate_prob_distribution(target)

        joint_prob = torch.bmm(pred_weight.permute(0, 2, 1), target_weight) / pred_weight.shape[1]
        prod_prob = torch.bmm(pred_prob.permute(0, 2, 1), target_prob)

        mi = torch.sum(
            joint_prob
            * torch.log((joint_prob + self.smooth_nr) / (prod_prob + self.smooth_dr) + self.smooth_dr),
            dim=(1, 2),
        )
        if self.reduction == "sum":
            return -torch.sum(mi)
        if self.reduction == "none":
            return -mi
        if self.reduction == "mean":
            return -torch.mean(mi)

        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )


class LNCC(nn.Module):

    def __init__(
        self,
        spatial_dims=3,
        kernel_size=3,
        kernel_type="rectangular",
        reduction="mean",
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        unsigned=True,
    ):
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported")
        self.reduction = reduction
        self.unsigned = unsigned

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        _kernel = kernel_dict[kernel_type]
        self.kernel = _kernel(self.kernel_size)
        self.kernel.requires_grad = False
        self.kernel_nd, self.kernel_vol = self.get_kernel_vol()  # get nD kernel and its volume
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return vol, torch.sum(vol)

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            pred: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        if pred.ndim - 2 != self.ndim:
            raise ValueError(
                f"expecting pred with {self.ndim} spatial dimensions, got pred of shape {pred.shape}"
            )
        if target.shape != pred.shape:
            raise ValueError(f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})")

        # sum over kernel
        def cc_checkpoint_fn(target, pred, kernel, kernel_vol):
            """
            This function is used to compute the intermediate results of the loss.
            """
            t2, p2, tp = target * target, pred * pred, target * pred
            kernel, kernel_vol = kernel.to(pred), kernel_vol.to(pred)
            # kernel_nd = self.kernel_nd.to(pred)
            kernels = [kernel] * self.ndim
            kernels_t = kernels_p = kernels
            kernel_vol_t = kernel_vol_p = kernel_vol
            # compute intermediates
            t_sum = seperate_filter(target, kernels=kernels_t)
            p_sum = seperate_filter(pred, kernels=kernels_p)
            t2_sum = seperate_filter(t2, kernels=kernels_t)
            p2_sum = seperate_filter(p2, kernels=kernels_p)
            tp_sum = seperate_filter(tp, kernels=kernels_t)  # use target device's output
            # average over kernel
            t_avg = t_sum / kernel_vol_t
            p_avg = p_sum / kernel_vol_p

            cross = tp_sum.to(pred) - p_avg * t_sum.to(pred)  # on pred device
            t_var = torch.max(
                t2_sum - t_avg * t_sum,
                torch.as_tensor(self.smooth_dr, dtype=t2_sum.dtype, device=t2_sum.device),
            ).to(pred)
            p_var = torch.max(
                p2_sum - p_avg * p_sum,
                torch.as_tensor(self.smooth_dr, dtype=p2_sum.dtype, device=p2_sum.device),
            )
            if self.unsigned:
                ncc: torch.Tensor = (cross * cross + self.smooth_nr) / ((t_var * p_var) + self.smooth_dr)
            else:
                ncc: torch.Tensor = (cross + self.smooth_nr) / (
                    (torch.sqrt(t_var) * torch.sqrt(p_var)) + self.smooth_dr
                )
            return ncc

        ncc = cc_checkpoint_fn(target, pred, self.kernel, self.kernel_vol)

        if mask is not None:
            maskmean = mask.flatten(2).mean(2)  # [B, N]
            for _ in range(self.ndim):
                maskmean = maskmean.unsqueeze(-1)  # [B, N, 1, 1, ...]
            ncc = ncc * mask / maskmean

        if self.reduction == "sum":
            return torch.sum(ncc).neg()  # sum over the batch, channel and spatial ndims
        if self.reduction == "none":
            return ncc.neg()
        if self.reduction == "mean":
            return torch.mean(ncc).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )


# def mi(
#     pred,
#     target,
#     kernel_type="gaussian",
#     num_bins=32,
#     sigma_ratio=0.5,
#     reduction="mean",
#     smooth_nr=1e-7,
#     smooth_dr=1e-7,
# ):
#     if num_bins <= 0:
#         raise ValueError("num_bins must > 0, got {num_bins}")

#     bin_centers = torch.linspace(0.0, 1.0, num_bins)
#     # Calculate sigma for Gaussian kernel
#     sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
#     bin_centers = bin_centers[None, None, ...]  # Add batch and channel dimensions

#     maxval = max(pred.max(), target.max())  # Noramlize the input
#     pred = pred / maxval
#     target = target / maxval

#     if target.shape != pred.shape:
#         raise ValueError(
#             f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})"
#         )

#     def parzen_windowing(img):
#         img = torch.clamp(img, 0, 1)  # Normalize the image to (0, 1)
#         img = img.reshape(img.shape[0], -1, 1)  #
#         preterm = 1 / (2 * sigma**2)
#         weight = torch.exp(-preterm.to(img) * (img - bin_centers.to(img)) ** 2)
#         weight = weight / torch.sum(weight, dim=-1, keepdim=True)
#         probability = torch.mean(weight, dim=-2, keepdim=True)
#         return weight, probability

#     wa, pa = parzen_windowing(pred)
#     wb, pb = parzen_windowing(target)

#     pab = torch.bmm(wa.permute(0, 2, 1), wb).div(wa.shape[1])
#     papb = torch.bmm(pa.permute(0, 2, 1), pb)
#     mi = torch.sum(
#         pab * torch.log((pab + smooth_nr) / (papb + smooth_dr) + smooth_dr), dim=(1, 2)
#     )

#     if reduction == "sum":
#         return -torch.sum(mi)
#     if reduction == "none":
#         return -mi.neg
#     if reduction == "mean":
#         return -torch.mean(mi)
#     raise ValueError(
#         f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].'
#     )
