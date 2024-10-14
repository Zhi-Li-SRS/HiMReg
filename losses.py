import torch
import torch.nn.functional as F

from utils import *


class LNCC(nn.Module):

    def __init__(
        self,
        spatial_dims: int = 3,
        kernel_size: int = 3,
        kernel_type: str = "rectangular",
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        unsigned: bool = True,
    ) -> None:
        super().__init__()
        self.ndim = spatial_dims
        if self.ndim not in {1, 2, 3}:
            raise ValueError(
                f"Unsupported ndim: {self.ndim}-d, only 1-d, 2-d, and 3-d inputs are supported"
            )
        self.reduction = reduction
        self.unsigned = unsigned

        self.kernel_size = kernel_size
        if self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {self.kernel_size}")

        _kernel = kernel_dict[kernel_type]
        self.kernel = _kernel(self.kernel_size)
        self.kernel.requires_grad = False
        self.kernel_nd, self.kernel_vol = (
            self.get_kernel_vol()
        )  # get nD kernel and its volume
        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def get_kernel_vol(self):
        vol = self.kernel
        for _ in range(self.ndim - 1):
            vol = torch.matmul(vol.unsqueeze(-1), self.kernel.unsqueeze(0))
        return vol, torch.sum(vol)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
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
            raise ValueError(
                f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})"
            )

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
                ncc: torch.Tensor = (cross * cross + self.smooth_nr) / (
                    (t_var * p_var) + self.smooth_dr
                )
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
            return torch.mean(
                ncc
            ).neg()  # average over the batch, channel and spatial ndims
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )


class MutualInformation(nn.Module):

    def __init__(
        self,
        kernel_type: str = "gaussian",
        num_bins: int = 32,
        sigma_ratio: float = 0.5,
        reduction: str = "mean",
        smooth_nr: float = 1e-7,
        smooth_dr: float = 1e-7,
    ) -> None:

        super().__init__()
        if num_bins <= 0:
            raise ValueError("num_bins must > 0, got {num_bins}")
        bin_centers = torch.linspace(0.0, 1.0, num_bins)  # (num_bins,)
        sigma = torch.mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
        self.kernel_type = kernel_type
        self.num_bins = num_bins
        self.kernel_type = kernel_type
        if self.kernel_type == "gaussian":
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        elif self.kernel_type == "b-spline":
            self.preterm = 1 / (2 * sigma**2)
            self.bin_centers = bin_centers[None, None, ...]
        self.reduction = reduction

        self.smooth_nr = float(smooth_nr)
        self.smooth_dr = float(smooth_dr)

    def parzen_windowing(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.kernel_type == "gaussian":
            pred_weight, pred_probability = self.parzen_windowing_gaussian(pred)
            target_weight, target_probability = self.parzen_windowing_gaussian(target)
        elif self.kernel_type == "b-spline":
            # a third order BSpline kernel is used for the pred image intensity PDF.
            pred_weight, pred_probability = self.parzen_windowing_b_spline(pred, order=3)
            # a zero order (box car) BSpline kernel is used for the target image intensity PDF.
            target_weight, target_probability = self.parzen_windowing_b_spline(
                target, order=0
            )
        else:
            raise ValueError
        return pred_weight, pred_probability, target_weight, target_probability

    def parzen_windowing_b_spline(
        self, img: torch.Tensor, order: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parzen windowing with b-spline kernel (adapted from ITK)
        Args:
            img: the shape should be B[NDHW].
            order: int.
        """
        _max, _min = torch.max(img), torch.min(img)
        padding = 2
        bin_size = (_max - _min) / (self.num_bins - 2 * padding)
        norm_min = torch.div(_min, bin_size) - padding

        # assign bin/window index to each voxel
        window_term = torch.div(img, bin_size) - norm_min  # B[NDHW]
        # make sure the extreme values are in valid (non-padded) bins
        window_term = torch.clamp(
            window_term, padding, self.num_bins - padding - 1
        )  # B[NDHW]
        window_term = window_term.reshape(
            window_term.shape[0], -1, 1
        )  # (batch, num_sample, 1)
        bins = torch.arange(self.num_bins, device=window_term.device).reshape(
            1, 1, -1
        )  # (1, 1, num_bins)
        sample_bin_matrix = torch.abs(bins - window_term)  # (batch, num_sample, num_bins)

        # b-spleen kernel
        # (4 - 6 * abs ** 2 + 3 * abs ** 3) / 6 when 0 <= abs < 1
        # (2 - abs) ** 3 / 6 when 1 <= abs < 2
        weight = torch.zeros_like(
            sample_bin_matrix, dtype=torch.float
        )  # (batch, num_sample, num_bins)
        if order == 0:
            weight = weight + (sample_bin_matrix < 0.5) + (sample_bin_matrix == 0.5) * 0.5
        elif order == 3:
            weight = (
                weight
                + (4 - 6 * sample_bin_matrix**2 + 3 * sample_bin_matrix**3)
                * (sample_bin_matrix < 1)
                / 6
            )
            weight = (
                weight
                + (2 - sample_bin_matrix) ** 3
                * (sample_bin_matrix >= 1)
                * (sample_bin_matrix < 2)
                / 6
            )
        else:
            raise ValueError(f"Do not support b-spline {order}-order parzen windowing")

        weight = weight / torch.sum(
            weight, dim=-1, keepdim=True
        )  # (batch, num_sample, num_bins)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bins)
        return weight, probability

    def parzen_windowing_gaussian(
        self, img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parzen windowing with gaussian kernel (adapted from DeepReg implementation)
        Note: the input is expected to range between 0 and 1
        Args:
            img: the shape should be B[NDHW].
        """
        img = torch.clamp(img, 0, 1)
        img = img.reshape(img.shape[0], -1, 1)  # (batch, num_sample, 1)
        weight = torch.exp(
            -self.preterm.to(img) * (img - self.bin_centers.to(img)) ** 2
        )  # (batch, num_sample, num_bin)
        weight = weight / torch.sum(
            weight, dim=-1, keepdim=True
        )  # (batch, num_sample, num_bin)
        probability = torch.mean(weight, dim=-2, keepdim=True)  # (batch, 1, num_bin)
        return weight, probability

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred: the shape should be B[NDHW].
            target: the shape should be same as the pred shape.
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
        """
        maxval = max(pred.max(), target.max())
        pred = pred / maxval
        target = target / maxval

        if target.shape != pred.shape:
            raise ValueError(
                f"ground truth has differing shape ({target.shape}) from pred ({pred.shape})"
            )
        wa, pa, wb, pb = self.parzen_windowing(
            pred, target
        )  # (batch, num_sample, num_bin), (batch, 1, num_bin)

        pab = torch.bmm(wa.permute(0, 2, 1), wb.to(wa)).div(
            wa.shape[1]
        )  # (batch, num_bins, num_bins)
        papb = torch.bmm(pa.permute(0, 2, 1), pb.to(pa))  # (batch, num_bins, num_bins)
        mi = torch.sum(
            pab
            * torch.log(
                (pab + self.smooth_nr) / (papb + self.smooth_dr) + self.smooth_dr
            ),
            dim=(1, 2),
        )  # (batch)

        ndim = len(pred.shape) - 2
        if mask is not None:
            maskmean = mask.flatten(2).mean(2)
            for i in range(ndim):
                maskmean = maskmean.unsqueeze(-1)
            mi = mi * mask / maskmean

        if self.reduction == "sum":
            return torch.sum(mi).neg()  # sum over the batch and channel ndims
        if self.reduction == "none":
            return mi.neg()
        if self.reduction == "mean":
            return torch.mean(mi).neg()  # average over the batch and channel ndims

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
