from typing import Any, List, Optional, Union

import numpy as np
import SimpleITK as sitk
import tifffile
import torch

from utils import integer_to_onehot


class Image:
    """Load the image data and perform preprocessing.
    Args:
        image_data (Union[str, sitk.Image, np.ndarray, torch.Tensor]): The image data to load.
        device (torch.device): The device to load the image data on.
        is_segmentation (bool): Whether the image is a segmentation mask. Defaults to False.
        max_seg_label (int): The maximum segmentation label. Defaults to None.
        background_seg_label (int): The background segmentation label. Defaults to 0.
        seg_preprocessor: A function to preprocess the segmentation mask. Defaults to None.
        spacing: The pixel spacing of the image. Defaults to None.
        direction: The direction of the image. Defaults to None.
        origin: The origin of the image. Defaults to None.
    """

    def __init__(
        self,
        image_data,
        device,
        is_segmentation=False,
        max_seg_label=None,
        background_seg_label=0,
        seg_preprocessor=None,
        spacing=None,
        direction=None,
        origin=None,
        center=None,
    ) -> None:

        self.device = device
        if isinstance(image_data, str):
            self.load_from_file(image_data)
        elif isinstance(image_data, sitk.Image):
            self.load_from_sitk(image_data)
        elif isinstance(image_data, np.ndarray):
            self.load_from_numpy(image_data)
        elif isinstance(image_data, torch.Tensor):
            self.load_from_torch(image_data)
        else:
            raise ValueError("Unsupported image_data type")

        self.set_dims()  # set the number of dimensions of the image

        assert self.dims in [2, 3], "Only 2D and 3D images are currently supported"

        self.preprocess()

        if is_segmentation:
            self._init_segmentation(max_seg_label, background_seg_label, seg_preprocessor)
        else:
            self._init_regular_image()

        self._init_transformations(spacing, direction, origin, center)

    def load_from_file(self, file_path: str):
        """Load an image from a file."""
        if file_path.lower().endswith(".tif") or file_path.lower().endswith(".tiff"):
            self.array = tifffile.imread(file_path)
            self.array = torch.from_numpy(self.array).to(self.device).float()
            self.itk_image = sitk.GetImageFromArray(self.array.cpu().numpy())
        else:
            itk_image = sitk.ReadImage(file_path)
            self.load_from_sitk(itk_image)

    def load_from_sitk(self, itk_image: sitk.Image):
        """Load an image from a SimpleITK image."""
        self.itk_image = itk_image
        self.dims = self.itk_image.GetDimension()
        self.array = torch.from_numpy(sitk.GetArrayFromImage(self.itk_image)).to(self.device).float()

    def load_from_numpy(self, np_array: np.ndarray):
        """Load an image from a numpy array."""
        self.array = torch.from_numpy(np_array).to(self.device).float()
        self.dims = self.array.ndim
        self.itk_image = sitk.GetImageFromArray(np_array)

    def load_from_torch(self, torch_tensor: torch.Tensor):
        """Load an image from a torch tensor."""
        self.array = torch_tensor.to(self.device).float()
        self.dims = self.array.ndim - 2
        self.itk_image = sitk.GetImageFromArray(self.array.cpu().numpy())

    def set_dims(self):
        if self.array.ndim == 2:
            self.dims = 2
        elif self.array.ndim == 3:
            self.dims = 3
        elif self.array.ndim > 3:
            self.dims = self.array.ndim - 2  # Assuming first two dims are batch and channel
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.array.ndim}")

    def preprocess(self):
        self.array = torch.nan_to_num(self.array, nan=0.0, posinf=1.0, neginf=0.0)

    def _init_regular_image(self):
        """Initialize a regular image."""
        self.array = self.array.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        self.channels = 1
        assert self.channels == 1, "Only single channel images are currently supported"

    def _init_segmentation(self, max_seg_label, background_seg_label, seg_preprocessor):
        """Initialize a segmentation mask."""
        array = torch.from_numpy(sitk.GetArrayFromImage(self.itk_image).astype(int)).to(self.device).long()
        array = seg_preprocessor(array)
        if max_seg_label is not None:
            array[array > max_seg_label] = background_seg_label
        array = integer_to_onehot(
            array, background_label=background_seg_label, max_label=max_seg_label
        ).unsqueeze(0)
        self.array = array.float()
        self.channels = array.shape[1]

    def _init_transformations(self, spacing, direction, origin, center):
        """Initialize the transformation matrices."""
        spacing = np.array(self.itk_image.GetSpacing())[None] if spacing is None else np.array(spacing)[None]
        origin = np.array(self.itk_image.GetOrigin())[None] if origin is None else np.array(origin)[None]
        direction = (
            np.array(self.itk_image.GetDirection()).reshape(self.dims, self.dims)
            if direction is None
            else np.array(direction).reshape(self.dims, self.dims)
        )

        if center is not None:
            print("Center location provided, recalibrating origin.")
            origin = (
                center
                - np.matmul(
                    direction, ((np.array(self.itk_image.GetSize()) * spacing / 2).squeeze())[:, None]
                ).T
            )

        px2phy = np.eye(self.dims + 1)
        px2phy[: self.dims, -1] = origin
        px2phy[: self.dims, : self.dims] = direction * spacing

        torch2px = np.eye(self.dims + 1)
        scaleterm = (np.array(self.itk_image.GetSize()) - 1) * 0.5
        torch2px[: self.dims, : self.dims] = np.diag(scaleterm)
        torch2px[: self.dims, -1] = scaleterm

        self.torch2phy = torch.from_numpy(np.matmul(px2phy, torch2px)).to(self.device).float().unsqueeze(0)
        self.phy2torch = torch.inverse(self.torch2phy[0]).float().unsqueeze(0)

    @classmethod
    def load_file(cls, image_path: str, *args, **kwargs) -> "Image":
        """
        Load an image from a file.
        Args:
            image_path (str): Path to the image file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Image: An Image object created from the file.
        """
        return cls(image_path, *args, **kwargs)

    @property
    def shape(self):
        """Get the shape of the image array."""
        return self.array.shape


class BatchedImages:

    def __init__(self, images: Union[Image, List[Image]]) -> None:
        """
        Initialize the BatchedImages object.

        Args:
            images (Union[Image, List[Image]]): A single Image object or a list of Image objects.

        Raises:
            ValueError: If no images are provided or if images have different shapes.
            TypeError: If any of the provided images is not an Image object.
        """
        if isinstance(images, Image):
            images = [images]
        self.images = images

        assert len(self.images) > 0, "At least one image must be provided"

        if not all(isinstance(image, Image) for image in self.images):
            raise TypeError("All images must be of type Image")

        shapes = [x.array.shape for x in self.images]
        if not all(x == shapes[0] for x in shapes):
            raise ValueError("All images must have the same shape")

        self.shape = shapes[0]
        self.n_images = len(self.images)
        self.interpolate_mode = "bilinear" if self.images[0].dims == 2 else "trilinear"

    def __call__(self) -> torch.Tensor:
        """
        Get the batch of images as a single tensor.

        Returns:
            torch.Tensor: A tensor containing all images in the batch.
        """
        return torch.cat([x.array for x in self.images], dim=0)

    @property
    def device(self):
        """Get the device of the images."""
        return self.images[0].device

    @property
    def dims(self):
        """Get the number of spatial dimensions of the images."""
        return self.images[0].dims

    def size(self):
        """Get the number of images in the batch."""
        return self.n_images

    def get_torch2phy(self):
        """Get the torch2phy transformation matrices for all images in the batch."""
        return torch.cat([x.torch2phy for x in self.images], dim=0)

    def get_phy2torch(self):
        """Get the phy2torch transformation matrices for all images in the batch."""
        return torch.cat([x.phy2torch for x in self.images], dim=0)
