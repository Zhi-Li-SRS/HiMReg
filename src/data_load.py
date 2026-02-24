import numpy as np
import SimpleITK as sitk
import tifffile
import torch

from src.preprocess import build_registration_view
from src.utils import integer_to_onehot


class Image:
    """Load the image data and perform preprocessing.
    Args:
        image_data (Union[str, sitk.Image, np.ndarray, torch.Tensor, List]): The image data to load.
        device (torch.device): The device to load the image data on.
        is_segmentation (bool): Whether the image is a segmentation mask. Defaults to False.
        max_seg_label (int): The maximum segmentation label. Defaults to None.
        background_seg_label (int): The background segmentation label. Defaults to 0.
        seg_preprocessor: A function to preprocess the segmentation mask. Defaults to None.
        spacing: The pixel spacing of the image. Defaults to None.
        origin: The origin of the image. Defaults to None.
    """

    def __init__(
        self,
        image_data,
        device,
        preprocessing=None,
        is_segmentation=False,
        max_seg_label=None,
        background_seg_label=0,
        seg_preprocessor=None,
        spacing=None,
        direction=None,
        origin=None,
    ) -> None:
        self._device = device if isinstance(device, torch.device) else torch.device(device)
        if isinstance(image_data, list):
            self.images = [
                self._load_single_image(
                    img,
                    device,
                    preprocessing,
                    is_segmentation,
                    max_seg_label,
                    background_seg_label,
                    seg_preprocessor,
                    spacing,
                    direction,
                    origin,
                )
                for img in image_data
            ]
            self._validate_images()
        else:
            self.images = [
                self._load_single_image(
                    image_data,
                    device,
                    preprocessing,
                    is_segmentation,
                    max_seg_label,
                    background_seg_label,
                    seg_preprocessor,
                    spacing,
                    direction,
                    origin,
                )
            ]

        self._shape = self.images[0].shape  # Change to private attribute
        self.n_images = len(self.images)
        self.interpolate_mode = "bilinear" if self.images[0].dims == 2 else "trilinear"

    def _load_single_image(
        self,
        image_data,
        device,
        preprocessing,
        is_segmentation,
        max_seg_label,
        background_seg_label,
        seg_preprocessor,
        spacing,
        direction,
        origin,
    ):
        image = SingleImage(
            image_data,
            device,
            preprocessing,
            is_segmentation,
            max_seg_label,
            background_seg_label,
            seg_preprocessor,
            spacing,
            direction,
            origin,
        )
        return image

    def _validate_images(self):
        assert len(self.images) > 0, "At least one image must be provided"
        shapes = [x.shape for x in self.images]
        if not all(x == shapes[0] for x in shapes):
            raise ValueError("All images must have the same shape")

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
        return self._device

    @device.setter
    def device(self, value):
        """Set the device of the images."""
        self._device = value
        if hasattr(self, "images"):
            for image in self.images:
                image.device = value

    @property
    def dims(self):
        """Get the number of spatial dimensions of the images."""
        return self.images[0].dims

    def size(self):
        """Get the number of images in the batch."""
        return self.n_images

    def get_pixel_to_physical(self):
        """Get the pixel_to_physical transformation matrices for all images in the batch."""
        return torch.cat([x.pixel_to_physical for x in self.images], dim=0)

    def get_physical_to_pixel(self):
        """Get the physical_to_pixel transformation matrices for all images in the batch."""
        return torch.cat([x.physical_to_pixel for x in self.images], dim=0)

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
        return self._shape


class SingleImage:
    """Load a single image data and perform preprocessing."""

    def __init__(
        self,
        image_data,
        device,
        preprocessing=None,
        is_segmentation=False,
        max_seg_label=None,
        background_seg_label=0,
        seg_preprocessor=None,
        spacing=None,
        direction=None,
        origin=None,
    ) -> None:
        self._device = device
        self.preprocessing = dict(preprocessing or {})
        self._source_spacing = None
        self._source_origin = None
        self._source_direction = None
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

        self.preprocess(is_segmentation=is_segmentation)
        self.set_dims()

        assert self.dims in [2, 3], "Only 2D and 3D images are currently supported"

        if is_segmentation:
            self._init_segmentation(max_seg_label, background_seg_label, seg_preprocessor)
        else:
            self._init_regular_image()

        self._init_transformations(spacing, direction, origin)

    def load_from_file(self, file_path: str):
        """Load an image from a file."""
        if file_path.lower().endswith(".tif") or file_path.lower().endswith(".tiff"):
            self.array = tifffile.imread(file_path)
            self.array = torch.from_numpy(np.asarray(self.array)).to(self.device).float()
            self.itk_image = sitk.GetImageFromArray(self.array.detach().cpu().numpy())
        else:
            itk_image = sitk.ReadImage(file_path)
            self.load_from_sitk(itk_image)

    def load_from_sitk(self, itk_image: sitk.Image):
        """Load an image from a SimpleITK image."""
        self.itk_image = itk_image
        self._source_spacing = tuple(self.itk_image.GetSpacing())
        self._source_origin = tuple(self.itk_image.GetOrigin())
        self._source_direction = tuple(self.itk_image.GetDirection())
        self.array = torch.from_numpy(sitk.GetArrayFromImage(self.itk_image)).to(self.device).float()

    def load_from_numpy(self, np_array: np.ndarray):
        """Load an image from a numpy array."""
        self.array = torch.from_numpy(np.asarray(np_array)).to(self.device).float()
        self.itk_image = sitk.GetImageFromArray(np.asarray(np_array))

    def load_from_torch(self, torch_tensor: torch.Tensor):
        """Load an image from a torch tensor."""
        self.array = torch_tensor.to(self.device).float()
        self.itk_image = sitk.GetImageFromArray(self.array.detach().cpu().numpy())

    def set_dims(self):
        if self.array.ndim == 2:
            self.dims = 2
        elif self.array.ndim == 3:
            self.dims = 3
        elif self.array.ndim > 3:
            self.dims = self.array.ndim - 2  # Assuming first two dims are batch and channel
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.array.ndim}")

    def preprocess(self, is_segmentation=False):
        self.array = torch.nan_to_num(self.array, nan=0.0, posinf=1.0, neginf=0.0)

        if is_segmentation:
            return

        array_np = self.array.detach().cpu().numpy()
        array_np = build_registration_view(array_np, self.preprocessing)

        self.array = torch.from_numpy(array_np).to(self.device).float()
        self.itk_image = sitk.GetImageFromArray(array_np.astype(np.float32))
        self._copy_source_metadata_if_compatible()

    def _copy_source_metadata_if_compatible(self):
        if self._source_spacing is None or self._source_origin is None:
            return

        if self.array.ndim == 2:
            spacing = tuple(self._source_spacing[:2])
            origin = tuple(self._source_origin[:2])
            direction = self._extract_2d_direction(self._source_direction)
            self.itk_image.SetSpacing(spacing)
            self.itk_image.SetOrigin(origin)
            self.itk_image.SetDirection(direction)
        elif self.array.ndim == 3 and len(self._source_spacing) >= 3 and len(self._source_origin) >= 3:
            self.itk_image.SetSpacing(tuple(self._source_spacing[:3]))
            self.itk_image.SetOrigin(tuple(self._source_origin[:3]))
            if self._source_direction is not None and len(self._source_direction) >= 9:
                self.itk_image.SetDirection(tuple(self._source_direction[:9]))

    @staticmethod
    def _extract_2d_direction(direction):
        if direction is None:
            return (1.0, 0.0, 0.0, 1.0)
        direction = tuple(direction)
        if len(direction) == 4:
            return direction
        if len(direction) >= 9:
            return (direction[0], direction[1], direction[3], direction[4])
        return (1.0, 0.0, 0.0, 1.0)

    def _init_regular_image(self):
        """Initialize a regular image."""
        if self.array.ndim == 2:
            self.array = self.array.unsqueeze(0).unsqueeze(0)
            self.channels = 1
        elif self.array.ndim == 3:
            self.array = self.array.unsqueeze(0).unsqueeze(0)
            self.channels = 1
        elif self.array.ndim == 4:
            self.channels = self.array.shape[1]
        else:
            raise ValueError(f"Unexpected tensor rank for regular image: {self.array.shape}")

    def _init_segmentation(self, max_seg_label, background_seg_label, seg_preprocessor):
        """Initialize a segmentation mask."""
        array = torch.from_numpy(sitk.GetArrayFromImage(self.itk_image).astype(int)).to(self.device).long()
        if seg_preprocessor is not None:
            array = seg_preprocessor(array)
        if max_seg_label is not None:
            array[array > max_seg_label] = background_seg_label
        array = integer_to_onehot(
            array, background_label=background_seg_label, max_label=max_seg_label
        ).unsqueeze(0)
        self.array = array.float()
        self.channels = array.shape[1]

    def _init_transformations(self, spacing, direction, origin):
        """Initialize the transformation matrices."""
        try:
            dims = self.dims
            spacing = (
                np.array(self.itk_image.GetSpacing())[None] if spacing is None else np.array(spacing)[None]
            )
            origin = np.array(self.itk_image.GetOrigin())[None] if origin is None else np.array(origin)[None]
            direction = (
                np.array(self.itk_image.GetDirection()).reshape(dims, dims)
                if direction is None
                else np.array(direction).reshape(dims, dims)
            )

            pixel_to_physical = np.eye(dims + 1)
            pixel_to_physical[:dims, -1] = origin
            pixel_to_physical[:dims, :dims] = direction * spacing

            physical_to_pixel = np.eye(dims + 1)
            scaleterm = (np.array(self.itk_image.GetSize()) - 1) * 0.5
            physical_to_pixel[:dims, :dims] = np.diag(scaleterm)
            physical_to_pixel[:dims, -1] = scaleterm

            pixel_to_physical = torch.from_numpy(np.matmul(pixel_to_physical, physical_to_pixel)).float()
            physical_to_pixel = torch.inverse(pixel_to_physical)
            self.pixel_to_physical = pixel_to_physical.to(self.device).unsqueeze(0)
            self.physical_to_pixel = physical_to_pixel.to(self.device).unsqueeze(0)

        except RuntimeError as e:
            print(f"Warning: CUDA error in transformation initialization. Falling back to CPU. Error: {e}")
            self.pixel_to_physical = torch.eye(self.dims + 1).float().unsqueeze(0)
            self.physical_to_pixel = torch.eye(self.dims + 1).float().unsqueeze(0)

    @property
    def device(self):
        return self._device

    @property
    def shape(self):
        """Get the shape of the image array."""
        return self.array.shape
