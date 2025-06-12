import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file


class CleanDIFTFeatureExtractor:
    """
    Handles feature extraction using CleanDIFT and computes an initial affine transform.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.vae, self.unet = self._load_models()
        self.sift = cv2.SIFT_create()

    def _load_models(self):
        """Loads the VAE and the fine-tuned CleanDIFT U-Net."""
        print("Loading Stable Diffusion models...")
        vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae").to(
            self.device
        )
        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet").to(
            self.device
        )

        print("Downloading and loading CleanDIFT weights...")

        ckpt_path = hf_hub_download(repo_id="CompVis/cleandift", filename="cleandift_sd21_unet.safetensors")
        state_dict = load_file(ckpt_path)
        unet.load_state_dict(state_dict, strict=True)
        print("Models loaded successfully.")
        return vae, unet

    def _preprocess_image(self, image_tensor):
        """Preprocesses a single image tensor for VAE encoding."""
        # Ensure image is in [0, 1] range and correct format [B, C, H, W]
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0

        # Expects a batch, so unsqueeze if it's a single image
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        if image_tensor.shape[1] == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        # Resize to a size VAE can handle (e.g., 512x512)
        return F.interpolate(image_tensor, size=(512, 512), mode="bilinear", align_corners=False)

    @torch.no_grad()
    def get_features(self, image_tensor):
        """
        Extracts features from an image tensor using the CleanDIFT U-Net.
        Args:
            image_tensor (torch.Tensor): The input image tensor on the correct device.
        Returns:
            A list of feature maps from the U-Net's down blocks.
        """
        processed_image = self._preprocess_image(image_tensor)
        latents = self.vae.encode(processed_image).latent_dist.sample() * self.vae.config.scaling_factor

        timestep = torch.zeros(1, device=self.device).long()
        context = torch.zeros((1, 1, 1024), device=self.device)  # SD 2.1 uses 1024-dim context

        # Get features from the down-blocks
        feature_maps = []
        for block in self.unet.down_blocks:
            latents, res_samples = block(hidden_states=latents, temb=timestep, encoder_hidden_states=context)
            feature_maps.extend(res_samples)

        return feature_maps

    def get_initial_transform(self, fixed_image, moving_image, good_match_percent=0.15):
        """
        Calculates the initial affine transformation matrix from moving to fixed image.
        Args:
            fixed_image (Image): HiMReg Image object for the fixed image.
            moving_image (Image): HiMReg Image object for the moving image.
            good_match_percent (float): Percentage of top matches to use for estimation.
        Returns:
            np.ndarray: A 2x3 affine transformation matrix, or None if matching fails.
        """
        print("Extracting features for initial alignment...")
        # Get features using a mid-level feature map (e.g., 6th one)
        fixed_features = self.get_features(fixed_image())[5].squeeze()
        moving_features = self.get_features(moving_image())[5].squeeze()

        # Convert images to CV2 format (grayscale, 8-bit)
        fixed_np = (fixed_image().squeeze().cpu().numpy() * 255).astype(np.uint8)
        moving_np = (moving_image().squeeze().cpu().numpy() * 255).astype(np.uint8)

        print("Finding keypoint correspondences...")
        # Detect keypoints
        kp1, _ = self.sift.detectAndCompute(moving_np, None)  # Moving
        kp2, _ = self.sift.detectAndCompute(fixed_np, None)  # Fixed

        # Extract descriptors from CleanDIFT feature maps
        des1 = self._get_descriptors_for_keypoints(
            kp1, moving_features, moving_np.shape
        )  # Descriptors for moving

        des2 = self._get_descriptors_for_keypoints(
            kp2, fixed_features, fixed_np.shape
        )  # Descriptors for fixed

        if des1 is None or des2 is None:
            print("Warning: Could not extract descriptors for keypoints.")
            return None

        # Match descriptors
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)

        # Keep best matches
        num_good_matches = int(len(matches) * good_match_percent)
        matches = matches[:num_good_matches]

        if len(matches) < 4:
            print("Warning: Not enough good matches found to estimate transform.")
            return None

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        print(f"Found {len(matches)} good matches.")
        print("Estimating initial affine transformation...")
        # Find affine transformation
        M, _ = cv2.estimateAffinePartial2D(points1, points2)

        return M

    def _get_descriptors_for_keypoints(self, keypoints, feature_map, image_shape):
        """Extracts feature vectors from a feature map for a list of keypoints."""
        if not keypoints:
            return None

        # Scale keypoint coordinates to match feature map size
        img_h, img_w = image_shape
        feat_c, feat_h, feat_w = feature_map.shape

        descriptors = []
        for kp in keypoints:

            x_feat = int(kp.pt[0] * (feat_w / img_w))
            y_feat = int(kp.pt[1] * (feat_h / img_h))

            # Boundary check
            x_feat = min(x_feat, feat_w - 1)
            y_feat = min(y_feat, feat_h - 1)

            # Extract descriptor
            descriptor = feature_map[:, y_feat, x_feat].cpu().numpy()
            descriptors.append(descriptor)

        return np.array(descriptors, dtype=np.float32)
