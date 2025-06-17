import os

import einops
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from scipy import ndimage
from skimage import feature, morphology
from torchvision.transforms.functional import to_tensor

# Setup loguru logging
logger.remove()  # Remove default handler
logger.add(
    "cleandift.log",
    format="{time:YYYY-MM-DD HH:mm} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
)
logger.add(
    lambda msg: print(msg, end=""),
    format="{time:HH:mm:ss} | {level} | {message}",
    level="INFO",
    colorize=True,
)


def load_model():
    """Load and initialize the CleanDIFT model."""
    logger.info("Loading model configuration...")
    config_dir = "cleandift_configs"
    cfg_model = OmegaConf.load(os.path.join(config_dir, "sd21_feature_extractor.yaml"))[
        "model"
    ]

    logger.info("Instantiating model...")
    cfg_model = hydra.utils.instantiate(cfg_model)
    model = cfg_model.cuda().bfloat16()

    logger.info("Loading model weights...")
    ckpt_pth = hf_hub_download(
        repo_id="CompVis/cleandift", filename="cleandift_sd21_full.safetensors"
    )
    state_dict = load_file(ckpt_pth)
    model.load_state_dict(state_dict, strict=True)
    model = model.eval()

    logger.success("Model loaded successfully")
    return model


def create_tissue_mask(image):
    """Create a binary mask to identify meaningful tissue regions with texture/detail."""

    gray = np.array(image.convert("L"))

    # Method 1: Texture-based masking using local standard deviation
    kernel_size = 15
    local_mean = ndimage.uniform_filter(gray.astype(float), size=kernel_size)
    local_sq_mean = ndimage.uniform_filter(gray.astype(float) ** 2, size=kernel_size)
    local_variance = local_sq_mean - local_mean**2
    local_variance = np.maximum(local_variance, 0)
    local_std = np.sqrt(local_variance)
    texture_threshold = np.percentile(local_std, 75)  # Top 25% most textured areas
    texture_mask = local_std > texture_threshold
    logger.debug(
        f"Texture threshold: {texture_threshold:.2f}, texture ratio: {np.mean(texture_mask):.3f}"
    )

    # Method 2: Intensity-based masking (avoid pure black and pure white)
    intensity_mask = (gray > np.percentile(gray, 5)) & (gray < np.percentile(gray, 95))

    # Method 3: Edge-based masking (areas with strong edges = structure)
    edges = feature.canny(gray, sigma=2, low_threshold=0.1, high_threshold=0.2)
    edge_regions = morphology.binary_dilation(edges, morphology.disk(8))

    logger.debug(f"Edge regions ratio: {np.mean(edge_regions):.3f}")

    combined_mask = texture_mask & intensity_mask & edge_regions

    # If combined mask is too restrictive, fall back to texture + intensity
    if np.mean(combined_mask) < 0.1:
        logger.debug("Combined mask too restrictive, using texture + intensity")
        combined_mask = texture_mask & intensity_mask

    # If still too restrictive, use just texture
    if np.mean(combined_mask) < 0.05:
        logger.debug("Still too restrictive, using texture only")
        combined_mask = texture_mask

    mask = morphology.binary_opening(combined_mask, morphology.disk(3))
    mask = morphology.binary_closing(mask, morphology.disk(5))
    mask = morphology.remove_small_objects(mask, min_size=50)

    final_ratio = np.mean(mask)
    logger.debug(f"Final meaningful tissue ratio: {final_ratio:.3f}")

    return mask


def convert_tiff_to_rgb(img):
    """Convert TIFF image to RGB, handling various formats."""
    logger.debug(f"Converting image with mode: {img.mode}")

    if img.mode in ["I", "I;16", "I;16B"]:
        img_array = np.array(img)
        if img_array.max() > 255:
            img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
    elif img.mode == "F":
        img_array = np.array(img)
        img_array = (img_array / img_array.max() * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")

    if img.mode != "RGB":
        img = img.convert("RGB")

    return img


def apply_tissue_mask_to_image(image, mask):
    """Apply tissue mask to image, setting non-tissue regions to black."""
    img_array = np.array(image)

    mask_3d = np.stack([mask, mask, mask], axis=-1)
    masked_img_array = img_array * mask_3d

    return Image.fromarray(masked_img_array.astype(np.uint8))


def load_and_preprocess_images(source_path, target_path, image_size=(768, 768)):
    """Load and preprocess images with tissue masking."""
    logger.info("Loading images...")

    # Load original images
    img_source_orig = Image.open(source_path)
    img_target_orig = Image.open(target_path)

    logger.info(f"Source: {img_source_orig.mode}, {img_source_orig.size}")
    logger.info(f"Target: {img_target_orig.mode}, {img_target_orig.size}")

    # Convert to RGB
    img_source = convert_tiff_to_rgb(img_source_orig)
    img_target = convert_tiff_to_rgb(img_target_orig)

    # Create tissue masks
    logger.info("Creating tissue masks...")
    mask_source = create_tissue_mask(img_source)
    mask_target = create_tissue_mask(img_target)

    # Apply masks to images for feature extraction
    logger.info("Applying tissue masks for feature extraction...")
    img_source_masked = apply_tissue_mask_to_image(img_source, mask_source)
    img_target_masked = apply_tissue_mask_to_image(img_target, mask_target)

    # Store original sizes
    original_sizes = {"source": img_source.size, "target": img_target.size}

    # Resize both original and masked images
    img_source_resized = img_source.resize(image_size)  # For visualization
    img_target_resized = img_target.resize(image_size)  # For visualization
    img_source_masked_resized = img_source_masked.resize(
        image_size
    )  # For feature extraction
    img_target_masked_resized = img_target_masked.resize(
        image_size
    )  # For feature extraction

    # Convert masked images to tensors for feature extraction
    img_source_tensor = (
        to_tensor(img_source_masked_resized)[None].to("cuda") * 2 - 1
    )  # dimension [1, 3, 768, 768]
    img_target_tensor = (
        to_tensor(img_target_masked_resized)[None].to("cuda") * 2 - 1
    )  # dimension [1, 3, 768, 768]

    return {
        "source_tensor": img_source_tensor,
        "target_tensor": img_target_tensor,
        "source_img": img_source_resized,  # Original for visualization
        "target_img": img_target_resized,  # Original for visualization
        "source_img_masked": img_source_masked_resized,  # Masked for visualization
        "target_img_masked": img_target_masked_resized,  # Masked for visualization
        "source_img_orig": img_source,
        "target_img_orig": img_target,
        "original_sizes": original_sizes,
        "masks": {"source": mask_source, "target": mask_target},
    }


def extract_features(model, images, caption="A photo", feat_key="us6"):
    """Extract features from preprocessed images."""
    logger.info(f"Extracting features using {feat_key}...")

    with torch.no_grad():
        source_features = model.get_features(
            images["source_tensor"].bfloat16(), [caption], t=None, feat_key=feat_key
        )
        target_features = model.get_features(
            images["target_tensor"].bfloat16(), [caption], t=None, feat_key=feat_key
        )

    logger.info(
        f"Feature map size: {source_features.shape[-1]} x {source_features.shape[-2]}"
    )
    return source_features, target_features


def find_point_correspondences(
    source_features, target_features, source_points, images, image_size=(768, 768)
):
    """Find correspondences for specific source points."""
    logger.info("Finding point correspondences...")

    original_source_size = images["original_sizes"]["source"]

    # Scale source points from original to resized coordinates
    scale_x = image_size[0] / original_source_size[0]
    scale_y = image_size[1] / original_source_size[1]
    source_points_scaled = [
        [int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in source_points
    ]

    # Scale to feature map coordinates
    feature_h, feature_w = source_features.shape[-2:]
    feat_scale_x = feature_w / image_size[0]
    feat_scale_y = feature_h / image_size[1]
    source_points_feat = [
        [int(pt[0] * feat_scale_x), int(pt[1] * feat_scale_y)]
        for pt in source_points_scaled
    ]

    logger.debug(f"Scaled points to feature space: {source_points_feat}")

    # Extract features for source points
    source_points_feat_tensor = (
        torch.tensor(source_points_feat).to(source_features.device).long()
    )
    source_point_feats = source_features[
        0, :, source_points_feat_tensor[:, 1], source_points_feat_tensor[:, 0]
    ].T[:, None]

    # Find matches
    target_features_norm_flat = einops.rearrange(
        target_features / target_features.norm(p=2, dim=1, keepdim=True),
        "1 c h w -> c (h w)",
    )
    source_point_feats_norm = source_point_feats / source_point_feats.norm(
        p=2, dim=-1, keepdim=True
    )
    target_feat_sims = einops.rearrange(
        source_point_feats_norm @ target_features_norm_flat,
        "b 1 (h w) -> b h w",
        h=target_features.shape[-2],
    )
    matches = torch.stack(
        torch.unravel_index(
            einops.rearrange(target_feat_sims, "b h w -> b (h w)").argmax(dim=-1),
            target_feat_sims.shape[1:],
        )
    ).T

    # Scale matches back to display coordinates
    matches_scaled = matches.float()
    matches_scaled[:, 0] = matches_scaled[:, 0] * image_size[1] / feature_h
    matches_scaled[:, 1] = matches_scaled[:, 1] * image_size[0] / feature_w

    return source_points_scaled, matches_scaled, target_feat_sims


def find_mutual_correspondences(
    source_features, target_features, images, similarity_threshold=0.75, top_k=50
):
    """Find mutual correspondences between feature maps, ensuring points are in meaningful tissue regions."""
    logger.info("Finding mutual correspondences...")

    # Get tissue masks
    mask_source = images["masks"]["source"]
    mask_target = images["masks"]["target"]

    # Resize masks to feature map size
    feature_h, feature_w = source_features.shape[-2:]
    mask_source_resized = (
        np.array(
            Image.fromarray(mask_source.astype(np.uint8) * 255).resize(
                (feature_w, feature_h), Image.NEAREST
            )
        )
        > 128
    )
    mask_target_resized = (
        np.array(
            Image.fromarray(mask_target.astype(np.uint8) * 255).resize(
                (feature_w, feature_h), Image.NEAREST
            )
        )
        > 128
    )

    logger.info(
        f"Meaningful tissue coverage in feature maps - Source: {np.mean(mask_source_resized):.3f}, Target: {np.mean(mask_target_resized):.3f}"
    )

    # Convert masks to flat indices
    mask_source_flat = mask_source_resized.flatten()
    mask_target_flat = mask_target_resized.flatten()

    # Get tissue pixel indices
    source_tissue_indices = np.where(mask_source_flat)[0]
    target_tissue_indices = np.where(mask_target_flat)[0]

    if len(source_tissue_indices) == 0 or len(target_tissue_indices) == 0:
        logger.warning("No meaningful tissue regions detected in feature maps!")
        return []

    logger.info(
        f"Meaningful tissue pixels - Source: {len(source_tissue_indices)}, Target: {len(target_tissue_indices)}"
    )

    # Extract features only from tissue regions
    feat1_flat = source_features.squeeze(0).view(
        source_features.size(1), -1
    )  # [C, H*W]
    feat2_flat = target_features.squeeze(0).view(
        target_features.size(1), -1
    )  # [C, H*W]

    # Get features only from tissue pixels
    feat1_tissue = feat1_flat[:, source_tissue_indices]  # [C, tissue_pixels1]
    feat2_tissue = feat2_flat[:, target_tissue_indices]  # [C, tissue_pixels2]

    # Additional quality filter: remove features with low variance (uniform regions)
    feat1_variance = torch.var(feat1_tissue, dim=0)
    feat2_variance = torch.var(feat2_tissue, dim=0)

    # Keep only features with above-median variance (more distinctive features)
    feat1_var_threshold = torch.median(feat1_variance)
    feat2_var_threshold = torch.median(feat2_variance)

    good_feat1_mask = feat1_variance > feat1_var_threshold
    good_feat2_mask = feat2_variance > feat2_var_threshold

    if good_feat1_mask.sum() < 10 or good_feat2_mask.sum() < 10:
        logger.warning(
            "Not enough distinctive features found, using all tissue features"
        )
        good_feat1_mask = torch.ones_like(feat1_variance, dtype=torch.bool)
        good_feat2_mask = torch.ones_like(feat2_variance, dtype=torch.bool)

    # Filter features and corresponding indices
    feat1_filtered = feat1_tissue[:, good_feat1_mask]
    feat2_filtered = feat2_tissue[:, good_feat2_mask]
    source_indices_filtered = source_tissue_indices[good_feat1_mask.cpu().numpy()]
    target_indices_filtered = target_tissue_indices[good_feat2_mask.cpu().numpy()]

    logger.info(
        f"High-quality tissue features - Source: {len(source_indices_filtered)}, Target: {len(target_indices_filtered)}"
    )

    # Normalize features
    feat1_norm = F.normalize(feat1_filtered, dim=0)
    feat2_norm = F.normalize(feat2_filtered, dim=0)

    logger.info(
        f"Computing similarity matrix: {feat1_norm.shape[1]} x {feat2_norm.shape[1]}"
    )

    # Compute similarity matrix
    similarity = torch.mm(
        feat1_norm.T, feat2_norm
    )  # [filtered_tissue_pixels1, filtered_tissue_pixels2]

    # Find all potential matches above threshold
    high_similarity_mask = similarity > similarity_threshold
    similarity_indices = torch.where(high_similarity_mask)

    if len(similarity_indices[0]) == 0:
        logger.warning(
            f"No correspondences found above threshold {similarity_threshold}, lowering threshold..."
        )
        similarity_threshold = 0.65
        high_similarity_mask = similarity > similarity_threshold
        similarity_indices = torch.where(high_similarity_mask)

        if len(similarity_indices[0]) == 0:
            logger.warning(
                "Still no correspondences found, using top matches regardless of threshold"
            )
            # Take top 50 matches regardless of threshold
            flat_similarity = similarity.flatten()
            top_indices = torch.topk(
                flat_similarity, min(50, len(flat_similarity)), largest=True
            )
            similarity_indices = torch.unravel_index(
                top_indices.indices, similarity.shape
            )

    # Get similarity scores for high-similarity matches
    similarity_scores = similarity[similarity_indices[0], similarity_indices[1]]

    # Sort by similarity score and take top_k
    sorted_indices = torch.argsort(similarity_scores, descending=True)[:top_k]

    # Get the best matches
    best_source_indices = similarity_indices[0][sorted_indices]
    best_target_indices = similarity_indices[1][sorted_indices]
    best_scores = similarity_scores[sorted_indices]

    # Convert back to mutual correspondences format
    mutual_matches = []

    for i in range(len(best_source_indices)):
        source_filtered_idx = best_source_indices[i].item()
        target_filtered_idx = best_target_indices[i].item()
        score = best_scores[i].item()

        # Convert filtered indices back to 2D coordinates
        source_flat_idx = source_indices_filtered[source_filtered_idx]
        target_flat_idx = target_indices_filtered[target_filtered_idx]

        y1, x1 = divmod(source_flat_idx, feature_w)
        y2, x2 = divmod(target_flat_idx, feature_w)

        mutual_matches.append(([x1, y1], [x2, y2], score))

    logger.success(
        f"Selected top {len(mutual_matches)} high-quality correspondences from meaningful tissue regions (similarity range: {best_scores[-1]:.3f} - {best_scores[0]:.3f})"
    )
    return mutual_matches


def visualize_point_correspondences(
    images, source_points_scaled, matches_scaled, target_feat_sims
):
    """Visualize individual point correspondences."""
    logger.info("Creating point correspondence visualizations...")

    for pt_idx in range(len(source_points_scaled)):
        plt.figure(figsize=(15, 5))

        # Use original images for better visibility
        plt.subplot(131)
        plt.imshow(images["source_img"])
        plt.scatter(
            *source_points_scaled[pt_idx],
            edgecolor="white",
            linewidth=2,
            color="lime",
            s=20,
        )
        plt.axis("off")
        plt.title("Source Point", fontsize=14, fontweight="bold")

        plt.subplot(132)
        plt.imshow(target_feat_sims[pt_idx].cpu().float(), cmap="coolwarm")
        plt.colorbar(shrink=0.8)
        plt.axis("off")
        plt.title("Similarity Map", fontsize=14, fontweight="bold")

        plt.subplot(133)
        plt.imshow(images["target_img"])
        plt.scatter(
            *matches_scaled[pt_idx].flip(-1).float().cpu().squeeze().tolist(),
            edgecolor="white",
            linewidth=2,
            color="orange",
            s=20,
        )
        plt.axis("off")
        plt.title("Target Match", fontsize=14, fontweight="bold")

        plt.tight_layout()
        filename = f"cleandift_source_points_point_{pt_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved {filename}")


def visualize_mutual_correspondences(
    images, mutual_matches, source_features, target_features
):
    """Visualize mutual correspondences with connecting lines."""
    if len(mutual_matches) == 0:
        logger.warning("No mutual correspondences to visualize")
        return

    logger.info("Creating mutual correspondences visualization...")

    # Create figure with side-by-side layout
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Select top matches by similarity
    top_matches = sorted(mutual_matches, key=lambda x: x[2], reverse=True)[:150]

    source_pts = [match[0] for match in top_matches]
    target_pts = [match[1] for match in top_matches]
    scores = [match[2] for match in top_matches]

    # Scale points to image size
    scale_x_source = images["source_img"].size[0] / source_features.shape[-1]
    scale_y_source = images["source_img"].size[1] / source_features.shape[-2]
    scale_x_target = images["target_img"].size[0] / target_features.shape[-1]
    scale_y_target = images["target_img"].size[1] / target_features.shape[-2]

    source_pts_scaled = [
        [int(pt[0] * scale_x_source), int(pt[1] * scale_y_source)] for pt in source_pts
    ]
    target_pts_scaled = [
        [int(pt[0] * scale_x_target), int(pt[1] * scale_y_target)] for pt in target_pts
    ]

    # Source image with keypoints
    ax1.imshow(images["source_img"])
    ax1.scatter(
        [pt[0] for pt in source_pts_scaled],
        [pt[1] for pt in source_pts_scaled],
        c=scores,
        cmap="hot",
        s=20,
        alpha=0.8,
        edgecolors="white",
        linewidth=1,
    )
    ax1.set_title("Source Image with Keypoints", fontsize=14, fontweight="bold")
    ax1.axis("off")

    # Target image with keypoints
    ax2.imshow(images["target_img"])
    scatter2 = ax2.scatter(
        [pt[0] for pt in target_pts_scaled],
        [pt[1] for pt in target_pts_scaled],
        c=scores,
        cmap="hot",
        s=20,
        alpha=0.8,
        edgecolors="white",
        linewidth=1,
    )
    ax2.set_title("Target Image with Keypoints", fontsize=14, fontweight="bold")
    ax2.axis("off")

    # Add colorbar
    plt.colorbar(scatter2, ax=ax2, shrink=0.8, label="Similarity Score")

    plt.tight_layout()
    filename = "cleandift_mutual_correspondences.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved {filename}")

    # Create a second figure showing correspondences with connecting lines
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))

    # Get image dimensions
    img_width = images["source_img"].size[0]

    # Create concatenated image (source on left, target on right)
    source_array = np.array(images["source_img"])
    target_array = np.array(images["target_img"])

    # Concatenate images horizontally
    combined_img = np.concatenate([source_array, target_array], axis=1)
    ax.imshow(combined_img)

    # Plot source points
    source_x = [pt[0] for pt in source_pts_scaled]
    source_y = [pt[1] for pt in source_pts_scaled]

    # Plot target points (shifted by image width)
    target_x = [pt[0] + img_width for pt in target_pts_scaled]
    target_y = [pt[1] for pt in target_pts_scaled]

    # Plot points
    ax.scatter(
        source_x,
        source_y,
        c=scores,
        cmap="hot",
        s=25,
        alpha=0.9,
        edgecolors="white",
        linewidth=1,
        label="Source",
    )
    ax.scatter(
        target_x,
        target_y,
        c=scores,
        cmap="hot",
        s=25,
        alpha=0.9,
        edgecolors="white",
        linewidth=1,
        label="Target",
    )

    # Draw connecting lines
    for i in range(len(source_pts_scaled)):
        # Color lines based on similarity score
        line_color = plt.cm.hot(scores[i])
        ax.plot(
            [source_x[i], target_x[i]],
            [source_y[i], target_y[i]],
            color=line_color,
            alpha=0.6,
            linewidth=0.8,
        )

    # Add vertical line to separate images
    ax.axvline(x=img_width, color="white", linestyle="--", linewidth=2, alpha=0.7)

    # Add labels
    ax.text(
        img_width // 2,
        -30,
        "Source Image",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="white",
    )
    ax.text(
        img_width + img_width // 2,
        -30,
        "Target Image",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        color="white",
    )

    ax.set_title(
        "Mutual Correspondences with Connecting Lines",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.axis("off")

    # Add colorbar for the lines
    sm = plt.cm.ScalarMappable(
        cmap="hot", norm=plt.Normalize(vmin=min(scores), vmax=max(scores))
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=30)
    cbar.set_label("Similarity Score", fontsize=12)

    plt.tight_layout()
    filename_lines = "cleandift_mutual_correspondences_with_lines.png"
    plt.savefig(filename_lines, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved {filename_lines}")


def save_tissue_mask_visualization(images):
    """Save visualization of tissue masks overlaid on original images."""
    logger.info("Creating tissue mask visualizations...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Source image and mask
    axes[0, 0].imshow(images["source_img"])
    axes[0, 0].set_title("Source Image", fontsize=12, fontweight="bold")
    axes[0, 0].axis("off")

    # Source mask
    mask_source_resized = np.array(
        Image.fromarray(images["masks"]["source"].astype(np.uint8) * 255).resize(
            images["source_img"].size
        )
    )
    axes[0, 1].imshow(mask_source_resized, cmap="hot")
    axes[0, 1].set_title(
        "Source Meaningful Tissue Mask", fontsize=12, fontweight="bold"
    )
    axes[0, 1].axis("off")

    # Source mask overlay
    source_array = np.array(images["source_img"])
    mask_overlay = np.zeros_like(source_array)
    mask_overlay[:, :, 0] = mask_source_resized  # Red channel for mask
    blended_source = 0.7 * source_array + 0.3 * mask_overlay
    axes[0, 2].imshow(blended_source.astype(np.uint8))
    axes[0, 2].set_title("Source + Mask Overlay", fontsize=12, fontweight="bold")
    axes[0, 2].axis("off")

    # Target image and mask
    axes[1, 0].imshow(images["target_img"])
    axes[1, 0].set_title("Target Image", fontsize=12, fontweight="bold")
    axes[1, 0].axis("off")

    # Target mask
    mask_target_resized = np.array(
        Image.fromarray(images["masks"]["target"].astype(np.uint8) * 255).resize(
            images["target_img"].size
        )
    )
    axes[1, 1].imshow(mask_target_resized, cmap="hot")
    axes[1, 1].set_title(
        "Target Meaningful Tissue Mask", fontsize=12, fontweight="bold"
    )
    axes[1, 1].axis("off")

    # Target mask overlay
    target_array = np.array(images["target_img"])
    mask_overlay_target = np.zeros_like(target_array)
    mask_overlay_target[:, :, 0] = mask_target_resized  # Red channel for mask
    blended_target = 0.7 * target_array + 0.3 * mask_overlay_target
    axes[1, 2].imshow(blended_target.astype(np.uint8))
    axes[1, 2].set_title("Target + Mask Overlay", fontsize=12, fontweight="bold")
    axes[1, 2].axis("off")

    plt.tight_layout()
    filename = "meaningful_tissue_masks_visualization.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    logger.success(f"Saved {filename}")


def main():
    """Main function to run the CleanDIFT correspondence finding."""
    logger.info("🚀 Starting CleanDIFT correspondence analysis")

    # Configuration
    source_path = "./data/K21/NADH-4.tif"
    target_path = "./data/K21/postaf-4.tif"
    source_points = [[2307, 783], [544, 904], [1284, 832], [2332, 1356], [1852, 2092]]
    image_size = (768, 768)

    try:
        # Load model
        model = load_model()
        images = load_and_preprocess_images(source_path, target_path, image_size)

        # Save tissue mask visualization to check quality
        save_tissue_mask_visualization(images)

        # Extract features (using masked images)
        source_features, target_features = extract_features(model, images)
        source_points_scaled, matches_scaled, target_feat_sims = (
            find_point_correspondences(
                source_features, target_features, source_points, images, image_size
            )
        )
        visualize_point_correspondences(
            images, source_points_scaled, matches_scaled, target_feat_sims
        )

        # Find mutual correspondences
        mutual_matches = find_mutual_correspondences(
            source_features, target_features, images
        )
        visualize_mutual_correspondences(
            images, mutual_matches, source_features, target_features
        )

        logger.success("✅ Analysis completed successfully")

    except Exception as e:
        logger.error(f"❌ Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()
