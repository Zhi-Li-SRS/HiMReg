import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        resnet = models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.features(x)


def extract_features(image, feature_extractor):
    # make the image 3 channel if it is 1 channel
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)

    with torch.no_grad():
        features = feature_extractor(image)
    return features


def match_features(features1, features2):
    kp1, des1 = features_to_keypoints(features1)
    kp2, des2 = features_to_keypoints(features2)

    if len(kp1) < 2 or len(kp2) < 2:
        return [], [], []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches, kp1, kp2


def features_to_keypoints(features):
    """Make keypoints and descriptors from features."""
    feature_map = features.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    gray = cv2.normalize(feature_map, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # Find the strongest corners in the image
    corners = cv2.goodFeaturesToTrack(
        np.max(gray, axis=2), maxCorners=1000, qualityLevel=0.01, minDistance=10
    )

    if corners is not None:
        corners = corners.reshape(-1, 2)
        kp = [
            cv2.KeyPoint(x=float(corner[0]), y=float(corner[1]), size=1)
            for corner in corners
        ]
        des = feature_map[corners[:, 1].astype(int), corners[:, 0].astype(int)]
    else:
        kp = []
        des = np.array([])

    return kp, des


def estimate_affine(kp1, kp2, good_matches):
    if len(good_matches) < 4:
        return None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    return M


class DeepFeatureRegistration:
    def __init__(self, device="cuda"):
        self.device = device
        self.feature_extractor = FeatureExtractor().to(device)
        self.feature_extractor.eval()

    def register(self, fixed_image, moving_image):

        fixed_features = extract_features(fixed_image, self.feature_extractor)
        moving_features = extract_features(moving_image, self.feature_extractor)

        matches, kp1, kp2 = match_features(fixed_features, moving_features)

        M = estimate_affine(kp1, kp2, matches)
        if M is None:
            print("Not enough matches found to estimate affine transformation")
            return moving_image, np.eye(3)[:2]

        height, width = fixed_image.shape[2:]
        moved_image = cv2.warpAffine(
            moving_image.squeeze().cpu().numpy(), M, (width, height)
        )
        moved_image = (
            torch.from_numpy(moved_image).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        return moved_image, torch.from_numpy(M).to(self.device)
