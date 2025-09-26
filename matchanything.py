from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import tifffile
import torch
from transformers import AutoImageProcessor, AutoModelForKeypointMatching
import os
import argparse

Image.MAX_IMAGE_PIXELS = None


def percentile_norm(x, p_low=0.5, p_high=99.5):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return (y * 255.0).astype(np.uint8)


def load_any(path, channel_axis="auto", channels=None, level=None):
    """
    Load an image from a file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with tifffile.TiffFile(path) as tf:
            ser = tf.series[0]
            if level is not None and level < len(ser.levels):
                arr = ser.levels[level].asarray()
            else:
                arr = ser.asarray()
    else:
        img = Image.open(path)
        if img.mode == "I;16":
            arr = np.array(img, dtype=np.uint16)
        else:
            arr = np.array(img)
    if arr.ndim == 2:
        raw = arr
        gray8 = percentile_norm(arr)
        vis_rgb = np.stack([gray8, gray8, gray8], axis=-1)
    elif arr.ndim == 3:
        H, W, C_last = arr.shape
        C_first = arr.shape[0]
        use_last = (channel_axis == "last") or (channel_axis == "auto" and C_last in (1, 2, 3, 4))
        use_first = (channel_axis == "first") or (
            channel_axis == "auto" and not use_last and C_first in (1, 2, 3, 4)
        )
        if use_last:
            x = arr
        elif use_first:
            x = np.moveaxis(arr, 0, -1)
        else:
            z = arr if arr.shape[0] < arr.shape[-1] else np.moveaxis(arr, -1, 0)
            g = percentile_norm(z.max(axis=0))
            raw = z.max(axis=0)  # raw 按2D处理
            vis_rgb = np.stack([g, g, g], axis=-1)
            return raw, vis_rgb

        if channels is None:
            if x.shape[-1] == 1:
                g = percentile_norm(x[..., 0])
                raw = x[..., 0]
                vis_rgb = np.stack([g, g, g], axis=-1)
            elif x.shape[-1] >= 3:
                sel = [0, 1, 2]
                raw = x[..., : x.shape[-1]]
                c0 = percentile_norm(x[..., sel[0]])
                c1 = percentile_norm(x[..., sel[1]])
                c2 = percentile_norm(x[..., sel[2]])
                vis_rgb = np.stack([c0, c1, c2], axis=-1)
            else:
                c0 = percentile_norm(x[..., 0])
                c1 = percentile_norm(x[..., 1])
                raw = x
                vis_rgb = np.stack([c0, c1, c1], axis=-1)
        else:
            sel = [int(i) for i in channels.split(",")]
            # 不足3个就补灰度
            bands = []
            for k in range(3):
                idx = sel[k] if k < len(sel) else sel[-1]
                bands.append(percentile_norm(x[..., idx]))
            raw = x
            vis_rgb = np.stack(bands, axis=-1)
    else:
        # Higher dimensions, do maximum projection
        z = arr
        for _ in range(z.ndim - 2):
            z = z.max(axis=0)
        g = percentile_norm(z)
        raw = z
        vis_rgb = np.stack([g, g, g], axis=-1)
    return raw, vis_rgb


def resize_keep_aspect(img_rgb_u8, max_dim=None):
    if max_dim is None:
        return img_rgb_u8, 1.0
    h, w = img_rgb_u8.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 1.0:
        new_size = (int(round(w * scale)), int(round(h * scale)))
        img = cv2.resize(img_rgb_u8, new_size, interpolation=cv2.INTER_LINEAR)
        return img, scale
    return img_rgb_u8, 1.0


def estimate_transform(pts_src, pts_dst, mode="affine", ransac_thr=3.0):
    if mode == "affine":
        M, inl = cv2.estimateAffinePartial2D(
            pts_src,
            pts_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
            maxIters=5000,
            confidence=0.999,
        )
        if M is None:
            M, inl = cv2.estimateAffine2D(
                pts_src,
                pts_dst,
                method=cv2.RANSAC,
                ransacReprojThreshold=ransac_thr,
                maxIters=5000,
                confidence=0.999,
            )
        return M, inl
    else:
        H, inl = cv2.findHomography(
            pts_src,
            pts_dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_thr,
            maxIters=5000,
            confidence=0.999,
        )
        return H, inl


def to_gray_u8(x):
    x = x.astype(np.float32)
    if x.ndim == 2:
        g = x
    else:
        r, g, b = x[..., 0], x[..., 1], x[..., 2]
        g = 0.2989 * r + 0.5870 * g + 0.1140 * b
    g = percentile_norm(g)
    return g


def save_overlay_rg(fixed_raw, warped_raw, out_path):
    g0 = to_gray_u8(fixed_raw)
    g1 = to_gray_u8(warped_raw)
    h, w = g0.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[..., 0] = g0  # fixed -> 红
    overlay[..., 1] = g1  # warped -> 绿
    Image.fromarray(overlay).save(out_path)


def draw_matches_panel(img0_u8, img1_u8, k0, k1, inliers, out_path):
    s0 = img0_u8
    s1 = img1_u8
    H = max(s0.shape[0], s1.shape[0])
    canvas = np.zeros((H, s0.shape[1] + s1.shape[1], 3), dtype=np.uint8)
    canvas[: s0.shape[0], : s0.shape[1]] = s0
    canvas[: s1.shape[0], s0.shape[1] :] = s1
    off = np.array([s0.shape[1], 0], dtype=np.float32)
    if inliers is None:
        mask = np.ones((k0.shape[0],), dtype=bool)
    else:
        mask = inliers.ravel() > 0
    for i, ok in enumerate(mask):
        p0 = tuple(np.round(k0[i]).astype(int))
        p1 = tuple(np.round(k1[i] + off).astype(int))
        color = (0, 255, 0) if ok else (128, 128, 128)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, (255, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, (0, 255, 255), -1, cv2.LINE_AA)
    Image.fromarray(canvas).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True, help="参考图（fixed/target）路径，支持 .tif/.tiff/.png/.jpg 等")
    ap.add_argument("--moving", required=True, help="需扭正的 moving/source 图路径")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--mode", choices=["affine", "homography"], default="affine")
    ap.add_argument("--resize", type=int, default=1400, help="匹配阶段最长边，None=原分辨率")
    ap.add_argument("--thr", type=float, default=0.05, help="匹配置信度阈值（越小匹配越多）")
    ap.add_argument(
        "--channel_axis",
        choices=["auto", "first", "last"],
        default="auto",
        help="TIFF通道轴位置自动/首维/末维",
    )
    ap.add_argument("--channels", default=None, help="选择用于可视化与匹配的通道索引，如 '0,1,2'")
    ap.add_argument("--tiff_level", type=int, default=None, help="金字塔TIFF的level（0=最高分辨率）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    fixed_raw, fixed_vis = load_any(
        args.fixed, channel_axis=args.channel_axis, channels=args.channels, level=args.tiff_level
    )
    moving_raw, moving_vis = load_any(
        args.moving, channel_axis=args.channel_axis, channels=args.channels, level=args.tiff_level
    )

    fixed_vis_rz, sF = resize_keep_aspect(fixed_vis, args.resize)
    moving_vis_rz, sM = resize_keep_aspect(moving_vis, args.resize)

    processor = AutoImageProcessor.from_pretrained("zju-community/matchanything_eloftr")
    model = AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr").to(args.device)
    model.eval()

    inp = processor([Image.fromarray(fixed_vis_rz), Image.fromarray(moving_vis_rz)], return_tensors="pt").to(
        args.device
    )
    with torch.no_grad():
        out = model(**inp)
    img_sizes = [
        [(fixed_vis_rz.shape[0], fixed_vis_rz.shape[1]), (moving_vis_rz.shape[0], moving_vis_rz.shape[1])]
    ]
    post = processor.post_process_keypoint_matching(out, img_sizes, threshold=args.thr)[0]
    # post返回的是在“当前输入尺寸”坐标系下的点，映射回原图坐标
    k0_rz = post["keypoints0"].cpu().numpy() / sF
    k1_rz = post["keypoints1"].cpu
