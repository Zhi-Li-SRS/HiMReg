#!/usr/bin/env python
"""Generate publication-quality figures from benchmark results.
Usage:
    python scripts/make_figures.py --results-dir comparison_results/benchmark
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np

# Optional — only needed for overlay PNGs
try:
    import tifffile
except ImportError:
    tifffile = None

# ── Global matplotlib config (Nature Comms style) ──
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
try:
    matplotlib.rcParams["font.family"] = "Arial"
    font_manager.findfont("Arial", fallback_to_default=False)
except Exception:
    matplotlib.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


# ── Colors (saturated, distinct, cool→warm) ──
COLORS = {
    "Before": "#5AADBB",  # teal — cool baseline
    "Elastix": "#F5D06E",  # warm gold — mid-tier
    "HiMReg": "#E05B4B",  # coral red — protagonist
}
METHOD_ORDER = ["Before", "Elastix", "HiMReg"]

# Metric metadata: (csv_key, display_name_with_arrow, higher_is_better)
METRICS = [
    ("MI", "MI \u2191", True),
    ("NMI", "NMI \u2191", True),
    ("GradNCC", "GradNCC \u2191", True),
    ("GradSSIM", "GradSSIM \u2191", True),
    ("TissueDice", "TissueDice \u2191", True),
    ("TissueIoU", "TissueIoU \u2191", True),
    ("HD95", "HD95 \u2193", False),
    ("ASSD", "ASSD \u2193", False),
]


# ── Image helpers ──


def robust_normalize(image: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


def blue_yellow_overlay(fixed_np: np.ndarray, moving_np: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    fixed = robust_normalize(fixed_np)
    moving = robust_normalize(moving_np)
    base = 0.18 * fixed
    r = np.clip(base + alpha * moving, 0.0, 1.0)
    g = np.clip(base + alpha * moving, 0.0, 1.0)
    b = np.clip(base + fixed, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def colorize_blue(image_np: np.ndarray) -> np.ndarray:
    img = robust_normalize(image_np)
    base = 0.18 * img
    r = base
    g = base
    b = np.clip(base + img, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def colorize_yellow(image_np: np.ndarray, alpha: float = 0.85) -> np.ndarray:
    img = robust_normalize(image_np)
    r = np.clip(alpha * img, 0.0, 1.0)
    g = np.clip(alpha * img, 0.0, 1.0)
    b = np.zeros_like(img)
    return np.stack([r, g, b], axis=-1)


def save_image_no_border(arr: np.ndarray, path: Path, cmap: str | None = None) -> None:
    h, w = arr.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if cmap:
        ax.imshow(arr, cmap=cmap)
    else:
        ax.imshow(arr)
    ax.axis("off")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ── Overlay PNGs ──


def generate_overlay_pngs(results_dir: Path, roi_ids: List[str]) -> None:
    if tifffile is None:
        print("[WARN] tifffile not installed, skipping overlay PNGs")
        return

    for roi in roi_ids:
        roi_dir = results_dir / roi
        if not roi_dir.exists():
            print(f"[WARN] {roi_dir} not found, skipping")
            continue

        fixed_path = roi_dir / "fixed_processed.tif"
        if not fixed_path.exists():
            print(f"[WARN] {fixed_path} not found, skipping {roi}")
            continue

        fixed_np = tifffile.imread(fixed_path)
        moving_np = tifffile.imread(roi_dir / "moving_scale_synced.tif")
        before_np = tifffile.imread(roi_dir / "moving_before_resampled.tif")
        himreg_np = tifffile.imread(roi_dir / "himreg_warped.tif")

        baseline_path = roi_dir / "elastix_warped.tif"
        if not baseline_path.exists():
            baseline_path = roi_dir / "simpleitk_warped.tif"
        if not baseline_path.exists():
            baseline_path = roi_dir / "baseline_warped.tif"
        baseline_np = tifffile.imread(baseline_path)

        out_dir = results_dir / "figures" / roi
        out_dir.mkdir(parents=True, exist_ok=True)

        save_image_no_border(colorize_blue(fixed_np), out_dir / "he_fixed.png")
        save_image_no_border(colorize_yellow(moving_np), out_dir / "srs_791_moving.png")
        save_image_no_border(blue_yellow_overlay(fixed_np, before_np), out_dir / "overlay_before.png")
        save_image_no_border(blue_yellow_overlay(fixed_np, himreg_np), out_dir / "overlay_himreg.png")
        save_image_no_border(blue_yellow_overlay(fixed_np, baseline_np), out_dir / "overlay_elastix.png")

        print(f"  {roi}: 5 images saved to {out_dir}")


# ── Metrics CSV ──


def load_metrics(results_dir: Path) -> List[Dict]:
    csv_path = results_dir / "metrics_all_cases.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for row in rows:
        if row["method"] == "SimpleITK":
            row["method"] = "Elastix"
    return rows


# ── Metrics SVG ──


def generate_metrics_svg(results_dir: Path, rows: List[Dict]) -> None:
    import seaborn as sns

    fig, axes = plt.subplots(2, 4, figsize=(7.09, 4.0))
    axes = axes.flatten()

    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    rng = np.random.default_rng(42)

    for idx, (key, title, _higher_better) in enumerate(METRICS):
        ax = axes[idx]
        means, stds, colors, scatter_vals, labels = [], [], [], [], []

        for method in methods:
            vals = np.array([float(r[key]) for r in rows if r["method"] == method], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            means.append(vals.mean())
            stds.append(vals.std(ddof=0))
            colors.append(COLORS.get(method, "#777777"))
            scatter_vals.append(vals)
            labels.append(method)

        x = np.arange(len(means))

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            capsize=3,
            color=colors,
            alpha=0.75,
            edgecolor=[_darken(c, 0.7) for c in colors],
            linewidth=0.5,
            width=0.65,
            zorder=2,
            error_kw={"elinewidth": 0.8, "capthick": 0.8, "color": "black"},
        )

        ax.grid(axis="y", color="#DDDDDD", linewidth=0.4, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
        for sep in [i + 0.5 for i in range(len(means) - 1)]:
            ax.axvline(sep, color="#DDDDDD", linewidth=0.4, linestyle=":", zorder=0)

        for xi, vals, c in zip(x, scatter_vals, colors):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(xi + jitter, vals, marker="D", s=20, c=c, edgecolors="none", alpha=0.75, zorder=4)

        for xi, mean_val in zip(x, means):
            ax.text(
                xi,
                mean_val * 0.02,
                f"{mean_val:.3f}" if mean_val < 10 else f"{mean_val:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="#333333",
                fontweight="medium",
                zorder=5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=0, ha="center")
        ax.set_title(title, fontsize=8, pad=4)
        ax.tick_params(axis="y", labelsize=6.5, width=0.6, length=2)
        ax.tick_params(axis="x", width=0, length=0)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_linewidth(0.5)
        sns.despine(ax=ax)

    fig.tight_layout()

    out_path = results_dir / "figures" / "metrics.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Metrics figure saved to {out_path}")


# ── Legend SVG (one row, standalone) ──


def generate_legend_svg(results_dir: Path) -> None:
    """Generate a standalone one-row legend SVG matching the metrics chart style."""
    methods = METHOD_ORDER
    colors = [COLORS[m] for m in methods]

    fig, ax = plt.subplots(figsize=(3.5, 0.4))
    ax.set_axis_off()

    # Create invisible bar + scatter handles for legend
    handles = []
    for method, color in zip(methods, colors):
        bar = ax.bar(
            [0], [0], color=color, alpha=0.75, edgecolor=_darken(color, 0.7), linewidth=0.5, label=method
        )
        handles.append(bar)

    legend = ax.legend(
        handles=handles,
        labels=methods,
        loc="center",
        ncol=len(methods),
        frameon=False,
        fontsize=8,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.5,
    )

    out_path = results_dir / "figures" / "legend.svg"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Legend saved to {out_path}")


def _darken(hex_color: str, factor: float = 0.7) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from benchmark results")
    parser.add_argument("--results-dir", type=str, required=True, help="Path to benchmark output directory")
    parser.add_argument("--roi-ids", type=str, default="roi2,roi3,roi4,roi5", help="Comma-separated ROI ids")
    parser.add_argument(
        "--skip-overlays",
        action="store_true",
        help="Skip overlay PNG generation (when .tif files are missing)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    roi_ids = [r.strip() for r in args.roi_ids.split(",")]

    if not args.skip_overlays:
        print("=== Generating individual overlay PNGs ===")
        generate_overlay_pngs(results_dir, roi_ids)
    else:
        print("=== Skipping overlay PNGs (--skip-overlays) ===")

    print("\n=== Generating metrics SVG ===")
    rows = load_metrics(results_dir)
    generate_metrics_svg(results_dir, rows)

    print("\n=== Generating legend SVG ===")
    generate_legend_svg(results_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
