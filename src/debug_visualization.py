"""
debug_visualization.py
Utility functions for saving intermediate results and plots produced by
the in-focus‐mask pipeline.

Author: you
"""
from __future__ import annotations
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ──────────────────────────  shared helpers  ────────────────────────── #

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save(path: str, img: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path))
    cv2.imwrite(path, img)


# ──────────────────────────  single-image dumps  ────────────────────── #

def gray(image: np.ndarray, debug_dir: str) -> None:
    _save(f"{debug_dir}/01_gray_image.jpg", image)


def gaussian(level: int, img: np.ndarray, debug_dir: str) -> None:
    _save(f"{debug_dir}/02_gaussian_pyramid_level{level}.jpg", img)


def dog(level: int, img: np.ndarray, debug_dir: str) -> None:
    # multiply by 10 just for visibility – same as original script
    _save(f"{debug_dir}/03_difference_of_gaussian_level{level}.jpg", 10 * img)


def extremum_mask(mask: np.ndarray, name: str, debug_dir: str) -> None:
    _save(f"{debug_dir}/04_{name}.jpg", 255 * mask.astype(np.uint8))


def depth_snapshot(depth: np.ndarray, debug_dir: str) -> None:
    _save(f"{debug_dir}/09_depth.jpg", depth)


def depth_colored(depth_vis: np.ndarray, debug_dir: str) -> None:
    _save(f"{debug_dir}/10_depth_with_infocus_highlight.jpg", depth_vis)


def image_part(name: str, img: np.ndarray, debug_dir: str) -> None:
    _save(f"{debug_dir}/{name}.jpg", img)


# ──────────────────────────  matplotlib plots  ─────────────────────── #

def plot_dog_distribution(ext_vals: np.ndarray,
                          percentiles: list[int],
                          debug_dir: str) -> None:
    pct_vals = np.percentile(ext_vals, percentiles)
    plt.figure(figsize=(10, 6))
    plt.hist(ext_vals, bins=50, density=True, alpha=0.7)
    for p, v in zip(percentiles, pct_vals):
        plt.axvline(v, linestyle="--", label=f"{p}%: {v:.2f}")
    plt.legend(); plt.grid(True, alpha=.3)
    plt.title("Distribution of DoG extremum values (after NMS)")
    plt.xlabel("DoG response"); plt.ylabel("Density")
    _ensure_dir(debug_dir)
    plt.savefig(f"{debug_dir}/04_dog_extremums_distribution.jpg",
                bbox_inches="tight", dpi=300)
    plt.close()


def plot_depth_bins(bin_counts: np.ndarray,
                    bin_edges: np.ndarray,
                    focus_span: tuple[int, int],
                    debug_dir: str) -> None:
    nbins = len(bin_counts)
    first, last = focus_span
    bars = plt.bar(range(nbins), bin_counts)
    for i in range(first, last + 1):
        bars[i].set_facecolor("salmon")
        bars[i].set_edgecolor("red")

    plt.title(f"Focus-point depth distribution ({nbins} bins)")
    plt.xlabel("Depth bin"); plt.ylabel("# focus pts")
    plt.xticks(range(nbins),
               [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
                for i in range(nbins)],
               rotation=45)
    plt.grid(axis="y", alpha=.3)
    for i, c in enumerate(bin_counts):
        plt.text(i, c + 0.5, str(c), ha="center")

    _ensure_dir(debug_dir)
    plt.tight_layout()
    plt.savefig(f"{debug_dir}/08_focus_points_depth_distribution_bins.jpg",
                bbox_inches="tight", dpi=300)
    plt.close()
