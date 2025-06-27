"""
infocus_detection.py
Pipeline for detecting the in-focus mask given an RGB image and its depth map.

Author: you
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Optional
from scipy.ndimage import maximum_filter

from . import debug_visualization as dbg   # local import – same folder
from .depth_estimation import _estimate_depth


# ──────────────────────────  core algorithm  ───────────────────────── #

def _generate_pog(gray: np.ndarray,
                  sigmas: list[float]) -> list[np.ndarray]:
    """Return Pyramid-of-Gaussians (one level per sigma)."""
    pogs: list[np.ndarray] = []
    for sigma in sigmas:
        if sigma > 0.0:
            pogs.append(cv2.GaussianBlur(gray, (15, 15), sigma))
        else:
            pogs.append(gray.copy())
    return pogs


def _generate_dog(pogs: list[np.ndarray]) -> list[np.ndarray]:
    """Return Difference-of-Gaussians between consecutive PoG levels."""
    return [np.abs(pogs[i + 1] - pogs[i]) for i in range(len(pogs) - 1)]


def _detect_extremums(dog0: np.ndarray,
                      dog1: np.ndarray,
                      percentile: float = 99.,
                      debug_dir: Optional[str] = None) -> tuple[np.ndarray, float]:
    """Non-maximum suppression across scale; return boolean mask + thresh."""
    local_max  = maximum_filter(dog0, size=3, mode='constant')
    same_scale = dog0 == local_max

    next_scale = dog0 >= maximum_filter(dog1, size=3, mode='constant')
    extremums_wrt_next_scale = np.logical_and(same_scale, next_scale)

    ext_vals  = dog0[same_scale]
    thresh    = np.percentile(ext_vals, percentile)
    extremums = extremums_wrt_next_scale & (dog0 >= thresh)

    return extremums, thresh


def _vote_depth_bins(depth: np.ndarray,
                     focus_mask: np.ndarray,
                     nbins: int = 120) -> tuple[np.ndarray, np.ndarray,
                                               tuple[int, int]]:
    """Histogram votes → return counts, edges, and winning [first,last] bins."""
    min_d, max_d = depth.min(), depth.max()
    edges = np.linspace(min_d, max_d, nbins + 1)
    counts, _ = np.histogram(depth[focus_mask], bins=edges)

    # Blur values in counts
    from scipy.ndimage import gaussian_filter1d
    counts = gaussian_filter1d(counts.astype(float), sigma=3.0)

    center = int(np.argmax(counts))
    threshold = counts[center] / 10

    first = center
    while first - 1 >= 0 and counts[first - 1] >= threshold:
        first -= 1
    last = center
    while last + 1 < nbins and counts[last + 1] >= threshold:
        last += 1
    return counts, edges, (first, last)


def _mask_from_bins(depth: np.ndarray,
                    edges: np.ndarray,
                    span: tuple[int, int]) -> np.ndarray:
    first, last = span
    return np.logical_and(depth >= edges[first],
                          depth <= edges[last + 1])

def _prune_by_focus_votes(mask: np.ndarray,
                          focus_pts: np.ndarray,
                          *,
                          min_votes: int = 10,
                          connectivity: int = 8) -> np.ndarray:
    """
    Keeps only those connected components within *mask*
    that received votes from ≥ *min_votes* points in *focus_pts*.

    connectivity    : 4 or 8 — how many neighbors to consider
    """
    if not np.any(mask):
        return mask                     # nothing to filter

    # 1) component labels: labels ∈ {0..N}, where 0 is background
    num, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity)

    # 2) count votes: each focus point casts +1 vote
    #    for the component it belongs to
    votes = np.bincount(labels[focus_pts], minlength=num)  # shape = (num,)

    # 3) determine which components to keep
    keep = votes >= min_votes
    keep[0] = False                     # never keep the background

    # 4) create the final pruned mask
    pruned_mask = keep[labels]
    return pruned_mask


def _mask_from_focus_circles(focus_pts: np.ndarray,
                             radius: int) -> np.ndarray:
    """
    Return boolean mask where every True pixel lies inside
    a filled circle of given *radius* around any True pixel in *focus_pts*.
    """
    # Morphological dilation with an elliptical kernel is faster
    # than drawing cv2.circle in a loop.
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    dilated = cv2.dilate(focus_pts.astype(np.uint8), se)
    return dilated.astype(bool)


# ──────────────────────────  public API  ───────────────────────────── #

def detect_infocus_mask(image: np.ndarray,
                        depth: Optional[np.ndarray] = None,
                        *,
                        sigmas: list[float] = (0.0, 0.75, 2.0),
                        nbins: int = 120,
                        debug_dir: Optional[str] = None) -> np.ndarray:
    """
    Detect the in-focus region of *image*.

    Parameters
    ----------
    image      : BGR uint8 image (H×W×3)
    depth      : float32 depth map (H×W) – if **None**, it will be
                 estimated on-the-fly with `_estimate_depth`
    sigmas     : Gaussian sigmas for the PoG
    nbins      : histogram bins for depth voting
    debug_dir  : if given, all intermediate artefacts are written here

    Returns
    -------
    in_focus_mask : bool ndarray of shape H×W
    """
    # 0) obtain or verify depth-map
    if depth is None:
        depth = _estimate_depth(image)
    else:
        assert depth.shape == image.shape[:2], \
            "Image and depth map sizes must match"

    # 1) grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if debug_dir:
        dbg.gray(gray, debug_dir)

    # 2) PoG
    pogs = _generate_pog(gray, list(sigmas))
    if debug_dir:
        for lvl, g in enumerate(pogs):
            dbg.gaussian(lvl, g, debug_dir)

    # 3) DoG
    dogs = _generate_dog(pogs)
    if debug_dir:
        for lvl, d in enumerate(dogs):
            dbg.dog(lvl + 1, d, debug_dir)         # level indices start at 1

    # 4) extremums (scale-space NMS)
    focus_pts, thresh = _detect_extremums(dogs[0], dogs[1], 99., debug_dir)
    if debug_dir:
        dbg.extremum_mask(focus_pts, "voting_points_mask", debug_dir)
        dbg.plot_dog_distribution(dogs[0][dogs[0] == dogs[0]],   # same mask
                                  [1, 25, 50, 75, 99],
                                  debug_dir)

    # 5) vote depth bins
    counts, edges, span = _vote_depth_bins(depth, focus_pts, nbins)
    if debug_dir:
        dbg.plot_depth_bins(counts, edges, span, debug_dir)

    # 6) produce initial mask
    mask_after_per_depth_bins_voting = _mask_from_bins(depth, edges, span)

    # 6b) draw circles around each focus point and intersect with mask
    height, width = mask_after_per_depth_bins_voting.shape
    radius = max(1, max(width, height) // 20)  # 5% of image size
    circles_mask = _mask_from_focus_circles(focus_pts, radius)
    mask_after_circles_around_focus_points = mask_after_per_depth_bins_voting & circles_mask  # final refined mask

    # 6c) prune mask by voting of focus points - to improve results when depth estimation is incorrect, f.e. DSCF5538.JPG
    mask_after_pruning = _prune_by_focus_votes(mask_after_circles_around_focus_points, focus_pts, min_votes=10)
    mask = mask_after_pruning

    # 7) optional extra visual outputs
    if debug_dir:
        dbg.mask_after_per_depth_bins_voting(mask_after_per_depth_bins_voting, "01_mask_after_per_depth_bins_voting", debug_dir)
        dbg.mask_after_per_depth_bins_voting(mask_after_circles_around_focus_points, "02_circles_around_focus_points", debug_dir)
        dbg.mask_after_per_depth_bins_voting(mask_after_pruning, "03_mask_after_pruning", debug_dir)
        dbg.depth_snapshot(depth.astype(np.uint16), debug_dir)

        # colourise depth + highlight focus bins
        norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_color = cv2.applyColorMap(norm, cv2.COLORMAP_OCEAN)
        depth_color[mask] = (0, 0, 255)            # red for focus
        dbg.depth_colored(depth_color, debug_dir)

        # split image parts
        img_focus = image.copy();  img_focus[~mask] = 0
        img_defoc = image.copy();  img_defoc[mask]  = 0
        dbg.image_part("12_image_infocus_part", img_focus, debug_dir)
        dbg.image_part("13_image_defocus_part", img_defoc, debug_dir)

    return mask
