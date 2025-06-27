"""CLI entry point for the `polarini-focusini` console script."""
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm.auto import tqdm          # auto picks the right backend (tty / notebook)

from polarini_focusini.infocus_detection import detect_infocus_mask

VALID_EXTS = (".jpg", ".jpeg", ".png")


def process_dir(indir: Path, outdir: Path,
                limit_with_circles_around_focus_points: bool = False,
                sigmas=(0.0, 0.75, 2.0), nbins: int = 120,
                debug_root: Optional[Path] = None,
                ignore_cuda: bool = False,
                verbose: bool = False) -> None:
    """
    Process all images in *indir* and write masks to *outdir*.
    If *debug_root* is given, per-image sub-folders with detailed debug
    artefacts will be created inside it.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    if debug_root:
        debug_root.mkdir(parents=True, exist_ok=True)

    # gather files first so tqdm knows the total
    files = [
        f for f in sorted(indir.iterdir())
        if f.suffix.lower() in VALID_EXTS
    ]

    for fname in tqdm(files, desc="Generating masks", unit="img"):
        img = cv2.imread(str(fname))

        # one sub-folder per image:  <debug_root>/<cat>/
        dbg_dir = (debug_root / fname.stem) if debug_root else None
        mask = detect_infocus_mask(
            img,
            limit_with_circles_around_focus_points=limit_with_circles_around_focus_points,
            sigmas=sigmas,
            nbins=nbins,
            debug_dir=str(dbg_dir) if dbg_dir else None,
            ignore_cuda=ignore_cuda,
            verbose=verbose,
        )

        out_name = outdir / f"{fname.stem}_mask.png"
        cv2.imwrite(str(out_name), np.uint8(mask) * 255)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="polarini-focusini",
        description="Generate in-focus masks for all images in a directory."
    )
    p.add_argument("input_dir",  type=Path,
                   help="Directory with input JPG/PNG images")
    p.add_argument("output_dir", type=Path,
                   help="Directory to place resulting *_mask.png files")
    p.add_argument("--limit-with-circles", action="store_true",
                   help="Limit mask to circular regions around focus points (optional refinement step)")
    p.add_argument("--sigmas", default="0.0,0.75,2.0",
                   help="Comma-separated Gaussian sigmas (default: %(default)s)")
    p.add_argument("--nbins",  type=int, default=120,
                   help="Depth-histogram bins (default: %(default)s)")
    p.add_argument("-d", "--debug-dir", type=Path, default=None,
                   help="Root folder for debug artefacts (omit to disable)")
    p.add_argument("--ignore-cuda", action="store_true",
                   help="Run inference on CPU, ignoring CUDA (default: try to use CUDA if available)")
    p.add_argument("--verbose", action="store_true",
                   help="Add verbose output, includes profiling timings for processing steps")
    args = p.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    process_dir(args.input_dir, args.output_dir,
                limit_with_circles_around_focus_points=args.limit_with_circles, sigmas=sigmas, nbins=args.nbins,
                debug_root=args.debug_dir, ignore_cuda=args.ignore_cuda, verbose=args.verbose)


if __name__ == "__main__":
    main()
