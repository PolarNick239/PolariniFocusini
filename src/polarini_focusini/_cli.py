"""CLI entry point for the `polarini-focusini` console script."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm          # auto picks the right backend (tty / notebook)

from .infocus_detection import detect_infocus_mask     # re-use your pipeline

VALID_EXTS = (".jpg", ".jpeg", ".png")


def process_dir(indir: Path, outdir: Path,
                sigmas=(0.0, 0.75, 2.0), nbins: int = 20) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # gather files first so tqdm knows the total
    files = [
        f for f in sorted(indir.iterdir())
        if f.suffix.lower() in VALID_EXTS
    ]

    # TODO add progress bar
    for fname in tqdm(files, desc="Generating masks", unit="img"):
        img = cv2.imread(str(fname))
        mask = detect_infocus_mask(img, sigmas=sigmas, nbins=nbins)

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
    p.add_argument("--sigmas", default="0.0,0.75,2.0",
                   help="Comma-separated Gaussian sigmas (default: %(default)s)")
    p.add_argument("--nbins",  type=int, default=20,
                   help="Depth-histogram bins (default: %(default)s)")
    args = p.parse_args()

    sigmas = [float(s) for s in args.sigmas.split(",")]
    process_dir(args.input_dir, args.output_dir, sigmas, args.nbins)


if __name__ == "__main__":
    main()
