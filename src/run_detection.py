import os
import cv2
import numpy as np
from infocus_detection import detect_infocus_mask

DATA_DIR = "data"
SIGMAS   = (0.0, 0.75, 2.0)

filenames = os.listdir(DATA_DIR)
for idx, fname in enumerate(filenames):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    if fname.split(".")[0].lower().endswith("_depth"):
        continue

    stem   = os.path.splitext(fname)[0]
    image  = cv2.imread(os.path.join(DATA_DIR, fname))
    depth  = np.load(os.path.join(DATA_DIR, f"{stem}_depth.npy"))

    debug_dir = os.path.join(DATA_DIR, f"{stem}_debug")
    mask = detect_infocus_mask(image, depth,
                               sigmas=SIGMAS,
                               nbins=20,
                               debug_dir=debug_dir)

    results_dir = os.path.join(DATA_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    cv2.imwrite(os.path.join(results_dir, f"{stem}_infocus_mask.jpg"), np.uint8(mask) * 255)
    print('{} processed'.format(stem))
