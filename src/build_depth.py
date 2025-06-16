import os
import cv2
import torch
import numpy as np

from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(DEVICE)

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl'  # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()

    dir_path = '../data'

    # List all files in directory (non-recursive)
    all_files = os.listdir(dir_path)

    # Filter image files without '_depth' suffix
    image_filenames = [
        f for f in all_files
        if os.path.isfile(os.path.join(dir_path, f))  # Ensure it's a file, not subdirectory
           and f.lower().endswith(('.jpg', '.jpeg', '.png'))
           and not f.lower().split('.')[0].endswith('_depth')  # Exclude files with '_depth' before extension
    ]

    for i, image_filename in enumerate(image_filenames):
        print("processing #{}/{}...".format(i+1, len(image_filenames)))
        image_path = os.path.join(dir_path, image_filename)

        # Extract image file name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        raw_img = cv2.imread(image_path)
        depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

        # Save depth map as image
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255
        cv2.imwrite(dir_path + f'/{image_name}_depth.jpg', depth_normalized)

        # Save as numpy array (preserves floating-point values and shape)
        np.save(dir_path + f'/{image_name}_depth.npy', depth)