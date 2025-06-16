import os
import cv2
import numpy as np
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

if __name__ == '__main__':
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
        print("____________________________________________")
        print("processing image {}/{}".format(i+1, len(image_filenames)))
        image_path = os.path.join(dir_path, image_filename)

        # Extract image file name without extension
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        image = cv2.imread(image_path)
        depth = np.load(dir_path + f'/{image_name}_depth.npy')  # This loads the original floating-point depth map

        h, w = image.shape[:2]
        assert (h, w) == depth.shape[:2]

        debug_path = dir_path + '/' + image_name + '_debug/'
        os.makedirs(debug_path, exist_ok=True)

        # Convert to grayscale (optional but often done for DoG)
        gray_image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(debug_path + '01_gray_image.jpg', gray_image)

        # Pyramid of Gaussians
        pogs = []
        for level, sigma in enumerate([0.0, 0.75, 2.0]):
            gaussian = gray_image.copy()
            if sigma > 0.0:
                gaussian = cv2.GaussianBlur(gaussian, (15, 15), sigmaX=sigma, sigmaY=sigma)
            pogs.append(gaussian)
            cv2.imwrite(debug_path + '02_gaussian_pyramid_level{}.jpg'.format(level), gaussian)

        # Difference of Gaussians
        dogs = []
        for level in range(1, len(pogs)):
            gaussian_prev = pogs[level - 1]
            gaussian_next = pogs[level]
            dog = np.abs(gaussian_next - gaussian_prev)
            dogs.append(dog)
            cv2.imwrite(debug_path + '03_difference_of_gaussian_level{}.jpg'.format(level), 10 * dog)

        dog = dogs[0]
        # Do non-maximum suppression (NMS)
        # Find local maxima using maximum filter
        local_max = maximum_filter(dog, size=3, mode='constant')
        # Only keep pixels that are equal to the local maximum
        local_extremums = dog == local_max
        cv2.imwrite(debug_path + '04_dog_extremums_after_nms.jpg', 255*local_extremums)

        extremum_values = dog[local_extremums]

        # Compute percentiles
        percentiles = [1, 25, 50, 75, 99]
        percentile_values = np.percentile(extremum_values, percentiles)

        # Print to console
        print("Extremum values percentiles:")
        for p, val in zip(percentiles, percentile_values):
            print(f"{p}%: {val:.4f}")

        # Plot distribution
        plt.figure(figsize=(10, 6))
        plt.hist(extremum_values, bins=50, density=True, alpha=0.7, color='blue')
        plt.title('Distribution of DoG Extremum Values After NMS')
        plt.xlabel('DoG Response Value')
        plt.ylabel('Density')

        # Add percentile markers
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        for p, val, col in zip(percentiles, percentile_values, colors):
            plt.axvline(val, color=col, linestyle='--',
                        label=f'{p}%: {val:.2f}')

        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save plot
        plot_path = os.path.join(debug_path, '04_dog_extremums_after_nms_plot_values.jpg')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        local_extremums = dog == local_max
        extremum_threshold = np.percentile(extremum_values, 99)
        cv2.imwrite(debug_path + '04_dog_extremums_after_nms_bigger_than_threshold.jpg', 255 * np.logical_and(local_extremums, dog >= extremum_threshold))

        print(f"Saved extremum values plot to: {plot_path}")

        dog_next = dogs[1]
        # Do NMS w.r.t. next DoG level
        next_local_max = maximum_filter(dog_next, size=3, mode='constant')
        local_next_extremums = dog >= next_local_max
        cv2.imwrite(debug_path + '05_dog_extremums_wrt_next_level.jpg', 255 * local_next_extremums)

        points_of_focus = np.logical_and(local_extremums, local_next_extremums)
        cv2.imwrite(debug_path + '06_points_of_focus_mask.jpg', 255 * points_of_focus)

        points_of_focus = np.logical_and(np.logical_and(local_extremums, dog >= extremum_threshold), local_next_extremums)
        cv2.imwrite(debug_path + '07_points_of_focus_mask_wrt_bigger_than_threshold.jpg', 255 * points_of_focus)

        # Get depth values at focus points
        focus_depth_values = depth[points_of_focus]

        nbins = 20

        # Calculate depth range and bin edges
        min_depth = np.min(depth)
        max_depth = np.max(depth)
        bin_edges = np.linspace(min_depth, max_depth, num=nbins + 1)

        # Count points in each bin
        bin_counts, _ = np.histogram(focus_depth_values, bins=bin_edges)

        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(range(nbins), bin_counts, width=0.8, color='skyblue', edgecolor='navy')

        # Customize plot
        plt.title(f'Focus Points Distribution Across {nbins} Depth Bins')
        plt.xlabel(f'Depth Bins (Range: {min_depth:.2f} to {max_depth:.2f})')
        plt.ylabel('Number of Focus Points')
        plt.xticks(range(nbins), [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(nbins)], rotation=45)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on top of bars
        for i, count in enumerate(bin_counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom')

        plt.tight_layout()

        print(f"Saved depth bin distribution plot to: {plot_path}")
        print(f"Bin edges: {bin_edges}")
        print(f"Points per bin: {bin_counts}")

        infocus_center_bin = np.argmax(bin_counts)
        print(infocus_center_bin)

        extremum_votes = bin_counts[infocus_center_bin]
        votes_threshold = extremum_votes / 10

        # Find first bin (searching backwards from center)
        infocus_first_bin = None
        for i in range(infocus_center_bin, -1, -1):
            if bin_counts[i] >= votes_threshold:
                infocus_first_bin = i
            else:
                break  # Stop when we find a bin below threshold

        # Find last bin (searching forwards from center)
        infocus_last_bin = None
        for i in range(infocus_center_bin, len(bin_counts)):
            if bin_counts[i] >= votes_threshold:
                infocus_last_bin = i
            else:
                break  # Stop when we find a bin below threshold

        bars = plt.bar(range(nbins), bin_counts, width=0.8, color='skyblue', edgecolor='navy')
        # Highlight the in-focus region
        for i in range(infocus_first_bin, infocus_last_bin + 1):
            bars[i].set_facecolor('salmon')  # Highlight in-focus bins
            bars[i].set_edgecolor('red')

        # Save the plot
        plot_path = debug_path + '08_focus_points_depth_distribution_bins.jpg'
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        cv2.imwrite(debug_path + '09_depth.jpg', depth)

        # Create a color-coded depth map
        normalized_depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_colored = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_OCEAN)

        # Create mask for in-focus depth range
        in_focus_mask = np.logical_and(depth >= bin_edges[infocus_first_bin],
                                       depth <= bin_edges[infocus_last_bin + 1])

        # Highlight in-focus regions in red
        depth_colored[in_focus_mask] = [0, 0, 255]  # Pure red (BGR format)

        # Save the result
        cv2.imwrite(debug_path + '10_depth_with_infocus_highlight.jpg', depth_colored)

        # Version 1: Only show in-focus regions (black out out-of-focus)
        image_infocus_part = image.copy()
        image_infocus_part[~in_focus_mask] = 0  # Black out non-focus areas
        cv2.imwrite(debug_path + '11_image_infocus_part.jpg', image_infocus_part)

        image_out_of_focus_part = image.copy()
        image_out_of_focus_part[in_focus_mask] = 0  # Black out in-focus areas
        cv2.imwrite(debug_path + '12_image_defocus_part.jpg', image_out_of_focus_part)