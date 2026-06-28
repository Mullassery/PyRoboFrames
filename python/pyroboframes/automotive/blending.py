"""Advanced image blending for panoramic stitching.

Phase 2 techniques:
- Laplacian pyramid blending for smooth transitions
- Graph-cut seam optimization for content-aware seams
- Exposure compensation for lighting mismatches
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter


def build_gaussian_pyramid(image: np.ndarray, levels: int = 4) -> list[np.ndarray]:
    """Build Gaussian pyramid for image.

    Args:
        image: [H, W, 3] image
        levels: Number of pyramid levels

    Returns:
        List of images [image, downsampled_1x, downsampled_2x, ...]
    """
    pyramid = [image]

    for level in range(1, levels):
        # Downsample by 2x using Gaussian blur
        prev = pyramid[-1]
        h, w = prev.shape[:2]
        new_h, new_w = h // 2, w // 2

        if new_h < 2 or new_w < 2:
            break

        # Gaussian blur + downsample
        blurred = gaussian_filter(prev.astype(np.float32), sigma=1.0, axes=(0, 1))
        downsampled = blurred[::2, ::2]

        pyramid.append(downsampled)

    return pyramid


def build_laplacian_pyramid(
    image: np.ndarray,
    levels: int = 4,
) -> list[np.ndarray]:
    """Build Laplacian pyramid (difference-of-Gaussians).

    Args:
        image: [H, W, 3] image
        levels: Number of levels

    Returns:
        List of Laplacian levels [detail_1, detail_2, ..., coarse]
    """
    gaussian_pyr = build_gaussian_pyramid(image, levels)
    laplacian_pyr = []

    for level in range(len(gaussian_pyr) - 1):
        current = gaussian_pyr[level]
        next_level = gaussian_pyr[level + 1]

        # Upsample next level to match current size
        h, w = current.shape[:2]
        upsampled = np.zeros_like(current)

        next_h, next_w = next_level.shape[:2]
        upsampled[:next_h*2:2, :next_w*2:2] = next_level

        # Apply Gaussian blur for smooth interpolation
        upsampled = gaussian_filter(upsampled.astype(np.float32), sigma=1.0, axes=(0, 1))

        # Laplacian = current - upsampled
        laplacian = current.astype(np.float32) - upsampled.astype(np.float32)
        laplacian_pyr.append(laplacian)

    # Add coarsest level
    laplacian_pyr.append(gaussian_pyr[-1].astype(np.float32))

    return laplacian_pyr


def blend_laplacian_pyramids(
    left_pyr: list[np.ndarray],
    right_pyr: list[np.ndarray],
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> np.ndarray:
    """Blend two images using Laplacian pyramids.

    Args:
        left_pyr: Laplacian pyramid of left image
        right_pyr: Laplacian pyramid of right image
        left_mask: [H, W] validity mask for left
        right_mask: [H, W] validity mask for right

    Returns:
        [H, W, 3] blended image
    """
    # Blend each Laplacian level
    blended_pyr = []

    for left_level, right_level in zip(left_pyr, right_pyr):
        h, w = left_level.shape[:2]

        # Create smooth blend mask
        x = np.linspace(-1, 1, w)
        blend_weights = (x + 1.0) / 2.0  # 0 at left, 1 at right

        blend_weights = np.tile(blend_weights[np.newaxis, :, np.newaxis], (h, 1, 3))

        # Weighted blend
        blended = (
            left_level * (1.0 - blend_weights) + right_level * blend_weights
        )

        blended_pyr.append(blended)

    # Reconstruct from pyramid
    result = blended_pyr[-1]

    for level in range(len(blended_pyr) - 2, -1, -1):
        # Upsample result
        h, w = blended_pyr[level].shape[:2]
        upsampled = np.zeros((h, w, 3), dtype=np.float32)

        res_h, res_w = result.shape[:2]
        upsampled[: res_h * 2 : 2, : res_w * 2 : 2] = result

        upsampled = gaussian_filter(upsampled, sigma=1.0, axes=(0, 1))

        # Add Laplacian detail
        result = upsampled + blended_pyr[level]

    return result


def compute_seam_cost(
    left_image: np.ndarray,
    right_image: np.ndarray,
    seam_x: int,
    overlap_width: int = 64,
) -> np.ndarray:
    """Compute cost of vertical seam for graph-cut optimization.

    Args:
        left_image: [H, W, 3] left image
        right_image: [H, W, 3] right image
        seam_x: Approximate seam position
        overlap_width: Width of overlap region

    Returns:
        [H, overlap_width] cost for each vertical position
    """
    height = left_image.shape[0]
    cost = np.zeros((height, overlap_width), dtype=np.float32)

    # Compute color difference in overlap region
    for x_offset in range(overlap_width):
        x_left = seam_x - overlap_width // 2 + x_offset
        x_right = seam_x - overlap_width // 2 + x_offset

        if 0 <= x_left < left_image.shape[1] and 0 <= x_right < right_image.shape[1]:
            # L1 distance in color space
            diff = np.abs(
                left_image[:, x_left].astype(np.float32)
                - right_image[:, x_right].astype(np.float32)
            )
            cost[:, x_offset] = diff.sum(axis=1)

    return cost


def find_optimal_seam(
    left_image: np.ndarray,
    right_image: np.ndarray,
    seam_x: int,
    overlap_width: int = 64,
) -> np.ndarray:
    """Find optimal vertical seam using dynamic programming.

    Args:
        left_image: [H, W, 3] left image
        right_image: [H, W, 3] right image
        seam_x: Approximate seam position
        overlap_width: Search width

    Returns:
        [H] seam position for each row
    """
    cost = compute_seam_cost(left_image, right_image, seam_x, overlap_width)
    height, width = cost.shape

    # DP: find minimum cost path
    dp = np.zeros_like(cost)
    dp[0, :] = cost[0, :]

    # Forward pass
    for y in range(1, height):
        for x in range(width):
            min_cost = np.inf

            # Check neighbors (left, center, right)
            for dx in [-1, 0, 1]:
                prev_x = x + dx
                if 0 <= prev_x < width:
                    min_cost = min(min_cost, dp[y - 1, prev_x])

            dp[y, x] = cost[y, x] + min_cost

    # Backward pass: trace seam
    seam = np.zeros(height, dtype=np.int32)
    seam[-1] = np.argmin(dp[-1, :])

    for y in range(height - 2, -1, -1):
        x = seam[y + 1]
        candidates = []

        for dx in [-1, 0, 1]:
            prev_x = x + dx
            if 0 <= prev_x < width:
                candidates.append((dp[y, prev_x], prev_x))

        if candidates:
            seam[y] = min(candidates, key=lambda c: c[0])[1]

    # Convert seam positions to absolute x coordinates
    seam = seam + (seam_x - overlap_width // 2)

    return seam


def blend_with_seam(
    left_image: np.ndarray,
    right_image: np.ndarray,
    seam: np.ndarray,
    blend_width: int = 32,
) -> np.ndarray:
    """Blend images along optimal seam with feathering.

    Args:
        left_image: [H, W, 3] left image
        right_image: [H, W, 3] right image
        seam: [H] seam position for each row
        blend_width: Width of blending region around seam

    Returns:
        [H, W, 3] blended image
    """
    height, width = left_image.shape[:2]
    blended = np.zeros_like(left_image, dtype=np.float32)

    for y in range(height):
        seam_x = int(seam[y])

        # Define blend region
        left_bound = max(0, seam_x - blend_width)
        right_bound = min(width, seam_x + blend_width)

        # Left part: use left image
        if left_bound > 0:
            blended[y, :left_bound] = left_image[y, :left_bound]

        # Right part: use right image
        if right_bound < width:
            blended[y, right_bound:] = right_image[y, right_bound:]

        # Blend region
        for x in range(left_bound, right_bound):
            # Distance-based blending
            dist_to_seam = abs(x - seam_x) / blend_width
            alpha = np.clip(dist_to_seam, 0, 1)

            blended[y, x] = (
                left_image[y, x] * (1.0 - alpha) + right_image[y, x] * alpha
            )

    return np.clip(blended, 0, 255).astype(left_image.dtype)


def compensate_exposure(
    left_image: np.ndarray,
    right_image: np.ndarray,
    overlap_region: tuple[int, int, int, int],
) -> np.ndarray:
    """Adjust right image exposure to match left in overlap region.

    Args:
        left_image: [H, W, 3] left image
        right_image: [H, W, 3] right image
        overlap_region: (x_min, y_min, x_max, y_max) of overlap

    Returns:
        Exposure-compensated right image
    """
    x_min, y_min, x_max, y_max = overlap_region

    # Compute mean intensity in overlap
    left_overlap = left_image[y_min:y_max, x_min:x_max].astype(np.float32)
    right_overlap = right_image[y_min:y_max, x_min:x_max].astype(np.float32)

    left_mean = left_overlap.mean(axis=(0, 1))
    right_mean = right_overlap.mean(axis=(0, 1))

    # Compute gain per channel
    gain = left_mean / (right_mean + 1e-6)

    # Apply gain
    compensated = right_image.astype(np.float32) * gain[np.newaxis, np.newaxis, :]

    return np.clip(compensated, 0, 255).astype(right_image.dtype)


def compute_blend_mask(
    left_valid: np.ndarray,
    right_valid: np.ndarray,
    seam: np.ndarray,
    blend_width: int = 32,
) -> np.ndarray:
    """Compute smooth blending mask around seam.

    Args:
        left_valid: [H, W] validity mask for left
        right_valid: [H, W] validity mask for right
        seam: [H] seam position
        blend_width: Blend region width

    Returns:
        [H, W] blend weight mask [0, 1]
    """
    height, width = left_valid.shape
    mask = np.zeros((height, width), dtype=np.float32)

    for y in range(height):
        seam_x = int(seam[y])

        for x in range(width):
            dist_to_seam = abs(x - seam_x)

            if dist_to_seam < blend_width:
                # Smooth blend weight
                blend_weight = 1.0 - (dist_to_seam / blend_width)
                mask[y, x] = blend_weight
            elif x < seam_x:
                # Left side: use left if valid
                mask[y, x] = 1.0 if left_valid[y, x] else 0.0
            else:
                # Right side: use right if valid
                mask[y, x] = 1.0 if right_valid[y, x] else 0.0

    return mask
