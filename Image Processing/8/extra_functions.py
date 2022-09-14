import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature
from skimage import filters



def _preprocess(image, mask, sigma, mode, cval):

    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval,
                           preserve_range=False)
    if mask is None:
        # Smooth the masked image
        smoothed_image = filters.gaussian(image, **gaussian_kwargs)
        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0
        return smoothed_image, eroded_mask

    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]

    # Compute the fractional contribution of masked pixels by applying
    # the function to the mask (which gets you the fraction of the
    # pixel data that's due to significant points)
    bleed_over = (
        filters.gaussian(mask.astype(float), **gaussian_kwargs) + np.finfo(float).eps
    )

    # Smooth the masked image
    smoothed_image = filters.gaussian(masked_image, **gaussian_kwargs)

    # Lower the result by the bleed-over fraction, so you can
    # recalibrate by dividing by the function on the mask to recover
    # the effect of smoothing from just the significant pixels.
    smoothed_image /= bleed_over

    # Make the eroded mask. Setting the border value to zero will wipe
    # out the image edges for us.
    s = ndi.generate_binary_structure(2, 2)
    eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    return smoothed_image, eroded_mask

def _set_local_maxima(magnitude, pts, w_num, w_denum, row_slices,
                      col_slices, out):
    """Get the magnitudes shifted left to make a matrix of the points to
    the right of pts. Similarly, shift left and down to get the points
    to the top right of pts.
    """
    r_0, r_1, r_2, r_3 = row_slices
    c_0, c_1, c_2, c_3 = col_slices
    c1 = magnitude[r_0, c_0][pts[r_1, c_1]]
    c2 = magnitude[r_2, c_2][pts[r_3, c_3]]
    m = magnitude[pts]
    w = w_num[pts] / w_denum[pts]
    c_plus = c2 * w + c1 * (1 - w) <= m
    c1 = magnitude[r_1, c_1][pts[r_0, c_0]]
    c2 = magnitude[r_3, c_3][pts[r_2, c_2]]
    c_minus = c2 * w + c1 * (1 - w) <= m
    out[pts] = c_plus & c_minus

    return out


def _get_local_maxima(isobel, jsobel, magnitude, eroded_mask):

    abs_isobel = np.abs(isobel)
    abs_jsobel = np.abs(jsobel)

    eroded_mask = eroded_mask & (magnitude > 0)

    # Normals' orientations
    is_horizontal = eroded_mask & (abs_isobel >= abs_jsobel)
    is_vertical = eroded_mask & (abs_isobel <= abs_jsobel)
    is_up = (isobel >= 0)
    is_down = (isobel <= 0)
    is_right = (jsobel >= 0)
    is_left = (jsobel <= 0)
    #
    # --------- Find local maxima --------------
    #
    # Assign each point to have a normal of 0-45 degrees, 45-90 degrees,
    # 90-135 degrees and 135-180 degrees.
    #
    local_maxima = np.zeros(magnitude.shape, bool)
    # ----- 0 to 45 degrees ------
    # Mix diagonal and horizontal
    pts_plus = is_up & is_right
    pts_minus = is_down & is_left
    pts = ((pts_plus | pts_minus) & is_horizontal)
    # Get the magnitudes shifted left to make a matrix of the points to the
    # right of pts. Similarly, shift left and down to get the points to the
    # top right of pts.
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_jsobel, abs_isobel,
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        [slice(None), slice(None), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 45 to 90 degrees ------
    # Mix diagonal and vertical
    #
    pts = ((pts_plus | pts_minus) & is_vertical)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_isobel, abs_jsobel,
        [slice(None), slice(None), slice(1, None), slice(-1)],
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 90 to 135 degrees ------
    # Mix anti-diagonal and vertical
    #
    pts_plus = is_down & is_right
    pts_minus = is_up & is_left
    pts = ((pts_plus | pts_minus) & is_vertical)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_isobel, abs_jsobel,
        [slice(None), slice(None), slice(-1), slice(1, None)],
        [slice(1, None), slice(-1), slice(1, None), slice(-1)],
        local_maxima)
    # ----- 135 to 180 degrees ------
    # Mix anti-diagonal and anti-horizontal
    #
    pts = ((pts_plus | pts_minus) & is_horizontal)
    local_maxima = _set_local_maxima(
        magnitude, pts, abs_jsobel, abs_isobel,
        [slice(-1), slice(1, None), slice(-1), slice(1, None)],
        [slice(None), slice(None), slice(1, None), slice(-1)],
        local_maxima)

    return local_maxima