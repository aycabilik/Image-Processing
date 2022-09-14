import cv2
import numpy as np
import matplotlib.pyplot as plt

def conv2(img, krn):

    h_krn, w_krn = krn.shape
    h_p = (h_krn - 1) // 2
    w_p = (w_krn - 1) // 2

    img_padded = np.pad(img, (h_p, w_p), 'constant')
    # nrows, ncols = img.shape
    #
    # h_pad_im = nrows + 2 * h_p
    # w_pad_im = ncols + 2 * w_p
    # img_padded = np.zeros((h_pad_im, w_pad_im), dtype=img.dtype)
    # img_padded[h_p:h_p + nrows, w_p:w_p + ncols] = img

    res = np.zeros(img.shape, dtype=krn.dtype)
    # mult = np.zeros_like(krn)
    for row, col in np.ndindex(img.shape):
        mult = np.multiply(img_padded[row:row+h_krn, col:col+w_krn], np.flip(krn))
        res[row, col] = mult.sum()

    return res


height_impulse_im = 6
width_impulse_im = 6
im_impulse = np.zeros((height_impulse_im, width_impulse_im), dtype=int)
im_impulse[height_impulse_im//2, width_impulse_im//2] = 1

np.random.seed(0)
mask_random = np.random.randint(0, 6, (3, 3))
im_impulse_response = conv2(im_impulse, mask_random)

# im_impulse = np.float32(im_impulse)
# im_impulse_response = cv2.filter2D(im_impulse, -1, mask_random)


print('\nInput impulse image')
print(im_impulse)
print('\nMask')
print(mask_random)
print('\nFiltered impulse image (Look at the location of the mask on the filtered image)')
print(im_impulse_response)

