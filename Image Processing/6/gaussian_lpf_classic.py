import cv2
import numpy as np
from matplotlib import pyplot as plt

def scale_freqmag_cv2(dft_im):
    return 20 * np.log(1 + cv2.magnitude(dft_im[:, :, 0], dft_im[:, :, 1]))


def scale_freqmag_np(dft_im):
    return 20 * np.log(1 + np.abs(dft_im))


def filt_freq(img, transf_shifted, double_pad=False):

    if double_pad:
        dft = np.fft.fft2(img, (2 * img.shape[0], 2 * img.shape[1]))
    else:
        dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    mag = scale_freqmag_np(dft_shift)

    # apply mask and inverse DFT
    fshift = dft_shift * transf_shifted

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    magf = scale_freqmag_np(fshift)

    return transf_shifted, mag, magf, img_back


def smoothsquare_freq(img):
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    return filt_freq(img, mask)


def smoothgauss_freq(img, cutoff_freq, double_pad=True):
    #     transf_shifted = np.zeros((*img.shape, 2))
    if double_pad:
        rows, cols = 2 * img.shape[0], 2 * img.shape[1]
    else:
        rows, cols = img.shape
    transf_shifted = np.zeros((rows, cols))
    u_cent = transf_shifted.shape[0] // 2
    v_cent = transf_shifted.shape[1] // 2
    for u in range(transf_shifted.shape[0]):
        for v in range(transf_shifted.shape[1]):
            dist2 = (u - u_cent) ** 2 + (v - v_cent) ** 2
            transf_shifted[u, v] = np.exp(-dist2 / (2 * cutoff_freq ** 2))
    return filt_freq(img, transf_shifted, double_pad)


filename = 'images/cube_indoor.jpg'
im = cv2.imread(filename)[..., 0]

D0_values = [10, 20, 40, 60]
fig, axes = plt.subplots(3, len(D0_values), figsize=(20, 16), dpi=100)
for n, D0 in enumerate(D0_values):
    transf, mag0, mag_filt, im_filt = smoothgauss_freq(im, D0, double_pad=True)
    # transf, mag0, mag_filt, im_filt = smoothsquare_freq(im)
    im_filt = im_filt[:im.shape[0], :im.shape[1]]
    axes[0, n].imshow(transf, cmap='gray')
    axes[0, n].set_title(f'Transfer function {transf.shape}')
    axes[1, n].imshow(mag_filt, cmap='gray')
    axes[1, n].set_title(f'Filtered Magnitude {im.shape}')
    axes[2, n].imshow(im_filt, cmap='gray')
    axes[2, n].set_title(f'Filtered image')
plt.show()
