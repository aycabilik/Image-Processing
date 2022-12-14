import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('images/architecture.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (img.shape[0]*2, img.shape[1]*2))
img_fourier = np.fft.fft2(img)
fshift = np.fft.fftshift(img_fourier)
magnitude_spectrum = 20*np.log(1 + np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


# rows, cols = img.shape
# crow, ccol = int(rows/2), int(cols/2)
#
# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
#
# f_ishift = np.fft.ifftshift(fshift)
# img_back = np.fft.ifft2(f_ishift)
# img_back = np.abs(img_back)
#
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(img_back, cmap='gray')
# plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(img_back)
# plt.title('Result in JET'), plt.xticks([]), plt.yticks([])
#
# plt.show()
