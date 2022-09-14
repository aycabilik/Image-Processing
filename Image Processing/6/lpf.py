import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('images/cube_indoor.jpg', cv2.IMREAD_GRAYSCALE)
img_fourier = np.fft.fft2(img)
fshift = np.fft.fftshift(img_fourier)
magnitude_spectrum = 20*np.log(1 + np.abs(fshift))

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)

r = 50

new_fshift = np.zeros(fshift.shape)
new_fshift[crow-r:crow+r, ccol-r:ccol+r] = fshift[crow-r:crow+r, ccol-r:ccol+r]

f_ishift = np.fft.ifftshift(new_fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

magnitude_spectrum = 20*np.log(1 + np.abs(new_fshift))

plt.subplot(121), plt.imshow(img_back, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Image after LPF'), plt.xticks([]), plt.yticks([])

plt.show()