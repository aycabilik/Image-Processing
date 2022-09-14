import cv2
import numpy as np
from matplotlib import pyplot as plt

image_1 = cv2.imread('images/cube_indoor.jpg', cv2.IMREAD_GRAYSCALE)
image1_fourier = np.fft.fft2(image_1)
fshift_1 = np.fft.fftshift(image1_fourier)
magnitude_spectrum_1 = 20*np.log(1 + np.abs(fshift_1))
ang_1 = np.angle(fshift_1)
real_1 = np.real(fshift_1)
imaginary_1 = np.imag(fshift_1)


plt.subplot(231), plt.imshow(image_1, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(magnitude_spectrum_1, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(ang_1, cmap='gray')
plt.title('Phase Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(real_1)
plt.title('Real Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(imaginary_1)
plt.title('Imaginary Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


