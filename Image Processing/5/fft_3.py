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

image_2 = cv2.imread('images/einstein.jpeg', cv2.IMREAD_GRAYSCALE)
image2_fourier = np.fft.fft2(image_2)
fshift_2 = np.fft.fftshift(image2_fourier)
magnitude_spectrum_2 = 20*np.log(1 + np.abs(fshift_2))
ang_2 = np.angle(fshift_2)
real_2 = np.real(fshift_2)
imaginary_2 = np.imag(fshift_2)


combined = np.multiply(np.abs(fshift_2), np.exp(1j*ang_1))
fftx = np.fft.ifftshift(combined)
ffty = np.fft.ifft2(fftx)
imgCombined = np.abs(ffty).astype(np.uint8)

plt.imshow(imgCombined, cmap="gray", interpolation=None)
plt.title("Reconstructed")
plt.show()
