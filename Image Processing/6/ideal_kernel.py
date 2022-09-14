import cv2
import numpy as np
from matplotlib import pyplot as plt
from gaussian_lpf_classic import filt_freq

image = cv2.imread("images/cube_indoor.jpg", cv2.IMREAD_GRAYSCALE)

h, w = image.shape
r = 6

xx, yy = np.mgrid[:h, :w]
circle = np.sqrt((xx - w/2) ** 2 + (yy - h/2) ** 2)
circle = circle < (2*r)

# circle = np.zeros((h, w))
# for y in range(h):
#     for x in range(w):
#         if np.sqrt((y - h/2) ** 2 + (x - w/2) ** 2) < 2*r:
#             circle[x, y] = 1

plt.imshow(circle, cmap="gray")
plt.show()

img_f = np.fft.fft2(circle)
fshift = np.fft.fftshift(img_f)
magnitude_spectrum = 20*np.log(1 + np.abs(fshift))

plt.subplot(121), plt.imshow(circle, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(np.abs(fshift), cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

transf_shifted, mag, magf, img_back = filt_freq(image, circle)

plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(img_back, cmap='gray')
plt.title('Image After Applying the Filter'), plt.xticks([]), plt.yticks([])
plt.show()
