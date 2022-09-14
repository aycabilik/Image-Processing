import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("camera.jpg", cv2.IMREAD_GRAYSCALE)

kernelSizes = [(3, 3), (9, 9), (15, 15)]
sigma_x = 3

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# Adding Some Noise
noise_std = 30
noised_image = image + np.random.randn(*image.shape) * noise_std

plt.imshow(noised_image, cmap='gray')
plt.title('Image After Noise Adding')
plt.show()

for (kX, kY) in kernelSizes:
    # apply an "average" blur to the image using the current kernel
    blurred = cv2.blur(noised_image, (kX, kY))

    plt.imshow(blurred, cmap='gray')
    plt.title("Average Blur ({}, {})".format(kX, kY))
    plt.show()

for (kX, kY) in kernelSizes:
    # apply an "average" blur to the image using the current kernel
    blurred = cv2.GaussianBlur(noised_image, (kX, kY), sigma_x)

    plt.imshow(blurred, cmap='gray')
    plt.title("Gaussian Blur ({}, {})".format(kX, kY))
    plt.show()
