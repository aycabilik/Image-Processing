import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mtimg

# Open the image.
img = cv2.imread('camera.jpg')

plt.hist(img.ravel(), 256, [0, 256])
plt.show()

cv2.imshow('original image', img)
cv2.waitKey(0)
# Trying 4 gamma values.
for gamma in [0.1, 0.5, 1.2, 2.2]:
    # Apply gamma correction.
    gamma_corrected = np.array(255 * (img / 255) ** gamma, dtype='uint8')

    plt.hist(gamma_corrected.ravel(), 256, [0, 256])
    plt.show()

    # Save edited images.
    cv2.imshow('gamma_transformed_' + str(gamma), gamma_corrected)

    cv2.waitKey(0)
