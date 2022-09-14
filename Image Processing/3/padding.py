import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("camera.jpg", cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.title('Before Adding Padding to Image')
plt.show()

# Adding zero padding
output = np.pad(image, (40, 50), 'constant')

# Full custom padding (pad_top, pad_bottom), (pad_left, pad_right)
# output = np.pad(image, ((10, 40), (30, 60)), 'constant')


plt.imshow(output, cmap='gray')
plt.title('After Adding Padding to Image')
plt.show()


# Adding mirror padding (Reflection Padding)
output = np.pad(image, (40, 50), 'reflect')

plt.imshow(output, cmap='gray')
plt.title('After Adding Padding to Image')
plt.show()
