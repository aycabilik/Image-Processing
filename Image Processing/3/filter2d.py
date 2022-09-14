import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("camera.jpg", cv2.IMREAD_GRAYSCALE)

# Apply identity kernel
kernel1 = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])


identity = cv2.filter2D(src=image, ddepth=-1, kernel=kernel1)

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

plt.imshow(identity, cmap='gray')
plt.title('Identity Image')
plt.show()


# Creating blurring kernel
kernel_size = 15
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

# Sum of the kernel should be 1
kernel /= (kernel_size * kernel_size)

# Apply blurring kernel
output = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

plt.imshow(output, cmap='gray')
plt.title('Blur Kernel Image')
plt.show()
