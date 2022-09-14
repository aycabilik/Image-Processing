import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("camera.jpg", cv2.IMREAD_GRAYSCALE)

plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()



def gauss_kernel_1d(sig):
    krn_width = 6 * sig + 1
    cent = krn_width // 2
    xvals = np.array(range(krn_width)).reshape(krn_width, 1)

    # 1D Gaussian #
    krn = np.exp(-(xvals - cent)**2 / sig ** 2 / 2)
    return krn / krn.sum()


def gauss_kernel_2d(sig):
    krn_1d = gauss_kernel_1d(sig)
    # Hint use np.dot() funtion
    krn_2d = np.dot(krn_1d, krn_1d.T)
    return krn_2d / krn_2d.sum()


sigma = 3
kernel_gs = gauss_kernel_2d(sigma)
print(kernel_gs.shape, kernel_gs.sum())

im_out_gs = cv2.filter2D(image, -1, kernel_gs)
plt.imshow(im_out_gs, cmap='gray')
plt.title('Manuel Gaussian Filtered Image')
plt.show()
