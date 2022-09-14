import cv2
import numpy as np
from matplotlib import pyplot as plt


image = cv2.imread("images/architecture.jpg", cv2.IMREAD_GRAYSCALE)
image = image.astype(float)
# image = cv2.resize(image, (512, 512))
plt.imshow(image, cmap="gray")
plt.title("{} x {} image".format(image.shape[0], image.shape[0]))
plt.show()

downsize_factor = 4
output_size = image.shape[0]

for _ in range(downsize_factor):
    input_size = output_size
    bin_size = 2
    output_size = input_size // bin_size
    image = image.reshape((1, output_size, bin_size,
                           output_size, bin_size)).max(4).max(2).reshape((output_size, output_size))
    plt.imshow(image, cmap="gray")
    plt.title("{} x {} image".format(output_size, output_size))
    plt.show()


    # img_fourier  = np.fft.fft2(image, s=(4096, 4096))
    # fshift = np.fft.fftshift(img_fourier)
    # magnitude_spectrum = 20 * np.log(1 + np.abs(fshift))
    #
    # plt.subplot(131), plt.imshow(image, cmap='gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    # plt.subplot(133), plt.imshow(image, cmap="gray")
    # plt.title("{} x {} image".format(output_size, output_size))
    # plt.show()
