import cv2
import matplotlib.pyplot as plt
import numpy as np


def reduce(img):
    # img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, dsize=(img.shape[1]//2, img.shape[0]//2))
    # img = img.astype("float")
    return img


def expand(img):
    # img = cv2.Laplacian(img, cv2.CV_64F)
    img = cv2.resize(img, dsize=(img.shape[1] * 2, img.shape[0] * 2))
    #img = img.astype("uint8")
    return img


def gaussian_pyramid(image, pyramid_level):
    image_copy = image.copy()
    gp = [image_copy]
    for i in range(pyramid_level):
        image_copy = reduce(image_copy)
        gp.append(image_copy)

    return gp


def laplacian_pyramid(gaussian_pyr, pyramid_level):
    image_copy = gaussian_pyr[pyramid_level-1]
    lp = [image_copy]
    for i in range(pyramid_level-1, 0, -1):
        gaussian_expanded = expand(gaussian_pyr[i])
        laplacian = cv2.subtract(gaussian_pyr[i-1], gaussian_expanded)
        lp.append(laplacian)
    return lp


def combine_laplacians(laplacians1, laplacians2, mask):
    combined_pyramid = []
    mask = mask / 255

    for lp1, lp2 in zip(laplacians1, laplacians2):
        m = cv2.resize(mask, dsize=lp1.shape)
        lp2_m = lp2 * m
        m = 1-m
        lp1_m = lp1 * m
        laplacian = lp2_m + lp1_m

        combined_pyramid.append(laplacian)

    return combined_pyramid


def pyramid_blending(image1, image2, mask, pyramid_level):
    gp1 = gaussian_pyramid(image1, pyramid_level)
    gp2 = gaussian_pyramid(image2, pyramid_level)

    lp1 = laplacian_pyramid(gp1, pyramid_level)
    lp2 = laplacian_pyramid(gp2, pyramid_level)

    # gpm = gaussian_pyramid(mask, pyramid_level)
    # lpm = laplacian_pyramid(gpm, pyramid_level)

    combined_laplacian = combine_laplacians(lp1, lp2, mask)

    # Reconstruction
    reconstructed = combined_laplacian[0]
    for i in range(1, pyramid_level):
        reconstructed = expand(reconstructed)
        reconstructed = cv2.add(combined_laplacian[i], reconstructed)

    return reconstructed


apple = cv2.imread('images/apple.pgm', -1)
orange = cv2.imread('images/orange.pgm', -1)
mask = cv2.imread('images/mask.pgm', -1)

apple = np.pad(apple, ((51, 51), (0, 0)), 'constant')
apple = apple[:, 33:-34]
orange = np.pad(orange, ((51, 51), (0, 0)), 'constant')
orange = orange[:, 33:-34]
mask = np.pad(mask, ((51, 51), (0, 0)), 'constant')
mask = mask[:, 33:-34]

# plt.imshow(mask, cmap="gray")
# plt.show()

reconstructed_image = pyramid_blending(apple, orange, mask, 6)
plt.imshow(reconstructed_image, cmap="gray")
plt.show()
