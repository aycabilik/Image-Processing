import copy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def read_img(path):
    return cv.imread(path, cv.IMREAD_GRAYSCALE)


def to_float(img):
    return img.astype(float)


def normalize(img):
    return img / 255


def empty_image():
    return np.zeros([256, 256])


def draw_L(img):
    # vertical rectangle
    img[48:210, 65:87] = 1
    # horizontal rectangle
    img[190:210, 87:200] = 1
    return img


def merge_img(img1, img2):
    for x in range(img2.shape[0]):
        for y in range(img2.shape[1]):
            if img2[x][y] == 1:
                img1[x, y] = 1
    return img1


def show_plots(img1, img2, img3):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Images")
    ax1.imshow(img1, cmap='gray')
    ax1.title.set_text("Input")
    ax2.imshow(img2, cmap='gray')
    ax2.title.set_text("L")
    ax3.imshow(img3, cmap='gray')
    ax3.title.set_text("Input+L")
    plt.savefig("figure.png")
    plt.show()


if __name__ == '__main__':
    path = "lab2.jpg"
    """QUESTION 1"""
    img = read_img(path)
    """QUESTION 2"""
    img = to_float(img)
    img = normalize(img)
    """QUESTION 3"""
    empty_img = empty_image()
    """QUESTION 4"""
    L_img = draw_L(empty_img)
    """QUESTION 5"""
    merg_img = merge_img(copy.deepcopy(img), empty_img)
    """QUESTION 6"""
    show_plots(img, L_img, merg_img)

    """ 
    cv.imshow("img", img)
    cv.waitKey()
    """
