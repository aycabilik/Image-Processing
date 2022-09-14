import cv2
import numpy as np
import math


def show_img(img):
    cv2.imshow("img",img)
    cv2.waitKey()

def conv2(img, krn):
    h_krn, w_krn = krn.shape
    h_p = (h_krn - 1) // 2
    w_p = (w_krn - 1) // 2
    img_padded = np.pad(img, (h_p, w_p), 'constant')
    res = np.zeros(img.shape, dtype=krn.dtype)
    for row, col in np.ndindex(img.shape):
        mult = np.multiply(img_padded[row:row + h_krn, col:col + w_krn], np.flip(krn))
        res[row, col] = mult.sum()
    res = res.astype("uint8")
    return res


def gauss(l=11, sig=4):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
    return kernel / np.sum(kernel)


apple = cv2.imread("apple.png", cv2.IMREAD_GRAYSCALE)
print(apple.shape[0]/2, apple.shape[1])
kernel = gauss()
res = apple
cv2.imwrite("img.png", res)
for i in range(3):
    print(res.shape)
    kernel = gauss(l=5)
    res = conv2(res, kernel)
    cv2.imwrite("img" + str(i) + ".png", res)
    res = cv2.resize(res, dsize=(math.floor(res.shape[1]/2), math.floor(res.shape[0]/2)))
cv2.imwrite("img.png", res)
