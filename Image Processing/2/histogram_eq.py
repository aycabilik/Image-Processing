import cv2
import numpy as np
import matplotlib.pyplot as plt

# read a image using imread

img = cv2.imread("dark.jpg", cv2.IMREAD_GRAYSCALE)

# # find frequency of pixels in range 0-255
# histr = cv2.calcHist([img],[0],None,[256],[0,256])
#
# # show the plotting graph of an image
# plt.plot(histr)
# plt.show()


# alternative way to find histogram of an image
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

# creating a Histograms Equalization
# of a image using cv2.equalizeHist()
equ = cv2.equalizeHist(img)

# stacking images side-by-side
res = np.hstack((img, equ))

plt.hist(equ.ravel(), 256, [0, 256])
plt.show()

# show image input vs output
cv2.imshow('image', res)

cv2.waitKey(0)
cv2.destroyAllWindows()
