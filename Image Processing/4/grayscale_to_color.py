import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/strawberries.jpg")

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", gray_image)
cv2.waitKey(0)

plt.imshow(gray_image, cmap="gist_rainbow")
plt.show()
