import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/strawberries.jpg")

b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()

cv2.imshow("b", b)
cv2.imshow("g", g)
cv2.imshow("r", r)
cv2.waitKey(0)

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")

plt.show()

