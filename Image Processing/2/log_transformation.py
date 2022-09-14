import cv2
import numpy as np
import matplotlib.pyplot as plt

#%%
# Open the image.
img = cv2.imread('light.jpg')

# Apply log transform.
c = 255 / (np.log(1 + np.max(img)))
log_transformed = c * np.log(1 + img)
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
# Specify the data type.
log_transformed = np.array(log_transformed, dtype=np.uint8)

plt.hist(log_transformed.ravel(), 256, [0, 256])
plt.show()

img2 = cv2.hconcat([img, log_transformed])

# Save the output.
cv2.imshow('log_transformed', img2)

#%%
img = cv2.imread('forest.jpg')
# Apply Gamma=2.2 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
gamma_two_point_two = np.array(255*(img/255)**2.2, dtype='uint8')
# Similarly, Apply Gamma=0.4
gamma_point_four = np.array(255*(img/255)**0.4, dtype='uint8')
# Display the images in subplots
img3 = cv2.hconcat([gamma_two_point_two, gamma_point_four])
cv2.imshow('a2', img3)
cv2.waitKey(0)
