import cv2
import numpy as np
import matplotlib.pyplot as plt

cube_indoor = cv2.imread("images/cube_indoor.jpg")
cube_outdoor = cv2.imread("images/cube_outdoor.jpg")

cube_indoor = cv2.cvtColor(cube_indoor, cv2.COLOR_BGR2RGB)
cube_outdoor = cv2.cvtColor(cube_outdoor, cv2.COLOR_BGR2RGB)
channel_names = ["R", "G", "B"]

o_r = cube_outdoor[:, :, 0]
o_g = cube_outdoor[:, :, 1]
o_b = cube_outdoor[:, :, 2]

i_r = cube_indoor[:, :, 0]
i_g = cube_indoor[:, :, 1]
i_b = cube_indoor[:, :, 2]


# create figure
fig = plt.figure()
# setting values to rows and column variables
rows = 2
columns = 4
# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(cube_outdoor, cmap="gray")
plt.axis('off')
plt.title("Outdoor")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(o_r, cmap="gray")
plt.axis('off')
plt.title(channel_names[0])

# Adds a subplot at the 3th position
fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(o_g, cmap="gray")
plt.axis('off')
plt.title(channel_names[1])

# Adds a subplot at the 4th position
fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(o_b, cmap="gray")
plt.axis('off')
plt.title(channel_names[2])

# Adds a subplot at the 5th position
fig.add_subplot(rows, columns, 5)

# showing image
plt.imshow(cube_indoor, cmap="gray")
plt.axis('off')
plt.title("Indoor")

# Adds a subplot at the 6th position
fig.add_subplot(rows, columns, 6)

# showing image
plt.imshow(i_r, cmap="gray")
plt.axis('off')
plt.title(channel_names[0])

# Adds a subplot at the 7th position
fig.add_subplot(rows, columns, 7)

# showing image
plt.imshow(i_g, cmap="gray")
plt.axis('off')
plt.title(channel_names[1])

# Adds a subplot at the 8th position
fig.add_subplot(rows, columns, 8)

# showing image
plt.imshow(i_b, cmap="gray")
plt.axis('off')
plt.title(channel_names[2])

plt.show()
