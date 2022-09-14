import numpy as np
import matplotlib.pyplot as plt

sigma_x = 4
sigma_y = 3

x = np.linspace(-15, 15, 100)
y = np.linspace(-15, 15, 100)
x, y = np.meshgrid(x, y)
# Gaussian Formula
z = (1/(2*np.pi*sigma_x*sigma_y) * np.exp(-(x**2/(2*sigma_x**2)
     + y**2/(2*sigma_y**2))))

# 2D Representation
plt.imshow(z, interpolation="bicubic", cmap='rainbow')
"plt.contourf(x, y, z)"
plt.colorbar()

# 3D Representation
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='rainbow',
           linewidths=0.01, alpha=.9,
           edgecolor='w',
                )
plt.show()
