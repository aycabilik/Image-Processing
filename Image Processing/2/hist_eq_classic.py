import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def hist(arr, levs):
    histbins = np.zeros(levs, dtype=int)
    for el in arr:
        histbins[el] += 1
    return histbins


def histeq(histvals, total=None):
    if total is None:
        total = sum(histvals)
    nlevs = len(histvals)
    trnsf = np.zeros(nlevs, dtype=np.uint)
    trnsf[0] = histvals[0]
    for k in range(1, nlevs):
        trnsf[k] = trnsf[k-1] + histvals[k]

    trnsf = trnsf.astype(float)*(nlevs-1) / total
    trnsf = np.round(trnsf).astype(np.uint8)

    return trnsf


def imhisteq(img):
    sz = np.prod(img.shape)
    he_trnsf = histeq(hist(img.flat, L), sz)
    im_new = np.zeros(img.shape, dtype=np.uint8)
    for i in range(sz):
        im_new.flat[i] = he_trnsf[img.flat[i]]

    return im_new, he_trnsf




im = cv2.imread("out.jpg")

# Show a single image
fig_size = 8  # Adjust your figure size on the page
fig = plt.figure(figsize=(fig_size, fig_size))

fntsize = 20

# plt.subplot(1, 2, 1)
plt.imshow(im, cmap='gray')

plt.show()

bytes_per_pixel = int(im.nbytes / np.prod(im.shape))
L = bytes_per_pixel * 2**8

hist_im = hist(im.flat, L)
# Let's normalize it so that it becomes probability mass (since unit is 1)
hist_im_normed = hist_im / sum(hist_im)

fig_size = 6  # Adjust your figure size on the page
fig = plt.figure(figsize=(1.5*fig_size, fig_size))

# Plot the normalized histogram
r = np.array(range(L))
plt.bar(r, hist_im_normed, width=1)
highest_prob = hist_im_normed.max()
plt.text(-4, highest_prob, f'{highest_prob:.3f}',
         horizontalalignment='right', verticalalignment='top')

plt.ylim(0, highest_prob)
plt.xlim(0, L-1)
plt.xlabel('r', fontsize=fntsize)
plt.title(f'Histogram of the input image', fontsize=fntsize)
plt.show()

im_heq, histeq_trnsf = imhisteq(im)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 10))
ax = axes.ravel()

ax[0].imshow(im, cmap=plt.cm.gray)
ax[0].set_title('Original image', fontsize=fntsize)
ax[1].imshow(im_heq, cmap=plt.cm.gray) # Takes care of intensity scaling
ax[1].set_title(f'Histogram Equalization', fontsize=fntsize)

for a in ax:
    a.axis('off')  # Don't show the axis
fig.tight_layout()


# Histogram of the histogram equalized image

hist_im = hist(im_heq.flat, L)
# Let's normalize it so that it becomes probability mass (since unit is 1)
hist_im_normed = hist_im / sum(hist_im)

fig_size = 6  # Adjust your figure size on the page
fig = plt.figure(figsize=(1.5*fig_size, fig_size))

# Plot the normalized histogram
r = np.array(range(L))
plt.bar(r, hist_im_normed, width=1)
highest_prob = hist_im_normed.max()
plt.text(-4, highest_prob, f'{highest_prob:.3f}',
         horizontalalignment='right', verticalalignment='top')

plt.ylim(0, highest_prob)
plt.xlim(0, L-1)
plt.xlabel('r', fontsize=fntsize)
plt.title(f'Histogram of the input image', fontsize=fntsize)
plt.show()

# Hist. eq. transfer function of the histogram equalized image
histeq_trnsf2 = histeq(hist_im)


fig_size = 6  # Adjust your figure size on the page
fig = plt.figure(figsize=(1.5*fig_size, fig_size))

r = np.array(range(L))
plt.step(r, histeq_trnsf2)  # Do a stair case plot instead of the plot function
plt.ylim(0, L-1)
plt.xlim(0, L-1)
plt.xlabel('r', fontsize=fntsize)
plt.ylabel('s', fontsize=fntsize)
plt.title(f'Transfer function of the histogram equalization', fontsize=fntsize)
plt.show()


