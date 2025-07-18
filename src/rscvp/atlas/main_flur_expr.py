import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gaussian

# Load and preprocess your image
img = iio.imread("/Volumes/BigDATA/data/user/yu-ting/analysis/hist/YW051/zproj/YW051_3_5_r.tif")
img_green = img[:, :, 0]  # extract green channel

# Smooth and rescale using robust method
img_smooth = gaussian(img_green, sigma=1)
p1, p99 = np.percentile(img_smooth, [0, 99.5])
img_contrast = np.clip((img_smooth - p1) / (p99 - p1), 0, 1)

# Suppress oversaturation (choose one)
img_contrast = img_contrast ** 0.5

# Subtract median of central 100x100
h, w = img_contrast.shape
center_crop = img_contrast[h // 2 - 50:h // 2 + 50, w // 2 - 50:w // 2 + 50]
center_median = np.median(center_crop)

img_subtracted = np.clip(img_contrast - center_median, 0, None)
img_final = img_subtracted / (img_subtracted.max() + 1e-6)

# Plot with axis objects
fig, ax = plt.subplots(figsize=(10, 10))

# Contour
levels = np.linspace(0.1, 0.9, 9)
cs = ax.contour(img_final, levels=levels, cmap='inferno')

# Overlay image
ax.imshow(img_final, cmap='gray', alpha=0.5)

# Colorbar
cbar = fig.colorbar(cs, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Normalized fluorescence")

# Aesthetics
ax.set_title("Fluorescence Contour Plot (Auto-contrast & median subtracted)")
ax.axis('off')
plt.tight_layout()
plt.show()
