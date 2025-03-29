import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (1, 1, 1)]  # black to white
cmap_name = 'custom_gray'
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
# Assuming Z1 and Z2 are already defined and have the same shape as Z3
Z2=np.load('1.npy')
Z1=np.load('2.npy')
# Initialize Z3
Z3 = np.zeros((1200, 1920))

# Create masks based on the given conditions
mask1 = (Z1 > 0) & (Z2 == 0)
mask2 = (Z1 == 0) & (Z2 > 0)
mask3 = (Z1 > 0) & (Z2 > 0)

# Apply the masks to Z3
Z3[mask1] = Z1[mask1]
Z3[mask2] = Z2[mask2]
Z3[mask3] = (Z1[mask3] + Z2[mask3]) / 2

#print(Z3)
def update_zeros(Z):
    # Copy the array to avoid modifying the original array inplace
    Z_modified = np.copy(Z)
    # Iterate over each element in the array
    for i in range(Z.shape[0]):  # Loop through rows
        for j in range(Z.shape[1]):  # Loop through columns
            # If the current element is 0
            if Z[i, j] == 0:
                # Check 15 columns to the right
                pixel_right = None
                for k in range(j + 1, min(j + 12, Z.shape[1])):
                    if Z[i, k] > 0:
                        pixel_right = Z[i, k]
                        break
                # Check 15 columns to the left
                pixel_left = None
                for k in range(j - 1, max(j - 12, -1), -1):
                    if Z[i, k] > 0:
                        pixel_left = Z[i, k]
                        break
                # If both pixel_left and pixel_right are found
                if pixel_left is not None and pixel_right is not None:
                    # Update the current 0 value with the average of pixel_left and pixel_right
                    Z_modified[i, j] = (pixel_left + pixel_right) / 2
    return Z_modified
Z3=update_zeros(Z3)
#Z3[Z3 < 10] = 0
maximum=np.max(Z3)
print(maximum)
np.savez_compressed("Z3 ",  Z3)
# Now Z3 is updated according to the algorithm
plt.imshow(Z3, cmap=custom_cmap, vmin=0, vmax=250)
cbar = plt.colorbar()  # Add a color bar for reference
cbar.set_label('Value in mm')  # Set a label for the color bar

plt.show()
