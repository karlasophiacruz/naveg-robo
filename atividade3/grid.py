import matplotlib.pyplot as plt
import numpy as np

# Create a 2D grid with obstacles
# Here, 0 represents free space and 1 represents obstacles
grid = np.zeros((1000, 1000))

# Add some obstacles
# Let's add a rectangular obstacle
grid[300:400, 200:400] = 1

# Add another rectangular obstacle
grid[600:700, 600:800] = 1

# Add circular obstacle
cx, cy, r = 800, 200, 50
y, x = np.ogrid[-cx:grid.shape[0]-cx, -cy:grid.shape[1]-cy]
mask = x**2 + y**2 <= r**2
grid[mask] = 1

# Save the grid
np.save('cspace.npy', grid)

# Plot the grid with obstacles

plt.figure(figsize=(10, 10))
plt.imshow(grid, cmap='binary')
plt.title('Grid with Obstacles')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
