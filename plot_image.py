import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import generic_filter

# Original polar_to_cartesian function
def polar_to_cartesian(polar_image, threshold=0):
    height, width = polar_image.shape
    max_radius = height
    angular_range = np.linspace(-45, 45, width)
    angular_range_rad = np.deg2rad(angular_range)

    x_coords = []
    y_coords = []
    values = []

    for r in range(height):
        for i, theta in enumerate(angular_range_rad):
            value = polar_image[r, i]
            if value > threshold:  # Simple threshold check
                y = max_radius + r * np.cos(theta)
                x = max_radius + r * np.sin(theta)
                x_coords.append(x)
                y_coords.append(-y)
                values.append(value)

    return np.array(x_coords), np.array(y_coords), np.array(values)

# Function to read the image from the given path
def read_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

# Initial global threshold filtering
def apply_global_thresholding(values, threshold):
    median_val = np.median(values)
    std_dev = np.std(values)
    global_threshold = median_val + std_dev
    return values > global_threshold

# Local thresholding using grid and window
def apply_local_thresholding(x_coords, y_coords, values, grid_size, window_size):
    # Filtering indices based on global thresholding first
    global_filtered_indices = apply_global_thresholding(values, 0)
    x_coords, y_coords, values = x_coords[global_filtered_indices], y_coords[global_filtered_indices], values[global_filtered_indices]

    # Grid setup
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))

    # Interpolating data onto the grid
    grid_values = np.zeros_like(grid_x)
    for x, y, val in zip(x_coords, y_coords, values):
        ix = np.argmin(abs(grid_x[0] - x))
        iy = np.argmin(abs(grid_y[:, 0] - y))
        grid_values[iy, ix] = val  # Simplest assignment approach

    # Local thresholding
    def local_stats(window):
        local_median = np.median(window)
        local_std = np.std(window)
        return local_median + local_std

    local_threshold_grid = generic_filter(grid_values, local_stats, size=window_size)
    filtered_mask = grid_values > local_threshold_grid

    # Extracting filtered coordinates and values
    filtered_x = grid_x[filtered_mask]
    filtered_y = grid_y[filtered_mask]
    filtered_values = grid_values[filtered_mask]

    return filtered_x, filtered_y, filtered_values

# Main script
image_path = 'test.png'
polar_image = read_image(image_path)
x_coords, y_coords, values = polar_to_cartesian(polar_image, threshold=0)
grid_size = 300  # Adjust grid resolution as needed
window_size = 5  # Local window size
x_coords_filtered, y_coords_filtered, values_filtered = apply_local_thresholding(x_coords, y_coords, values, grid_size, window_size)

# Plotting
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(x_coords, y_coords, c=values, cmap='jet', s=20, edgecolor='none')
plt.colorbar()
plt.title('Before Local Thresholding')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().set_facecolor('black')
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.scatter(x_coords_filtered, y_coords_filtered, c=values_filtered, cmap='jet', s=20, edgecolor='none')
plt.colorbar()
plt.title('After Local Thresholding')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().set_facecolor('black')
plt.axis('equal')

plt.tight_layout()
plt.show()
