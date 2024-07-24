import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import generic_filter

#TODO add a good compensation to the sonar data to compensate for weaker signals when objects are further away, and the angle is not perpendicular to the object

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

# Function to compensate for stronger signals when objects are further away and assuming perpendicular reflection
def compensate_signal_strength(polar_image):
    height, width = polar_image.shape
    compensated_polar_image = np.copy(polar_image)
    
    max_distance = height
    angular_range = np.linspace(-45, 45, width)
    angular_range_rad = np.deg2rad(angular_range)
    
    for r in range(height):
        distance_compensation_factor = max_distance / (r + 1)
        for i, theta in enumerate(angular_range_rad):
            angle_compensation_factor = np.abs(np.cos(theta))  # Stronger signal when perpendicular
            compensated_polar_image[r, i] *= distance_compensation_factor * angle_compensation_factor
    
    return compensated_polar_image

# Main script
image_path = 'test.png'
polar_image = read_image(image_path)

# Compensate signal strength
compensated_polar_image = compensate_signal_strength(polar_image)

# Convert compensated polar image to cartesian coordinates
x_coords_comp, y_coords_comp, values_comp = polar_to_cartesian(compensated_polar_image, threshold=0)

# Apply local thresholding
grid_size = 300  # Adjust grid resolution as needed
window_size = 5  # Local window size
x_coords_filtered, y_coords_filtered, values_filtered = apply_local_thresholding(x_coords_comp, y_coords_comp, values_comp, grid_size, window_size)

# Plotting
plt.figure(figsize=(20, 10))

# Original polar image
plt.subplot(1, 4, 1)
plt.imshow(polar_image, cmap='gray', aspect='auto')
plt.colorbar()
plt.title('Original Polar Image')
plt.xlabel('Angle')
plt.ylabel('Distance')

# Compensated polar image
plt.subplot(1, 4, 2)
plt.imshow(compensated_polar_image, cmap='gray', aspect='auto')
plt.colorbar()
plt.title('Compensated Polar Image')
plt.xlabel('Angle')
plt.ylabel('Distance')

# Compensated Cartesian image
plt.subplot(1, 4, 3)
plt.scatter(x_coords_comp, y_coords_comp, c=values_comp, cmap='jet', s=20, edgecolor='none')
plt.colorbar()
plt.title('Compensated Cartesian Image')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().set_facecolor('black')
plt.axis('equal')

# Filtered Cartesian image
plt.subplot(1, 4, 4)
plt.scatter(x_coords_filtered, y_coords_filtered, c=values_filtered, cmap='jet', s=20, edgecolor='none')
plt.colorbar()
plt.title('Filtered Cartesian Image')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.gca().set_facecolor('black')
plt.axis('equal')

plt.tight_layout()
plt.show()
