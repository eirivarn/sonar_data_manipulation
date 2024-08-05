import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import generic_filter
import cv2

def read_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

def polar_to_cartesian(polar_image, threshold=100):
    height, width = polar_image.shape
    max_radius = height
    angular_range = np.linspace(-45, 45, width)
    angular_range_rad = np.deg2rad(angular_range)
    x_coords, y_coords, values = [], [], []
    for r in range(height):
        for i, theta in enumerate(angular_range_rad):
            value = polar_image[r, i]
            if value > threshold:
                y = max_radius + r * np.cos(theta)
                x = max_radius + r * np.sin(theta)
                x_coords.append(x)
                y_coords.append(-y)
                values.append(value)
    return np.array(x_coords), np.array(y_coords), np.array(values)

def compensate_signal_strength(polar_image):
    height, width = polar_image.shape
    compensated_polar_image = np.copy(polar_image)
    max_distance = height
    angular_range = np.linspace(-45, 45, width)
    angular_range_rad = np.deg2rad(angular_range)
    for r in range(height):
        distance_compensation_factor = max_distance / (r + 1)
        for i, theta in enumerate(angular_range_rad):
            angle_compensation_factor = np.abs(np.cos(theta))
            compensated_polar_image[r, i] *= distance_compensation_factor * angle_compensation_factor
    return compensated_polar_image

def apply_local_thresholding(x_coords, y_coords, values, grid_size, window_size):
    global_filtered_indices = apply_global_thresholding(values, 0)
    x_coords, y_coords, values = x_coords[global_filtered_indices], y_coords[global_filtered_indices], values[global_filtered_indices]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
    grid_values = np.zeros_like(grid_x)
    for x, y, val in zip(x_coords, y_coords, values):
        ix = np.argmin(abs(grid_x[0] - x))
        iy = np.argmin(abs(grid_y[:, 0] - y))
        grid_values[iy, ix] = val
    local_threshold_grid = generic_filter(grid_values, lambda x: np.median(x) + np.std(x), size=window_size)
    filtered_mask = grid_values > local_threshold_grid
    return grid_x[filtered_mask], grid_y[filtered_mask], grid_values[filtered_mask]

def apply_global_thresholding(values, threshold):
    median_val = np.median(values)
    std_dev = np.std(values)
    return values > (median_val + std_dev)

def save_plot_and_detect_circles(x, y, values, figure_size):
    # Create and save the scatter plot as an image
    plt.figure(figsize=figure_size)
    plt.scatter(x, y, c=values, cmap='jet', s=20, edgecolor='none')
    plt.colorbar()
    plt.title('Filtered Cartesian Image')
    plt.axis('equal')
    plt.axis('off')  # Hide axis
    plt.savefig('scatter_plot.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Read the saved image using OpenCV
    img = cv2.imread('scatter_plot.png', cv2.IMREAD_COLOR)

    # Crop the image to remove the rightmost 150 pixels
    cropped_img = img[:, :-150, :]

    # Save or process the cropped image further
    # Save the cropped image if needed for further use
    cv2.imwrite('cropped_scatter_plot.png', cropped_img)

    # If you want to proceed with further processing, like detecting circles:
    img_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 5)

    # Detect circles using the Hough Transform
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=20, param2=20, minRadius=1500, maxRadius=3000)

    # Draw circles on the cropped image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(cropped_img, (i[0], i[1]), i[2], (255, 0, 0), 2)  # Outer circle
            cv2.circle(cropped_img, (i[0], i[1]), 2, (0, 255, 0), 3)  # Center of the circle

    # Display the result with detected circles
    plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles in Cropped Image')
    plt.axis('off')
    plt.show()
    
    # if circles is not None:
    #     circles = np.uint16(np.around(circles[0]))
    #     highest_circle = max(circles, key=lambda c: c[1])  # Lambda function to find the circle with the largest y-value

    #     # Draw only the highest circle on the image
    #     cv2.circle(img, (highest_circle[0], highest_circle[1]), highest_circle[2], (255, 0, 0), 2)  # Outer circle
    #     cv2.circle(img, (highest_circle[0], highest_circle[1]), 2, (0, 255, 0), 3)  # Center of the circle

    # # Display the result with detected circle
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.title('Detected Circle with Highest Y-Coordinate')
    # plt.axis('off')
    # plt.show()
    
    
image_path = 'test.png'
polar_image = read_image(image_path)
compensated_polar_image = compensate_signal_strength(polar_image)

# Plot polar images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(polar_image, cmap='gray', aspect='auto')
plt.colorbar()
plt.title('Original Polar Image')
plt.subplot(1, 2, 2)
plt.imshow(compensated_polar_image, cmap='gray', aspect='auto')
plt.colorbar()
plt.title('Compensated Polar Image')
plt.show()

x_coords_comp, y_coords_comp, values_comp = polar_to_cartesian(compensated_polar_image, threshold=0)

# Plot first Cartesian image
plt.figure(figsize=(6, 6))
plt.scatter(x_coords_comp, y_coords_comp, c=values_comp, cmap='jet', s=20, edgecolor='none')
plt.colorbar()
plt.title('Compensated Cartesian Image')
plt.axis('equal')
plt.show()

x_coords_filtered, y_coords_filtered, values_filtered = apply_local_thresholding(x_coords_comp, y_coords_comp, values_comp, 300, 5)

# Calculate figure size based on range of coordinates
x_range = np.max(x_coords_filtered) - np.min(x_coords_filtered)
y_range = np.max(y_coords_filtered) - np.min(y_coords_filtered)
scale_ratio = y_range / x_range
figure_size = (6, 6 * scale_ratio) if scale_ratio > 1 else (6 / scale_ratio, 6)


# Detect and display the circles
save_plot_and_detect_circles(x_coords_filtered, y_coords_filtered, values_filtered, figure_size)
