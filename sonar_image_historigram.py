import cv2
import numpy as np
import matplotlib.pyplot as plt

def scale_image(image, scale_factor=2, interpolation=cv2.INTER_CUBIC):
    # Calculate new dimensions
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    # Resize image
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    return scaled_image

def select_roi(image, scale_factor, window_name):
    scaled_image = scale_image(image, scale_factor)
    roi = cv2.selectROI(window_name, scaled_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    # Adjust ROI coordinates to match the original image scale
    x, y, w, h = [int(coord / scale_factor) for coord in roi]
    sub_image = image[y:y+h, x:x+w]
    return sub_image, x, y, w, h

def calculate_statistics(sub_image):
    mean_val = np.mean(sub_image)
    variance_val = np.var(sub_image)
    min_val = np.min(sub_image)
    max_val = np.max(sub_image)
    return mean_val, variance_val, min_val, max_val

def plot_x_position_values(sub_image1, sub_image2, start_x1, start_x2):
    mean_values1 = np.mean(sub_image1, axis=0)
    mean_values2 = np.mean(sub_image2, axis=0)
    x_positions1 = np.arange(start_x1, start_x1 + len(mean_values1))
    x_positions2 = np.arange(start_x2, start_x2 + len(mean_values2))
    plt.plot(x_positions1, mean_values1, label='Region 1')
    plt.plot(x_positions2, mean_values2, label='Region 2')
    plt.title('Pixel Values for Each X Position')
    plt.xlabel('X Position')
    plt.ylabel('Mean Pixel Value')
    plt.legend(loc='upper right')
    plt.show()

# Main flow
image_path = 'test.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Could not open or find the image!")

# Use a scale factor that makes selection easier
scale_factor = 8  # Example scale factor, adjust as needed

# Select zoom area
zoomed_image, zx, zy, zw, zh = select_roi(image, scale_factor, "Select area to zoom in")

# Select ROIs within the zoomed image
sub_image1, x1, y1, w1, h1 = select_roi(zoomed_image, scale_factor, "Select ROI 1 in zoomed image")
sub_image2, x2, y2, w2, h2 = select_roi(zoomed_image, scale_factor, "Select ROI 2 in zoomed image")

# Calculate and print statistics
mean1, var1, min1, max1 = calculate_statistics(sub_image1)
mean2, var2, min2, max2 = calculate_statistics(sub_image2)
print(f"Region 1 - Mean: {mean1}, Variance: {var1}, Min value: {min1}, Max value: {max1}")
print(f"Region 2 - Mean: {mean2}, Variance: {var2}, Min value: {min2}, Max value: {max2}")

# Plotting values for each x-position
plot_x_position_values(sub_image1, sub_image2, zx + x1, zx + x2)
