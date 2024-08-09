import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Predefined set of distinct colors
colors = [
    '#1f77b4',  # Muted blue
    '#ff7f0e',  # Safety orange
    '#2ca02c',  # Cooked asparagus green
    '#d62728',  # Brick red
    '#9467bd',  # Muted purple
    '#8c564b',  # Chestnut brown
    '#e377c2',  # Raspberry yogurt pink
    '#7f7f7f',  # Middle gray
    '#bcbd22',  # Curry yellow-green
    '#17becf'   # Blue-teal
]

def create_next_run_folder(base_path="Histogram_analysis"):
    # Create the base directory if it does not exist
    os.makedirs(base_path, exist_ok=True)
    run_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    run_numbers = sorted([int(d.replace('run', '')) for d in run_folders if d.startswith('run') and d[3:].isdigit()])
    next_run_number = run_numbers[-1] + 1 if run_numbers else 1
    next_run_folder = os.path.join(base_path, f'run{next_run_number}')
    os.makedirs(next_run_folder, exist_ok=True)
    return next_run_folder

def apply_gain(image, gain_value):
    return cv2.convertScaleAbs(image, alpha=gain_value, beta=0)

def on_trackbar(val):
    global gain_value, image, enhanced_image
    gain_value = val / 10.0
    enhanced_image = apply_gain(image, gain_value)
    cv2.imshow('Adjust Gain', enhanced_image)

def scale_image(image, scale_factor=2, interpolation=cv2.INTER_CUBIC):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=interpolation)

def select_roi(image, scale_factor, window_name):
    scaled_image = scale_image(image, scale_factor)
    roi = cv2.selectROI(window_name, scaled_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()
    x, y, w, h = [int(coord / scale_factor) for coord in roi]
    sub_image = image[y:y+h, x:x+w]
    return sub_image, x, y, w, h

def draw_bounding_boxes(image, rois, colors):
    if len(image.shape) == 2 or image.shape[2] == 1:
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_with_boxes = image.copy()

    for roi, color in zip(rois, colors):
        x, y, w, h = roi
        cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image_with_boxes, 'ROI', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image_with_boxes

def save_zoomed_image_with_boxes(scaled_zoomed_image, rois, colors, output_path):
    image_with_boxes = draw_bounding_boxes(scaled_zoomed_image, rois, colors)
    cv2.imwrite(output_path, image_with_boxes)
    print(f"Zoomed image with ROIs saved as {output_path}")

def save_global_mean_histogram_plot(sub_images, x_starts, colors, output_path):
    plt.figure()
    for idx, (sub_image, color, start_x) in enumerate(zip(sub_images, colors, x_starts)):
        mean_values = np.mean(sub_image, axis=0)
        x_positions = np.arange(start_x, start_x + len(mean_values))
        normalized_color = [c / 255.0 for c in color]
        plt.plot(x_positions, mean_values, color=normalized_color, label=f'Mean ROI {idx+1}')
    
    plt.title('Mean Pixel Values Across All ROIs (Global X Coordinates)')
    plt.xlabel('Global X Position')
    plt.ylabel('Mean Pixel Value')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved global mean histograms plot to {output_path}")

def save_histogram(sub_image, color, output_path, roi_index):
    plt.figure()
    hist, _ = np.histogram(sub_image.ravel(), bins=256, range=[0, 256])
    normalized_color = [c / 255.0 for c in color]
    plt.plot(hist, color=normalized_color)
    plt.title(f'Histogram for ROI {roi_index + 1}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved histogram for ROI {roi_index + 1} to {output_path}")

def select_multiple_rois(scaled_image, window_name):
    rois = []
    colors = []
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while True:
        roi = cv2.selectROI(window_name, scaled_image, fromCenter=False, showCrosshair=True)
        if roi[2] == 0 or roi[3] == 0:
            break
        colors.append(tuple(int(x) for x in np.random.choice(range(256), size=3)))
        rois.append(roi)
    cv2.destroyAllWindows()
    return rois, colors

# Main flow
output_folder = create_next_run_folder()

# Set paths for outputs
global_mean_histogram_path = os.path.join(output_folder, "global_mean_histogram_plot.png")
zoomed_boxes_path = os.path.join(output_folder, "zoomed_image_with_rois.png")

image_path = 'test.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Could not open or find the image!")

gain_value = 1.0
enhanced_image = apply_gain(image, gain_value)

cv2.namedWindow('Adjust Gain')
cv2.createTrackbar('Gain', 'Adjust Gain', 10, 30, on_trackbar)
on_trackbar(10)
cv2.waitKey(0)
cv2.destroyAllWindows()

scale_factor = 8
zoomed_image, zx, zy, zw, zh = select_roi(enhanced_image, scale_factor, "Select Zoom Area")
scaled_zoomed_image = scale_image(zoomed_image, scale_factor)

rois, colors = select_multiple_rois(scaled_zoomed_image, "Select ROIs in Zoomed Image")
sub_images = [scaled_zoomed_image[y:y+h, x:x+w] for x, y, w, h in rois]
x_starts = [zx + x for x, _, _, _ in rois]

zoomed_image_with_boxes = draw_bounding_boxes(scaled_zoomed_image, rois, colors)
cv2.imshow('Zoomed Image with Bounding Boxes', zoomed_image_with_boxes)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save global mean histogram plot
save_global_mean_histogram_plot(sub_images, x_starts, colors[:len(sub_images)], global_mean_histogram_path)

# Save individual frequency histograms for each ROI
for i, (sub_image, color) in enumerate(zip(sub_images, colors)):
    histogram_path = os.path.join(output_folder, f'histogram_roi_{i+1}.png')
    save_histogram(sub_image, color, histogram_path, i)

# Save the zoomed image with bounding boxes
save_zoomed_image_with_boxes(scaled_zoomed_image, rois, colors[:len(rois)], zoomed_boxes_path)
