import cv2
import os
import pandas as pd

# Function to adjust the alpha (scaling factor) of the image
def adjust_alpha(image, alpha_value):
    return cv2.convertScaleAbs(image, alpha=alpha_value, beta=0)

# Function to update the image display when the trackbar is adjusted
def on_trackbar(val):
    global alpha_value, original_image
    alpha_value = val / 100.0
    adjusted_image = adjust_alpha(original_image, alpha_value)
    flipped_image = cv2.flip(adjusted_image, 0)
    cv2.imshow('Adjusted Image', flipped_image)

# Function to overlay text (timestamp) on the image
def overlay_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_color=(255, 255, 255), font_thickness=1):
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness)

# Function to load timestamps from the CSV file
def load_timestamps_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df['Timestamp'] = df['Transmission Time'].apply(lambda x: x.split(' ')[1])
    return df['Timestamp'].tolist()

# Function to create a video from images with overlayed timestamps
def create_video_from_images(image_folder, output_video_path, alpha_value, timestamps, frame_rate=17.2):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")], 
                    key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not images:
        raise ValueError("No images found in the folder.")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    frame = adjust_alpha(frame, alpha_value)
    frame = cv2.flip(frame, 0)

    height, width = frame.shape[:2]
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for i, image in enumerate(images):
        if i >= len(timestamps):
            timestamp = timestamps[-1]
        else:
            timestamp = timestamps[i]

        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        adjusted_frame = adjust_alpha(frame, alpha_value)
        flipped_frame = cv2.flip(adjusted_frame, 0)

        # Overlay the timestamp on the upper right corner
        overlay_text(flipped_frame, timestamp, (width - 150, 30))  # Adjust the position as needed

        video.write(flipped_frame)
        
    video.release()
    cv2.destroyAllWindows()

# Load the first image and set up the initial display
image_folder = 'runs/run_2/extracted_images_run_2'
first_image_path = os.path.join(image_folder, 'image_200.png')
original_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

if original_image is None:
    raise ValueError("First image not found or could not be loaded.")

alpha_value = 1.0
cv2.namedWindow('Adjusted Image')
cv2.createTrackbar('Alpha', 'Adjusted Image', 1, 5000, on_trackbar)
cv2.imshow('Adjusted Image', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Select ROI and apply adjustments
adjusted_image = adjust_alpha(original_image, alpha_value)
flipped_image = cv2.flip(adjusted_image, 0)
roi = cv2.selectROI("Select ROI", flipped_image, showCrosshair=True)
cv2.destroyAllWindows()

# Load timestamps and create the video
csv_file = 'image_records.csv'
timestamps = load_timestamps_from_csv(csv_file)
output_video_path = 'output_video_with_adjusted_alpha_and_cropped.mp4'
create_video_from_images(image_folder, output_video_path, alpha_value, timestamps, frame_rate=17.2)
