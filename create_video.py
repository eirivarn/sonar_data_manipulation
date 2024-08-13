import cv2
import os
import numpy as np

def adjust_alpha(image, alpha_value):
    # Juster alfaen basert på alpha_value
    adjusted = cv2.convertScaleAbs(image, alpha=alpha_value, beta=0)
    return adjusted

def on_trackbar(val):
    global alpha_value
    alpha_value = val / 100.0
    adjusted_image = adjust_alpha(original_image, alpha_value)
    # Flip the Y-axis for display
    flipped_image = cv2.flip(adjusted_image, 0)
    # Resize the image
    resized_image = resize_image(flipped_image, target_width=800)  # Adjust the width as needed
    cv2.imshow('Adjusted Image', resized_image)

def resize_image(image, target_width):
    height, width = image.shape[:2]
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)
    resized_image = cv2.resize(image, (target_width, new_height))
    return resized_image

# Load det første bildet
image_folder = 'extracted_images'
first_image_path = os.path.join(image_folder, 'image_0.png')  # eller hva det første bildet heter
original_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)

if original_image is None:
    raise ValueError("Første bilde ikke funnet eller kunne ikke lastes.")

# Initial alpha-verdi
alpha_value = 1.0

# Opprett et vindu for å vise bildet
cv2.namedWindow('Adjusted Image')

# Opprett en trackbar for å justere alfaen
cv2.createTrackbar('Alpha', 'Adjusted Image', 1, 10000, on_trackbar)  

# Vis det originale bildet først
cv2.imshow('Adjusted Image', original_image)

# Kjør trackbar vinduet og vent til brukeren er ferdig
cv2.waitKey(0)
cv2.destroyAllWindows()

def create_video_from_images(image_folder, output_video_path, alpha_value, target_width=1200, frame_rate=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

    if not images:
        raise ValueError("Ingen bilder funnet i mappen.")

    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    
    # Flip the first frame and resize it to get the correct video dimensions
    frame = adjust_alpha(frame, alpha_value)
    frame = cv2.flip(frame, 0)
    frame = resize_image(frame, target_width)
    
    height, width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        adjusted_frame = adjust_alpha(frame, alpha_value)
        flipped_frame = cv2.flip(adjusted_frame, 0)
        resized_frame = resize_image(flipped_frame, target_width)
        video.write(resized_frame)

    video.release()
    cv2.destroyAllWindows()

# Lag videoen med den justerte alfa-verdien, flippet Y-akse, og endret bredde
output_video_path = 'output_video_with_adjusted_alpha_and_flip.mp4'
create_video_from_images(image_folder, output_video_path, alpha_value, target_width=800)
