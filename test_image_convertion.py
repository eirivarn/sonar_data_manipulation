from PIL import Image
import numpy as np

# Load the image
image_path = 'test.png'
polar_image = Image.open(image_path).convert('L')  # Convert to grayscale
polar_array = np.array(polar_image)

# Display the properties of the image
polar_array.shape, polar_array.dtype

import numpy as np

def polar_to_cartesian(polar_image, threshold=0):
    height, width = polar_image.shape
    max_radius = height // 2
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

# Apply the polar_to_cartesian transformation
x_coords, y_coords, values = polar_to_cartesian(polar_array, threshold=10)

# To visualize, create a scatter plot of the Cartesian points with the intensity as color
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter(x_coords, y_coords, c=values, cmap='gray', s=1)
ax.set_aspect('equal', 'box')
ax.axis('off')  # Turn off the axis
plt.show()

# Save the scatter plot to a PNG file
output_path = 'cartesian_image.png'
fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

output_path
