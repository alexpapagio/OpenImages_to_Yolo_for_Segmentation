import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# Paths
img_path = './Dataset/images/train/0a00de3a8c699ab1.jpg'
label_path = './Dataset/labels/train/0a00de3a8c699ab1.txt'

# Check file existence
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {os.path.abspath(img_path)}")
if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label not found: {os.path.abspath(label_path)}")


# Load image
img = cv2.imread(img_path)
h, w = img.shape[:2]

# Read label file
with open(label_path, 'r') as f:
    lines = f.readlines()

for line in lines:
    parts = list(map(float, line.strip().split()))
    class_id = int(parts[0])
    poly_coords = parts[1:]

    # Convert normalized coords to pixel coords
    points = []
    for i in range(0, len(poly_coords), 2):
        x = int(poly_coords[i] * w)
        y = int(poly_coords[i+1] * h)
        points.append([x, y])

    points = np.array([points], dtype=np.int32)

    # Draw polygon
    cv2.polylines(img, points, isClosed=True, color=(0, 255, 0), thickness=2)

# Convert BGR to RGB for matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(img_rgb)
plt.title(f"Polygon overlay on image")
plt.axis('off')
plt.show()
