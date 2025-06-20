from ultralytics import YOLO
import cv2
import numpy as np

model_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Model/runs/train5/weights/last.pt'
img_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Dataset/images/test/ad6a4b8bfef7ceac.jpg'

# Load image
img = cv2.imread(img_path)
H, W, _ = img.shape

# Load model and run prediction
model = YOLO(model_path)
results = model(img)

# Initialize blank mask
combined_mask = np.zeros((H, W), dtype=np.uint8)

for result in results:
    print(f"Detected {len(result.masks.data)} mask(s)")

    for j, mask in enumerate(result.masks.data):
        mask = mask.cpu().numpy() * 255
        mask = cv2.resize(mask, (W, H)).astype(np.uint8)

        # Combine the mask
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Optional: Save individual mask
        cv2.imwrite(f'./output_mask_{j}.png', mask)

# Overlay combined mask on image
overlay = cv2.addWeighted(img, 0.7, cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR), 0.3, 0)

# Display
cv2.imshow('Overlay', overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
