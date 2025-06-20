import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

def remove_background(image_path, model, save_path="output.png"):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)

    if not results or results[0].masks is None:
        print("❌ No mask found in prediction.")
        return

    masks = results[0].masks.data.cpu().numpy()  # (n_instances, h, w)
    combined_mask = np.any(masks, axis=0).astype(np.uint8)

    # Resize mask to match original image shape
    mask_resized = cv2.resize(combined_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    alpha = mask_resized * 255
    b, g, r = cv2.split(image)
    rgba = cv2.merge((b, g, r, alpha))  # All channels now same shape

    output = Image.fromarray(rgba)
    output.save(save_path)
    print(f"✅ Saved with transparent background: {save_path}")

model_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Model/runs/train7/weights/last.pt'
img_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Dataset/images/test/ad6a4b8bfef7ceac.jpg'

# Load image
img = cv2.imread(img_path)


# Load model and run prediction
model = YOLO(model_path)

remove_background(img_path, model, save_path="output.png")
