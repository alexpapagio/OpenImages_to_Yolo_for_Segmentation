from ultralytics import YOLO

import cv2


model_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Model/runs/train5/weights/last.pt'
img_path = '/home/alexp/code/alexpapagio/computer_vision_engineer/07_segmentation_with_Yolov8/Dataset/images/test/ad6a4b8bfef7ceac.jpg'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):

        mask = mask.cpu().numpy() * 255

        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)
