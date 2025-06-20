from ultralytics import YOLO
'''refer to https://docs.ultralytics.com/tasks/segment/#train
but running on custom config.yaml'''

model = YOLO("yolov8n-seg.pt")

results = model.train(data='Model/config.yaml',project='Model/runs', epochs=20, imgsz=320, batch=16)
