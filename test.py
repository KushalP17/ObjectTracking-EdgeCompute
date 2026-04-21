import sys
import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

# Load the exported TensorRT model
trt_model = YOLO(model=f"yolov8n.engine", task="detect")

# Run inference
# results = trt_model.predict("https://ultralytics.com/images/bus.jpg", show=True, save=True)
results = trt_model.val(data="coco128.yaml", imgsz=512)