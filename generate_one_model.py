import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())


model = YOLO(f"yolov8x.pt")
model.export(format='onnx', imgsz=512, half=True)
