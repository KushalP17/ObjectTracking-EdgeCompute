import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

size_map = ['n', 's', 'm', 'l', 'x']

export_map = ['onnx', 'engine']

for model_size in size_map:
    model = YOLO(f"yolov8{model_size}.pt")

    for export_type in export_map:
        model.export(format=export_type, dynamic=True)
