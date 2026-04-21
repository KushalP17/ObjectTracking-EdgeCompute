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
        if(export_type == 'onnx'):
            model.export(format=export_type, imgsz=512, half=True)
        else:
            model.export(format=export_type, imgsz=512, int8=True, data='coco128.yaml')
