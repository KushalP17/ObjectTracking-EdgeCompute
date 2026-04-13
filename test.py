import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

# Load a YOLOv8n PyTorch model
# model = YOLO("yolov8n.pt")

# Export the model to TensorRT
# model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO("yolov8n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")