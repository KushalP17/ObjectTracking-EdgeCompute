import sys
import torch
import torchvision
from ultralytics import YOLO

print(torch.__version__)
print(torchvision.__version__)
print(torch.cuda.is_available())

yolo_size_map = ['n', 's', 'm', 'l', 'x']
model_type = ['pt', 'onnx', 'engine']

if len(sys.argv) == 1 or sys.argv[1] == None:
    model_size = 'n'
elif sys.argv[1] not in yolo_size_map:
    raise Exception(f"Invalid YOLO Model Size: {sys.argv[1]}")
else:
    model_size = sys.argv[1] 

if len(sys.argv) == 2 or sys.argv[2] == None:
    model_backend = 'pt'
elif sys.argv[2] not in model_type:
    raise Exception(f"Invalid YOLO Model Backend: {sys.argv[2]}")
else:
    model_backend = sys.argv[2]


# Load the exported TensorRT model
trt_model = YOLO(model=f"yolov8{model_size}.{model_backend}", task="detect")

# Run inference
# results = trt_model.predict("https://ultralytics.com/images/bus.jpg", show=True, save=True)
results = trt_model.val(data="coco128.yaml", imgsz=512)
# print(results)
# print(results.speed)

with open("latency_readings.csv", 'a') as latency_readings:
    latency_readings.write(f"YOLO v8{model_size} with {model_backend} backend Latency Metrics\n")
    latency_readings.write(str(results.speed) + "\n\n")
