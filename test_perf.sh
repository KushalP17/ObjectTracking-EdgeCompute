# yolo_size_array=("n" "s" "m" "l" "x")
yolo_size_array=("n")
model_type=("pt" "onnx" "engine")

for size in "${yolo_size_array[@]}"; do
    
    echo "YOLOv8${size} Metrics" >> power_mem_readings.csv

    tegrastats >> power_mem_readings.csv &
    BGPID=$!

    ./.venv/bin/python test.py "$size"

    sleep 3
    kill $BGPID

    echo "" >> power_mem_readings.csv

done