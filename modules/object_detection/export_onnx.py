from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8m.pt")  # Or "yolov8s.pt" / "yolov8l.pt"

# Convert the model to ONNX format
model.export(format="onnx", imgsz=[480, 640])

print("Model exported successfully as yolov8m.onnx")
