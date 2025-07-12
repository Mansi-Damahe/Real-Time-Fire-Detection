from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Run prediction on webcam (source=0)
model.predict(source=0, imgsz=640, conf=0.6, save=True)
