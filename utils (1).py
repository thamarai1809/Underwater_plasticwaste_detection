from ultralytics import YOLO
import cv2

# Load your custom-trained model
model = YOLO("best.pt")  # Make sure this path is correct

def detect_objects(image_path):
    # Run inference
    results = model(image_path)[0]

    boxes = []
    class_ids = []
    confidences = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        boxes.append((x1, y1, x2, y2))
        class_ids.append(cls)
        confidences.append(conf)

    # Get class names from model
    class_names = model.names

    return boxes, class_ids, confidences, class_names
