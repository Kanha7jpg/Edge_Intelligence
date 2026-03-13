import cv2
import numpy as np
from ultralytics import YOLO

class MLEngine:
    def __init__(self, model_path="yolov8s.pt", conf_threshold=0.5):
        """
        Initializes the YOLOv8 model for person detection.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        # Class 0 in COCO dataset is person
        self.target_classes = [0]

    def process_frame(self, frame):
        """
        Runs inference on a single frame and returns bounding boxes of detected persons.
        Returns: list of (x1, y1, x2, y2, conf)
        """
        results = self.model(frame, classes=self.target_classes, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections

    def get_padded_crop(self, frame, bbox, padding=10):
        """
        Extracts a bounding box crop from the frame with given padding.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2, _ = bbox
        
        # Add padding but stay within frame bounds
        px1 = max(0, x1 - padding)
        py1 = max(0, y1 - padding)
        px2 = min(w, x2 + padding)
        py2 = min(h, y2 + padding)
        
        crop = frame[py1:py2, px1:px2]
        return crop
