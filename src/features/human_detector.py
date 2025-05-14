from ultralytics import YOLO
import numpy as np

class HumanDetector:
    def __init__(self):
        self.model = YOLO('yolov9e.pt')

    def detect_humans(self, image):
        if image is None:
            return {'count': 0, 'locations': []}

        results = self.model(image)
        human_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls == 0:
                    human_detections.append({
                        'confidence': conf,
                        'bbox': box.xyxy.cpu().numpy()[0].tolist()
                    })

        return {
            'count': len(human_detections),
            'locations': human_detections
        }
