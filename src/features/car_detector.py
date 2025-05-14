from ultralytics import YOLO


class CarDetector:
    def __init__(self):
        self.model = YOLO('yolov9e.pt')
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def detect_vehicles(self, image):
        if image is None:
            return {'count': 0, 'detections': []}

        results = self.model(image)
        vehicles = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                if cls in self.vehicle_classes:
                    vehicles.append({
                        'type': self.vehicle_classes[cls],
                        'confidence': conf,
                        'bbox': box.xyxy.cpu().numpy()[0].tolist()
                    })

        return {
            'count': len(vehicles),
            'detections': vehicles
        }
