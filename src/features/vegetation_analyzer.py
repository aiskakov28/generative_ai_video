import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class VegetationAnalyzer:
    def __init__(self):
        self.model = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.vegetation_class = 15

    def preprocess_image(self, image):
        h, w = image.shape[:2]
        aspect = w / h
        if aspect > 1:
            new_w = 224
            new_h = int(224 / aspect)
        else:
            new_h = 224
            new_w = int(224 * aspect)

        resized = cv2.resize(image, (new_w, new_h))

        square_img = np.zeros((224, 224, 3), dtype=np.uint8)

        y_offset = (224 - new_h) // 2
        x_offset = (224 - new_w) // 2
        square_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return square_img

    def analyze_vegetation(self, image):
        if image is None:
            return 0.0

        processed_image = self.preprocess_image(image)

        pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        input_tensor = self.transform(pil_image).unsqueeze(0)

        try:
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]
                vegetation_mask = (output.argmax(0) == self.vegetation_class).byte().cpu().numpy()

            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)

            green_mask = cv2.inRange(hsv, np.array([30, 30, 40]), np.array([85, 255, 255]))

            if green_mask.shape != vegetation_mask.shape:
                green_mask = cv2.resize(green_mask, (vegetation_mask.shape[1], vegetation_mask.shape[0]))

            combined_mask = (vegetation_mask.astype(np.uint8) & (green_mask > 0).astype(np.uint8))

            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            vegetation_pixels = np.sum(combined_mask > 0)
            vegetation_ratio = vegetation_pixels / total_pixels

            normalized_score = min(1.0, vegetation_ratio / (1 + vegetation_ratio))

            return float(normalized_score)

        except Exception as e:
            print(f"Error in vegetation analysis: {str(e)}")
            return 0.0

    def analyze_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return 0.0

        return self.analyze_vegetation(image)