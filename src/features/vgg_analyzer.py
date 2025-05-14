import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VGGAnalyzer:
    def __init__(self):
        self.model = models.vgg16(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image):
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(image_pil).unsqueeze(0)

        with torch.no_grad():
            features = self.model.features(input_tensor)
            features = features.mean([2, 3])

        return features.numpy().flatten()

    def compare_features(self, features1, features2):
        similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        return float(similarity)
