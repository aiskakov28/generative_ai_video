import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class DeepLabV3PlusAnalyzer:
    def __init__(self):
        self.model = torchvision.models.segmentation.deeplabv3_plus_resnet50(pretrained=True)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def analyze(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]

        output_predictions = output.argmax(0).byte().cpu().numpy()

        return {
            'segmentation_mask': output_predictions,
            'num_classes': len(torch.unique(output.argmax(0)))
        }
