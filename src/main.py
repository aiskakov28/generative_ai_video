import os
import cv2
import numpy as np
import json
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
from PIL import Image
from sklearn.cluster import KMeans
from src.features.vegetation_analyzer import VegetationAnalyzer


class Analyzer:
    def __init__(self):
        self.model = YOLO('yolov9e.pt')
        self.vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg.eval()
        self.vegetation_analyzer = VegetationAnalyzer()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.n_colors = 5

    def get_color_distribution(self, image):
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        pixels = image_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(pixels)
        return colors.astype(int), percentages

    def extract_frame(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError(f"No frames extracted from video: {video_path}")
        return frames

    def analyze_image(self, image):
        results = self.model(image)
        boxes = results[0].boxes.data
        classes = boxes[:, 5].cpu().numpy()
        counts = {
            'humans': int(sum(classes == 0)),
            'vehicles': int(sum((classes == 2) | (classes == 5) | (classes == 7))),
        }

        vegetation_score = self.vegetation_analyzer.analyze_vegetation(image)
        counts['vegetation'] = int(vegetation_score * 100)

        colors, color_dist = self.get_color_distribution(image)

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(pil_image).unsqueeze(0)

        with torch.no_grad():
            features = self.vgg.features(img_tensor)
            segmentation = self.vegetation_analyzer.model(img_tensor)['out'][0]
            seg_mask = segmentation.argmax(0).byte().cpu().numpy()
            seg_classes = len(torch.unique(segmentation.argmax(0)))

        return {
            'counts': counts,
            'colors': colors.tolist(),
            'color_distribution': color_dist.tolist(),
            'features': features.mean().item(),
            'segmentation_mask': seg_mask.tolist(),
            'segmentation_classes': int(seg_classes)
        }

def normalize_bias_score(score):
    return min(1.0, score / (1 + score))

def calculate_color_similarity(colors1, dist1, colors2, dist2):
    total_diff = 0
    for (c1, d1), (c2, d2) in zip(zip(colors1, dist1), zip(colors2, dist2)):
        color_diff = np.sqrt(np.sum((np.array(c1) - np.array(c2)) ** 2)) / 441.67
        dist_diff = abs(d1 - d2)
        total_diff += (color_diff + dist_diff) / 2
    return 1 - (total_diff / len(colors1))


def calculate_bias_metrics(original_metrics, ai_metrics):
    bias_scores = {}
    for key in original_metrics['counts']:
        original_count = original_metrics['counts'][key]
        ai_count = ai_metrics['counts'][key]
        if original_count > 0:
            ratio = ai_count / original_count
            bias_scores[key] = normalize_bias_score(abs(1 - ratio))
        else:
            bias_scores[key] = 1.0 if ai_count > 0 else 0.0

    color_similarity = calculate_color_similarity(
        original_metrics['colors'],
        original_metrics['color_distribution'],
        ai_metrics['colors'],
        ai_metrics['color_distribution']
    )
    bias_scores['color'] = 1 - color_similarity

    feature_diff = abs(original_metrics['features'] - ai_metrics['features'])
    bias_scores['visual_similarity'] = 1 - normalize_bias_score(feature_diff)

    seg_class_diff = abs(original_metrics['segmentation_classes'] - ai_metrics['segmentation_classes'])
    bias_scores['segmentation'] = 1 - normalize_bias_score(seg_class_diff)

    weights = {
        'humans': 0.20,
        'vehicles': 0.15,
        'vegetation': 0.15,
        'color': 0.15,
        'visual_similarity': 0.15,
        'segmentation': 0.20
    }
    bias_scores['overall'] = sum(bias_scores[k] * weights[k] for k in weights)

    return bias_scores

def main():
    data_dir = Path('../data')
    output_dir = Path('../output/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = Analyzer()
    locations = [
        'Adliya_Bahrain', 'Al-Hamra', 'Ballsbridge', 'Bur_Dubai', 'Doha_Corniche',
        'Downtown_Dubai', 'Dubai_Marina', 'Dublin_Docklands', 'Hadar', 'Hawally',
        'Houthavens', 'Juffair', 'Kirchberg', 'Mirqab', 'Muttrah', 'Neve_Tzedek',
        'Nydalen', 'Nytorv', 'Old_City', 'Qurum', 'Rehavia', 'Reykjavik_South_Harbor',
        'Rolla', 'Ruwi', 'Saadiyat_Island', 'Salmiya', 'Sarona', 'Schiphol',
        'seef', 'Sharq_District', 'Sloterdijk', 'Swords', 'Tahlia_Street',
        'Temple_Bar', 'The_Pearl_Doha', 'Westbay_Doha', 'Zurich_West'
    ]

    results = {}
    all_metrics = []

    for location in locations:
        print(f"\nAnalyzing {location}...")
        original_path = data_dir / 'original' / f"{location}_Original.png"
        ai_path = data_dir / 'ai_generated' / f"{location}_AI.mp4"

        try:
            original_image = cv2.imread(str(original_path))
            if original_image is None:
                raise ValueError(f"Failed to read original image: {original_path}")

            original_metrics = analyzer.analyze_image(original_image)
            frames = analyzer.extract_frame(str(ai_path))
            frame_metrics = [analyzer.analyze_image(frame) for frame in frames]

            ai_metrics = {
                'counts': {
                    key: np.mean([m['counts'][key] for m in frame_metrics])
                    for key in frame_metrics[0]['counts']
                },
                'colors': np.mean([m['colors'] for m in frame_metrics], axis=0).tolist(),
                'color_distribution': np.mean([m['color_distribution'] for m in frame_metrics], axis=0).tolist(),
                'features': np.mean([m['features'] for m in frame_metrics]),
                'segmentation_mask': frame_metrics[0]['segmentation_mask'],
                'segmentation_classes': int(np.mean([m['segmentation_classes'] for m in frame_metrics]))
            }

            bias_metrics = calculate_bias_metrics(original_metrics, ai_metrics)
            results[location] = {
                'original': original_metrics,
                'ai_generated': ai_metrics,
                'bias_metrics': bias_metrics
            }

            all_metrics.append({
                'Location': location,
                'Human Bias': bias_metrics['humans'],
                'Vehicle Bias': bias_metrics['vehicles'],
                'Vegetation Bias': bias_metrics['vegetation'],
                'Color Bias': bias_metrics['color'],
                'Visual Similarity': bias_metrics['visual_similarity'],
                'Segmentation Bias': bias_metrics['segmentation'],
                'Overall Bias': bias_metrics['overall']
            })

        except Exception as e:
            print(f"Error analyzing {location}: {str(e)}")
            continue

    if all_metrics:
        df = pd.DataFrame(all_metrics)

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 1, 1)
        sns.barplot(data=df.melt(
            id_vars=['Location'],
            value_vars=['Human Bias', 'Vehicle Bias', 'Vegetation Bias',
                        'Color Bias', 'Visual Similarity', 'Segmentation Bias'],
            var_name='Metric', value_name='Bias Score'
        ), x='Location', y='Bias Score', hue='Metric')
        plt.title('Bias Metrics by Location')
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        sns.barplot(data=df, x='Location', y='Overall Bias')
        plt.title('Overall Bias Score by Location')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'bias_analysis.png')
        plt.close()

        with open(output_dir / 'analysis_results.json', 'w') as f:
            json.dump(results, f, indent=4, cls=NumpyEncoder)

        print("\nAnalysis Summary:")
        print(tabulate(df, headers='keys', tablefmt='grid', floatfmt='.3f'))
    else:
        print("\nNo valid analysis results to visualize")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

if __name__ == "__main__":
    main()