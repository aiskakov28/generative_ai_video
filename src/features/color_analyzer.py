import cv2
import numpy as np
from scipy.spatial.distance import cosine

class ColorAnalyzer:
    def analyze_colors(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()

        dominant_colors = self.get_dominant_colors(image, k=5)

        return {
            'color_distribution': {
                'hue': h_hist,
                'saturation': s_hist,
                'value': v_hist
            },
            'dominant_colors': dominant_colors
        }

    def get_dominant_colors(self, image, k=5):
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)

        unique_labels, counts = np.unique(labels, return_counts=True)

        percentages = counts / len(labels)

        return {
            'colors': centers.tolist(),
            'percentages': percentages.tolist()
        }

    def calculate_color_bias(self, original_hist, generated_hist):
        hue_similarity = 1 - cosine(original_hist['hue'], generated_hist['hue'])
        sat_similarity = 1 - cosine(original_hist['saturation'], generated_hist['saturation'])
        val_similarity = 1 - cosine(original_hist['value'], generated_hist['value'])

        color_similarity = np.mean([hue_similarity, sat_similarity, val_similarity])

        color_bias = 1 - color_similarity

        return {
            'color_bias': color_bias,
            'channel_similarities': {
                'hue': hue_similarity,
                'saturation': sat_similarity,
                'value': val_similarity
            }
        }
