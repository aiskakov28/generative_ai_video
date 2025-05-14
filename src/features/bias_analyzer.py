import numpy as np
from scipy.stats import zscore

class BiasAnalyzer:
    def __init__(self):
        self.metrics = ['humans', 'vehicles', 'vegetation', 'style']

    def normalize_scores(self, scores):
        normalized = {}
        for metric, value in scores.items():
            if isinstance(value, dict):
                if 'count' in value:
                    normalized[metric] = value['count']
                elif 'vegetation_ratio' in value:
                    normalized[metric] = value['vegetation_ratio']
                else:
                    normalized[metric] = np.mean(list(value.values()))
            else:
                normalized[metric] = value
        return normalized

    def calculate_bias(self, original_metrics, generated_metrics):
        orig_norm = self.normalize_scores(original_metrics)
        gen_norm = self.normalize_scores(generated_metrics)

        differences = {}
        for metric in self.metrics:
            if metric in orig_norm and metric in gen_norm:
                diff = abs(orig_norm[metric] - gen_norm[metric])
                differences[metric] = diff

        max_diff = max(differences.values()) if differences else 1
        normalized_differences = {k: v / max_diff for k, v in differences.items()}
        overall_bias = np.mean(list(normalized_differences.values()))

        return {
            'normalized_metrics': {
                'original': orig_norm,
                'generated': gen_norm
            },
            'differences': normalized_differences,
            'overall_bias_score': float(overall_bias)
        }
