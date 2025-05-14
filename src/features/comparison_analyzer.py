import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

class ComparisonAnalyzer:
    def __init__(self):
        self.metrics = ['human_presence', 'vehicle_density', 'vegetation_coverage', 'architectural_style']
        self.weights = {
            'human_presence': 0.3,
            'vehicle_density': 0.2,
            'vegetation_coverage': 0.25,
            'architectural_style': 0.25
        }

    def create_comparison_table(self, original_metrics, ai_metrics):
        data = {
            'Metric': self.metrics,
            'Original': [
                original_metrics['humans']['count'],
                original_metrics['vehicles']['count'],
                original_metrics['vegetation']['vegetation_ratio'],
                original_metrics['style_similarity']
            ],
            'AI Generated': [
                ai_metrics['humans']['count'],
                ai_metrics['vehicles']['count'],
                ai_metrics['vegetation']['vegetation_ratio'],
                ai_metrics['style_similarity']
            ]
        }

        df = pd.DataFrame(data)
        df['Difference'] = abs(df['Original'] - df['AI Generated'])
        df['Normalized Difference'] = df['Difference'] / df['Original']
        return df

    def generate_bias_report(self, comparison_df):
        weighted_bias = sum(
            comparison_df['Normalized Difference'] *
            [self.weights[m] for m in self.metrics]
        )

        report = {
            'table': tabulate(comparison_df, headers='keys', tablefmt='grid'),
            'overall_bias_score': weighted_bias,
            'bias_by_category': dict(zip(
                self.metrics,
                comparison_df['Normalized Difference'].tolist()
            ))
        }
        return report

    def plot_comparisons(self, comparison_df, location_name, output_path):
        plt.figure(figsize=(12, 6))
        x = np.arange(len(self.metrics))
        width = 0.35

        plt.bar(x - width / 2, comparison_df['Original'], width, label='Original')
        plt.bar(x + width / 2, comparison_df['AI Generated'], width, label='AI Generated')

        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Comparison of Metrics for {location_name}')
        plt.xticks(x, self.metrics, rotation=45)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{output_path}/{location_name}_comparison.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        diff_matrix = comparison_df['Normalized Difference'].values.reshape(1, -1)
        sns.heatmap(diff_matrix,
                    xticklabels=self.metrics,
                    yticklabels=['Bias Score'],
                    cmap='RdYlGn_r',
                    annot=True)

        plt.title(f'Bias Analysis Heatmap for {location_name}')
        plt.tight_layout()
        plt.savefig(f"{output_path}/{location_name}_heatmap.png")
        plt.close()
