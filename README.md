# SoraVs  
## Urban Embeddings for AI Fairness: A Comparative Study of AI-Generated and Real-World Streetscapes

---

A comparative research project examining the fairness and bias of AI-generated video renderings of European and Middle Eastern cities versus genuine street-level imagery. The pipeline leverages deep learning and computer vision techniques to assess object distribution, color, vegetation, and visual style, quantifying and visualizing any identified disparities.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Pipeline Overview](#pipeline-overview)
- [Directory Structure](#directory-structure)
- [Core Features](#core-features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Reproducibility & Extension](#reproducibility--extension)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

---

## Project Overview

This project investigates whether generative AI models like Sora and Kling produce biased, incomplete, or skewed representations of city environments when compared to authentic Google Street View imagery. It systematically compares distributions of people, vehicles, vegetation, color palettes, and architectural segmentation in cities of Europe and the Middle East, aiming to highlight unintended bias and inspire more equitable generative models.

---

## Pipeline Overview

```mermaid
flowchart TD
    A[Data Collection] --> B1[Collect Ground-Truth Images (Google Maps Street View)]
    A --> B2[Generate AI Videos (Sora & Kling AI)]
    B1 --> C1[Split Images Vertically]
    B2 --> C2[Sample Video Frames]
    C1 --> D[Feature Analysis]
    C2 --> D
    D --> E1[Vegetation Analyzer (GVI)]
    D --> E2[VGG Feature Extraction]
    D --> E3[Color Distribution Analysis]
    D --> E4[Car & Human Detection (YOLOv9e)]
    D --> E5[Urban Feature Detection (DeepLabv3+)]
    E1 --> F[Results Compilation]
    E2 --> F
    E3 --> F
    E4 --> F
    E5 --> F
    F --> G[Bias Analysis]
    G --> H1[Style Consistency]
    G --> H2[Color Bias]
    G --> H3[Content Preservation]
    G --> H4[Artifact Detection]
    G --> H5[Overall Bias]
```

---

## Directory Structure

```text
.
├── data/
│   ├── ai_generated/
│   └── original/
├── output/
│   ├── frames/
│   └── results/
│       ├── analysis_results.json
│       └── bias_analysis.png
├── src/
│   ├── features/
│   │   ├── bias_analyzer.py
│   │   ├── car_detector.py
│   │   ├── color_analyzer.py
│   │   ├── comparison_analyzer.py
│   │   ├── deeplabv3plus_analyzer.py
│   │   ├── human_detector.py
│   │   ├── vegetation_analyzer.py
│   │   └── vgg_analyzer.py
│   ├── utils/
│   │   ├── image_processor.py
│   │   └── video_processor.py
│   ├── main.py
│   └── yolov9e.pt
├── requirements.txt
└── README.md
```

**Essential files:**
- All scripts in `src/features/` and `src/utils/`
- `src/main.py` (delete any other `main.py`)
- Data folders: `data/ai_generated/`, `data/original/`
- YOLO model: `src/yolov9e.pt`
- Output: `output/results/analysis_results.json`, `bias_analysis.png`
- `requirements.txt`

---

## Core Features

- **Vegetation Analysis**: Calculates the Green View Index (GVI) for quantifying greenery using DeepLabV3 segmentation.
- **Object Detection**: Detects people and vehicles (cars, buses, trucks) using YOLOv9e.
- **Visual Similarity**: VGG16 feature extraction for comparing overall "style."
- **Color Analysis**: Computes HSV histograms & clusters dominant colors for palette and distribution comparison.
- **Urban Feature Detection**: Semantic segmentation for identifying architectural and urban elements.
- **Bias Quantification**: Computes human, vehicle, vegetation, color, segmentation, and style biases using normalized differences and weighted combinations.
- **Visualization**: Produces bar/heatmap plots summarizing results for each city and region.

---

## Getting Started

### Requirements

- Python 3.10+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Download YOLOv9e weights (`yolov9e.pt`) to `src/`

### Data Preparation

1. Place original Google Maps images in `data/original/`, named `{city}_Original.png`
2. Put AI-generated videos in `data/ai_generated/`, named `{city}_AI.mp4`

---

## Usage

**Run the full analysis:**
```bash
python src/main.py
```
- Output summary (`bias_analysis.png`) and data dump (`analysis_results.json`) will be in `output/results/`.

---

## Sample Output

- **analysis_results.json**: Raw features and bias scores for each location.
- **bias_analysis.png**: Multi-metric bar/heatmap visualization per city and region.

Example entry in `analysis_results.json`:
```json
{
  "Dublin_Docklands": {
    "original": { ... },
    "ai_generated": { ... },
    "bias_metrics": {
      "humans": 0.32,
      "vehicles": 0.12,
      "vegetation": 0.65,
      "color": 0.20,
      "visual_similarity": 0.74,
      "segmentation": 0.91,
      "overall": 0.46
    }
  }
}
```

---

## Reproducibility & Extension

- To add a new feature or detection class, create a new file in `src/features/` and hook it into the main pipeline.
- All metric weighting and normalization logic can be adjusted in `src/features/bias_analyzer.py` and `src/main.py`.
- Aggregation logic (Europe vs. ME) is customizable for other regions or data splits.

---

## Contributing

Contributions, extensions, and bug-fixes are welcome!
- Fork the repository, create a feature branch, and open a pull request.
- Please document new metric or feature additions with specific docstrings and update this README.

---

## Credits

- Core code and pipeline: **Abylay Iskakov** , **Yuanzhu Li**
- Built on: [ultralytics/YOLOv9](https://github.com/ultralytics/ultralytics), [PyTorch/torchvision](https://pytorch.org/vision/stable/models.html), [Google Maps Street View](https://www.google.com/maps), Sora & Kling AI

---

## License

*MIT License. 2025.*

---
