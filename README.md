# Multi-Cancer Histology Classification

This repository contains **two** Google Colab notebooks for classifying histological images from the **Multi-Cancer** dataset using TensorFlow/Keras:

1. **ALL.ipynb** – A step-by-step pipeline that downloads the dataset, preprocesses images from the `ALL/` folder (all\_benign, all\_early, all\_pre, all\_pro), constructs data generators, trains a CNN, and evaluates baseline performance.
2. **ALL\_EfficientNetB0.ipynb** – An optimized version using **EfficientNetB0** as the backbone, with detailed model architecture, training callbacks, and advanced visualization of training/test metrics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Notebooks](#notebooks)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Kaggle API Setup](#kaggle-api-setup)
- [Running the Notebooks](#running-the-notebooks)
- [Project Structure](#project-structure)
- [Shared Workflow](#shared-workflow)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

Both notebooks implement image classification on histological slides from the `ALL/` subfolder of the Multi-Cancer dataset, covering **four classes**:

- `all_benign`
- `all_early`
- `all_pre`
- `all_pro`

They demonstrate how to:

- Download and unzip the Kaggle dataset
- Assemble a Pandas DataFrame of image file paths and labels
- Split data into train/validation/test with stratification
- Build Keras `ImageDataGenerator` pipelines
- Define, train, and evaluate CNN models
- Visualize training curves, confusion matrices, and sample predictions

## Notebooks

### 1. ALL.ipynb

- Uses a custom Sequential CNN with Conv2D/Pooling layers and dense heads.
- Illustrates basic data loading, augmentation, and model evaluation.
- Good for learning the end-to-end workflow.

### 2. ALL\_EfficientNetB0.ipynb

- Builds upon the first notebook by integrating `tf.keras.applications.EfficientNetB0` (ImageNet pretrained) as the feature extractor.
- Adds callbacks (EarlyStopping, ModelCheckpoint) and plots best-epoch indicators.
- Provides detailed result analysis (classification report, heatmap, sample grid).

## Dataset

Hosted on Kaggle as `obulisainaren/multi-cancer`. It follows this folder structure once unzipped:

```
multi-cancer.zip
└─ Multi Cancer/
   └─ Multi Cancer/
      ├─ ALL/
      │  ├─ all_benign/
      │  ├─ all_early/
      │  ├─ all_pre/
      │  └─ all_pro/
      └─ [other classes if included]
```

Each subfolder contains .png histology images.

## Prerequisites

- Python ≥ 3.8
- `pip` package manager
- A Kaggle account for dataset download

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/multi-cancer-classification.git
cd multi-cancer-classification

# Install dependencies
pip install -r requirements.txt
```

*(Both notebooks share the same requirements: TensorFlow 2.x, sklearn, pandas, joblib, seaborn, plotly, etc.)*

## Kaggle API Setup

1. Generate a Kaggle API token on your account settings and download `kaggle.json`.
2. Place it in `~/.kaggle/` and secure permissions:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
3. In each notebook, run:
   ```bash
   !kaggle datasets download -d obulisainaren/multi-cancer
   !unzip multi-cancer.zip -d /content/multi-cancer
   ```

## Running the Notebooks

- **ALL.ipynb:** [Open in Colab](https://colab.research.google.com/drive/<YOUR_NOTEBOOK_ID_ALL>)
- **ALL\_EfficientNetB0.ipynb:** [Open in Colab](https://colab.research.google.com/drive/<YOUR_NOTEBOOK_ID_EFFNET>)

Or run locally:

```bash
jupyter notebook ALL.ipynb ALL_EfficientNetB0.ipynb
```

## Project Structure

```
├── ALL.ipynb                   # Base CNN pipeline
├── ALL_EfficientNetB0.ipynb    # EfficientNetB0 optimized pipeline
├── requirements.txt            # Shared Python dependencies
├── data/                       # Unzipped dataset (user-provided)
└── README.md                   # This documentation
```

## Shared Workflow

1. **DataFrame assembly:** Scan `ALL/` subfolders, collect file paths and labels.
2. **Train/Val/Test split:** 70% / 15% / 15% stratified by label.
3. **Generators:** Use `ImageDataGenerator.flow_from_dataframe`.
4. **Modeling:**
   - ALL.ipynb: Custom CNN layers.
   - EfficientNetB0.ipynb: Pretrained base + dense head.
5. **Training:**
   - Loss: Categorical crossentropy
   - Optimizer: Adam/Adamax
   - Callbacks: EarlyStopping (patience=5)
6. **Evaluation:** Compute loss/accuracy on each split, plot metrics, show confusion matrix and sample predictions.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- Kaggle user **obulisainaren** for the dataset
- TensorFlow community for model implementations
- Colab environment for reproducible analysis

