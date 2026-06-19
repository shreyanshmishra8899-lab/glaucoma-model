# 🔬 Glaucoma Detection — EfficientNetB4 Transfer Learning

A deep learning pipeline for automated glaucoma detection from retinal fundus images, built with TensorFlow/Keras and EfficientNetB4.

---

## 📋 Overview

This project classifies retinal fundus images into two categories:
- **NRG** — Non-Referable Glaucoma (healthy)
- **RG** — Referable Glaucoma (disease present)

The model uses a two-phase transfer learning strategy on top of an **EfficientNetB4** backbone pretrained on ImageNet, with CLAHE contrast enhancement as a clinically-motivated preprocessing step.

---

## 🗂️ Dataset

**Source:** [Glaucoma Dataset — EyePACS AIROGS Light v2](https://www.kaggle.com/datasets/deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2)

| Split      | NRG   | RG   | Total |
|------------|-------|------|-------|
| Train      | 4,000 | 4,000 | 8,000 |
| Validation | 385   | 385  | 770   |
| Test       | 385   | 385  | 770   |

Expected folder structure:
```
dataset/
├── train/
│   ├── NRG/
│   └── RG/
├── validation/
│   ├── NRG/
│   └── RG/
└── test/
    ├── NRG/
    └── RG/
```

---

## 🏗️ Model Architecture — GlaucomaNet B4

```
Input (380×380×3)
    └── EfficientNetB4 Backbone (ImageNet weights)
        └── Feature Refinement (Conv1×1 → BN → ReLU → Dropout)
            └── Dual Pooling (GAP + GMP → Concatenate → 1024-d)
                └── Classification Head
                    ├── Dense(512) → BN → ReLU → Dropout(0.5)
                    ├── Dense(256) → BN → ReLU → Dropout(0.3)
                    ├── Dense(128) → BN → ReLU → Dropout(0.2)
                    └── Dense(2) → Softmax
```

Key design choices:
- **Dual pooling** (GAP + GMP concatenated) for richer feature aggregation
- **Mixed precision** (`float16`) training for faster GPU throughput
- **L2 regularization** on all Dense layers
- **BatchNorm frozen** during fine-tuning to prevent feature drift

---

## ⚙️ Pipeline

### 1. EDA
- Class distribution across splits
- Sample image grids
- Per-channel pixel intensity distributions (KDE plots)
- Image resolution audit

### 2. Preprocessing — CLAHE
CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied on the **LAB L-channel** to enhance optic disc visibility, which is clinically relevant for glaucoma diagnosis.

### 3. Data Augmentation
Applied during training only:
- Random horizontal & vertical flips
- Random rotation (±20°)
- Random zoom (±15%)
- Random brightness & contrast (±15%)
- Random translation (±10%)

### 4. Two-Phase Training

| Phase | Backbone | LR | Epochs |
|-------|----------|----|--------|
| 1 — Head Training | Frozen | 1e-3 | 20 |
| 2 — Fine-Tuning | Top 40 layers unfrozen | 1e-5 | 30 |

**Callbacks:** ModelCheckpoint (best val AUC), EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger

### 5. Evaluation
- Confusion matrix
- ROC curve + AUC
- Precision-Recall curve + AUC
- Full classification report

---

## 📦 Requirements

```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python pandas
```

| Package | Version |
|---------|---------|
| TensorFlow | ≥ 2.12 |
| scikit-learn | ≥ 1.0 |
| OpenCV | ≥ 4.5 |
| NumPy | ≥ 1.21 |
| Pandas | ≥ 1.3 |
| Matplotlib | ≥ 3.5 |
| Seaborn | ≥ 0.11 |

---

## 🚀 Usage

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/glaucoma-detection.git
cd glaucoma-detection
```

### 2. Configure paths
Open the notebook and update the paths in **Section 1 — Paths & Configuration**:
```python
TRAIN_DIR = "/path/to/your/train"
VAL_DIR   = "/path/to/your/validation"
TEST_DIR  = "/path/to/your/test"
```

### 3. Run the notebook
```bash
jupyter notebook Glaucoma_model.ipynb
```
Or run on Kaggle directly with the dataset linked above.

---

## 📁 Outputs

After training, all artifacts are saved to `outputs/`:

```
outputs/
├── glaucoma_efficientnetb4_final.keras       # Final model (Keras format)
├── glaucoma_efficientnetb4_savedmodel/       # TF SavedModel (for serving)
├── checkpoints/
│   ├── phase1_best.keras
│   └── phase2_best.keras
├── history_phase1.csv
├── history_phase2.csv
├── class_distribution.png
├── sample_images.png
├── pixel_intensity.png
├── augmentation_preview.png
├── training_curves.png
├── confusion_matrix.png
└── roc_pr_curves.png
```

---

## 📊 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 380 × 380 |
| Batch size | 64 |
| Phase 1 LR | 1e-3 |
| Phase 2 LR | 1e-5 |
| Dropout | 0.5 (head) |
| Weight decay | 1e-4 |
| Unfrozen layers (Phase 2) | 40 |
| Optimizer | AdamW |
| Loss | Categorical Crossentropy |
| Seed | 42 |

---

## 📌 Notes

- The dataset is **balanced** (equal NRG/RG samples), so class weights are approximately 1.0, but the weight computation is kept in the pipeline for extensibility with imbalanced datasets.
- **BatchNormalization layers are always kept frozen** during fine-tuning to preserve stable feature statistics learned in Phase 1.
- Mixed precision (`float16`) is enabled for GPU efficiency; outputs use `float32` softmax to avoid numerical instability.

---

## 📄 License

This project is for research and educational purposes. Please refer to the dataset's original license before any commercial use.
