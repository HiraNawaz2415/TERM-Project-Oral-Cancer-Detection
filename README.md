# Term Project AIHC
# 🦷 Oral Cancer Detection using Deep Learning (VGG16 + CNN)

![Accuracy](https://img.shields.io/badge/Accuracy-88%25-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-orange)

This repository implements a **binary image classification system** for **Oral Cancer Detection** using **Deep Learning and Transfer Learning (VGG16)**.  
The model classifies oral cavity images into **Oral Cancer** and **Normal** classes and includes evaluation and explainability using **Grad-CAM**.

---

## Dataset Description

The dataset is organized into two folders:

- **Oral Cancer photos/** → Images showing oral cancer
- **normal/** → Healthy oral images
- Images are resized to **224 × 224**
- RGB images
- Labels:
  - `0` → Normal  
  - `1` → Oral Cancer

---

##  Model Overview

###  Base Model
- **VGG16**
- Pretrained on **ImageNet**
- `include_top = False`
- Convolutional layers are frozen

###  Custom Classification Head
- Flatten
- Dense (256, ReLU)
- Dropout (0.5)
- Dense (1, Sigmoid)

This approach improves performance with limited medical data.

---

##  Model Architecture

| Component | Details |
|---------|--------|
| Input Shape | (224, 224, 3) |
| Base Network | VGG16 |
| Activation | ReLU, Sigmoid |
| Loss Function | Binary Crossentropy |
| Optimizer | Adam |
| Output Classes | Normal / Oral Cancer |

---

##  Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 📈 Performance Results

###  Classification Report

| Class | Precision | Recall | F1-score | Support |
|------|----------|--------|---------|---------|
| Normal (0) | 0.85 | 0.89 | 0.87 | 83 |
| Oral Cancer (1) | 0.91 | 0.87 | 0.89 | 103 |
| **Overall Accuracy** | | | **0.88** | 186 |

---

###  Confusion Matrix

| Actual \ Predicted | Normal | Cancer |
|-------------------|--------|--------|
| Normal | 74 | 9 |
| Oral Cancer | 13 | 90 |

---

## 📉 ROC Curve & AUC

- ROC curve shows the trade-off between **True Positive Rate** and **False Positive Rate**
- **AUC (Area Under Curve)** indicates how well the model separates cancer vs normal images  
- Higher AUC means better diagnostic performance

---

##  Explainable AI (XAI)

To understand model decisions:

- **Grad-CAM** is used
- Highlights important regions in oral images
- Improves trust and interpretability in medical AI

---

##  Technologies Used

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebook

---

---

##  How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/oral-cancer-detection.git

