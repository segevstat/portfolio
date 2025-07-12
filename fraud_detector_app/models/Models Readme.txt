# 📦 models/ — Machine Learning Models & Processing

This folder contains all Python modules related to model development, training, saving, and evaluation for the **Credit Card Fraud Detection** project.

---

## 📁 Folder Purpose

The `models/` directory encapsulates the core logic for building and using machine learning models, including:
- Preprocessing functions
- Baseline model training
- Ensemble model evaluation

It forms the **backbone** of the fraud detection system.

---

## 📂 File Overview

### 🔹 `feature_engineering.py`
> Handles all preprocessing and feature transformation steps.

- **Key Function**:
  ```python
  preprocess(df) → (X, y)
 