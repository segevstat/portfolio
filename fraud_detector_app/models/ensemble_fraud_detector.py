#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
models/

│
├── feature_engineering.py        → Clean + scale features (e.g. Amount, Time)
│
├── baseline_models.py            → Train + save baseline models (LogReg, RF)
│
├── ensemble_fraud_detector.py    → Load saved models, combine them (Ensemble), evaluate
│
├── data_utils.py                 → Load dataset, calculate % of rows, split train/test
"""

# models/ensemble_fraud_detector.py
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
# If you need to import from other modules in the 'models' package, use relative imports:
# from .feature_engineering import preprocess
# from .data_utils import load_and_prepare_data

def run_ensemble_model(X_test, y_test):
    """
    Loads saved baseline models, combines their predictions, and evaluates
    the ensemble model. Accepts pre-split test data.
    """
    logreg_path = r"C:\Users\segev\code_notebooks\fraud_ml_project\models\logistic_regression.joblib"
    rf_path = r"C:\Users\segev\code_notebooks\fraud_ml_project\models\random_forest.joblib"

    try:
        logreg = joblib.load(logreg_path)
        rf = joblib.load(rf_path)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}. Make sure baseline models are trained and saved first.")
        return

    logreg_probs = logreg.predict_proba(X_test)[:, 1]
    rf_probs = rf.predict_proba(X_test)[:, 1]

    ensemble_probs = (logreg_probs + rf_probs) / 2
    ensemble_preds = (ensemble_probs >= 0.5).astype(int)

    print("📊 תוצאות המודל המשולב (Ensemble):")
    print(f"ROC AUC: {roc_auc_score(y_test, ensemble_probs):.4f}")
    print("🔹 Confusion Matrix:")
    print(confusion_matrix(y_test, ensemble_preds))
    print("🔹 Classification Report:")
    print(classification_report(y_test, ensemble_preds, zero_division=0))