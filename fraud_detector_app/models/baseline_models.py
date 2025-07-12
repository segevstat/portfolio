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


# models/baseline_models.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import joblib
import os
from .data_utils import load_and_prepare_data # <--- THIS MUST BE 'from .data_utils'

def train_and_save_baselines(X_train, X_test, y_train, y_test):
    """
    Trains and saves baseline models (Logistic Regression, Random Forest).
    Accepts pre-split data.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = model.predict(X_test)

        print(f"\nEvaluation results for {name}:")
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_preds))
        print("Classification Report:")
        print(classification_report(y_test, y_preds, zero_division=0))
        save_path = r'C:\Users\segev\code_notebooks\fraud_ml_project\models'
        filename = os.path.join(save_path, f"{name.replace(' ', '_').lower()}.joblib")
        joblib.dump(model, filename)
        print(f"Saved {name} model to {filename}")