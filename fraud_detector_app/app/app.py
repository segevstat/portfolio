#!/usr/bin/env python
# coding: utf-8

# pip install streamlit

import sys
import os

# Allow Python to import from the parent directory (for models package)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import joblib
from models.feature_engineering import preprocess

# Configure the Streamlit app
st.set_page_config(page_title="Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")

# --- Load trained models ---
logreg_path = r"C:\Users\segev\code_notebooks\fraud_ml_project\models\logistic_regression.joblib"
rf_path = r"C:\Users\segev\code_notebooks\fraud_ml_project\models\random_forest.joblib"

logreg = joblib.load(logreg_path)
rf = joblib.load(rf_path)

# --- Input form for transaction data ---
st.subheader("Enter Transaction Details")

# Example inputs: replace V1, V2 with clearer placeholders
pca1 = st.number_input("PCA_Feature_1", value=0.0)
pca2 = st.number_input("PCA_Feature_2", value=0.0)
amount = st.number_input("Amount (Transaction Value)", value=50.0)
time = st.number_input("Time (Seconds Since First Transaction)", value=10000)

# Create DataFrame for input
input_df = pd.DataFrame([[pca1, pca2, amount, time]], columns=["V1", "V2", "Amount", "Time"])

# Add placeholder values for the rest of the anonymized PCA components
for i in range(3, 29):
    input_df[f"V{i}"] = 0.0

# --- Preprocess ---
X, _ = preprocess(input_df)

# --- Predict using ensemble ---
logreg_prob = logreg.predict_proba(X)[:, 1]
rf_prob = rf.predict_proba(X)[:, 1]
ensemble_prob = (logreg_prob + rf_prob) / 2
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

# --- Show prediction ---
st.subheader("🧠 Prediction Result")
label = "🚨 Fraud Detected!" if ensemble_pred[0] == 1 else "✅ Legitimate Transaction"
st.write(label)
st.write(f"Fraud Probability: **{ensemble_prob[0]:.2%}**")
