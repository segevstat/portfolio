#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[14]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from .feature_engineering import preprocess # <--- THIS MUST BE 'from .feature_engineering'

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

# def preprocess(df):
#     # Make a copy to avoid side-effects
#     df = df.copy()
#     scaler = StandardScaler()
#     df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])
#     df.drop(['Amount', 'Time'], axis=1, inplace=True)
#     X = df.drop('Class', axis=1)
#     y = df['Class']
#     return X, y

def load_and_prepare_data(percent=0.25, test_size=0.2, seed=42):
    """
    Loads a percentage of the dataset, applies preprocessing, and returns a train/test split.
    """
    if percent < 1:
        total_rows = sum(1 for _ in pd.read_csv(url, chunksize=10000))
        total_rows -= 1
        nrows = int(total_rows * percent)
        df = pd.read_csv(url, nrows=nrows)
    else:
        df = pd.read_csv(url)
    X, y = preprocess(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)

