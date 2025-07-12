#!/usr/bin/env python
# coding: utf-8

# In[6]:


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


# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[5]:


url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)



# In[16]:



def preprocess(df):
    df = df.copy()
    scaler = StandardScaler()

    df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(df[['Amount', 'Time']])
    df.drop(['Amount', 'Time'], axis=1, inplace=True)

    # הגנה: רק אם 'Class' קיימת, נסיר ונגדיר y
    if 'Class' in df.columns:
        X = df.drop('Class', axis=1)
        y = df['Class']
    else:
        X = df
        y = None

    return X, y


