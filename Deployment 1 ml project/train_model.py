# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 02:43:33 2025

@author: sjyti
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load your dataset
data = pd.read_csv("D:\\deployment project\\diabetes.csv")  # replace with your dataset path

X = data.drop("Outcome", axis=1)  # Features
y = data["Outcome"]               # Target

# 2. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save the model correctly as .pkl
with open("D:\\deployment project\\diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")
