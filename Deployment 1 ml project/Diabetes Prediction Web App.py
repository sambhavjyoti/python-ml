# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 02:47:40 2025

@author: sjyti
"""

import streamlit as st
import pickle

# 1. Load the trained model
with open("D:\\deployment project\\diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

def main():
    st.title("Diabetes Prediction Web App")

    # 2. Get user input
    Pregnancies = st.number_input("Number of Pregnancies", 0, 20, 0)
    Glucose = st.number_input("Glucose Level", 0, 200, 100)
    BloodPressure = st.number_input("Blood Pressure", 0, 140, 70)
    SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)
    Insulin = st.number_input("Insulin Level", 0, 900, 80)
    BMI = st.number_input("BMI", 0.0, 70.0, 20.0)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.number_input("Age", 0, 120, 25)

    # 3. Predict
    if st.button("Predict"):
        result = model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        st.success(f"Prediction: {'Diabetic' if result[0]==1 else 'Non-Diabetic'}")

if __name__ == "__main__":
    main()
