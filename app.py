import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("breast_cancer_model_5.h5")
scaler = joblib.load("scaler_5.pkl")

st.title("🔬 Breast Cancer Prediction App (5 Features)")
st.write("Enter patient details to predict whether the tumor is Benign or Malignant.")

# Only 5 selected features
radius = st.number_input("Mean Radius", min_value=0.0, value=14.0)
texture = st.number_input("Mean Texture", min_value=0.0, value=20.0)
smoothness = st.number_input("Mean Smoothness", min_value=0.0, value=0.1)
concavity = st.number_input("Mean Concavity", min_value=0.0, value=0.05)
symmetry = st.number_input("Mean Symmetry", min_value=0.0, value=0.2)

if st.button("Predict"):
    input_data = np.array([[radius, texture, smoothness, concavity, symmetry]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0][0]
    result = "Benign" if prediction > 0.5 else "Malignant"
    st.subheader(f"🔍 Prediction: {result}")
    st.write(f"Confidence Score (probability of Benign): {prediction:.4f}")

