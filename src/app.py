import streamlit as st
import numpy as np
from predict import predict_transaction

st.title("Credit Card Fraud Detection")

st.write("Enter transaction details to check if it's fraudulent.")

time = st.number_input("Time", min_value=0.0)
amount = st.number_input("Amount", min_value=0.0)

v_inputs = []
for i in range(1, 29):
    v_inputs.append(st.number_input(f"V{i}", value=0.0))

if st.button("Predict"):
    features = [time, amount] + v_inputs
    pred, prob = predict_transaction(features)
    if pred == 1:
        st.error(f"⚠️ Fraud Detected! (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Legitimate Transaction (Probability: {prob:.2f})")
