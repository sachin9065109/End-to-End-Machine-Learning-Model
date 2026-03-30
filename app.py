import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Loan Prediction App")

age = st.number_input("Enter Age")
income = st.number_input("Enter Income")

if st.button("Predict"):
    data = np.array([[age, income]])
    result = model.predict(data)
    st.success(f"Loan Prediction: {result[0]}")