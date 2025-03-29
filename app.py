import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🏡 House Price Prediction")
st.write("Enter the area (in sq. ft.) to predict the house price.")

area = st.number_input("Enter Area (sq. ft.)", min_value=100, max_value=10000, step=50)

if st.button("Predict Price"):
    prediction = model.predict(np.array([[area]]))[0]
    st.success(f"🏠 Estimated Price: ${prediction:.2f}K")
