import streamlit as st
from model_utils import load_model, predict_price

st.title("SmartCity — Traffic ML Dashboard")

model = load_model()

route = st.selectbox("Route", ["Route_1", "Route_2"])
date = st.date_input("Date")
time = st.time_input("Time")
cab_type = st.selectbox("Cab type", ["yellow", "green"])

if st.button("Predict"):
    result = predict_price(model, route, date, time, cab_type)
    st.success(f"Predicted price: {result}")
