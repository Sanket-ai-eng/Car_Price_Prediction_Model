import streamlit as st
import pandas as pd
import joblib

model = joblib.load("linear_regression_model.pkl")
df = pd.read_csv("Cleaned_Car_data.csv")

st.title("🚗 Car Price Prediction")

company = st.selectbox("Select Company", sorted(df['company'].unique()))
name = st.selectbox(
    "Select Car Model",
    sorted(df[df['company'] == company]['name'].unique())
)

year = st.number_input("Year", 1990, 2025, step=1)
kms_driven = st.number_input("Kilometers Driven", min_value=0)
fuel_type = st.selectbox("Fuel Type", df['fuel_type'].unique())

if st.button("Predict Price"):
    input_df = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
    price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹{int(price):,}")
