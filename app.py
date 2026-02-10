app_py_content = """import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the pre-trained model
model = joblib.load('linear_regression_model.pkl')

# Load the cleaned data to get unique values for dropdowns and ranges
df = pd.read_csv('Cleaned_Car_data.csv')

# Streamlit app title
st.title('Car Price Predictor')

# Input widgets
company = st.selectbox('Company', sorted(df['company'].unique()))
name = st.selectbox('Model', sorted(df[df['company'] == company]['name'].unique()))
year = st.number_input('Year', min_value=int(df['year'].min()), max_value=int(df['year'].max()), value=2015, step=1)
kms_driven = st.number_input('Kilometers Driven', min_value=0, value=50000, step=1000)
fuel_type = st.selectbox('Fuel Type', sorted(df['fuel_type'].unique()))

# Prediction button
if st.button('Predict Price'):
    # Create a DataFrame from user inputs
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the predicted price
    st.success(f'Predicted Price: ₹ {int(prediction):,}')

"""
with open('app.py', 'w') as f:
    f.write(app_py_content)
