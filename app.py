import streamlit as st
import pandas as pd
import pickle

# --- Load Assets ---
try:
    # Use the correct file name for your cleaned data
    df = pd.read_csv('cleaned_car_data.csv')
    with open('car_price_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error(
        "Required files are not found. Make sure 'cleaned_car_data.csv' and 'car_price_pipeline.pkl' are in the same directory.")
    st.stop()

st.set_page_config(layout="wide")
st.title("Car Price Predictor 🚗")
st.write("Fill in the details below to get a price estimate for your car.")
st.markdown("---")

# --- Create UI Elements in Columns ---
col1, col2, col3 = st.columns(3)

with col1:
    st.header("Car Details")
    # Using correct column names from your file
    brands = sorted(df['brand'].dropna().unique())
    selected_brand = st.selectbox("Brand", brands)

    models = sorted(df[df['brand'] == selected_brand]['model'].dropna().unique())
    selected_model = st.selectbox("Model", models)

    body_types = sorted(df['body_type'].dropna().unique())
    selected_body = st.selectbox("Body Type", body_types)

    cities = sorted(df['city'].dropna().unique())
    selected_city = st.selectbox("City", cities)

with col2:
    st.header("Technical Specs")
    # Using correct column names
    car_age = st.number_input("Car Age (in years)", min_value=0, max_value=50, step=1)
    kms_driven = st.number_input("Kilometres Driven", min_value=0, step=1000)

    engine_capacity = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, step=100)
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, step=0.5)
    seats = st.selectbox("Number of Seats", sorted(df['seats'].dropna().unique()))

with col3:
    st.header("Ownership & Condition")
    fuel_types = sorted(df['fuel_type'].dropna().unique())
    selected_fuel = st.selectbox("Fuel Type", fuel_types)

    transmission_types = sorted(df['transmission_type'].dropna().unique())
    selected_transmission = st.selectbox("Transmission", transmission_types)

    owner_types = sorted(df['owner_type'].dropna().unique())
    selected_owner = st.selectbox("Owner Type", owner_types)

    insurance_types = sorted(df['insurance'].dropna().unique())
    selected_insurance = st.selectbox("Insurance Type", insurance_types)

st.markdown("---")

# --- Prediction Logic ---
if st.button("Predict Price", use_container_width=True, type="primary"):
    # Create a DataFrame from the inputs using the EXACT column names from your file
    input_data = pd.DataFrame({
        'engine_capacity': [engine_capacity],
        'insurance': [selected_insurance],
        'transmission_type': [selected_transmission],
        'kms_driven': [kms_driven],
        'owner_type': [selected_owner],
        'fuel_type': [selected_fuel],
        'seats': [seats],
        'body_type': [selected_body],
        'city': [selected_city],
        'mileage_numeric': [mileage],
        'car_age': [car_age],
        'brand': [selected_brand],
        'model': [selected_model]
    })

    try:
        # Predict the price
        predicted_price_lakhs = pipeline.predict(input_data)[0]

        # Display the result
        st.success(f"## Estimated Price: ₹ {predicted_price_lakhs:.2f} Lakhs")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")



