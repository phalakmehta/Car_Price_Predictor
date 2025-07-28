import streamlit as st
import pandas as pd
import pickle

# --- Load Assets ---
# This section attempts to load the dataset and the trained model pipeline.
# It will display an error and stop the app if the files are not found.
try:
    # Load the pre-cleaned dataset to populate the dropdowns
    df = pd.read_csv('cleaned_car_data.csv')
    # Load the trained machine learning pipeline
    with open('car_price_pipeline.pkl', 'rb') as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    st.error(
        "Required files are not found. Make sure 'cleaned_car_data.csv' and 'car_price_pipeline.pkl' are in the same directory.")
    st.stop()

# --- Page Configuration ---
st.set_page_config(layout="wide")
st.title("Car Price Predictor 🚗")
st.write("Fill in the details below to get a price estimate for your car.")
st.markdown("---")

# --- Create UI Elements in Columns ---
# The UI is organized into three columns for a clean layout.
col1, col2, col3 = st.columns(3)

# --- Column 1: Car Details ---
with col1:
    st.header("Car Details")
    # Dropdown for car brand, populated from the dataframe
    brands = sorted(df['brand'].dropna().unique())
    selected_brand = st.selectbox("Brand", brands)

    # Dropdown for car model, dynamically updated based on the selected brand
    models = sorted(df[df['brand'] == selected_brand]['model'].dropna().unique())
    selected_model = st.selectbox("Model", models)

    # Dropdown for city
    cities = sorted(df['city'].dropna().unique())
    selected_city = st.selectbox("City", cities)

# --- Column 2: Technical Specs ---
with col2:
    st.header("Technical Specs")
    # Number inputs for various technical specifications of the car
    car_age = st.number_input("Car Age (in years)", min_value=0, max_value=50, step=1)
    kms_driven = st.number_input("Kilometres Driven", min_value=0, step=1000)
    engine_capacity = st.number_input("Engine Capacity (CC)", min_value=600, max_value=6000, step=100)
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=40.0, step=0.5)
    seats = st.selectbox("Number of Seats", sorted(df['seats'].dropna().unique()))

# --- Column 3: Ownership & Condition ---
with col3:
    st.header("Ownership & Condition")
    # Dropdowns for ownership and condition details
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
# This block runs when the user clicks the "Predict Price" button.
if st.button("Predict Price", use_container_width=True, type="primary"):
    # Create a pandas DataFrame from the user's inputs.
    # The column names here MUST EXACTLY match the column names your model was trained on.
    # 'body_type' has been removed from this DataFrame.
    input_data = pd.DataFrame({
        'engine_capacity': [engine_capacity],
        'insurance': [selected_insurance],
        'transmission_type': [selected_transmission],
        'kms_driven': [kms_driven],
        'owner_type': [selected_owner],
        'fuel_type': [selected_fuel],
        'seats': [seats],
        'city': [selected_city],
        'mileage_numeric': [mileage],
        'car_age': [car_age],
        'brand': [selected_brand],
        'model': [selected_model]
    })

    try:
        # Use the loaded pipeline to make a prediction on the input data
        predicted_price_lakhs = pipeline.predict(input_data)[0]

        # Display the prediction in a success box, formatted to two decimal places
        st.success(f"## Estimated Price: ₹ {predicted_price_lakhs:.2f} Lakhs")
    except Exception as e:
        # Display an error message if the prediction fails for any reason
        st.error(f"An error occurred during prediction: {e}")




