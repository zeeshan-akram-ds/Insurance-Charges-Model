import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load trained model
model = joblib.load("charges_pipeline.pkl")

# Page config
st.set_page_config(page_title="Insurance Charges Estimator", layout="centered")
st.title("Insurance Charges Estimator")
st.markdown("Fill out the form below to estimate your insurance charges.")

# Reset 
if "reset" not in st.session_state:
    st.session_state.reset = False

# Form Layout
with st.form("insurance_form", clear_on_submit=st.session_state.reset):
    age = st.slider(
        "Age", 18, 100, 30,
        help="Your age in years"
    )
    sex = st.selectbox(
        "Sex", ["male", "female"],
        help="Biological sex"
    )
    bmi = st.slider(
        "BMI (Body Mass Index)", 10.0, 50.0, 22.0,
        help="Body Mass Index, a measure of body fat"
    )
    children = st.number_input(
        "Number of Children", 0, 10, 2,
        help="Number of dependent children"
    )
    smoker = st.selectbox(
        "Do you smoke?", ["yes", "no"],
        help="Do you currently smoke?"
    )
    region = st.selectbox(
        "Region", ["northeast", "northwest", "southeast", "southwest"],
        help="Region where you live"
    )

    # Buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        predict_button = st.form_submit_button("Predict Insurance Charges")
    with col2:
        reset_button = st.form_submit_button("Reset Form")
        if reset_button:
            st.session_state.reset = True
            st.rerun()

# Prediction 
if predict_button:
    st.session_state.reset = False

    # input
    input_df = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })

    # apply inverse log
    log_prediction = model.predict(input_df)[0]
    insurance_cost = np.expm1(log_prediction)

    st.subheader(f"Estimated Insurance Charges: ${insurance_cost:,.2f}")

    # Bar chart of numeric inputs
    st.markdown("### Input Overview")
    chart_data = pd.DataFrame({
        'Feature': ['Age', 'BMI', 'Children'],
        'Value': [age, bmi, children]
    })

    fig, ax = plt.subplots()
    ax.bar(chart_data['Feature'], chart_data['Value'], color=["#6baed6", "#fd8d3c", "#74c476"])
    ax.set_ylabel("Value")
    ax.set_title("Input Summary")
    st.pyplot(fig)
