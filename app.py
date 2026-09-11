import pickle
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Loan Approval Predictor", page_icon="🏦", layout="centered"
)


@st.cache_resource
def load_model():
    with open("decision.pkl", "rb") as f:
        model = pickle.load(f)
    return model


model = load_model()

st.title("🏦 Loan Approval Prediction App")
st.write(
    "Enter applicant details below to check loan eligibility status."
)

st.subheader("Applicant Details")
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input(
        "Number of Dependents", min_value=0, max_value=20, value=2, step=1
    )
    income_annum = st.number_input(
        "Annual Income (₹)", min_value=0, value=5000000, step=100000
    )
    loan_amount = st.number_input(
        "Loan Amount Requested (₹)", min_value=0, value=15000000, step=100000
    )
    cibil_score = st.slider(
        "CIBIL Score", min_value=300, max_value=900, value=750, step=1
    )

with col2:
    residential_assets_value = st.number_input(
        "Residential Asset Value (₹)", min_value=0, value=2000000, step=100000
    )
    commercial_assets_value = st.number_input(
        "Commercial Asset Value (₹)", min_value=0, value=0, step=100000
    )
    luxury_assets_value = st.number_input(
        "Luxury Asset Value (₹)", min_value=0, value=5000000, step=100000
    )
    bank_asset_value = st.number_input(
        "Bank Asset Value (₹)", min_value=0, value=1000000, step=100000
    )

st.markdown("---")

if st.button("Predict Approval Status", use_container_width=True):
    # Features in exact positional order expected by the model
    features_list = [
        no_of_dependents,
        income_annum,
        loan_amount,
        cibil_score,
        residential_assets_value,
        commercial_assets_value,
        luxury_assets_value,
        bank_asset_value,
    ]

    # Convert to a 2D DataFrame with exact column names stored inside the model
    input_df = pd.DataFrame([features_list], columns=model.feature_names_in_)

    # Predict approval status
    prediction = model.predict(input_df)[0]

    if str(prediction).strip() == "Approved":
        st.success("🎉 **Status: Loan Approved**")
    else:
        st.error("❌ **Status: Loan Rejected**")
