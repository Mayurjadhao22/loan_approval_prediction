import os
import pickle
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        background-color: #2563eb;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
    }
    .result-approved {
        padding: 20px;
        border-radius: 10px;
        background-color: #dcfce7;
        color: #15803d;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        border: 1px solid #86efac;
    }
    .result-rejected {
        padding: 20px;
        border-radius: 10px;
        background-color: #fee2e2;
        color: #b91c1c;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        margin-top: 20px;
        border: 1px solid #fca5a5;
    }
    </style>
""", unsafe_allow_html=True)

# Cache model loading to prevent repeated file reading
@st.cache_resource
def load_model():
    # Looks for any model pickle file in the workspace
    possible_names = ["model.pkl", "loan_model.pkl", "decision_tree.pkl"]
    base_dir = os.path.dirname(__file__)
    
    for filename in possible_names:
        file_path = os.path.join(base_dir, filename)
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                return pickle.load(f)
                
    # Fallback to check any .pkl file in current directory
    for file in os.listdir(base_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(base_dir, file), "rb") as f:
                return pickle.load(f)

    return None

model = load_model()

st.title("🏦 Loan Eligibility Predictor")
st.write("Fill in the financial details below to check loan approval status.")

# Input Form
with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
        income_annum = st.number_input("Annual Income ($)", min_value=0, value=5000000, step=50000)
        loan_amount = st.number_input("Loan Amount Requested ($)", min_value=0, value=15000000, step=100000)
        cibil_score = st.number_input("CIBIL / Credit Score", min_value=300, max_value=900, value=750, step=1)

    with col2:
        residential_assets_value = st.number_input("Residential Assets Value ($)", min_value=0, value=4000000, step=50000)
        commercial_assets_value = st.number_input("Commercial Assets Value ($)", min_value=0, value=2000000, step=50000)
        luxury_assets_value = st.number_input("Luxury Assets Value ($)", min_value=0, value=10000000, step=50000)
        bank_asset_value = st.number_input("Bank Assets Value ($)", min_value=0, value=3000000, step=50000)

    submit_btn = st.form_submit_button("Predict Approval Status")

# Prediction Handler
if submit_btn:
    if model is None:
        st.error("Model file (.pkl) not found in directory. Please upload your model pickle file.")
    else:
        try:
            # Construct feature vector in exact model order
            features = np.array([[
                float(no_of_dependents),
                float(income_annum),
                float(loan_amount),
                float(cibil_score),
                float(residential_assets_value),
                float(commercial_assets_value),
                float(luxury_assets_value),
                float(bank_asset_value)
            ]])

            # Predict outcome
            prediction = model.predict(features)[0]

            # Output Formatting
            if str(prediction).strip().lower() in ["approved", "1"]:
                st.markdown(
                    f'<div class="result-approved">Status: Loan Approved ✅</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-rejected">Status: Loan Rejected ❌</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
