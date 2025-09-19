# Step 1: Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import warnings

# Ignore all warnings to keep the output clean
warnings.filterwarnings('ignore')

# Step 2: Load the trained model
# The model file 'loan_approval_model.pkl' must exist in the same directory as this script.
try:
    model = joblib.load('loan_approval_model.pkl')
    # Use the model's feature names to ensure our input order is correct.
    if hasattr(model, 'feature_names_in_'):
        expected_feature_order = list(model.feature_names_in_)
    else:
        # If the model doesn't have `feature_names_in_`, we rely on our predefined order.
        expected_feature_order = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area'
        ]
    st.write("Model loaded successfully.")
except FileNotFoundError:
    st.error("Error: The file 'loan_approval_model.pkl' was not found. Please ensure it's in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model: {e}")
    st.stop()

# Step 3: Define the feature mapping
feature_mapping = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 3},
    'Education': {'Graduate': 0, 'Not Graduate': 1},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
}

# --- Streamlit UI Components ---

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

# Use columns for a clean, two-column layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://placehold.co/150x150/F87171/FFFFFF?text=Loan", width=150)

with col2:
    st.title("Loan Approval Predictor")
    st.markdown("Enter the applicant's details to predict their loan approval status.")

st.markdown("---")

# Use a form to group all input widgets
with st.form(key='loan_form'):
    # Create input widgets for each feature using Streamlit
    gender = st.selectbox("Gender", options=list(feature_mapping['Gender'].keys()))
    married = st.selectbox("Married", options=list(feature_mapping['Married'].keys()))
    dependents = st.selectbox("Dependents", options=list(feature_mapping['Dependents'].keys()))
    education = st.selectbox("Education", options=list(feature_mapping['Education'].keys()))
    self_employed = st.selectbox("Self Employed", options=list(feature_mapping['Self_Employed'].keys()))

    applicant_income = st.number_input("Applicant Income", min_value=0, value=50000, step=1000)
    coapplicant_income = st.number_input("Co-applicant Income", min_value=0, value=15000, step=1000)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=120000, step=1000)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, value=360, step=12)
    credit_history = st.selectbox("Credit History", options=[1.0, 0.0])
    property_area = st.selectbox("Property Area", options=list(feature_mapping['Property_Area'].keys()))

    # Create a submit button for the form
    submit_button = st.form_submit_button(label='Predict')

# Step 4: Make a prediction when the form is submitted
if submit_button:
    # Gather the input data into a dictionary
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Convert categorical data to numerical format
    processed_data = {}
    for key, value in input_data.items():
        if key in feature_mapping:
            processed_data[key] = feature_mapping[key].get(value, -1)
        else:
            processed_data[key] = float(value)

    # Create a list of values in the correct order
    ordered_values = [processed_data[feature] for feature in expected_feature_order]

    # Create a pandas DataFrame from the ordered values
    input_df = pd.DataFrame([ordered_values], columns=expected_feature_order)

    # Make the prediction
    prediction = model.predict(input_df)[0]

    st.markdown("---")

    # Display the result
    if prediction == 1:
        st.success("Predicted Status: Loan Approved")
    else:
        st.error("Predicted Status: Loan Rejected")

    st.markdown("This prediction is based on the trained machine learning model.")
