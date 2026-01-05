import streamlit as st
import pickle
import numpy as np

# Load model
with open("loan_approval_naive_bayes.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🏦 Loan Approval Prediction (Naive Bayes)")

st.write("Customer details enter pannunga 👇")

# ---- Inputs ----
dependents = st.number_input("No of Dependents", min_value=0, value=1)

education = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed = st.selectbox("Self Employed", ["No", "Yes"])

income_annum = st.number_input("Annual Income", value=500000)
loan_amount = st.number_input("Loan Amount", value=200000)
loan_term = st.number_input("Loan Term (months)", value=360)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750)

residential_assets_value = st.number_input("Residential Assets Value", value=100000)
commercial_assets_value = st.number_input("Commercial Assets Value", value=0)
luxury_assets_value = st.number_input("Luxury Assets Value", value=0)
bank_asset_value = st.number_input("Bank Asset Value", value=50000)

# ---- Encoding (same as training) ----
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# ---- Prediction ----
if st.button("Predict Loan Status"):
    input_data = np.array([[dependents, education, self_employed,
                            income_annum, loan_amount, loan_term,
                            cibil_score, residential_assets_value,
                            commercial_assets_value, luxury_assets_value,
                            bank_asset_value]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")