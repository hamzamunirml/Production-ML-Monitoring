"""
Bank Marketing Prediction - Streamlit App
"""

import streamlit as st
import requests
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title
st.title("🏦 Bank Marketing Prediction System")
st.markdown("### Will the customer subscribe to a term deposit?")

# Sidebar for input
st.sidebar.header("📝 Customer Information")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    job = st.selectbox("Job", [
        "management", "technician", "blue-collar", "admin", "services",
        "retired", "entrepreneur", "student", "unemployed", "housemaid"
    ])
    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["tertiary", "secondary", "primary", "unknown"])
    balance = st.number_input("Account Balance", value=1000)

with col2:
    housing = st.selectbox("Housing Loan", ["yes", "no"])
    loan = st.selectbox("Personal Loan", ["yes", "no"])
    duration = st.number_input("Call Duration (seconds)", min_value=0, value=200)
    default = st.selectbox("Credit Default", ["no", "yes"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])

# Advanced options (optional)
with st.expander("🔧 Advanced Options (Optional)"):
    day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
    month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    campaign = st.number_input("Number of Contacts", min_value=1, value=1)
    pdays = st.number_input("Days Since Last Contact", value=-1)
    previous = st.number_input("Previous Contacts", value=0)
    poutcome = st.selectbox("Previous Outcome", ["unknown", "success", "failure"])

# Predict button
predict_button = st.button("🔮 Predict", type="primary", use_container_width=True)

# API URL (change to your deployed URL or local)
# For local testing: "http://localhost:8000/predict"
# For deployed: "https://hamzamunirml.pythonanywhere.com/predict"
API_URL = "https://hamzamunirml.pythonanywhere.com/predict"

if predict_button:
    with st.spinner("Making prediction..."):
        # Prepare data
        customer_data = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "duration": duration,
            "default": default,
            "contact": contact,
            "day": day,
            "month": month,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }
        
        try:
            # Make API request
            response = requests.post(API_URL, json=customer_data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                st.success("✅ Prediction Complete!")
                
                # Create columns for results
                col1, col2, col3 = st.columns(3)
                
                prediction = result.get('prediction', 'No')
                probability = result.get('probability', 0.0)
                
                with col1:
                    if prediction == "Yes":
                        st.metric("Prediction", "✅ YES", "Will Subscribe")
                    else:
                        st.metric("Prediction", "❌ NO", "Will Not Subscribe")
                
                with col2:
                    st.metric("Probability (Yes)", f"{probability:.2%}")
                
                with col3:
                    confidence = "High" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "Low")
                    st.metric("Confidence Level", confidence)
                
                # Gauge chart for probability
                st.markdown("### Probability Visualization")
                st.progress(probability)
                
                # Display customer summary
                with st.expander("📋 Customer Summary"):
                    st.json(customer_data)
                    
            else:
                st.error(f"API Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"Connection Error: {str(e)}")
            st.info("Please make sure the API is running at: " + API_URL)

# Footer
st.markdown("---")
st.markdown("💡 **Note:** This prediction is based on historical bank marketing data.")