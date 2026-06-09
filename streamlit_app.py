"""
Bank Marketing Prediction - Streamlit App (Direct Prediction)
"""

import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

# Load model (make sure files are in the same directory)
@st.cache_resource
def load_model():
    model = joblib.load('saved_models/random_forest_model.pkl')
    preprocessor = joblib.load('saved_encoders/preprocessor.pkl')
    return model, preprocessor

try:
    model, preprocessor = load_model()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model files not found! Please upload model files.")

# Title
st.title("🏦 Bank Marketing Prediction System")
st.markdown("### Will the customer subscribe to a term deposit?")

if model_loaded:
    # Sidebar for input
    st.sidebar.header("📝 Customer Information")
    
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
    
    # Advanced options
    with st.expander("🔧 Advanced Options (Optional)"):
        day = st.number_input("Last Contact Day", min_value=1, max_value=31, value=15)
        month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
        campaign = st.number_input("Number of Contacts", min_value=1, value=1)
        pdays = st.number_input("Days Since Last Contact", value=-1)
        previous = st.number_input("Previous Contacts", value=0)
        poutcome = st.selectbox("Previous Outcome", ["unknown", "success", "failure"])
    
    # Predict button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        # Create DataFrame
        customer_data = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'balance': balance, 'housing': housing, 'loan': loan, 'duration': duration,
            'default': default, 'contact': contact, 'day': day, 'month': month,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
        }])
        
        # Add age group
        customer_data['age_group'] = customer_data['age'].apply(
            lambda x: 'Young' if x <= 30 else ('Middle-Aged' if x <= 50 else 'Senior')
        )
        
        # Add missing columns
        for col in ['default', 'contact', 'poutcome']:
            if col not in customer_data.columns:
                customer_data[col] = 'unknown'
        
        # Predict
        processed = preprocessor.transform(customer_data)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        
        # Display result
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.metric("Prediction", "✅ YES", "Will Subscribe")
            else:
                st.metric("Prediction", "❌ NO", "Will Not Subscribe")
        
        with col2:
            st.metric("Probability (Yes)", f"{probability:.2%}")
        
        with col3:
            confidence = "High" if probability >= 0.7 else ("Medium" if probability >= 0.4 else "Low")
            st.metric("Confidence Level", confidence)
        
        # Progress bar
        st.markdown("### Probability Visualization")
        st.progress(probability)
else:
    st.warning("⚠️ Model not loaded. Please upload model files to continue.")
    st.info("""
    **How to fix:**
    1. Upload `random_forest_model.pkl` to `saved_models/` folder
    2. Upload `preprocessor.pkl` to `saved_encoders/` folder
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Note:** This prediction is based on historical bank marketing data.")