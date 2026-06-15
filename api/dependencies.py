"""
Dependencies for Bank Marketing Prediction API
Pure FastAPI version - No Streamlit dependencies
"""

import joblib
import pandas as pd
import numpy as np
from fastapi import HTTPException
import os
from typing import Dict, Any, List

# Model paths - All files in saved_models folder
MODEL_PATH = "saved_models/random_forest_model.pkl"
PREPROCESSOR_PATH = "saved_models/preprocessor.pkl"


class ModelManager:
    """Manage loading and prediction of ML models"""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_models()

    def load_models(self):
        """Load model and preprocessor from saved_models folder"""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
                self.model = joblib.load(MODEL_PATH)
                self.preprocessor = joblib.load(PREPROCESSOR_PATH)
                print("✅ Models loaded successfully from 'saved_models/' folder!")

                # Verify model type
                print(f"📊 Model Type: {type(self.model).__name__}")
                print(f"📊 Preprocessor Type: {type(self.preprocessor).__name__}")
            else:
                print(f"⚠️ Model files not found at: {MODEL_PATH}")
                print(f"⚠️ Preprocessor not found at: {PREPROCESSOR_PATH}")
                self.model = None
                self.preprocessor = None
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model = None
            self.preprocessor = None

    def is_ready(self) -> bool:
        """Check if models are ready for predictions"""
        return self.model is not None and self.preprocessor is not None

    def preprocess_customer(self, customer_data: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess customer data for prediction"""
        try:
            # Create DataFrame
            df = pd.DataFrame([customer_data])

            # Add age group feature
            def age_binning(age):
                if age <= 30:
                    return "Young"
                elif age <= 50:
                    return "Middle-Aged"
                else:
                    return "Senior"

            df["age_group"] = df["age"].apply(age_binning)

            return df
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Preprocessing error: {str(e)}"
            )

    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for a single customer"""
        try:
            # Preprocess
            df = self.preprocess_customer(customer_data)

            # Transform using preprocessor
            X_processed = self.preprocessor.transform(df)

            # Predict
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0][1]

            return {
                "prediction": int(prediction),
                "prediction_label": "Yes" if prediction == 1 else "No",
                "probability_yes": float(probability),
                "probability_no": float(1 - probability),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

    def predict_batch(
        self, customers_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Predict for multiple customers"""
        try:
            results = []
            for customer in customers_data:
                result = self.predict_single(customer)
                results.append(result)
            return results
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Batch prediction error: {str(e)}"
            )


# ============================================================
# FUNCTIONS FOR FastAPI (Matching main.py imports)
# ============================================================


def preprocess_customer(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and preprocess customer data
    This is the function imported by main.py
    """
    # Required fields
    required_fields = [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "loan",
        "duration",
    ]

    # Check for required fields
    for field in required_fields:
        if field not in customer_data:
            raise HTTPException(
                status_code=400, detail=f"Missing required field: {field}"
            )

    # Validate age (18-100)
    if not (18 <= customer_data["age"] <= 100):
        raise HTTPException(status_code=400, detail="Age must be between 18 and 100")

    # Validate balance (reasonable range)
    if customer_data["balance"] < -10000 or customer_data["balance"] > 1000000:
        raise HTTPException(
            status_code=400, detail="Balance out of valid range (-10000 to 1,000,000)"
        )

    # Validate duration (0-5000 seconds)
    if not (0 <= customer_data["duration"] <= 5000):
        raise HTTPException(
            status_code=400, detail="Duration must be between 0 and 5000 seconds"
        )

    # Validate campaign contacts (1-50)
    if "campaign" in customer_data:
        if not (1 <= customer_data["campaign"] <= 50):
            raise HTTPException(
                status_code=400, detail="Campaign contacts must be between 1 and 50"
            )

    # Validate pdays
    if "pdays" in customer_data:
        if customer_data["pdays"] < -1:
            raise HTTPException(status_code=400, detail="pdays must be -1 or greater")

    # Valid job types
    valid_jobs = [
        "management",
        "technician",
        "entrepreneur",
        "blue-collar",
        "unknown",
        "retired",
        "admin.",
        "services",
        "self-employed",
        "unemployed",
        "housemaid",
        "student",
    ]

    if customer_data["job"] not in valid_jobs:
        raise HTTPException(
            status_code=400, detail=f"Invalid job. Must be one of: {valid_jobs}"
        )

    # Valid marital status
    valid_marital = ["married", "single", "divorced"]
    if customer_data["marital"] not in valid_marital:
        raise HTTPException(
            status_code=400, detail=f"Invalid marital status. Must be: {valid_marital}"
        )

    # Valid education
    valid_education = ["tertiary", "secondary", "primary", "unknown"]
    if customer_data["education"] not in valid_education:
        raise HTTPException(
            status_code=400, detail=f"Invalid education. Must be: {valid_education}"
        )

    # Add default values for optional fields
    defaults = {
        "default": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    for key, value in defaults.items():
        if key not in customer_data:
            customer_data[key] = value

    return customer_data


def get_confidence_level(probability: float) -> str:
    """
    Get confidence level based on probability
    """
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    elif probability >= 0.2:
        return "Low"
    else:
        return "Very Low"


def get_recommendation(prediction: int, probability: float) -> str:
    """
    Generate recommendation based on prediction
    """
    if prediction == 1:
        if probability >= 0.7:
            return "Strongly recommend focusing marketing efforts on this customer"
        elif probability >= 0.5:
            return "Recommended to include in marketing campaign"
        else:
            return "Consider including in campaign with low priority"
    else:
        if probability <= 0.3:
            return "Not recommended for current campaign. Consider different targeting strategy"
        elif probability <= 0.5:
            return "Low priority. May need more engagement before offering"
        else:
            return "Borderline case. Consider additional qualification"


# Initialize global model manager
model_manager = ModelManager()


# ============================================================
# ALIAS for backward compatibility
# ============================================================
validate_customer_data = preprocess_customer
