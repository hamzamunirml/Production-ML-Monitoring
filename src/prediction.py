"""
Prediction Module
"""

import pandas as pd
import joblib

class Predictor:
    def __init__(self, model_path='saved_models/random_forest_model.pkl', 
                 preprocessor_path='saved_encoders/preprocessor.pkl'):
        print("Loading model and preprocessor...")
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        print("✅ Model and preprocessor loaded successfully!")
    
    def age_binning(self, age):
        """Convert age to age group"""
        if age <= 30:
            return 'Young'
        elif age <= 50:
            return 'Middle-Aged'
        else:
            return 'Senior'
    
    def predict_single_customer(self, customer_data):
        """Predict for a single customer"""
        if isinstance(customer_data, dict):
            customer_data = pd.DataFrame([customer_data])
        
        # Add age_group if age exists
        if 'age' in customer_data.columns:
            customer_data = customer_data.copy()
            customer_data['age_group'] = customer_data['age'].apply(self.age_binning)
        
        # Preprocess
        processed_data = self.preprocessor.transform(customer_data)
        
        # Predict
        prediction = self.model.predict(processed_data)[0]
        probability = self.model.predict_proba(processed_data)[0][1]
        
        result = {
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability_yes': probability,
            'probability_no': 1 - probability
        }
        
        return result