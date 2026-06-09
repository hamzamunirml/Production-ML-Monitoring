"""
API dependencies - Model loading and initialization
"""

import sys
import os
import joblib
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class ModelManager:
    """Singleton class to manage model loading"""
    
    _instance = None
    _model = None
    _preprocessor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def load_models(self):
        """Load trained models and preprocessors"""
        if self._model is None:
            try:
                print("Loading Random Forest model...")
                self._model = joblib.load('saved_models/random_forest_model.pkl')
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                raise
            
        if self._preprocessor is None:
            try:
                print("Loading preprocessor...")
                self._preprocessor = joblib.load('saved_encoders/preprocessor.pkl')
                print("✅ Preprocessor loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading preprocessor: {e}")
                raise
        
        return self._model, self._preprocessor
    
    def get_model(self):
        """Get loaded model"""
        if self._model is None:
            self.load_models()
        return self._model
    
    def get_preprocessor(self):
        """Get loaded preprocessor"""
        if self._preprocessor is None:
            self.load_models()
        return self._preprocessor

# Global instance
model_manager = ModelManager()

def age_binning(age):
    """Convert age to age group"""
    if age <= 30:
        return 'Young'
    elif age <= 50:
        return 'Middle-Aged'
    else:
        return 'Senior'

def preprocess_customer(customer_dict, preprocessor):
    """Preprocess single customer data"""
    # Create DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Add age group
    df['age_group'] = df['age'].apply(age_binning)
    
    # Ensure all required columns exist
    required_columns = ['age', 'job', 'marital', 'education', 'default', 
                        'balance', 'housing', 'loan', 'contact', 'day', 
                        'month', 'duration', 'campaign', 'pdays', 'previous', 
                        'poutcome', 'age_group']
    
    for col in required_columns:
        if col not in df.columns:
            if col in ['default', 'contact', 'poutcome']:
                df[col] = 'unknown'
            elif col in ['day', 'campaign', 'pdays', 'previous']:
                df[col] = 0
            elif col in ['month']:
                df[col] = 'may'
            else:
                df[col] = 'unknown'
    
    # Transform
    processed = preprocessor.transform(df)
    return processed

def get_confidence_level(probability):
    """Get confidence level based on probability"""
    if probability >= 0.7:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"