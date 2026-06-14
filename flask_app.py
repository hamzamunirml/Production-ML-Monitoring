"""
Flask wrapper for FastAPI app (for PythonAnywhere)
"""

import os
import sys
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessor
print("Loading model...")
model = joblib.load('saved_models/random_forest_model.pkl')
preprocessor = joblib.load('saved_encoders/preprocessor.pkl')
print("Model loaded successfully!")

# Create Flask app
app = Flask(__name__)

def age_binning(age):
    if age <= 30:
        return 'Young'
    elif age <= 50:
        return 'Middle-Aged'
    else:
        return 'Senior'

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'model_loaded': True,
        'message': 'Bank Marketing Prediction API is running!'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a single customer"""
    try:
        data = request.get_json()
        
        # Create DataFrame
        df = pd.DataFrame([data])
        
        # Add age group
        df['age_group'] = df['age'].apply(age_binning)
        
        # Ensure all required columns exist
        required_cols = ['age', 'job', 'marital', 'education', 'default', 
                        'balance', 'housing', 'loan', 'contact', 'day', 
                        'month', 'duration', 'campaign', 'pdays', 'previous', 
                        'poutcome', 'age_group']
        
        for col in required_cols:
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
        
        # Predict
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        
        return jsonify({
            'prediction': 'Yes' if prediction == 1 else 'No',
            'probability_yes': round(probability, 4),
            'probability_no': round(1 - probability, 4),
            'confidence_level': 'High' if probability >= 0.7 else ('Medium' if probability >= 0.4 else 'Low')
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/docs', methods=['GET'])
def docs():
    """Simple documentation page"""
    return jsonify({
        'endpoints': {
            '/': 'GET - Health check',
            '/predict': 'POST - Make prediction',
            '/model/info': 'GET - Model information'
        },
        'sample_request': {
            'age': 35,
            'job': 'management',
            'marital': 'married',
            'education': 'tertiary',
            'balance': 1000,
            'housing': 'yes',
            'loan': 'no',
            'duration': 200
        }
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'Random Forest Classifier',
        'n_estimators': model.n_estimators,
        'max_depth': model.max_depth,
        'classes': ['No', 'Yes']
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)