"""
FastAPI application for Bank Marketing Prediction
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .models import (
    CustomerRequest, 
    SinglePredictionResponse, 
    BatchPredictionResponse,
    HealthResponse
)
from .dependencies import model_manager, preprocess_customer, get_confidence_level

# Initialize FastAPI app
app = FastAPI(
    title="Bank Marketing Prediction API",
    description="Predict if a bank client will subscribe to a term deposit",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load models when API starts"""
    print("\n" + "="*60)
    print("🚀 BANK MARKETING PREDICTION API")
    print("="*60)
    print("Starting API server...")
    model_manager.load_models()
    print("✅ API is ready to accept requests!")
    print("="*60 + "\n")

# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check if API is running and model is loaded"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=model_manager.get_model() is not None,
        timestamp=datetime.now().isoformat()
    )

# Single prediction endpoint
@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict_single(customer: CustomerRequest):
    """
    Make a single prediction for one customer
    
    - **customer**: Customer data including age, job, balance, duration, etc.
    - **returns**: Prediction (Yes/No) and probability scores
    """
    try:
        # Get model and preprocessor
        model = model_manager.get_model()
        preprocessor = model_manager.get_preprocessor()
        
        # Convert request to dictionary
        customer_dict = customer.dict()
        
        # Preprocess
        processed_data = preprocess_customer(customer_dict, preprocessor)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]
        
        # Prepare response
        response = SinglePredictionResponse(
            prediction="Yes" if prediction == 1 else "No",
            probability_yes=round(probability, 4),
            probability_no=round(1 - probability, 4),
            confidence_level=get_confidence_level(probability)
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(customers: List[CustomerRequest]):
    """
    Make batch predictions for multiple customers
    
    - **customers**: List of customer data objects
    - **returns**: Predictions for all customers with summary statistics
    """
    try:
        # Get model and preprocessor
        model = model_manager.get_model()
        preprocessor = model_manager.get_preprocessor()
        
        # Convert to list of dicts
        customers_list = [customer.dict() for customer in customers]
        
        # Process each customer
        predictions = []
        for i, customer_dict in enumerate(customers_list):
            processed_data = preprocess_customer(customer_dict, preprocessor)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            predictions.append({
                "customer_id": i + 1,
                "prediction": "Yes" if prediction == 1 else "No",
                "probability_yes": round(probability, 4),
                "probability_no": round(1 - probability, 4),
                "confidence_level": get_confidence_level(probability)
            })
        
        # Calculate summary
        yes_count = sum(1 for p in predictions if p['prediction'] == 'Yes')
        avg_probability = sum(p['probability_yes'] for p in predictions) / len(predictions)
        
        summary = {
            "total_customers": len(predictions),
            "predicted_yes": yes_count,
            "predicted_no": len(predictions) - yes_count,
            "yes_percentage": round(yes_count / len(predictions) * 100, 2),
            "average_probability": round(avg_probability, 4),
            "high_confidence_count": sum(1 for p in predictions if p['confidence_level'] == 'High'),
            "medium_confidence_count": sum(1 for p in predictions if p['confidence_level'] == 'Medium'),
            "low_confidence_count": sum(1 for p in predictions if p['confidence_level'] == 'Low')
        }
        
        return BatchPredictionResponse(
            total_customers=len(predictions),
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

# CSV upload endpoint
@app.post("/predict/csv", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_from_csv(file: bytes):
    """
    Upload CSV file and get predictions for all customers
    
    - **file**: CSV file with customer data
    - **returns**: Predictions for all customers in the file
    """
    try:
        # Read CSV from bytes
        from io import StringIO
        import csv
        
        csv_string = StringIO(file.decode('utf-8'))
        df = pd.read_csv(csv_string)
        
        # Get model and preprocessor
        model = model_manager.get_model()
        preprocessor = model_manager.get_preprocessor()
        
        # Validate required columns
        required_columns = ['age', 'job', 'marital', 'education', 'balance', 'duration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Process each row
        predictions = []
        for idx, row in df.iterrows():
            customer_dict = row.to_dict()
            
            # Add defaults for missing fields
            default_fields = {
                'default': 'no',
                'housing': 'yes',
                'loan': 'no',
                'contact': 'cellular',
                'day': 15,
                'month': 'may',
                'campaign': 1,
                'pdays': -1,
                'previous': 0,
                'poutcome': 'unknown'
            }
            
            for field, default_value in default_fields.items():
                if field not in customer_dict:
                    customer_dict[field] = default_value
            
            processed_data = preprocess_customer(customer_dict, preprocessor)
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0][1]
            
            predictions.append({
                "customer_id": idx + 1,
                "prediction": "Yes" if prediction == 1 else "No",
                "probability_yes": round(probability, 4),
                "probability_no": round(1 - probability, 4),
                "confidence_level": get_confidence_level(probability),
                **{k: v for k, v in customer_dict.items() if k in ['age', 'job', 'balance']}
            })
        
        # Calculate summary
        yes_count = sum(1 for p in predictions if p['prediction'] == 'Yes')
        avg_probability = sum(p['probability_yes'] for p in predictions) / len(predictions)
        
        summary = {
            "total_customers": len(predictions),
            "predicted_yes": yes_count,
            "predicted_no": len(predictions) - yes_count,
            "yes_percentage": round(yes_count / len(predictions) * 100, 2),
            "average_probability": round(avg_probability, 4)
        }
        
        return BatchPredictionResponse(
            total_customers=len(predictions),
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV processing failed: {str(e)}"
        )

# Model info endpoint
@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the loaded model"""
    try:
        model = model_manager.get_model()
        
        # Get feature names
        preprocessor = model_manager.get_preprocessor()
        
        # Get feature importance if available
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_.tolist()
        
        return {
            "model_type": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "feature_importances": feature_importance[:10] if feature_importance else None,
            "classes": ["No", "Yes"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )