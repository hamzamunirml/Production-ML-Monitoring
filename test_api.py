"""
Test the Bank Marketing Prediction API
"""

import requests
import json
import pandas as pd

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("1. Testing Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("2. Testing Single Prediction")
    print("="*60)
    
    customer = {
        "age": 35,
        "job": "management",
        "marital": "married",
        "education": "tertiary",
        "balance": 1000,
        "housing": "yes",
        "loan": "no",
        "duration": 200,
        "default": "no",
        "contact": "cellular",
        "day": 15,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown"
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("3. Testing Batch Prediction")
    print("="*60)
    
    customers = [
        {
            "age": 28,
            "job": "student",
            "marital": "single",
            "education": "tertiary",
            "balance": 500,
            "housing": "no",
            "loan": "no",
            "duration": 120,
            "default": "no",
            "contact": "cellular",
            "day": 10,
            "month": "apr",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        },
        {
            "age": 45,
            "job": "management",
            "marital": "married",
            "education": "tertiary",
            "balance": 2000,
            "housing": "yes",
            "loan": "no",
            "duration": 300,
            "default": "no",
            "contact": "cellular",
            "day": 15,
            "month": "jun",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        },
        {
            "age": 65,
            "job": "retired",
            "marital": "married",
            "education": "secondary",
            "balance": 3000,
            "housing": "yes",
            "loan": "no",
            "duration": 400,
            "default": "no",
            "contact": "telephone",
            "day": 20,
            "month": "jul",
            "campaign": 1,
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown"
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(f"Total Customers: {result['total_customers']}")
    print(f"Summary: {json.dumps(result['summary'], indent=2)}")
    print("\nIndividual Predictions:")
    for pred in result['predictions']:
        print(f"  Customer {pred['customer_id']}: {pred['prediction']} ({pred['probability_yes']:.1%})")
    
    return response.status_code == 200

def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("4. Testing Model Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 BANK MARKETING PREDICTION API TEST")
    print("="*60)
    
    # First, check if API is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=2)
    except requests.exceptions.ConnectionError:
        print("\n❌ API is not running!")
        print("Please start the API first using: python -m api.main")
        print("OR: uvicorn api.main:app --reload")
        exit(1)
    
    # Run all tests
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Model Info", test_model_info)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")