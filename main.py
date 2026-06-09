"""
Main script to run the entire pipeline
"""

import sys
import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import Predictor

def main():
    print("="*60)
    print("BANK MARKETING PREDICTION PIPELINE")
    print("="*60)
    
    # Step 1: Data Preprocessing
    print("\n" + "="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data()
    X, y = preprocessor.prepare_features(df)
    preprocessor.create_preprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_processed, X_test_processed = preprocessor.preprocess_data(X_train, X_test)
    preprocessor.save_preprocessor()
    
    # Step 2: Model Training
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    trainer = ModelTrainer()
    X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train_processed, y_train)
    dt_model = trainer.train_decision_tree(X_train_balanced, y_train_balanced)
    rf_model = trainer.train_random_forest(X_train_balanced, y_train_balanced)
    trainer.save_models()
    
    # Step 3: Model Evaluation
    print("\n" + "="*60)
    print("STEP 3: MODEL EVALUATION")
    print("="*60)
    
    evaluator = ModelEvaluator()
    
    # Evaluate models
    evaluator.evaluate_model(dt_model, X_test_processed, y_test, "Decision Tree")
    evaluator.evaluate_model(rf_model, X_test_processed, y_test, "Random Forest")
    
    # Compare models
    comparison_df, best_model = evaluator.compare_models()
    print("\n📊 Model Comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\n🏆 Best Model: {best_model}")
    
    # Step 4: Test Prediction
    print("\n" + "="*60)
    print("STEP 4: TEST PREDICTION")
    print("="*60)
    
    predictor = Predictor()
    
    # Sample customer data
    sample_customer = {
        'age': 35,
        'job': 'management',
        'marital': 'married',
        'education': 'tertiary',
        'default': 'no',
        'balance': 1000,
        'housing': 'yes',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'may',
        'duration': 200,
        'campaign': 1,
        'pdays': 10,
        'previous': 2,
        'poutcome': 'success'
    }
    
    result = predictor.predict_single_customer(sample_customer)
    print(f"\n📊 Sample Customer Prediction:")
    print(f"   Will subscribe to term deposit: {result['prediction']}")
    print(f"   Probability (Yes): {result['probability_yes']:.2%}")
    
    # Step 5: Show saved files
    print("\n" + "="*60)
    print("STEP 5: SAVED FILES")
    print("="*60)
    
    for folder in ['saved_models', 'saved_encoders', 'saved_features']:
        if os.path.exists(folder):
            print(f"\n📁 {folder}/")
            for f in os.listdir(folder):
                print(f"   - {f}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"🏆 Best Model: {best_model}")
    print("="*60)
# Add this at the VERY END of your main.py file

# ============================================================
# PREDICTION FUNCTIONS FOR NEW CUSTOMERS
# ============================================================

def predict_single_customer_manual():
    """
    Function to manually enter customer details and get prediction
    """
    print("\n" + "="*60)
    print("PREDICT FOR A NEW CUSTOMER (MANUAL INPUT)")
    print("="*60)
    
    # Load model and preprocessor
    import joblib
    
    try:
        model = joblib.load('saved_models/random_forest_model.pkl')
        preprocessor = joblib.load('saved_encoders/preprocessor.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model not found! Error: {e}")
        return
    
    # Get customer details
    print("\n📝 Enter Customer Details:")
    print("-" * 40)
    
    try:
        age = int(input("Enter age: "))
        job = input("Enter job (management/technician/blue-collar/admin/services/retired/student): ")
        marital = input("Enter marital status (married/single/divorced): ")
        education = input("Enter education (tertiary/secondary/primary/unknown): ")
        balance = float(input("Enter balance amount: "))
        housing = input("Has housing loan? (yes/no): ")
        loan = input("Has personal loan? (yes/no): ")
        duration = int(input("Enter last contact duration (seconds): "))
    except ValueError as e:
        print(f"❌ Invalid input! Please enter correct values. Error: {e}")
        return
    
    # Create customer dataframe
    customer = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': ['no'],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': ['cellular'],
        'day': [15],
        'month': ['may'],
        'duration': [duration],
        'campaign': [1],
        'pdays': [-1],
        'previous': [0],
        'poutcome': ['unknown']
    })
    
    # Add age group
    def age_binning(age):
        if age <= 30:
            return 'Young'
        elif age <= 50:
            return 'Middle-Aged'
        else:
            return 'Senior'
    
    customer['age_group'] = customer['age'].apply(age_binning)
    
    # Predict
    processed = preprocessor.transform(customer)
    prediction = model.predict(processed)[0]
    probability = model.predict_proba(processed)[0][1]
    
    # Display result
    print("\n" + "="*60)
    print("📊 PREDICTION RESULT")
    print("="*60)
    print(f"   Customer Age: {age}")
    print(f"   Customer Job: {job}")
    print(f"   Account Balance: ${balance:,.2f}")
    print(f"   Contact Duration: {duration} seconds")
    print("-" * 40)
    print(f"   Will subscribe to term deposit: {'✅ YES' if prediction == 1 else '❌ NO'}")
    print(f"   Probability (Yes): {probability:.2%}")
    print(f"   Probability (No): {(1-probability):.2%}")
    print("="*60)

def predict_sample_customers():
    """
    Function to test prediction on sample customer profiles
    """
    print("\n" + "="*60)
    print("TEST PREDICTIONS ON SAMPLE CUSTOMER PROFILES")
    print("="*60)
    
    import joblib
    
    # Load model
    try:
        model = joblib.load('saved_models/random_forest_model.pkl')
        preprocessor = joblib.load('saved_encoders/preprocessor.pkl')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model not found! Error: {e}")
        return
    
    # Define sample customer profiles
    test_customers = [
        {
            'name': 'Young Professional',
            'data': {
                'age': 28, 'job': 'management', 'marital': 'single', 'education': 'tertiary',
                'default': 'no', 'balance': 2500, 'housing': 'yes', 'loan': 'no',
                'contact': 'cellular', 'day': 15, 'month': 'may', 'duration': 350,
                'campaign': 1, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'
            }
        },
        {
            'name': 'Student',
            'data': {
                'age': 22, 'job': 'student', 'marital': 'single', 'education': 'tertiary',
                'default': 'no', 'balance': 300, 'housing': 'no', 'loan': 'no',
                'contact': 'cellular', 'day': 10, 'month': 'apr', 'duration': 120,
                'campaign': 2, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'
            }
        },
        {
            'name': 'Rich Retired',
            'data': {
                'age': 65, 'job': 'retired', 'marital': 'married', 'education': 'tertiary',
                'default': 'no', 'balance': 15000, 'housing': 'yes', 'loan': 'no',
                'contact': 'telephone', 'day': 20, 'month': 'jun', 'duration': 500,
                'campaign': 1, 'pdays': 5, 'previous': 1, 'poutcome': 'success'
            }
        },
        {
            'name': 'Low Balance Customer',
            'data': {
                'age': 35, 'job': 'blue-collar', 'marital': 'married', 'education': 'secondary',
                'default': 'no', 'balance': -500, 'housing': 'yes', 'loan': 'yes',
                'contact': 'cellular', 'day': 5, 'month': 'feb', 'duration': 80,
                'campaign': 3, 'pdays': -1, 'previous': 0, 'poutcome': 'unknown'
            }
        },
        {
            'name': 'High Duration Call',
            'data': {
                'age': 42, 'job': 'entrepreneur', 'marital': 'married', 'education': 'tertiary',
                'default': 'no', 'balance': 5000, 'housing': 'yes', 'loan': 'no',
                'contact': 'cellular', 'day': 25, 'month': 'jul', 'duration': 800,
                'campaign': 1, 'pdays': 10, 'previous': 2, 'poutcome': 'success'
            }
        }
    ]
    
    # Test each customer
    def age_binning(age):
        if age <= 30:
            return 'Young'
        elif age <= 50:
            return 'Middle-Aged'
        else:
            return 'Senior'
    
    print("\n📊 TEST RESULTS:")
    print("="*80)
    
    results = []
    for customer in test_customers:
        df = pd.DataFrame([customer['data']])
        df['age_group'] = df['age'].apply(age_binning)
        
        processed = preprocessor.transform(df)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]
        
        results.append({
            'Name': customer['name'],
            'Age': customer['data']['age'],
            'Job': customer['data']['job'],
            'Balance': customer['data']['balance'],
            'Duration': customer['data']['duration'],
            'Prediction': 'Yes' if prediction == 1 else 'No',
            'Probability': f"{probability:.1%}"
        })
    
    # Display results in a nice table
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    print("\n" + "="*80)
    
    # Summary
    yes_predictions = sum(1 for r in results if r['Prediction'] == 'Yes')
    print(f"\n📈 Summary: {yes_predictions}/{len(test_customers)} customers predicted to subscribe")

def prediction_menu():
    """
    Interactive menu for making predictions
    """
    while True:
        print("\n" + "="*60)
        print("🎯 BANK MARKETING PREDICTION SYSTEM")
        print("="*60)
        print("\nSelect an option:")
        print("1. Predict for a single customer (manual input)")
        print("2. Test on sample customer profiles")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            predict_single_customer_manual()
        elif choice == '2':
            predict_sample_customers()
        elif choice == '3':
            print("\n✅ Thank you for using Bank Marketing Prediction System!")
            break
        else:
            print("\n❌ Invalid choice! Please enter 1, 2, or 3.")

# Run the prediction menu after main pipeline
if __name__ == "__main__":
    # First run the main pipeline
    main()
    
    # Then ask if user wants to make predictions
    print("\n" + "="*60)
    print("🎯 PREDICTION SYSTEM")
    print("="*60)
    
    while True:
        choice = input("\nDo you want to make predictions for new customers? (yes/no): ").lower()
        if choice in ['yes', 'y']:
            prediction_menu()
            break
        elif choice in ['no', 'n']:
            print("\n✅ Thank you! Your models are saved in 'saved_models/' folder.")
            print("   You can use them anytime for predictions.")
            break
        else:
            print("Please enter 'yes' or 'no'")