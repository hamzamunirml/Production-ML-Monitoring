"""
Integration tests for complete pipeline
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator
from src.prediction import Predictor
from src.config import Config

class TestIntegration(unittest.TestCase):
    """Integration tests for entire pipeline"""
    
    def setUp(self):
        """Set up test data"""
        # Create a small synthetic dataset
        np.random.seed(42)
        n_samples = 500
        
        self.test_df = pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'job': np.random.choice(['management', 'technician', 'blue-collar'], n_samples),
            'marital': np.random.choice(['married', 'single', 'divorced'], n_samples),
            'education': np.random.choice(['tertiary', 'secondary', 'primary'], n_samples),
            'default': np.random.choice(['no', 'yes'], n_samples, p=[0.95, 0.05]),
            'balance': np.random.randint(-1000, 10000, n_samples),
            'housing': np.random.choice(['no', 'yes'], n_samples),
            'loan': np.random.choice(['no', 'yes'], n_samples, p=[0.85, 0.15]),
            'contact': np.random.choice(['cellular', 'telephone'], n_samples),
            'day': np.random.randint(1, 31, n_samples),
            'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun'], n_samples),
            'duration': np.random.randint(10, 1000, n_samples),
            'campaign': np.random.randint(1, 10, n_samples),
            'pdays': np.random.choice([-1, 1, 5, 10, 20], n_samples),
            'previous': np.random.randint(0, 5, n_samples),
            'poutcome': np.random.choice(['unknown', 'success', 'failure'], n_samples),
            'y': np.random.choice(['no', 'yes'], n_samples, p=[0.88, 0.12])
        })
        
        # Save temporary test data
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, 'test_bank.csv')
        self.test_df.to_csv(self.test_data_path, index=False)
        
        # Store original path
        self.original_data_path = Config.DATA_PATH
        
        # Update config to use test data
        Config.DATA_PATH = self.test_data_path
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original config
        Config.DATA_PATH = self.original_data_path
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_pipeline(self):
        """Test the complete pipeline end-to-end"""
        
        # 1. Data Preprocessing
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        X, y = preprocessor.prepare_features(df)
        preprocessor.create_preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_processed, X_test_processed = preprocessor.preprocess_data(X_train, X_test)
        
        # Assertions for preprocessing
        self.assertIsNotNone(X_train_processed)
        self.assertIsNotNone(X_test_processed)
        self.assertGreater(X_train_processed.shape[0], 0)
        self.assertGreater(X_test_processed.shape[0], 0)
        
        # 2. Model Training
        trainer = ModelTrainer()
        X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train_processed, y_train)
        dt_model = trainer.train_decision_tree(X_train_balanced, y_train_balanced)
        rf_model = trainer.train_random_forest(X_train_balanced, y_train_balanced)
        
        # Assertions for training
        self.assertIsNotNone(dt_model)
        self.assertIsNotNone(rf_model)
        
        # 3. Model Evaluation
        evaluator = ModelEvaluator()
        dt_metrics = evaluator.evaluate_model(dt_model, X_test_processed, y_test, "Decision Tree")
        rf_metrics = evaluator.evaluate_model(rf_model, X_test_processed, y_test, "Random Forest")
        
        # Assertions for evaluation
        self.assertIn('accuracy', dt_metrics)
        self.assertIn('f1_score', dt_metrics)
        self.assertIn('roc_auc', dt_metrics)
        
        # 4. Predictions should work
        comparison_df, best_model = evaluator.compare_models()
        self.assertIsNotNone(best_model)
        self.assertIn(best_model, ['Decision Tree', 'Random Forest'])
    
    def test_prediction_after_training(self):
        """Test predictions after training"""
        
        # Train models
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        X, y = preprocessor.prepare_features(df)
        preprocessor.create_preprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        X_train_processed, X_test_processed = preprocessor.preprocess_data(X_train, X_test)
        
        trainer = ModelTrainer()
        X_train_balanced, y_train_balanced = trainer.handle_imbalance(X_train_processed, y_train)
        rf_model = trainer.train_random_forest(X_train_balanced, y_train_balanced)
        
        # Make prediction
        sample_data = X_test.iloc[0:1]
        
        # Preprocess manually
        sample_processed = preprocessor.preprocessor.transform(sample_data)
        
        # Predict
        prediction = rf_model.predict(sample_processed)[0]
        probability = rf_model.predict_proba(sample_processed)[0][1]
        
        # Assertions
        self.assertIn(prediction, [0, 1])
        self.assertGreaterEqual(probability, 0)
        self.assertLessEqual(probability, 1)
    
    def test_data_consistency(self):
        """Test data consistency throughout pipeline"""
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_data()
        X, y = preprocessor.prepare_features(df)
        
        # Check no missing values after preprocessing
        self.assertFalse(X.isnull().any().any())
        
        # Check target values are 0 or 1
        self.assertTrue(y.isin([0, 1]).all())

if __name__ == '__main__':
    unittest.main()