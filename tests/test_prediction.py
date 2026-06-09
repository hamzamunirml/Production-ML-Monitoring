"""
Unit tests for Prediction
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
import tempfile
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.prediction import Predictor

class TestPredictor(unittest.TestCase):
    """Test prediction functions"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary model files if they don't exist
        self.test_model_path = 'test_model.pkl'
        self.test_preprocessor_path = 'test_preprocessor.pkl'
        
        # Skip if real models don't exist
        self.skip_tests = not (os.path.exists('saved_models/random_forest_model.pkl') and 
                               os.path.exists('saved_encoders/preprocessor.pkl'))
        
        if not self.skip_tests:
            self.predictor = Predictor(
                model_path='saved_models/random_forest_model.pkl',
                preprocessor_path='saved_encoders/preprocessor.pkl'
            )
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        if self.skip_tests:
            self.skipTest("Model files not found")
        
        self.assertIsNotNone(self.predictor)
        self.assertTrue(hasattr(self.predictor, 'model'))
        self.assertTrue(hasattr(self.predictor, 'preprocessor'))
    
    def test_age_binning(self):
        """Test age binning function"""
        if self.skip_tests:
            self.skipTest("Model files not found")
        
        self.assertEqual(self.predictor.age_binning(25), 'Young')
        self.assertEqual(self.predictor.age_binning(40), 'Middle-Aged')
        self.assertEqual(self.predictor.age_binning(65), 'Senior')
    
    def test_single_prediction(self):
        """Test single customer prediction"""
        if self.skip_tests:
            self.skipTest("Model files not found")
        
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
        
        result = self.predictor.predict_single_customer(sample_customer)
        
        self.assertIn('prediction', result)
        self.assertIn('probability_yes', result)
        self.assertIn('probability_no', result)
        self.assertIsInstance(result['prediction'], str)
        self.assertIsInstance(result['probability_yes'], float)
        
        # Probability should be between 0 and 1
        self.assertGreaterEqual(result['probability_yes'], 0)
        self.assertLessEqual(result['probability_yes'], 1)
    
    def test_dataframe_prediction(self):
        """Test prediction with DataFrame input"""
        if self.skip_tests:
            self.skipTest("Model files not found")
        
        sample_df = pd.DataFrame({
            'age': [35, 45],
            'job': ['management', 'technician'],
            'marital': ['married', 'single'],
            'education': ['tertiary', 'secondary'],
            'default': ['no', 'no'],
            'balance': [1000, 2000],
            'housing': ['yes', 'no'],
            'loan': ['no', 'no'],
            'contact': ['cellular', 'cellular'],
            'day': [15, 20],
            'month': ['may', 'june'],
            'duration': [200, 300],
            'campaign': [1, 2],
            'pdays': [10, -1],
            'previous': [2, 0],
            'poutcome': ['success', 'unknown']
        })
        
        # Test single customer
        result = self.predictor.predict_single_customer(sample_df.iloc[0].to_dict())
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()