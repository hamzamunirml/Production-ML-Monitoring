"""
Unit tests for Data Preprocessing
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import DataPreprocessor, BinaryEncoder
from src.config import Config

class TestBinaryEncoder(unittest.TestCase):
    """Test Binary Encoder"""
    
    def setUp(self):
        self.encoder = BinaryEncoder()
        
        # Sample binary data
        self.sample_data = pd.DataFrame({
            'default': ['no', 'yes', 'unknown', 'no'],
            'housing': ['yes', 'no', 'yes', 'unknown'],
            'loan': ['no', 'no', 'yes', 'yes']
        })
    
    def test_binary_encoding(self):
        """Test binary encoding transformation"""
        encoded = self.encoder.transform(self.sample_data)
        
        self.assertEqual(encoded.shape, (4, 3))
        self.assertTrue(all(x in [0, 1] for row in encoded for x in row))
    
    def test_mapping_values(self):
        """Test specific mapping values"""
        # Test single value transformation
        test_df = pd.DataFrame({'test': ['no', 'yes', 'unknown']})
        encoded = self.encoder.transform(test_df)
        
        # Expected: no->0, yes->1, unknown->0
        self.assertEqual(encoded[0][0], 0)  # 'no' -> 0
        self.assertEqual(encoded[1][0], 1)  # 'yes' -> 1
        self.assertEqual(encoded[2][0], 0)  # 'unknown' -> 0

class TestDataPreprocessor(unittest.TestCase):
    """Test Data Preprocessor"""
    
    def setUp(self):
        self.preprocessor = DataPreprocessor()
    
    def test_load_data(self):
        """Test data loading"""
        # Skip if file not found in CI/CD
        if not os.path.exists(Config.DATA_PATH):
            self.skipTest("Data file not found")
        
        df = self.preprocessor.load_data()
        
        self.assertIsNotNone(df)
        self.assertGreater(df.shape[0], 0)
        self.assertGreater(df.shape[1], 0)
    
    def test_create_preprocessor(self):
        """Test preprocessor creation"""
        preprocessor = self.preprocessor.create_preprocessor()
        
        self.assertIsNotNone(preprocessor)
        self.assertTrue(hasattr(preprocessor, 'fit_transform'))
        self.assertTrue(hasattr(preprocessor, 'transform'))
    
    def test_split_data(self):
        """Test data splitting"""
        # Create sample data
        X = pd.DataFrame({
            'age': [25, 35, 45, 55],
            'balance': [1000, 2000, 3000, 4000]
        })
        y = pd.Series([0, 1, 0, 1])
        
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Check split ratios
        self.assertEqual(len(X_train) + len(X_test), len(X))
        self.assertGreater(len(X_train), 0)
        self.assertGreater(len(X_test), 0)

if __name__ == '__main__':
    unittest.main()
