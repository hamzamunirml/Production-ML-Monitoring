"""
Unit tests for Model Training
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model_training import ModelTrainer
from src.config import Config

class TestModelTraining(unittest.TestCase):
    """Test model training functions"""
    
    def setUp(self):
        self.trainer = ModelTrainer()
        
        # Create synthetic balanced data
        np.random.seed(42)
        n_samples = 1000
        
        self.X_train = np.random.randn(n_samples, 10)
        self.y_train = np.random.randint(0, 2, n_samples)
        
        # Create imbalanced data for SMOTE test
        self.X_imbalanced = np.random.randn(1000, 10)
        self.y_imbalanced = np.array([0]*900 + [1]*100)  # 90% No, 10% Yes
    
    def test_handle_imbalance_smote(self):
        """Test SMOTE balancing"""
        X_balanced, y_balanced = self.trainer.handle_imbalance(
            self.X_imbalanced, self.y_imbalanced
        )
        
        # Check if balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        
        self.assertEqual(counts[0], counts[1])
        self.assertGreater(len(X_balanced), len(self.X_imbalanced))
    
    def test_decision_tree_training(self):
        """Test Decision Tree training"""
        model = self.trainer.train_decision_tree(self.X_train, self.y_train)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
    
    def test_random_forest_training(self):
        """Test Random Forest training"""
        model = self.trainer.train_random_forest(self.X_train, self.y_train)
        
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))
        self.assertTrue(hasattr(model, 'predict_proba'))
        self.assertTrue(hasattr(model, 'feature_importances_'))
    
    def test_model_accuracy(self):
        """Test model accuracy on simple data"""
        # Create simple linear separable data
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # Train Decision Tree
        dt_model = self.trainer.train_decision_tree(X, y)
        dt_pred = dt_model.predict(X)
        dt_accuracy = (dt_pred == y).mean()
        
        # Train Random Forest
        rf_model = self.trainer.train_random_forest(X, y)
        rf_pred = rf_model.predict(X)
        rf_accuracy = (rf_pred == y).mean()
        
        # Both models should perform well on training data
        self.assertGreater(dt_accuracy, 0.8)
        self.assertGreater(rf_accuracy, 0.8)

if __name__ == '__main__':
    unittest.main()