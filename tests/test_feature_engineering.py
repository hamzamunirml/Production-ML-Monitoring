"""
Unit tests for Feature Engineering
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineer
from src.config import Config

class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functions"""
    
    def setUp(self):
        """Set up test data"""
        self.engineer = FeatureEngineer()
        
        # Create sample data
        self.sample_df = pd.DataFrame({
            'age': [25, 35, 45, 55, 65, 18, 70, 30, 40, 50],
            'balance': [1000, 2000, 3000, 4000, 5000, 500, 6000, 1500, 2500, 3500],
            'y': ['no', 'yes', 'no', 'yes', 'no', 'no', 'yes', 'no', 'yes', 'no']
        })
    
    def test_age_binning(self):
        """Test age binning function"""
        df = self.engineer.create_age_groups(self.sample_df.copy())
        
        # Check if age_group column created
        self.assertIn('age_group', df.columns)
        
        # Check age group values
        self.assertEqual(df[df['age'] == 25]['age_group'].iloc[0], 'Young')
        self.assertEqual(df[df['age'] == 45]['age_group'].iloc[0], 'Middle-Aged')
        self.assertEqual(df[df['age'] == 65]['age_group'].iloc[0], 'Senior')
    
    def test_age_binning_boundaries(self):
        """Test age binning at boundaries"""
        df = self.engineer.create_age_groups(self.sample_df.copy())
        
        # Check boundary values
        young_ages = df[df['age'] <= 30]['age_group'].unique()
        middle_ages = df[(df['age'] > 30) & (df['age'] <= 50)]['age_group'].unique()
        senior_ages = df[df['age'] > 50]['age_group'].unique()
        
        self.assertTrue(all(g == 'Young' for g in young_ages))
        self.assertTrue(all(g == 'Middle-Aged' for g in middle_ages))
        self.assertTrue(all(g == 'Senior' for g in senior_ages))
    
    def test_detect_outliers(self):
        """Test outlier detection"""
        # Create data with outliers
        test_df = pd.DataFrame({
            'balance': [100, 200, 300, 10000, 150, 250, 20000, 180, 220, 99999]
        })
        
        outliers = self.engineer.detect_outliers(test_df, ['balance'])
        
        self.assertIn('balance', outliers)
        self.assertGreater(outliers['balance']['outliers_count'], 0)
    
    def test_get_feature_info(self):
        """Test feature info retrieval"""
        info = self.engineer.get_feature_info(self.sample_df)
        
        self.assertIn('numerical_features', info)
        self.assertIn('categorical_features', info)
        self.assertIn('total_features', info)
        self.assertIn('created_features', info)

if __name__ == '__main__':
    unittest.main()
