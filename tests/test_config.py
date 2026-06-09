"""
Unit tests for Configuration
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config

class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_attributes(self):
        """Test if all config attributes exist"""
        self.assertTrue(hasattr(Config, 'DATA_PATH'))
        self.assertTrue(hasattr(Config, 'DATA_SEP'))
        self.assertTrue(hasattr(Config, 'NUMERIC_FEATURES'))
        self.assertTrue(hasattr(Config, 'CATEGORICAL_NOMINAL'))
        self.assertTrue(hasattr(Config, 'CATEGORICAL_BINARY'))
        self.assertTrue(hasattr(Config, 'TEST_SIZE'))
        self.assertTrue(hasattr(Config, 'RANDOM_STATE'))
    
    def test_data_path(self):
        """Test data path configuration"""
        self.assertIsInstance(Config.DATA_PATH, str)
        self.assertTrue(Config.DATA_PATH.endswith('.csv'))
    
    def test_test_size(self):
        """Test test size is between 0 and 1"""
        self.assertGreater(Config.TEST_SIZE, 0)
        self.assertLess(Config.TEST_SIZE, 1)
    
    def test_random_state(self):
        """Test random state is integer"""
        self.assertIsInstance(Config.RANDOM_STATE, int)
    
    def test_features_lists(self):
        """Test feature lists are not empty"""
        self.assertGreater(len(Config.NUMERIC_FEATURES), 0)
        self.assertGreater(len(Config.CATEGORICAL_NOMINAL), 0)
        self.assertGreater(len(Config.CATEGORICAL_BINARY), 0)

if __name__ == '__main__':
    unittest.main()