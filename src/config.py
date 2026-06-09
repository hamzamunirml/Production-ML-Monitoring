"""
Configuration file
"""

class Config:
    # Local dataset file
    DATA_PATH = "bank-full.csv"
    DATA_SEP = ','  # Your file is comma-separated
    
    # Age binning rules
    AGE_BINS = {
        'Young': 30,
        'Middle-Aged': 50,
        'Senior': float('inf')
    }
    
    NUMERIC_FEATURES = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    CATEGORICAL_NOMINAL = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']
    CATEGORICAL_BINARY = ['default', 'housing', 'loan']
    
    TARGET_COLUMN = 'y'
    TARGET_MAPPING = {'no': 0, 'yes': 1}
    
    TEST_SIZE = 0.3
    RANDOM_STATE = 42
    
    DT_PARAMS = {
        'random_state': 42,
        'max_depth': 10,
        'min_samples_split': 20
    }
    
    RF_PARAMS = {
        'random_state': 42,
        'n_estimators': 100,
        'max_depth': 15,
        'min_samples_split': 20,
        'n_jobs': -1
    }