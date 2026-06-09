"""
Data Preprocessing Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from .config import Config
from .feature_engineering import FeatureEngineer

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            X[col] = X[col].map({'no': 0, 'yes': 1, 'unknown': 0})
        return X.values

class DataPreprocessor:
    def __init__(self):
        self.config = Config()
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = None
        
    def load_data(self):
        """Load dataset from local file"""
        print("Loading dataset from local file...")
        
        if not os.path.exists(Config.DATA_PATH):
            raise FileNotFoundError(f"Dataset file not found: {Config.DATA_PATH}")
        
        df = pd.read_csv(Config.DATA_PATH, sep=Config.DATA_SEP)
        print(f"✅ Dataset loaded! Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()[:5]}...")
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target"""
        print(f"\nOriginal columns: {df.columns.tolist()[:10]}...")
        
        # Create age groups
        df = self.feature_engineer.create_age_groups(df)
        
        # Separate features and target
        X = df.drop(Config.TARGET_COLUMN, axis=1)
        y = df[Config.TARGET_COLUMN].map(Config.TARGET_MAPPING)
        
        print(f"\n✅ Features prepared:")
        print(f"   Features shape: {X.shape}")
        print(f"   Target distribution:")
        print(f"   Class 0 (No): {(y == 0).sum()} ({(y == 0).mean()*100:.2f}%)")
        print(f"   Class 1 (Yes): {(y == 1).sum()} ({(y == 1).mean()*100:.2f}%)")
        
        return X, y
    
    def create_preprocessor(self):
        """Create preprocessing pipeline"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_nominal_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, Config.NUMERIC_FEATURES),
                ('nom', categorical_nominal_transformer, Config.CATEGORICAL_NOMINAL + ['age_group']),
                ('bin', BinaryEncoder(), Config.CATEGORICAL_BINARY)
            ])
        print("✅ Preprocessor created!")
        return self.preprocessor
    
    def split_data(self, X, y):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training: {X_train.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"\n   Training class distribution:")
        print(f"   Class 0 (No): {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.2f}%)")
        print(f"   Class 1 (Yes): {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess_data(self, X_train, X_test):
        """Preprocess data using the fitted preprocessor"""
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print(f"\n✅ Processed data shape:")
        print(f"   Training: {X_train_processed.shape}")
        print(f"   Test: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed
    
    def save_preprocessor(self, path='saved_encoders/preprocessor.pkl'):
        """Save preprocessor to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.preprocessor, path)
        print(f"✅ Preprocessor saved to {path}")