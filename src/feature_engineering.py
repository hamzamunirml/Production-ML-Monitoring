"""
Feature Engineering Module
"""

import pandas as pd
from .config import Config

class FeatureEngineer:
    def create_age_groups(self, df):
        """Create age groups from age column"""
        def age_binning(age):
            if age <= Config.AGE_BINS['Young']:
                return 'Young'
            elif age <= Config.AGE_BINS['Middle-Aged']:
                return 'Middle-Aged'
            else:
                return 'Senior'
        
        df = df.copy()
        df['age_group'] = df['age'].apply(age_binning)
        print(f"✅ Age groups created: {df['age_group'].value_counts().to_dict()}")
        return df
    
    def detect_outliers(self, df, columns):
        """Detect outliers in numerical columns"""
        outliers_info = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_info[col] = {
                'outliers_count': len(outliers),
                'outliers_percentage': (len(outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        return outliers_info
    
    def get_feature_info(self, df):
        """Get feature information"""
        info = {
            'numerical_features': Config.NUMERIC_FEATURES,
            'categorical_features': Config.CATEGORICAL_NOMINAL + Config.CATEGORICAL_BINARY,
            'total_features': len(df.columns) - 1,
            'created_features': ['age_group']
        }
        return info