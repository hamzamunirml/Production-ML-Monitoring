"""
Model Training Module
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Ab yeh kaam karega
from src.config import Config

class ModelTrainer:
    def __init__(self):
        self.config = Config()
        self.dt_model = None
        self.rf_model = None
        self.smote = None

    def handle_imbalance(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance"""
        print("\n" + "="*50)
        print("HANDLING IMBALANCE WITH SMOTE")
        print("="*50)

        print(f"Before SMOTE - Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")

        self.smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)

        print(f"After SMOTE - Class 0: {(y_train_balanced == 0).sum()}, Class 1: {(y_train_balanced == 1).sum()}")
        print("✅ Dataset balanced successfully!")

        return X_train_balanced, y_train_balanced

    def train_decision_tree(self, X_train, y_train):
        """Train Decision Tree classifier"""
        print("\n" + "="*50)
        print("TRAINING DECISION TREE")
        print("="*50)

        self.dt_model = DecisionTreeClassifier(**Config.DT_PARAMS)
        self.dt_model.fit(X_train, y_train)

        print("✅ Decision Tree trained successfully!")
        return self.dt_model

    def train_random_forest(self, X_train, y_train):
        """Train Random Forest classifier"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)

        self.rf_model = RandomForestClassifier(**Config.RF_PARAMS)
        self.rf_model.fit(X_train, y_train)

        print("✅ Random Forest trained successfully!")
        return self.rf_model

    def save_models(self, path='saved_models/'):
        """Save trained models"""
        import os
        os.makedirs(path, exist_ok=True)

        if self.dt_model:
            joblib.dump(self.dt_model, f'{path}decision_tree_model.pkl')
            print(f"✅ Decision Tree saved to {path}decision_tree_model.pkl")

        if self.rf_model:
            joblib.dump(self.rf_model, f'{path}random_forest_model.pkl')
            print(f"✅ Random Forest saved to {path}random_forest_model.pkl")

        if self.smote:
            joblib.dump(self.smote, f'{path}../saved_encoders/smote.pkl')
            print(f"✅ SMOTE saved to saved_encoders/smote.pkl")