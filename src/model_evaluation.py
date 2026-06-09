"""
Model Evaluation Module
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

class ModelEvaluator:
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        print(f"\n📊 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"✅ {model_name} evaluated!")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1-Score: {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self):
        """Compare all models"""
        comparison = []
        for model_name, result in self.results.items():
            comparison.append({
                'Model': model_name,
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1-Score': result['metrics']['f1_score'],
                'ROC-AUC': result['metrics']['roc_auc']
            })
        
        df_comparison = pd.DataFrame(comparison)
        
        # Find best model
        best_model = df_comparison.loc[df_comparison['F1-Score'].idxmax(), 'Model']
        
        return df_comparison, best_model