"""
Credit Card Fraud Detection - Complete ML Pipeline
Implements 6 classification models with comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve, auc
)
import pickle
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Complete pipeline for credit card fraud detection
    Trains and evaluates 6 different classification models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_and_prepare_data(self, filepath):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_csv(filepath)
        
        # Display basic info
        print(f"\nDataset shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum().sum()}")
        print(f"\nClass distribution:\n{df['Class'].value_counts()}")
        print(f"Fraud percentage: {df['Class'].mean() * 100:.2f}%")
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Split data (80-20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all required evaluation metrics"""
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_true, y_pred),
        }
        
        # AUC requires probability predictions
        if y_pred_proba is not None:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        else:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
            
        return metrics
    
    def train_logistic_regression(self):
        """Train Logistic Regression"""
        print("\n" + "="*60)
        print("Training Logistic Regression...")
        print("="*60)
        
        model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_decision_tree(self):
        """Train Decision Tree Classifier"""
        print("\n" + "="*60)
        print("Training Decision Tree Classifier...")
        print("="*60)
        
        model = DecisionTreeClassifier(random_state=self.random_state, max_depth=15)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['Decision Tree'] = model
        self.results['Decision Tree'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_knn(self):
        """Train K-Nearest Neighbors Classifier"""
        print("\n" + "="*60)
        print("Training K-Nearest Neighbors Classifier...")
        print("="*60)
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['KNN'] = model
        self.results['KNN'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_naive_bayes(self):
        """Train Gaussian Naive Bayes Classifier"""
        print("\n" + "="*60)
        print("Training Gaussian Naive Bayes Classifier...")
        print("="*60)
        
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['Naive Bayes'] = model
        self.results['Naive Bayes'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_random_forest(self):
        """Train Random Forest Classifier (Ensemble)"""
        print("\n" + "="*60)
        print("Training Random Forest Classifier (Ensemble)...")
        print("="*60)
        
        model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_xgboost(self):
        """Train XGBoost Classifier (Ensemble)"""
        print("\n" + "="*60)
        print("Training XGBoost Classifier (Ensemble)...")
        print("="*60)
        
        # Scale class weight for imbalanced data
        scale_pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=100,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = metrics
        
        print(f"Accuracy: {metrics['Accuracy']:.4f}")
        print(f"AUC: {metrics['AUC']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall: {metrics['Recall']:.4f}")
        print(f"F1 Score: {metrics['F1']:.4f}")
        print(f"MCC: {metrics['MCC']:.4f}")
        
        return self
    
    def train_all_models(self):
        """Train all 6 models"""
        print("\n\n" + "#"*60)
        print("# STARTING TRAINING OF ALL 6 MODELS")
        print("#"*60)
        
        self.train_logistic_regression()
        self.train_decision_tree()
        self.train_knn()
        self.train_naive_bayes()
        self.train_random_forest()
        self.train_xgboost()
        
        return self
    
    def generate_comparison_table(self):
        """Generate comparison table of all models"""
        print("\n\n" + "="*100)
        print("COMPARISON TABLE - ALL MODELS")
        print("="*100)
        
        df_results = pd.DataFrame(self.results).T
        df_results = df_results[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
        
        print(df_results.to_string())
        
        return df_results
    
    def save_models(self, save_dir='model'):
        """Save all trained models"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for name, model in self.models.items():
            filename = f"{save_dir}/{name.replace(' ', '_').lower()}_model.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {name} to {filename}")
        
        # Save scaler
        with open(f"{save_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {save_dir}/scaler.pkl")
    
    def get_results_dict(self):
        """Return results as dictionary for Streamlit app"""
        return self.results, self.models, self.scaler


if __name__ == "__main__":
    # Download dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    # Or use: https://drive.google.com/file/d/1r52Xk-nrU5OQa5xw7VoYcjKdj1PN10KI/view
    
    try:
        # Initialize pipeline
        pipeline = FraudDetectionPipeline(random_state=42)
        
        # Load and prepare data
        pipeline.load_and_prepare_data('creditcard.csv')
        
        # Train all models
        pipeline.train_all_models()
        
        # Generate comparison table
        comparison_df = pipeline.generate_comparison_table()
        
        # Save models
        pipeline.save_models()
        
        print("\n\nTraining completed successfully!")
        print("Models saved in 'model' directory")
        
    except FileNotFoundError:
        print("Error: creditcard.csv not found!")
        print("Please download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("Or from: https://drive.google.com/file/d/1r52Xk-nrU5OQa5xw7VoYcjKdj1PN10KI/view")
