"""
Credit Card Fraud Detection - Interactive Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    models = {}
    try:
        model_files = {
            'Logistic Regression': 'model/logistic_regression_model.pkl',
            'Decision Tree': 'model/decision_tree_model.pkl',
            'KNN': 'model/knn_model.pkl',
            'Naive Bayes': 'model/naive_bayes_model.pkl',
            'Random Forest': 'model/random_forest_model.pkl',
            'XGBoost': 'model/xgboost_model.pkl',
        }
        
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    models[name] = pickle.load(f)
        
        scaler = None
        if os.path.exists('model/scaler.pkl'):
            with open('model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }
    
    if y_pred_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_pred)
        except:
            metrics['AUC'] = 0.0
    
    return metrics

def main():
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page:", ["Home", "Model Evaluation", "About"])
    
    if page == "Home":
        st.header("Welcome to Credit Card Fraud Detection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### About This Project
            
            This application demonstrates a comprehensive machine learning solution 
            for detecting fraudulent credit card transactions.
            
            **Key Features:**
            - 6 different classification models
            - Comprehensive evaluation metrics
            - Interactive model comparison
            
            **Models Implemented:**
            1. Logistic Regression
            2. Decision Tree Classifier
            3. K-Nearest Neighbors (KNN)
            4. Naive Bayes
            5. Random Forest (Ensemble)
            6. XGBoost (Ensemble)
            """)
        
        with col2:
            st.markdown("""
            ### Dataset Information
            
            **Dataset:** Credit Card Fraud Detection
            - **Total Records:** 284,807
            - **Features:** 30 (PCA-transformed)
            - **Target Variable:** Class (0: Normal, 1: Fraud)
            - **Fraud Percentage:** 0.172%
            
            **Problem Statement:**
            Develop a machine learning model to identify fraudulent 
            credit card transactions with high precision and recall.
            """)
    
    elif page == "Model Evaluation":
        st.header("📊 Model Evaluation & Comparison")
        
        models, scaler = load_models()
        
        if not models:
            st.warning("Models not loaded.")
            return
        
        st.subheader("Upload Test Data")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Data Shape: {df.shape}")
                
                if 'Class' not in df.columns:
                    st.error("Class column not found")
                    return
                
                X = df.drop('Class', axis=1)
                y = df['Class']
                
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    st.error("Scaler not loaded")
                    return
                
                st.success(f"Data loaded! {len(X)} samples")
                
                selected_models = st.multiselect("Choose models:", list(models.keys()), default=list(models.keys()))
                
                if st.button("Evaluate Models"):
                    results = {}
                    
                    for model_name in selected_models:
                        model = models[model_name]
                        y_pred = model.predict(X_scaled)
                        
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                        else:
                            y_pred_proba = y_pred
                        
                        metrics = calculate_metrics(y, y_pred, y_pred_proba)
                        results[model_name] = metrics
                    
                    st.success("Evaluation Complete!")
                    
                    results_df = pd.DataFrame(results).T
                    results_df = results_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
                    st.dataframe(results_df, use_container_width=True)
                    
                    csv = results_df.to_csv()
                    st.download_button(
                        label="Download Results",
                        data=csv,
                        file_name="model_results.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    elif page == "About":
        st.header("About This Project")
        
        st.markdown("""
        ### Project Overview
        
        This is a machine learning assignment demonstrating end-to-end ML workflow:
        - Data preprocessing and feature scaling
        - Training multiple classification models
        - Comprehensive evaluation and comparison
        - Interactive web application deployment
        
        ### Models Implemented
        
        | Model | Type | Description |
        |-------|------|-------------|
        | **Logistic Regression** | Linear | Baseline model |
        | **Decision Tree** | Tree-based | Non-linear classification |
        | **K-Nearest Neighbors** | Instance-based | Distance-based |
        | **Naive Bayes** | Probabilistic | Fast algorithm |
        | **Random Forest** | Ensemble | Multiple trees |
        | **XGBoost** | Ensemble | Gradient boosting |
        
        ### Technologies Used
        
        - **Python 3.8+**
        - **Scikit-learn**: ML algorithms
        - **XGBoost**: Gradient boosting
        - **Pandas**: Data manipulation
        - **Streamlit**: Web application framework
        """)

if __name__ == "__main__":
    main()
