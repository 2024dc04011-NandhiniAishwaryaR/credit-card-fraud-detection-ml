"""
Credit Card Fraud Detection - Interactive Streamlit App
Allows users to upload data, select models, and view evaluation metrics
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

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_models():
    """Load all trained models and scaler"""
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
        
        # Load scaler
        scaler = None
        if os.path.exists('model/scaler.pkl'):
            with open('model/scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
        
        return models, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}, None

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
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

def create_confusion_matrix_plot(y_true, y_pred):
    """Create confusion matrix visualization"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=['No Fraud', 'Fraud'],
                yticklabels=['No Fraud', 'Fraud'])
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    return fig

def create_classification_report(y_true, y_pred):
    """Create classification report"""
    report = classification_report(y_true, y_pred, 
                                   target_names=['No Fraud', 'Fraud'],
                                   output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    return report_df

# Main app
def main():
    # Header
    st.title("💳 Credit Card Fraud Detection System")
    st.markdown("---")
    
    # Sidebar
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
        
        st.markdown("---")
        st.markdown("""
        ### Why This Matters
        
        Credit card fraud detection is a critical real-world problem:
        - **Cost Impact**: Fraudulent transactions cost billions annually
        - **Business Impact**: Better fraud detection reduces chargeback losses
        - **Technical Challenge**: Highly imbalanced dataset (99.83% legitimate)
        - **ML Application**: Perfect use case for classification algorithms
        """)
    
    elif page == "Model Evaluation":
        st.header("📊 Model Evaluation & Comparison")
        
        # Load models
        models, scaler = load_models()
        
        if not models:
            st.warning("⚠️ Models not loaded. Please ensure model files are in the 'model' directory.")
            return
        
        # File uploader section
        st.subheader("Upload Test Data")
        
        # Download test data button
        st.markdown("**Don't have test data?** Download the sample:")
        if os.path.exists('test_data.csv'):
            with open('test_data.csv', 'rb') as f:
                st.download_button(
                    label="📥 Download Test Data (test_data.csv)",
                    data=f,
                    file_name="test_data.csv",
                    mime="text/csv"
                )
        else:
            st.info("Test data file not available")
        
        st.markdown("---")
        
        # File upload
        uploaded_file = st.file_uploader("Choose a CSV file with 30 features and Class column", type="csv")
        
        if uploaded_file is not None:
            try:
                # Load data
                df = pd.read_csv(uploaded_file)
                st.write(f"**Data Shape:** {df.shape}")
                st.write(f"**Columns Found:** {list(df.columns)}")
                
                # Check if Class column exists
                if 'Class' not in df.columns:
                    st.error("⚠️ 'Class' column not found in the uploaded file. Please include the Class column (0 or 1).")
                    return
                
                # Separate features and target
                X = df.drop('Class', axis=1)
                y = df['Class']
                
                # Ensure features are in correct order and have correct names
                expected_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                                   'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                                   'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']
                
                # Check if we have the right features
                if len(X.columns) != 30:
                    st.error(f"⚠️ Expected 30 features, but got {len(X.columns)}. Please check your data.")
                    return
                
                # Reorder columns if needed
                if list(X.columns) != expected_features:
                    try:
                        X = X[expected_features]
                    except KeyError:
                        st.error("⚠️ Feature names don't match expected features. Please ensure your data has the correct column names.")
                        return
                
                # Scale features
                if scaler:
                    X_scaled = scaler.transform(X)
                else:
                    st.error("Scaler not loaded!")
                    return
                
                st.success(f"✅ Data loaded successfully! Test samples: {len(X)}")
                
                # Model selection
                st.subheader("Select Model(s) to Evaluate")
                selected_models = st.multiselect(
                    "Choose models to compare:",
                    list(models.keys()),
                    default=list(models.keys())
                )
                
                if not selected_models:
                    st.warning("Please select at least one model")
                    return
                
                # Evaluate models
                if st.button("🚀 Evaluate Selected Models", key="eval_button"):
                    results = {}
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    for idx, model_name in enumerate(selected_models):
                        model = models[model_name]
                        
                        # Make predictions
                        y_pred = model.predict(X_scaled)
                        
                        # Get probability predictions
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                        else:
                            y_pred_proba = y_pred
                        
                        # Calculate metrics
                        metrics = calculate_metrics(y, y_pred, y_pred_proba)
                        results[model_name] = metrics
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(selected_models))
                    
                    st.success("✅ Evaluation Complete!")
                    
                    # Display results table
                    st.subheader("📈 Results Comparison")
                    results_df = pd.DataFrame(results).T
                    results_df = results_df[['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']]
                    
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv()
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="model_evaluation_results.csv",
                        mime="text/csv"
                    )
                    
                    # Display detailed metrics for each model
                    st.subheader("🔍 Detailed Metrics per Model")
                    
                    for model_name in selected_models:
                        with st.expander(f"{model_name}", expanded=False):
                            col1, col2, col3 = st.columns(3)
                            metrics = results[model_name]
                            
                            with col1:
                                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                                st.metric("AUC", f"{metrics['AUC']:.4f}")
                            
                            with col2:
                                st.metric("Precision", f"{metrics['Precision']:.4f}")
                                st.metric("Recall", f"{metrics['Recall']:.4f}")
                            
                            with col3:
                                st.metric("F1 Score", f"{metrics['F1']:.4f}")
                                st.metric("MCC", f"{metrics['MCC']:.4f}")
                            
                            # Confusion matrix
                            model = models[model_name]
                            y_pred = model.predict(X_scaled)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = create_confusion_matrix_plot(y, y_pred)
                                st.pyplot(fig)
                                plt.close()
                            
                            with col2:
                                report_df = create_classification_report(y, y_pred)
                                st.dataframe(report_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"⚠️ Error processing file: {str(e)}")
    
    elif page == "About":
        st.header("ℹ️ About This Project")
        
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
        | **Logistic Regression** | Linear | Baseline model, interpretable |
        | **Decision Tree** | Tree-based | Non-linear, easy to visualize |
        | **K-Nearest Neighbors** | Instance-based | Distance-based classification |
        | **Naive Bayes** | Probabilistic | Fast, based on Bayes theorem |
        | **Random Forest** | Ensemble | Multiple trees, robust |
        | **XGBoost** | Ensemble | Gradient boosting, high performance |
        
        ### Evaluation Metrics
        
        - **Accuracy**: Overall correctness
        - **AUC Score**: Area under ROC curve
        - **Precision**: True positives / All predicted positives
        - **Recall**: True positives / All actual positives
        - **F1 Score**: Harmonic mean of Precision and Recall
        - **MCC**: Matthews Correlation Coefficient
        
        ### Why These Metrics Matter
        
        In fraud detection, we care about:
        - **Recall** (catching all frauds) - Don't want to miss frauds
        - **Precision** (false positive rate) - Don't want false alarms
        - **AUC** (overall performance) - How well we separate classes
        - **MCC** (balanced view) - Accounts for class imbalance
        
        ### Dataset Challenge
        
        The dataset is **highly imbalanced**:
        - 99.83% legitimate transactions
        - 0.17% fraudulent transactions
        
        This requires special handling:
        - Class weight adjustment
        - Stratified splitting
        - Metric selection (Recall > Accuracy)
        
        ### Key Insights
        
        1. **Ensemble models** (Random Forest, XGBoost) typically outperform single models
        2. **Recall is critical** - We must catch fraudulent transactions
        3. **Precision matters too** - Too many false positives are costly
        4. **AUC is a good metric** for imbalanced data
        5. **Feature scaling** is essential for distance-based models (KNN)
        
        ### Technologies Used
        
        - **Python 3.8+**
        - **Scikit-learn**: ML algorithms
        - **XGBoost**: Gradient boosting
        - **Pandas**: Data manipulation
        - **NumPy**: Numerical computing
        - **Matplotlib & Seaborn**: Visualization
        - **Streamlit**: Web application framework
        """)

if __name__ == "__main__":
    main()
