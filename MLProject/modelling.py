import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load the preprocessed dataset from file path."""
    try:
        data = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully from {file_path}")
        print(f"📊 Dataset shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"❌ Error: File {file_path} not found.")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None

def prepare_features_and_target(data, target_col='stroke'):
    """Separate features and target variable."""
    features = data.drop(target_col, axis=1)
    target = data[target_col]
    
    print(f"🎯 Features shape: {features.shape}")
    print(f"🎯 Target shape: {target.shape}")
    print(f"📈 Target distribution: {target.value_counts().to_dict()}")
    
    return features, target

def split_and_scale_data(X, y, test_ratio=0.2, random_seed=42):
    """Split data into train/test sets and apply scaling."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y
    )
    print(f"🔄 Data split - Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Apply feature scaling
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    print("⚖️ Features scaled using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'auc_roc': roc_auc_score(y_true, y_pred_proba)
    }
    return metrics

def train_stroke_prediction_model(dataset, target_column='stroke'):
    """
    Train stroke prediction model using Logistic Regression with MLflow tracking.
    """
    if dataset is None:
        print("❌ No dataset provided for training")
        return None

    # Prepare data
    X, y = prepare_features_and_target(dataset, target_column)
    X_train_scaled, X_test_scaled, y_train, y_test = split_and_scale_data(X, y)
    
    # Configure MLflow autologging before starting run
    mlflow.sklearn.autolog(
        log_model_signatures=True, 
        log_input_examples=True, 
        log_post_training_metrics=True
    )
    print("🔧 MLflow autolog enabled for sklearn models")

    # Start MLflow run for tracking
    with mlflow.start_run(run_name="Stroke_Prediction_LogisticRegression") as active_run:
        print(f"🚀 Started MLflow run: {active_run.info.run_id}")

        # Initialize model with optimized parameters
        classifier = LogisticRegression(
            C=100,
            random_state=42,
            max_iter=1000,
            solver='liblinear',
            penalty='l1'
        )
        
        print(f"🤖 Training {classifier.__class__.__name__} model...")
        
        # Train the model
        classifier.fit(X_train_scaled, y_train)
        
        # Generate predictions
        test_predictions = classifier.predict(X_test_scaled)
        test_probabilities = classifier.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate performance metrics
        performance_metrics = calculate_metrics(y_test, test_predictions, test_probabilities)
        
        # Display results
        print(f"\n📊 Model Performance Results:")
        print(f"   Accuracy:  {performance_metrics['accuracy']:.4f}")
        print(f"   Precision: {performance_metrics['precision']:.4f}")
        print(f"   Recall:    {performance_metrics['recall']:.4f}")
        print(f"   F1-Score:  {performance_metrics['f1_score']:.4f}")
        print(f"   AUC-ROC:   {performance_metrics['auc_roc']:.4f}")
        
        print(f"\n✅ Training completed. MLflow run ID: {active_run.info.run_id}")
        print("💡 View results with: mlflow ui")
        
        return active_run.info.run_id

if __name__ == "__main__":
    # Setup command line argument parsing
    arg_parser = argparse.ArgumentParser(
        description='Train stroke prediction model with MLflow tracking'
    )
    arg_parser.add_argument(
        '--data_path', 
        type=str,
        default='dataset_preprocessing/train_data_processed.csv',
        help='Path to the preprocessed training data'
    )
    arg_parser.add_argument(
        '--experiment_name', 
        type=str,
        default='stroke_prediction_github_actions',
        help='Name of the MLflow experiment'
    )
    
    # Parse arguments
    args = arg_parser.parse_args()
    
    # Configure MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    print(f"🎯 MLflow experiment set to: {args.experiment_name}")
    
    # Load and validate data
    training_data = load_data(args.data_path)
    
    if training_data is not None:
        # Check for missing values
        missing_values = training_data.isnull().sum()
        print(f"\n🔍 Missing values check:")
        print(missing_values)
        
        # Clean data by removing missing values
        original_size = len(training_data)
        training_data.dropna(inplace=True)
        cleaned_size = len(training_data)
        
        print(f"🧹 Data cleaning: {original_size} → {cleaned_size} rows")
        
        # Proceed with training if data is sufficient
        if not training_data.empty and cleaned_size > 50:
            model_run_id = train_stroke_prediction_model(
                training_data.copy(), 
                target_column='stroke'
            )
            
            if model_run_id:
                print(f"\n🎉 Model training successful!")
                print(f"📝 Run ID: {model_run_id}")
                print(f"🔗 Access results: mlflow ui")
            else:
                print("❌ Model training failed")
        else:
            print("❌ Insufficient data for training after cleaning")
    else:
        print(f"❌ Failed to load data from: {args.data_path}")
        print("💡 Ensure preprocessing has been completed first")