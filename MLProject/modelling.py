import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import argparse
import os

def load_stroke_data(filepath):
    """Load stroke prediction dataset from CSV file."""
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Successfully loaded dataset from {filepath}")
        print(f"📊 Dataset dimensions: {data.shape}")
        return data
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def prepare_data_for_training(data, target='stroke'):
    """Prepare features and target for model training."""
    features = data.drop(columns=[target])
    target_values = data[target]
    
    print(f"🎯 Features: {features.shape}, Target: {target_values.shape}")
    print(f"📈 Class distribution: {target_values.value_counts().to_dict()}")
    
    return features, target_values

def create_train_test_splits(X, y):
    """Create train and test splits with proper scaling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"🔄 Training set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
    print("⚖️ Feature scaling completed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model_performance(model, X_test, y_test):
    """Calculate comprehensive model evaluation metrics."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    results = {
        'accuracy': accuracy_score(y_test, predictions),
        'precision': precision_score(y_test, predictions),
        'recall': recall_score(y_test, predictions),
        'f1': f1_score(y_test, predictions),
        'auc': roc_auc_score(y_test, probabilities)
    }
    
    return results, predictions, probabilities

def execute_model_training(dataset, experiment_name="Default_Experiment"):
    """Execute complete model training pipeline with MLflow tracking."""
    if dataset is None:
        print("❌ No valid dataset provided")
        return None
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    print(f"🧪 MLflow experiment set to: {experiment_name}")
    
    # Prepare data
    X, y = prepare_data_for_training(dataset)
    X_train, X_test, y_train, y_test = create_train_test_splits(X, y)
    
    # Enable MLflow autologging
    mlflow.sklearn.autolog(
        log_model_signatures=True,
        log_input_examples=True, 
        log_post_training_metrics=True
    )
    print("🔧 MLflow autologging activated")
    
    # Execute training with MLflow tracking
    with mlflow.start_run(run_name="StrokePredictor_LogisticRegression") as run:
        print(f"🚀 MLflow run started: {run.info.run_id}")
        
        # Initialize optimized logistic regression model
        stroke_classifier = LogisticRegression(
            C=100,
            solver='liblinear', 
            penalty='l1',
            random_state=42,
            max_iter=1000
        )
        
        print(f"🤖 Training classifier: {stroke_classifier.__class__.__name__}")
        
        # Train the model
        stroke_classifier.fit(X_train, y_train)
        
        # Evaluate performance
        metrics, preds, probs = evaluate_model_performance(stroke_classifier, X_test, y_test)
        
        # Display results
        print(f"\n📊 Model Performance Summary:")
        print(f"   🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   🎯 Precision: {metrics['precision']:.4f}")
        print(f"   🎯 Recall:    {metrics['recall']:.4f}")
        print(f"   🎯 F1-Score:  {metrics['f1']:.4f}")
        print(f"   🎯 AUC-ROC:   {metrics['auc']:.4f}")
        
        print(f"\n✅ Training completed successfully!")
        print(f"📝 MLflow run ID: {run.info.run_id}")
        
        return run.info.run_id

def main():
    """Main function to handle command line arguments and execute training."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Stroke Prediction Model Training')
    parser.add_argument('--data_path', type=str, 
                       default='dataset_preprocessing/train_data_processed.csv',
                       help='Path to the training data CSV file')
    parser.add_argument('--experiment_name', type=str,
                       default='CI_Experiment_GitHubActions_Stroke',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    print(f"🔧 Configuration:")
    print(f"   📁 Data path: {args.data_path}")
    print(f"   🧪 Experiment: {args.experiment_name}")
    
    # Check if data file exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return
    
    # Load and validate dataset
    stroke_dataset = load_stroke_data(args.data_path)
    
    if stroke_dataset is not None:
        # Check data quality
        print(f"\n🔍 Data Quality Check:")
        missing_data = stroke_dataset.isnull().sum()
        print(missing_data)
        
        # Clean dataset
        original_rows = len(stroke_dataset)
        stroke_dataset.dropna(inplace=True)
        cleaned_rows = len(stroke_dataset)
        print(f"🧹 Data cleaning: {original_rows} → {cleaned_rows} rows")
        
        # Execute training if sufficient data
        if cleaned_rows > 100:
            training_run_id = execute_model_training(
                stroke_dataset.copy(), 
                args.experiment_name
            )
            
            if training_run_id:
                print(f"\n🎉 Training pipeline completed!")
                print(f"🔗 Run ID: {training_run_id}")
                print(f"💡 View results: mlflow ui")
            else:
                print("❌ Training pipeline failed")
        else:
            print("❌ Insufficient data after cleaning")
    else:
        print("❌ Unable to proceed - dataset loading failed")

if __name__ == "__main__":
    main()