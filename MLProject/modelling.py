import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import time
import warnings
import argparse
import os
import sys

warnings.filterwarnings('ignore')


class StrokeModelTrainer:
    """
    Train multiple ML models for stroke prediction with MLflow tracking
    Compatible with MLflow Projects
    """
    
    def __init__(self, experiment_name="stroke_prediction_ci", data_path="stroke_data_preprocessing"):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.models = {}
        self.results = {}
        
        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)
        print(f"🎯 MLflow experiment: {experiment_name}")
        print(f"📁 Data path: {data_path}")
    
    def load_processed_data(self):
        """Load preprocessed training and test data from specified path"""
        try:
            # Construct file paths
            train_path = os.path.join(self.data_path, "train_data_processed.csv")
            test_path = os.path.join(self.data_path, "test_data_processed.csv")
            
            print(f"📂 Loading data from:")
            print(f"   Train: {train_path}")
            print(f"   Test: {test_path}")
            
            # Check if files exist
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training data not found: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test data not found: {test_path}")
            
            # Load data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            # Split features and target
            self.X_train = train_data.drop('stroke', axis=1)
            self.y_train = train_data['stroke']
            self.X_test = test_data.drop('stroke', axis=1)
            self.y_test = test_data['stroke']
            
            print(f"✅ Data loaded successfully:")
            print(f"   Training: {self.X_train.shape[0]} samples × {self.X_train.shape[1]} features")
            print(f"   Testing: {self.X_test.shape[0]} samples × {self.X_test.shape[1]} features")
            print(f"   Class distribution (train): {self.y_train.value_counts().to_dict()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            print(f"💡 Current working directory: {os.getcwd()}")
            print(f"💡 Available files: {os.listdir('.')}")
            return False
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive model metrics"""
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        # ROC AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        # Additional metrics
        metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        metrics['sensitivity'] = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        
        return metrics
    
    def train_logistic_regression(self):
        """Train Logistic Regression model with MLflow tracking"""
        model_name = "Logistic Regression"
        print(f"\n🤖 Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_ci"):
            start_time = time.time()
            
            # Initialize model with default parameters
            model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
            
            # Log parameters manually
            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_iter", 1000)
            mlflow.log_param("solver", "liblinear")
            mlflow.log_param("C", 1.0)
            mlflow.log_param("ci_run", True)  # Mark as CI run
            
            # Train model
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics manually
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("specificity", metrics['specificity'])
            mlflow.log_metric("sensitivity", metrics['sensitivity'])
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log confusion matrix elements
            mlflow.log_metric("true_positives", metrics['true_positives'])
            mlflow.log_metric("true_negatives", metrics['true_negatives'])
            mlflow.log_metric("false_positives", metrics['false_positives'])
            mlflow.log_metric("false_negatives", metrics['false_negatives'])
            
            # Log model
            mlflow.sklearn.log_model(model, name="logistic_regression_ci_model")
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            print(f"✅ {model_name} completed:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def train_random_forest(self):
        """Train Random Forest model with MLflow tracking"""
        model_name = "Random Forest"
        print(f"\n🌳 Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_ci"):
            start_time = time.time()
            
            # Initialize model with default parameters
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            
            # Log parameters manually
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_depth", None)
            mlflow.log_param("min_samples_split", 2)
            mlflow.log_param("min_samples_leaf", 1)
            mlflow.log_param("n_jobs", -1)
            mlflow.log_param("ci_run", True)
            
            # Train model
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics manually
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("specificity", metrics['specificity'])
            mlflow.log_metric("sensitivity", metrics['sensitivity'])
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log confusion matrix elements
            mlflow.log_metric("true_positives", metrics['true_positives'])
            mlflow.log_metric("true_negatives", metrics['true_negatives'])
            mlflow.log_metric("false_positives", metrics['false_positives'])
            mlflow.log_metric("false_negatives", metrics['false_negatives'])
            
            # Log feature importance
            feature_importance = model.feature_importances_
            mlflow.log_metric("mean_feature_importance", np.mean(feature_importance))
            mlflow.log_metric("std_feature_importance", np.std(feature_importance))
            
            # Log model
            mlflow.sklearn.log_model(model, name="random_forest_ci_model")
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            print(f"✅ {model_name} completed:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def train_xgboost(self):
        """Train XGBoost model with MLflow tracking"""
        model_name = "XGBoost"
        print(f"\n🚀 Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_ci"):
            start_time = time.time()
            
            # Initialize model with default parameters
            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
            
            # Log parameters manually
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.3)
            mlflow.log_param("subsample", 1.0)
            mlflow.log_param("colsample_bytree", 1.0)
            mlflow.log_param("eval_metric", "logloss")
            mlflow.log_param("ci_run", True)
            
            # Train model
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log metrics manually
            mlflow.log_metric("accuracy", metrics['accuracy'])
            mlflow.log_metric("precision", metrics['precision'])
            mlflow.log_metric("recall", metrics['recall'])
            mlflow.log_metric("f1_score", metrics['f1_score'])
            mlflow.log_metric("roc_auc", metrics['roc_auc'])
            mlflow.log_metric("specificity", metrics['specificity'])
            mlflow.log_metric("sensitivity", metrics['sensitivity'])
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log confusion matrix elements
            mlflow.log_metric("true_positives", metrics['true_positives'])
            mlflow.log_metric("true_negatives", metrics['true_negatives'])
            mlflow.log_metric("false_positives", metrics['false_positives'])
            mlflow.log_metric("false_negatives", metrics['false_negatives'])
            
            # Log feature importance
            feature_importance = model.feature_importances_
            mlflow.log_metric("mean_feature_importance", np.mean(feature_importance))
            mlflow.log_metric("std_feature_importance", np.std(feature_importance))
            
            # Log model
            mlflow.xgboost.log_model(model, name="xgboost_ci_model")
            
            # Store results
            self.models[model_name] = model
            self.results[model_name] = metrics
            
            print(f"✅ {model_name} completed:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    
    def train_all_models(self):
        """Train all models and compare results"""
        print("🚀 Starting CI model training pipeline...")
        
        # Train each model
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Print comparison summary
        self.print_model_comparison()
    
    def print_model_comparison(self):
        """Print comparison of all trained models"""
        print("\n" + "="*60)
        print("📊 CI MODEL TRAINING SUMMARY")
        print("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Specificity': f"{metrics['specificity']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Find best model by F1-score
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_f1 = self.results[best_model_name]['f1_score']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Best F1-Score: {best_f1:.4f}")
        
        return best_model_name, comparison_df


def parse_arguments():
    """Parse command line arguments for MLflow Project"""
    parser = argparse.ArgumentParser(description="Stroke Prediction ML Training Pipeline")
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="stroke_prediction_ci",
        help="MLflow experiment name (default: stroke_prediction_ci)"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="stroke_data_preprocessing",
        help="Path to preprocessed data directory (default: stroke_data_preprocessing)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function for MLflow Project execution"""
    print("🎯 Starting MLflow Project: Stroke Prediction Training")
    print("="*60)
    print(f"🐍 Python version: {sys.version}")
    print(f"📦 Working directory: {os.getcwd()}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    if args.verbose:
        print(f"📋 Configuration:")
        print(f"   Experiment name: {args.experiment_name}")
        print(f"   Data path: {args.data_path}")
        print(f"   Working directory: {os.getcwd()}")
    
    try:
        # Initialize trainer with parsed arguments
        trainer = StrokeModelTrainer(
            experiment_name=args.experiment_name,
            data_path=args.data_path
        )
        
        # Load data
        if not trainer.load_processed_data():
            print("❌ Failed to load data. Please check file paths.")
            sys.exit(1)
        
        # Train all models
        trainer.train_all_models()
        
        print("\n🎉 CI Model training completed successfully!")
        print("📊 All models trained and metrics logged to MLflow")
        
        # Return success exit code
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error in model training pipeline: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()