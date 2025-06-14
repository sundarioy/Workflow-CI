import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, make_scorer
)
from mlflow.models import infer_signature
import xgboost as xgb
import time
import warnings
import argparse
import os
import sys

warnings.filterwarnings('ignore')


class StrokeModelTrainer:
    """
    Train multiple ML models for stroke prediction with hyperparameter tuning
    Compatible with MLflow Projects - FIXED VERSION
    """
    
    def __init__(self, experiment_name="stroke_prediction_ci", data_path="stroke_data_preprocessing", enable_tuning=True):
        self.experiment_name = experiment_name
        self.data_path = data_path
        self.enable_tuning = enable_tuning
        self.models = {}
        self.results = {}
        self.best_params = {}
        
        # Detect MLflow Project context
        self.is_mlflow_project = (
            os.getenv('MLFLOW_RUN_ID') is not None or
            os.getenv('MLFLOW_EXPERIMENT_ID') is not None
        )
        
        if not self.is_mlflow_project:
            mlflow.set_experiment(experiment_name)
            print(f"🎯 Experiment: {experiment_name}")
        else:
            print(f"🎯 MLflow Project mode detected - models will be saved")
            
        print(f"📁 Data path: {data_path}")
        print(f"🔧 Hyperparameter tuning: {'Enabled' if enable_tuning else 'Disabled'}")
    
    def load_processed_data(self):
        """Load preprocessed training and test data"""
        try:
            train_path = os.path.join(self.data_path, "train_data_processed.csv")
            test_path = os.path.join(self.data_path, "test_data_processed.csv")
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                raise FileNotFoundError("Training or test data not found")
            
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            self.X_train = train_data.drop('stroke', axis=1)
            self.y_train = train_data['stroke']
            self.X_test = test_data.drop('stroke', axis=1)
            self.y_test = test_data['stroke']
            
            print(f"✅ Data loaded: {self.X_train.shape[0]} train, {self.X_test.shape[0]} test samples")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return False
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive model metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1_score': f1_score(y_true, y_pred, average='binary')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix elements
        cm = confusion_matrix(y_true, y_pred)
        metrics.update({
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1]),
            'specificity': cm[0, 0] / (cm[0, 0] + cm[0, 1]),
            'sensitivity': cm[1, 1] / (cm[1, 1] + cm[1, 0])
        })
        
        return metrics
    
    def safe_log_param(self, key, value):
        """Safely log parameter"""
        try:
            mlflow.log_param(key, value)
        except Exception:
            pass
    
    def safe_log_metric(self, key, value):
        """Safely log metric"""
        try:
            mlflow.log_metric(key, value)
        except Exception:
            pass
    
    def get_logistic_regression_params(self):
        """Get parameter grid for Logistic Regression"""
        if self.enable_tuning:
            return {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000],
                'penalty': ['l1', 'l2']
            }
        else:
            return {}
    
    def get_random_forest_params(self):
        """Get parameter grid for Random Forest"""
        if self.enable_tuning:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        else:
            return {}
    
    def get_xgboost_params(self):
        """Get parameter grid for XGBoost"""
        if self.enable_tuning:
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        else:
            return {}
    
    def tune_model(self, model, param_grid, model_name, cv=3, scoring='f1'):
        """Perform hyperparameter tuning"""
        if not param_grid:
            print(f"   No tuning for {model_name} - using default parameters")
            # FIX: Fit model with default parameters when no tuning
            model.fit(self.X_train, self.y_train)
            return model, {}
        
        print(f"   🔍 Tuning {model_name} with {len(param_grid)} parameter combinations...")
        
        # Use RandomizedSearchCV for faster tuning with large parameter spaces
        if len(param_grid) > 0:
            param_combinations = 1
            for key, values in param_grid.items():
                param_combinations *= len(values)
            
            if param_combinations > 50:
                # Use RandomizedSearchCV for large parameter spaces
                search = RandomizedSearchCV(
                    model, param_grid, 
                    n_iter=20, cv=cv, scoring=scoring,
                    random_state=42, n_jobs=-1
                )
            else:
                # Use GridSearchCV for smaller parameter spaces
                search = GridSearchCV(
                    model, param_grid, 
                    cv=cv, scoring=scoring, n_jobs=-1
                )
        
        search.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Best {model_name} score: {search.best_score_:.4f}")
        print(f"   🎯 Best {model_name} params: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
    
    def train_model_with_tuning(self):
        """Train models with hyperparameter tuning - FIXED VERSION"""
        print("🚀 Training models with hyperparameter tuning...")
        
        models_config = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': self.get_logistic_regression_params(),
                'log_func': mlflow.sklearn.log_model
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': self.get_random_forest_params(),
                'log_func': mlflow.sklearn.log_model
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': self.get_xgboost_params(),
                'log_func': mlflow.xgboost.log_model
            }
        ]
        
        best_overall_model = None
        best_overall_f1 = 0
        best_overall_name = ""
        
        for config in models_config:
            model_name = config['name']
            print(f"\n🤖 Training {model_name}...")
            
            start_time = time.time()
            
            # Hyperparameter tuning
            tuned_model, best_params = self.tune_model(
                config['model'], 
                config['params'], 
                model_name
            )
            tuning_time = time.time() - start_time
            
            # Train final model and evaluate
            eval_start = time.time()
            y_pred = tuned_model.predict(self.X_test)
            y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1]
            eval_time = time.time() - eval_start
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            # Log parameters (both tuned and default)
            prefix = model_name.lower().replace(' ', '_')
            self.safe_log_param(f"{prefix}_model_type", tuned_model.__class__.__name__)
            self.safe_log_param(f"{prefix}_tuning_enabled", self.enable_tuning)
            
            # Log best parameters
            for param_name, param_value in best_params.items():
                self.safe_log_param(f"{prefix}_{param_name}", param_value)
            
            # Log timing
            self.safe_log_metric(f"{prefix}_tuning_time", tuning_time)
            self.safe_log_metric(f"{prefix}_eval_time", eval_time)
            self.safe_log_metric(f"{prefix}_total_time", tuning_time + eval_time)
            
            # Log performance metrics
            for metric_name, metric_value in metrics.items():
                self.safe_log_metric(f"{prefix}_{metric_name}", metric_value)
            
            # Log feature importance if available
            if hasattr(tuned_model, 'feature_importances_'):
                self.safe_log_metric(f"{prefix}_feature_importance_mean", np.mean(tuned_model.feature_importances_))
                self.safe_log_metric(f"{prefix}_feature_importance_std", np.std(tuned_model.feature_importances_))
            
            # Track best overall model
            if metrics['f1_score'] > best_overall_f1:
                best_overall_f1 = metrics['f1_score']
                best_overall_model = tuned_model
                best_overall_name = model_name
            
            print(f"✅ {model_name}:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   F1-Score: {metrics['f1_score']:.4f}")
            print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
            print(f"   Tuning time: {tuning_time:.2f}s")
            
            # Store results
            self.models[model_name] = tuned_model
            self.results[model_name] = metrics
            self.best_params[model_name] = best_params
        
        # FIXED: Always save the best model regardless of MLflow Project mode
        if best_overall_model is not None:
            print(f"\n🏆 Saving best model: {best_overall_name} (F1-Score: {best_overall_f1:.4f})")
            
            # Generate signature and example
            signature = infer_signature(self.X_train, best_overall_model.predict_proba(self.X_test)[:, 1])
            input_example = self.X_train.iloc[:1]
            
            # Save best model with 'model' artifact name
            if best_overall_name == 'XGBoost':
                mlflow.xgboost.log_model(
                    best_overall_model,
                    "model",  # Standard artifact name
                    signature=signature,
                    input_example=input_example
                )
            else:
                mlflow.sklearn.log_model(
                    best_overall_model,
                    "model",  # Standard artifact name
                    signature=signature,
                    input_example=input_example
                )
            
            # Log best model info
            self.safe_log_param("best_model_name", best_overall_name)
            self.safe_log_metric("best_model_f1_score", best_overall_f1)
            
            print("✅ Best model saved successfully!")
        
        self.print_summary()
    
    def train_model_standalone_with_individual_runs(self):
        """Train models with individual MLflow runs and tuning"""
        print("🚀 Training models in standalone mode with individual runs...")
        
        models_config = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=42, max_iter=2000),
                'params': self.get_logistic_regression_params(),
                'log_func': mlflow.sklearn.log_model
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': self.get_random_forest_params(),
                'log_func': mlflow.sklearn.log_model
            },
            {
                'name': 'XGBoost',
                'model': xgb.XGBClassifier(
                    random_state=42, 
                    eval_metric='logloss',
                    use_label_encoder=False
                ),
                'params': self.get_xgboost_params(),
                'log_func': mlflow.xgboost.log_model
            }
        ]
        
        for config in models_config:
            model_name = config['name']
            run_name = f"{model_name.replace(' ', '_')}_Tuned" if self.enable_tuning else model_name.replace(' ', '_')
            
            with mlflow.start_run(run_name=run_name):
                print(f"\n🤖 Training {model_name}...")
                
                start_time = time.time()
                
                # Hyperparameter tuning
                tuned_model, best_params = self.tune_model(
                    config['model'], 
                    config['params'], 
                    model_name
                )
                tuning_time = time.time() - start_time
                
                # Evaluate model
                eval_start = time.time()
                y_pred = tuned_model.predict(self.X_test)
                y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1]
                eval_time = time.time() - eval_start
                
                # Calculate metrics
                metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
                
                # Log parameters
                mlflow.log_param("model_type", tuned_model.__class__.__name__)
                mlflow.log_param("tuning_enabled", self.enable_tuning)
                mlflow.log_param("tuning_method", "RandomizedSearchCV" if len(config['params']) > 50 else "GridSearchCV")
                
                # Log best parameters
                for param_name, param_value in best_params.items():
                    mlflow.log_param(f"best_{param_name}", param_value)
                
                # Log timing metrics
                mlflow.log_metric("tuning_time_seconds", tuning_time)
                mlflow.log_metric("evaluation_time_seconds", eval_time)
                mlflow.log_metric("total_training_time_seconds", tuning_time + eval_time)
                
                # Log performance metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log feature importance if available
                if hasattr(tuned_model, 'feature_importances_'):
                    mlflow.log_metric("feature_importance_mean", np.mean(tuned_model.feature_importances_))
                    mlflow.log_metric("feature_importance_std", np.std(tuned_model.feature_importances_))
                
                # Log model with signature and example
                signature = infer_signature(self.X_train, y_pred_proba)
                input_example = self.X_train.iloc[:1]
                
                config['log_func'](
                    tuned_model,
                    name=f"{model_name.lower().replace(' ', '_')}_tuned_model",
                    signature=signature,
                    input_example=input_example
                )
                
                print(f"✅ {model_name}:")
                print(f"   Accuracy: {metrics['accuracy']:.4f}")
                print(f"   F1-Score: {metrics['f1_score']:.4f}")
                print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
                print(f"   Tuning time: {tuning_time:.2f}s")
                
                # Store results
                self.models[model_name] = tuned_model
                self.results[model_name] = metrics
                self.best_params[model_name] = best_params
        
        self.print_summary()
    
    def print_summary(self):
        """Print model comparison summary with tuning results"""
        print("\n" + "="*70)
        print("📊 HYPERPARAMETER TUNING & TRAINING SUMMARY")
        print("="*70)
        
        if not self.results:
            print("❌ No results to display")
            return
        
        # Create comparison table
        data = []
        for name, metrics in self.results.items():
            best_params_str = str(self.best_params.get(name, {}))[:50] + "..." if len(str(self.best_params.get(name, {}))) > 50 else str(self.best_params.get(name, {}))
            
            data.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'Best Params': best_params_str
            })
        
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        
        # Best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_f1 = self.results[best_model]['f1_score']
        best_params = self.best_params[best_model]
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")
        print(f"   Best Parameters: {best_params}")
        
        # Log summary metrics
        self.safe_log_metric("best_f1_score", best_f1)
        self.safe_log_param("best_model_name", best_model)
        self.safe_log_param("total_models_trained", len(self.results))
        self.safe_log_param("hyperparameter_tuning_enabled", self.enable_tuning)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Stroke Prediction ML Training Pipeline with Hyperparameter Tuning")
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="stroke_prediction_ci",
        help="MLflow experiment name"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default="stroke_data_preprocessing",
        help="Path to preprocessed data directory"
    )
    
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Disable hyperparameter tuning (use default parameters)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function"""
    print("🎯 Stroke Prediction Training Pipeline with Hyperparameter Tuning - FIXED VERSION")
    print("="*70)
    
    args = parse_arguments()
    
    if args.verbose:
        print(f"📋 Configuration:")
        print(f"   Experiment: {args.experiment_name}")
        print(f"   Data path: {args.data_path}")
        print(f"   Tuning enabled: {not args.no_tuning}")
        print(f"   MLflow Run ID: {os.getenv('MLFLOW_RUN_ID', 'Not set')}")
    
    try:
        trainer = StrokeModelTrainer(
            experiment_name=args.experiment_name,
            data_path=args.data_path,
            enable_tuning=not args.no_tuning
        )
        
        if not trainer.load_processed_data():
            print("❌ Failed to load data")
            sys.exit(1)
        
        # Choose training method based on context
        if trainer.is_mlflow_project:
            trainer.train_model_with_tuning()
        else:
            trainer.train_model_standalone_with_individual_runs()
        
        print("\n🎉 Training with hyperparameter tuning completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()