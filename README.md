# MLOps Workflow CI/CD - Stroke Prediction Model

## 🎯 Overview

This repository demonstrates a complete **MLOps CI/CD pipeline** for stroke prediction using **MLflow Projects** and **GitHub Actions**. The project implements automated model training, hyperparameter tuning, and artifact management with comprehensive MLflow tracking.

### ✨ Key Features

- 🤖 **Automated ML Pipeline** with GitHub Actions CI/CD
- 🔧 **Hyperparameter Tuning** with GridSearchCV and RandomizedSearchCV
- 📊 **MLflow Tracking** for experiment management
- 🎯 **Multiple Models** (Logistic Regression, Random Forest, XGBoost)
- 🚀 **Smart Training Strategy** (Fast mode for PRs, Full mode for production)
- 📁 **Artifact Management** with automatic upload
- 🔄 **Flexible Triggers** (Push, PR, Manual)

## 🏗️ Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD pipeline
├── MLProject/                     # MLflow Project directory
│   ├── MLProject                  # MLflow project configuration
│   ├── python_env.yaml           # Python environment specification
│   ├── requirements.txt           # Python dependencies
│   ├── modelling.py              # Main training script with tuning
│   └── stroke_data_preprocessing/ # Preprocessed dataset
│       ├── train_data_processed.csv
│       ├── test_data_processed.csv
│       └── feature_info.json
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- MLflow 2.0+
- Git
- GitHub account

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/sundarioy/Workflow-CI.git
   cd Workflow-CI
   ```

2. **Install dependencies**
   ```bash
   cd MLProject
   pip install -r requirements.txt
   ```

3. **Run training locally**
   ```bash
   # Quick training (no hyperparameter tuning)
   python modelling.py --experiment-name "local_test" --no-tuning

   # Full training with hyperparameter tuning
   python modelling.py --experiment-name "local_tuned" --verbose
   ```

4. **Run as MLflow Project**
   ```bash
   # Fast mode
   mlflow run . --env-manager local --entry-point "fast"
   
   # Full mode with tuning
   mlflow run . --env-manager local --entry-point "main"
   ```

## 🔧 CI/CD Pipeline

### Workflow Triggers

| Trigger | Training Mode | Hyperparameter Tuning | Duration |
|---------|---------------|----------------------|----------|
| **Pull Request** | Fast | ❌ Disabled | ~3-5 minutes |
| **Push to Main** | Full | ✅ Enabled | ~15-30 minutes |
| **Manual Dispatch** | Configurable | ✅ Optional | Variable |

### Pipeline Jobs

1. **🔍 Validate** - Checks MLProject structure and dependencies
2. **🤖 Train** - Executes model training with MLflow tracking
3. **📊 Summary** - Generates training reports and uploads artifacts

### Artifact Storage

After successful training, artifacts are automatically uploaded to GitHub Actions:

- **MLflow Runs** (`mlruns/`) - Complete experiment tracking data
- **Model Files** - Trained models with signatures and examples
- **Training Summary** - Detailed execution report
- **Performance Metrics** - Model comparison and tuning results

## 🤖 Model Training

### Supported Models

| Model | Hyperparameters Tuned | Tuning Method |
|-------|----------------------|---------------|
| **Logistic Regression** | C, solver, penalty, max_iter | GridSearchCV |
| **Random Forest** | n_estimators, max_depth, min_samples_split, min_samples_leaf | RandomizedSearchCV |
| **XGBoost** | n_estimators, max_depth, learning_rate, subsample, colsample_bytree | RandomizedSearchCV |

### Training Modes

#### 🚀 Fast Mode (PR Validation)
- **Purpose**: Quick validation for pull requests
- **Duration**: 3-5 minutes
- **Hyperparameter Tuning**: Disabled
- **Models**: All models with default parameters

#### 🔧 Full Mode (Production Training)
- **Purpose**: Complete training for production deployment
- **Duration**: 15-30 minutes
- **Hyperparameter Tuning**: Enabled
- **Cross-validation**: 3-fold
- **Strategy**: Smart Grid/Randomized search

### Performance Metrics

All models are evaluated using:
- **Accuracy** - Overall prediction accuracy
- **F1-Score** - Balanced precision and recall
- **ROC-AUC** - Area under ROC curve
- **Precision** - Positive prediction accuracy
- **Recall** - True positive rate
- **Specificity** - True negative rate

## 📊 MLflow Integration

### Experiment Tracking

- **Automatic Logging** - Parameters, metrics, and artifacts
- **Model Signatures** - Input/output schema inference
- **Model Registry** - Versioned model storage
- **Artifact Storage** - Complete model and data lineage

### Accessing Results

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("file:./mlruns")

# List experiments
experiments = mlflow.search_experiments()

# Get best model
runs = mlflow.search_runs(experiment_ids=['0'])
best_run = runs.loc[runs['metrics.f1_score'].idxmax()]

# Load model
model_uri = f"runs:/{best_run.run_id}/model"
model = mlflow.sklearn.load_model(model_uri)
```

## 🛠️ Configuration

### Environment Variables

```yaml
# GitHub Actions Environment
PYTHON_VERSION: '3.11'
MLFLOW_TRACKING_URI: 'file:./mlruns'
```

### MLflow Project Configuration

```yaml
# MLProject file
name: stroke_prediction_mlflow_project
python_env: python_env.yaml

entry_points:
  main:
    parameters:
      experiment_name: {type: string, default: "stroke_prediction_ci"}
      data_path: {type: string, default: "stroke_data_preprocessing"}
    command: "python modelling.py --experiment-name {experiment_name} --data-path {data_path}"
```

## 📊 Usage Examples

### Manual Workflow Dispatch

1. Go to **Actions** tab in GitHub
2. Select **MLflow Project CI** workflow
3. Click **Run workflow**
4. Configure options:
   - **Experiment Name**: Custom experiment identifier
   - **Verbose Logging**: Enable detailed logging

### Custom Training Scripts

```python
from modelling import StrokeModelTrainer

# Initialize trainer
trainer = StrokeModelTrainer(
    experiment_name="custom_experiment",
    data_path="stroke_data_preprocessing",
    enable_tuning=True
)

# Load data and train
trainer.load_processed_data()
trainer.train_model_with_tuning()
```

## 📈 Model Performance

### Latest Results

| Model | Accuracy | F1-Score | ROC-AUC | Training Time |
|-------|----------|----------|---------|---------------|
| **Logistic Regression** | 0.8973 | 0.3312 | 0.8406 | ~2 minutes |
| **Random Forest** | 0.9247 | 0.1538 | 0.7575 | ~5 minutes |
| **XGBoost** | 0.9295 | 0.1429 | 0.7898 | ~8 minutes |

*Results from latest automated training run*

### Performance Improvements

- **Hyperparameter Tuning**: +5-15% improvement in F1-score
- **Feature Engineering**: Comprehensive preprocessing pipeline
- **Cross-validation**: 3-fold validation for robust performance
- **Automated Selection**: Best model selection based on F1-score

---

## 📄 Assignment Information

This project is developed as part of **Dicoding MLOps Learning Path** - demonstrating the implementation of automated ML pipeline using MLflow Projects and GitHub Actions CI/CD.

**Key Learning Objectives Achieved:**
- ✅ MLflow Project structure and configuration
- ✅ Automated model training with CI/CD
- ✅ Hyperparameter tuning and model comparison  
- ✅ Artifact management and MLflow tracking
- ✅ GitHub Actions workflow implementation