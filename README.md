# Workflow-CI: Stroke Prediction MLflow Project

🎯 **Automated ML training pipeline using MLflow Projects and GitHub Actions CI/CD**

## 📋 Project Overview

This repository implements a complete CI/CD workflow for stroke prediction machine learning models using MLflow Projects. It automatically trains and evaluates multiple ML models when triggered by code changes or manual dispatch.

## 🏗️ Repository Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI workflow
├── MLProject/                        # MLflow Project directory
│   ├── modelling.py                  # ML training script (CLI-enabled)
│   ├── conda.yaml                    # Conda environment specification
│   ├── MLProject                     # MLflow project configuration
│   ├── requirements.txt              # Python dependencies
│   └── stroke_data_preprocessing/    # Preprocessed dataset
│       ├── train_data_processed.csv
│       └── test_data_processed.csv
└── README.md                         # This file
```

## 🚀 Features

### ✅ MLflow Project Integration
- **Configurable entry points** with parameters
- **Conda environment** management
- **Reproducible ML training** pipeline

### ✅ CI/CD Automation
- **Automatic triggering** on push to main/develop
- **Manual workflow dispatch** with custom parameters
- **Validation** of MLProject structure
- **Artifact upload** and retention

### ✅ Multi-Model Training
- **Logistic Regression** baseline model
- **Random Forest** ensemble method
- **XGBoost** gradient boosting
- **Comprehensive metrics** logging

## 🎯 Usage

### Local Development

#### 1. Clone Repository
```bash
git clone <repository-url>
cd Workflow-CI
```

#### 2. Run MLflow Project Locally
```bash
# Basic run with defaults
mlflow run MLProject/

# Custom experiment name
mlflow run MLProject/ -P experiment_name="my_experiment"

# Custom data path
mlflow run MLProject/ -P data_path="custom_data_path"

# Verbose logging
mlflow run MLProject/ -P verbose=true
```

#### 3. View Results
```bash
mlflow ui --port 5000
# Open http://localhost:5000 in browser
```

### CI/CD Workflow

#### Automatic Triggers
- **Push to main/develop**: Automatically trains models
- **Pull Request**: Validates code changes

#### Manual Trigger
1. Go to **Actions** tab in GitHub
2. Select **"MLflow Project CI - Stroke Prediction"**
3. Click **"Run workflow"**
4. Configure parameters:
   - `experiment_name`: Custom experiment name
   - `verbose`: Enable detailed logging

## 📊 Models & Metrics

### Trained Models
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble method with 100