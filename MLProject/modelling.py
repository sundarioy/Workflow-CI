import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost as xgb
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_preprocessed_data(file_path):
    """Memuat dataset yang sudah diproses."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset preprocessed berhasil dimuat dari {file_path}")
        print(f"Shape dataset: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        return None

def train_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Melatih model Logistic Regression dengan MLflow autolog."""
    # Inisialisasi MLflow autolog
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    print("MLflow autolog diaktifkan untuk Scikit-Learn.")

    with mlflow.start_run(run_name="Logistic_Regression_Run") as run:
        print(f"Memulai MLflow Run ID: {run.info.run_id}")

        # Inisialisasi dan latih model
        model = LogisticRegression(
            C=100,
            random_state=42,
            max_iter=1000,
            solver='liblinear',
            penalty='l1'
        )
        print(f"Melatih model: {type(model).__name__}")
        model.fit(X_train_scaled, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nMetrik Model pada Data Uji ({type(model).__name__}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        print(f"MLflow Run Selesai. Cek MLflow Tracking UI di folder 'mlruns'.")
        return run.info.run_id

def train_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Melatih model Random Forest dengan MLflow autolog."""
    # Inisialisasi MLflow autolog
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)
    print("MLflow autolog diaktifkan untuk Scikit-Learn.")

    with mlflow.start_run(run_name="Random_Forest_Run") as run:
        print(f"Memulai MLflow Run ID: {run.info.run_id}")

        # Inisialisasi dan latih model
        model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            min_samples_split=5,
            min_samples_leaf=1,
            max_features='log2',
            max_depth=None,
            bootstrap=False
        )
        print(f"Melatih model: {type(model).__name__}")
        model.fit(X_train_scaled, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nMetrik Model pada Data Uji ({type(model).__name__}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        print(f"MLflow Run Selesai. Cek MLflow Tracking UI di folder 'mlruns'.")
        return run.info.run_id

def train_xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test):
    """Melatih model XGBoost dengan MLflow autolog."""
    # Inisialisasi MLflow autolog
    mlflow.xgboost.autolog(log_model_signatures=True, log_input_examples=True)
    print("MLflow autolog diaktifkan untuk XGBoost.")

    with mlflow.start_run(run_name="XGBoost_Run") as run:
        print(f"Memulai MLflow Run ID: {run.info.run_id}")

        # Inisialisasi dan latih model
        model = xgb.XGBClassifier(
            n_estimators=500,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            reg_lambda=1,
            subsample=0.9,
            reg_alpha=0.1,
            max_depth=6,
            learning_rate=0.1,
            colsample_bytree=1.0
        )
        print(f"Melatih model: {type(model).__name__}")
        model.fit(X_train_scaled, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nMetrik Model pada Data Uji ({type(model).__name__}):")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        print(f"MLflow Run Selesai. Cek MLflow Tracking UI di folder 'mlruns'.")
        return run.info.run_id

def train_models(df, target_column='stroke'):
    """
    Melatih semua model machine learning dengan MLflow autolog.
    """
    if df is None:
        return

    # 1. Pemisahan Fitur (X) dan Target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Bentuk X: {X.shape}, Bentuk y: {y.shape}")
    print(f"Distribusi target: {y.value_counts().to_dict()}")

    # 2. Pembagian Data Latih dan Uji
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dibagi menjadi data latih ({len(X_train)} baris) dan data uji ({len(X_test)} baris).")

    # 3. Scaling Fitur (untuk konsistensi dengan preprocessing)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Fitur telah di-scale menggunakan StandardScaler.")

    # 4. Melatih semua model
    print("\n🚀 Memulai pelatihan semua model...")
    
    results = {}
    
    # Train Logistic Regression
    print("\n" + "="*50)
    print("📊 TRAINING LOGISTIC REGRESSION")
    print("="*50)
    lr_run_id = train_logistic_regression_model(X_train_scaled, X_test_scaled, y_train, y_test)
    if lr_run_id:
        results['Logistic Regression'] = lr_run_id
    
    # Train Random Forest
    print("\n" + "="*50)
    print("🌳 TRAINING RANDOM FOREST")
    print("="*50)
    rf_run_id = train_random_forest_model(X_train_scaled, X_test_scaled, y_train, y_test)
    if rf_run_id:
        results['Random Forest'] = rf_run_id
    
    # Train XGBoost
    print("\n" + "="*50)
    print("🚀 TRAINING XGBOOST")
    print("="*50)
    xgb_run_id = train_xgboost_model(X_train_scaled, X_test_scaled, y_train, y_test)
    if xgb_run_id:
        results['XGBoost'] = xgb_run_id

    # Summary
    print("\n" + "="*60)
    print("✅ SUMMARY - SEMUA MODEL TELAH DILATIH")
    print("="*60)
    for model_name, run_id in results.items():
        print(f"  {model_name}: {run_id}")
    
    return results

if __name__ == "__main__":
    # Argument parsing untuk MLproject compatibility
    parser = argparse.ArgumentParser(description='Train stroke prediction models')
    parser.add_argument('--data_path', 
                       default='dataset_preprocessing/train_data_processed.csv',
                       help='Path to preprocessed data file')
    parser.add_argument('--experiment_name', 
                       default='stroke_prediction_github_actions',
                       help='MLflow experiment name')
    
    args = parser.parse_args()
    
    # Set experiment name
    mlflow.set_experiment(args.experiment_name)
    print(f"MLflow experiment: {args.experiment_name}")
    
    # Memuat data
    df_preprocessed = load_preprocessed_data(args.data_path)

    if df_preprocessed is not None:
        # Pastikan tidak ada nilai NaN sebelum melatih model
        print(f"\nJumlah nilai NaN sebelum penanganan:")
        print(df_preprocessed.isnull().sum())
        
        # Handle missing values
        initial_rows = len(df_preprocessed)
        df_preprocessed.dropna(inplace=True)
        final_rows = len(df_preprocessed)
        
        print(f"Jumlah baris setelah menghapus NaN: {final_rows} (dari {initial_rows})")

        if not df_preprocessed.empty:
            # Melatih semua model
            results = train_models(df_preprocessed.copy(), target_column='stroke')
            
            if results:
                print(f"\n🎉 Pelatihan selesai! Total {len(results)} model berhasil dilatih.")
                print("📊 Hasil tracking tersimpan di MLflow.")
                print("🔗 Untuk melihat hasil: mlflow ui")
            else:
                print("❌ Tidak ada model yang berhasil dilatih.")
        else:
            print("DataFrame kosong setelah menghapus NaN, pelatihan dibatalkan.")
    else:
        print(f"❌ Gagal memuat data dari {args.data_path}")
        print("Pastikan file preprocessing telah dijalankan terlebih dahulu.")