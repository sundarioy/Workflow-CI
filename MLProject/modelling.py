import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

def load_preprocessed_data(file_path):
    """Memuat dataset yang sudah diproses."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset preprocessed berhasil dimuat dari {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} tidak ditemukan.")
        return None

def train_model(df, target_column='stroke'):
    """
    Melatih model machine learning dengan MLflow autolog.
    """
    if df is None:
        return

    # 1. Pemisahan Fitur (X) dan Target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Bentuk X: {X.shape}, Bentuk y: {y.shape}")

    # 2. Pembagian Data Latih dan Uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Data dibagi menjadi data latih ({len(X_train)} baris) dan data uji ({len(X_test)} baris).")

    # 3. Scaling Fitur (Opsional, tapi seringkali baik untuk model linear)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Fitur telah di-scale menggunakan StandardScaler.")

    # 4. Inisialisasi MLflow autolog
    # Autolog akan secara otomatis mencatat parameter, metrik, dan artefak model
    mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True, log_post_training_metrics=True)
    print("MLflow autolog diaktifkan untuk Scikit-Learn.")

    # 5. Melatih Model
    # Kita akan melatih model di dalam 'with mlflow.start_run():'
    # Ini akan membuat sebuah "run" baru di MLflow Tracking UI lokal
    with mlflow.start_run(run_name="Stroke_Prediction_LogisticRegression") as run:
        print(f"Memulai MLflow Run ID: {run.info.run_id}")

        # Inisialisasi dan latih model dengan parameter yang dioptimasi
        model = LogisticRegression(
            C=100,                 # Regularization strength
            random_state=42,       # Reproducibility  
            max_iter=1000,         # Max iterations
            solver='liblinear',    # Solver algorithm
            penalty='l1'           # L1 regularization
        )
        print(f"Melatih model: {type(model).__name__}")
        model.fit(X_train_scaled, y_train)

        # Prediksi pada data uji
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] # Probabilitas untuk kelas positif

        # Evaluasi model (autolog akan mencatat ini, tapi kita bisa hitung juga untuk dilihat)
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

        # Autolog akan otomatis menyimpan model, parameter, dan metrik.
        # Tidak perlu mlflow.log_param, mlflow.log_metric, atau mlflow.sklearn.log_model secara manual
        # untuk kriteria Basic.

        print(f"MLflow Run Selesai. Cek MLflow Tracking UI di folder 'mlruns' atau dengan perintah 'mlflow ui'.")
        return run.info.run_id


if __name__ == "__main__":
    # Path ke dataset yang sudah diproses
    preprocessed_data_path = "dataset_preprocessing/train_data_processed.csv"

    # Memuat data
    df_preprocessed = load_preprocessed_data(preprocessed_data_path)

    if df_preprocessed is not None:
        # Pastikan tidak ada nilai NaN sebelum melatih model
        # Anda bisa memilih untuk menghapus baris dengan NaN atau melakukan imputasi
        print(f"\nJumlah nilai NaN sebelum penanganan di modelling.py:")
        print(df_preprocessed.isnull().sum())
        df_preprocessed.dropna(inplace=True) # Contoh: menghapus baris dengan NaN
        print(f"Jumlah baris setelah menghapus NaN: {len(df_preprocessed)}")

        if not df_preprocessed.empty:
            # Melatih model
            run_id = train_model(df_preprocessed.copy(), target_column='stroke')
            if run_id:
                print(f"\nModel telah dilatih. Run ID di MLflow: {run_id}")
                print("Untuk melihat hasil tracking, jalankan 'mlflow ui' di terminal pada direktori yang sama dengan 'mlruns'.")
        else:
            print("DataFrame kosong setelah menghapus NaN, pelatihan model dibatalkan.")