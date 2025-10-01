import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os # Untuk membuat direktori output jika belum ada

# --- 1. Fungsi untuk Memuat Data Mentah ---
def load_raw_data(file_path):
    """Memuat data mentah dari path yang diberikan."""
    try:
        df = pd.read_csv(file_path)
        print(f"Data mentah berhasil dimuat dari: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File mentah '{file_path}' tidak ditemukan. Pastikan path file sudah benar.")
        return None

# --- 2. Fungsi untuk Pembersihan Awal ---
def initial_cleaning(df_raw):
    """Melakukan pembersihan data awal sesuai langkah di notebook."""
    if df_raw is None:
        print("Pembersihan awal dilewati karena data mentah tidak dimuat.")
        return None
    
    df = df_raw.copy()

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print("Kolom 'customerID' telah dihapus.")

    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        df.loc[(df['tenure'] == 0) & (df['TotalCharges'].isnull()), 'TotalCharges'] = 0.0
        if df['TotalCharges'].isnull().sum() > 0:
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        print("'TotalCharges' telah diproses (konversi tipe, imputasi NaN).")

    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].dtype != 'object':
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        print("Kolom 'SeniorCitizen' diubah ke 'No'/'Yes'.")
    
    return df

# --- 3. Fungsi untuk Preprocessing Fitur dan Target ---
def preprocess_data_for_output(df_cleaned, target_col='Churn'):
    """
    Melakukan encoding target dan preprocessing fitur (scaling, one-hot encoding)
    pada keseluruhan dataset yang sudah dibersihkan.
    Mengembalikan DataFrame yang sudah sepenuhnya diproses.
    """
    if df_cleaned is None:
        print("Preprocessing dilewati karena data hasil pembersihan awal tidak ada.")
        return None

    df = df_cleaned.copy()

    if target_col not in df.columns:
        print(f"Error: Kolom target '{target_col}' tidak ditemukan dalam data yang dibersihkan.")
        return None
    
    df[target_col] = df[target_col].apply(lambda x: 1 if x == 'Yes' else 0)
    print(f"Kolom target '{target_col}' telah di-encode menjadi 0/1.")

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    X_processed_array = preprocessor.fit_transform(X)
    print("Fitur telah diproses (scaling dan one-hot encoding).")

    try:
        onehot_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_feature_names_processed = numerical_features + list(onehot_feature_names)
    except AttributeError:
        print("Warning: Gagal mendapatkan nama fitur dari OneHotEncoder secara otomatis. Nama kolom mungkin generik.")
        # Hitung jumlah kolom yang dihasilkan oleh OHE secara manual jika get_feature_names_out gagal
        # Ini adalah perkiraan dan mungkin perlu disesuaikan jika ada fitur 'passthrough' yang bukan numerik
        num_processed_cols = X_processed_array.shape[1]
        num_onehot_cols = num_processed_cols - len(numerical_features)
        onehot_feature_names = [f"ohe_feature_{i}" for i in range(num_onehot_cols)]
        all_feature_names_processed = numerical_features + list(onehot_feature_names)
        
    X_processed_df = pd.DataFrame(X_processed_array, columns=all_feature_names_processed, index=X.index)
    df_final_processed = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)
    # Pastikan nama kolom target adalah 'Churn' atau sesuai target_col
    df_final_processed.rename(columns={df_final_processed.columns[-1]: target_col}, inplace=True)


    print(f"DataFrame akhir yang diproses memiliki shape: {df_final_processed.shape}")
    return df_final_processed

# --- 4. Fungsi untuk Menyimpan Data ---
def save_processed_data(df_processed, output_path):
    """Menyimpan DataFrame yang sudah diproses ke file CSV."""
    if df_processed is None:
        print("Penyimpanan data dilewati karena tidak ada data yang diproses.")
        return

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"Data yang sudah diproses berhasil disimpan di: {output_path}")
    except Exception as e:
        print(f"Error saat menyimpan data yang diproses: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Path input data mentah (sesuaikan nama file jika berbeda)
    # Relatif terhadap lokasi script automate_Nama-kamu.py di folder preprocessing/
    raw_data_input_path = "telco-dataset_raw/telco_dataset.csv"

    # Path output data yang sudah diproses
    # Akan disimpan di dalam preprocessing/namadataset_preprocessing/
    processed_data_output_path = "./preprocessing/telco-dataset_preprocessing/dataset_processed.csv"

    print("--- Memulai Proses Otomatisasi Preprocessing ---")
    
    df_raw = load_raw_data(raw_data_input_path)
    df_cleaned = initial_cleaning(df_raw)
    df_final_processed = preprocess_data_for_output(df_cleaned, target_col='Churn')
    save_processed_data(df_final_processed, processed_data_output_path)
    
    print("--- Proses Otomatisasi Preprocessing Selesai ---")