# Eksperimen dan Otomatisasi Preprocessing Data Churn Pelanggan Telco

Repository ini berisi pekerjaan untuk Kriteria 1 Proyek Sistem Machine Learning, yang berfokus pada eksperimen data, analisis data eksploratif (EDA), dan otomatisasi pipeline preprocessing untuk dataset Churn Pelanggan Telco.

## Deskripsi Dataset

Dataset yang digunakan adalah **Telco Customer Churn**, yang umumnya berisi informasi mengenai pelanggan layanan telekomunikasi, atribut layanan mereka, dan apakah mereka melakukan churn (berhenti berlangganan) atau tidak.

* **Sumber Data Mentah**: Dataset mentah (`telco_dataset.csv`) disimpan dalam folder `telco-dataset_raw/`.
* **Target Variabel**: `Churn` (Yes/No), yang telah di-encode menjadi 1 (Yes) dan 0 (No) pada data yang diproses.

## Tujuan Repository

Tujuan utama dari repository ini adalah untuk mendokumentasikan proses:
1.  Melakukan analisis data eksploratif (EDA) untuk memahami karakteristik dataset.
2.  Mengembangkan langkah-langkah preprocessing data untuk membersihkan dan menyiapkan data untuk pemodelan machine learning.
3.  Mengotomatisasi pipeline preprocessing menggunakan script Python.
4.  Mengimplementasikan workflow GitHub Actions untuk menjalankan pipeline preprocessing secara otomatis ketika ada perubahan pada data mentah atau script preprocessing.

## Struktur Repository

* `.github/workflows/main.yml`: Berisi konfigurasi GitHub Actions untuk otomatisasi preprocessing.
* `telco-dataset_raw/`:
    * `telco_dataset.csv`: File dataset mentah yang digunakan.
* `preprocessing/`:
    * `Eksperimen_Reisya-Junita.ipynb`: Jupyter Notebook yang berisi analisis data eksploratif (EDA) dan langkah-langkah eksperimen preprocessing secara manual.
    * `automate_Reisya-Junita.py`: Script Python untuk menjalankan seluruh pipeline preprocessing data secara otomatis.
    * `requirements.txt`: Daftar library Python yang dibutuhkan untuk menjalankan notebook dan script preprocessing.
    * `telco-dataset_preprocessing/`:
        * `dataset_processed.csv`: File output dari script `automate_Reisya-Junita.py`, berisi data yang sudah bersih dan siap untuk tahap pemodelan.

## Cara Menjalankan Otomatisasi Preprocessing

### 1. Secara Lokal

Pastikan Python dan pip sudah terinstal.

a.  **Clone Repository (jika belum)**:
    ```bash
    git clone [URL_REPOSITORY_ANDA]
    cd Eksperimen_SML_Reisya-Junita 
    ```

b.  **Install Dependencies**:
    Dianjurkan untuk membuat virtual environment terlebih dahulu.
    ```bash
    # Dari root repository
    python -m venv env
    source env/bin/activate  # Untuk Linux/macOS
    # env\Scripts\activate    # Untuk Windows

    pip install -r preprocessing/requirements.txt
    ```

c.  **Jalankan Script Otomasi**:
    Script dijalankan dari dalam folder `preprocessing`.
    ```bash
    cd preprocessing
    python automate_Reisya-Junita.py
    ```
    Output berupa file `dataset_processed.csv` akan tersimpan di `preprocessing/telco-dataset_preprocessing/`.

### 2. Melalui GitHub Actions

Workflow GitHub Actions (`.github/workflows/main.yml`) dikonfigurasi untuk berjalan secara otomatis setiap kali ada `push` ke branch `main` yang mengubah file di dalam:
* `telco-dataset_raw/telco_dataset.csv`
* `preprocessing/automate_Reisya-Junita.py`

Workflow ini akan menjalankan script `automate_Reisya-Junita.py` dan secara otomatis melakukan commit serta push file `dataset_processed.csv` yang terupdate kembali ke repository. Status eksekusi dapat dilihat pada tab "Actions" di halaman repository GitHub.

## Output

Hasil utama dari proses ini adalah file `dataset_processed.csv` yang terletak di `preprocessing/telco-dataset_preprocessing/`. Dataset ini berisi fitur-fitur yang telah dibersihkan, di-transformasi (scaling untuk numerik, one-hot encoding untuk kategorikal), dan siap digunakan untuk melatih model machine learning pada Kriteria 2.

## Teknologi yang Digunakan

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib & Seaborn (untuk EDA di notebook)
* GitHub Actions

---
