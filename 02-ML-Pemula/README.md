# Proyek Machine Learning: Clustering dan Klasifikasi Lagu Billboard

Proyek ini diajukan untuk memenuhi kriteria kelulusan kelas **"Belajar Machine Learning untuk Pemula"** dari [Dicoding Indonesia](https://www.dicoding.com/).

Proyek ini mendemonstrasikan alur kerja machine learning secara menyeluruh, dimulai dari **analisis *unsupervised learning* (Clustering)** untuk menemukan grup atau pola tersembunyi dalam data, yang kemudian dilanjutkan dengan **analisis *supervised learning* (Klasifikasi)** untuk memprediksi grup tersebut berdasarkan fitur-fitur lagu.

## 📝 Alur Proyek

1.  **Tahap 1: Clustering (Unsupervised Learning)**
    * Dataset lagu Billboard yang awalnya tanpa label dianalisis menggunakan algoritma K-Means.
    * Tujuannya adalah untuk mengelompokkan lagu-lagu ke dalam beberapa klaster berdasarkan kesamaan karakteristiknya (misalnya, popularitas, metrik audio, dan data tangga lagu).
    * Hasil dari tahap ini adalah **label klaster** untuk setiap lagu.

2.  **Tahap 2: Klasifikasi (Supervised Learning)**
    * Label klaster yang didapat dari Tahap 1 digunakan sebagai target (variabel y) untuk model klasifikasi.
    * Sebuah model klasifikasi dilatih untuk memprediksi klaster sebuah lagu berdasarkan fitur-fiturnya.
    * Tujuannya adalah untuk membuat model yang dapat secara otomatis mengklasifikasikan lagu baru ke dalam grup yang sudah ada.

## 📊 Dataset

Dataset yang dianalisis dalam proyek ini adalah **"Billboard Top Score"**, yang bersumber dari Kaggle. Dataset ini mencakup sekitar 5000 data lagu.

### Karakteristik Dataset
Dataset ini terdiri dari 13 kolom dengan detail sebagai berikut:

| Fitur              | Deskripsi                                           |
| ------------------ | --------------------------------------------------- |
| **Song** | Judul lagu                                          |
| **Artist** | Nama penyanyi/grup musik                            |
| **Streams** | Jumlah total streaming (sepanjang waktu)            |
| **Daily Streams** | Rata-rata streaming per hari                        |
| **Genre** | Genre musik (Pop, Hip-Hop, Rock, dll.)              |
| **Release Year** | Tahun lagu dirilis                                  |
| **Peak Position** | Peringkat tertinggi di tangga lagu                  |
| **Weeks on Chart** | Total minggu berada di tangga lagu                  |
| **Lyrics Sentiment**| Analisis sentimen lirik (-1 hingga +1)              |
| **TikTok Virality**| Skor popularitas berdasarkan tren TikTok (0-100)      |
| **Danceability** | Tingkat kemudahan lagu untuk dibuat menari (0-1)    |
| **Acousticness** | Tingkat elemen akustik dalam lagu (0-1)             |
| **Energy** | Tingkat energi keseluruhan lagu (0-1)               |

## 🛠️ Instalasi & Penggunaan

Untuk menjalankan proyek ini, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/reisyajunita/Dicoding_Machine-Learning-Pemula.git](https://github.com/reisyajunita/Dicoding_Machine-Learning-Pemula.git)
    cd Dicoding_Machine-Learning-Pemula
    ```

2.  **Install library yang dibutuhkan:**
    Sangat disarankan untuk membuat *virtual environment* terlebih dahulu.
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn jupyter
    ```

3.  **Jalankan Notebook:**
    Proses analisis dibagi menjadi dua notebook. Jalankan secara berurutan:
    
    a. **Untuk Clustering:**
    ```bash
    jupyter notebook "[Clustering]_Submission_Akhir_BMLP_Reisya_Junita.ipynb"
    ```

    b. **Untuk Klasifikasi:**
    ```bash
    jupyter notebook "[Klasifikasi]_Submission_Akhir_BMLP_Reisya_Junita.ipynb"
    ```

## 🧑‍💻 Author

* **Nama:** Reisya Junita
* **GitHub:** [@reisyajunita](https://github.com/reisyajunita)
