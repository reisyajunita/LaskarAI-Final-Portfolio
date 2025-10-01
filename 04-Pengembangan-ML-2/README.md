# Proyek Analisis Sentimen: Ulasan Game Play Together

Proyek ini dibuat sebagai submission untuk kelas **"Belajar Pengembangan Machine Learning"** dari [Dicoding Indonesia](https://www.dicoding.com/).

Tujuan utama dari proyek ini adalah untuk membangun model *deep learning* yang mampu melakukan analisis sentimen terhadap ulasan dari para pemain game **Play Together**. Model ini dilatih untuk mengklasifikasikan sentimen ulasan ke dalam **tiga kategori**: **positif, negatif, dan netral**.

## 📊 Dataset

Dataset yang digunakan adalah kumpulan ulasan pemain game Play Together yang diambil dari platform distribusi aplikasi. Dataset ini (`data_sentiment.csv`) berisi dua kolom utama:
1.  **text**: Teks ulasan yang diberikan oleh pemain.
2.  **sentiment**: Label sentimen (`positive`, `negative`, atau `neutral`) dari ulasan tersebut.

Proses pengumpulan data dilakukan melalui *scraping*, dan detail teknisnya dapat dilihat pada notebook `scraping_data.ipynb`.

## ⚙️ Alur Kerja Proyek

1.  **Pengumpulan Data:** Data ulasan di-*scrape* dari halaman aplikasi game Play Together.
2.  **Pra-pemrosesan Teks:** Teks ulasan dibersihkan dari karakter yang tidak perlu, diubah menjadi huruf kecil, dan melalui proses tokenisasi. Sebuah kamus kata (*Kamus Besar Bahasa Alay*) juga digunakan (`kbba.py`) untuk menormalisasi kata-kata gaul ke dalam bentuk baku agar lebih mudah dipahami oleh model.
3.  **Pembuatan Sekuens dan Padding:** Teks yang sudah bersih diubah menjadi sekuens numerik dan dilakukan *padding* untuk menyamakan panjang setiap ulasan.
4.  **Pengembangan Model:** Model *deep learning* dibangun menggunakan arsitektur **LSTM (Long Short-Term Memory)** dengan lapisan Embedding, yang sangat efektif untuk memahami konteks dari data sekuensial seperti teks.
5.  **Pelatihan Model:** Model dilatih pada data training dengan menggunakan *callback* untuk mengoptimalkan proses, seperti `EarlyStopping` (menghentikan pelatihan jika tidak ada peningkatan) dan `ReduceLROnPlateau` (mengurangi *learning rate* jika performa stagnan).
6.  **Evaluasi:** Performa model dievaluasi berdasarkan metrik akurasi pada data validasi untuk memastikan kemampuannya dalam menggeneralisasi pada data baru.

## 🚀 Instalasi & Penggunaan

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repository ini:**
    ```bash
    git clone [https://github.com/reisyajunita/dicoding_analisis_sentimen_pengembangan_ml.git](https://github.com/reisyajunita/dicoding_analisis_sentimen_pengembangan_ml.git)
    cd dicoding_analisis_sentimen_pengembangan_ml
    ```

2.  **Install library yang dibutuhkan:**
    Sangat disarankan untuk membuat *virtual environment* terlebih dahulu.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Jalankan Notebook Pelatihan:**
    Buka dan jalankan file `notebook_training.ipynb` untuk melihat seluruh proses, mulai dari memuat data hingga melatih dan mengevaluasi model.
    ```bash
    jupyter notebook notebook_training.ipynb
    ```

## 📈 Hasil

Model yang dikembangkan berhasil mencapai akurasi yang baik pada set data validasi, menunjukkan kemampuannya dalam memahami dan mengklasifikasikan sentimen dari teks ulasan pemain game Play Together secara efektif. Visualisasi dari histori pelatihan (akurasi dan loss) juga disajikan di dalam notebook.

## 🧑‍💻 Author

* **Nama:** Reisya Junita
* **GitHub:** [@reisyajunita](https://github.com/reisyajunita)
