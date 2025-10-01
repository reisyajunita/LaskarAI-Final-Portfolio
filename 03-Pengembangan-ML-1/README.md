# Proyek Klasifikasi Gambar Hewan dengan Transfer Learning (MobileNet)

Proyek ini dibuat sebagai submission untuk kelas **"Belajar Pengembangan Machine Learning"**. Tujuannya adalah membangun sistem klasifikasi gambar yang mampu mengidentifikasi 3 jenis hewan: **kucing, anjing, dan ular**.

Untuk mencapai akurasi tinggi dan efisiensi, proyek ini mengimplementasikan teknik **Transfer Learning** dengan memanfaatkan arsitektur **MobileNetV2** yang sudah terlatih pada dataset ImageNet. Model yang telah dilatih kemudian dikonversi ke format **TensorFlow Lite (TFLite)** dan **TensorFlow.js (TFJS)** untuk potensi deployment di berbagai platform seperti perangkat mobile atau web.

## 📊 Dataset

Dataset yang digunakan adalah **Animal Image Classification Dataset** yang bersumber dari [Hugging Face](https://huggingface.co/datasets/AlvaroVasquezAI/Animal_Image_Classification_Dataset).

### Detail Dataset
* **Total Gambar:** 3.000 gambar.
* **Format & Resolusi:** Semua gambar dalam format JPG dengan resolusi seragam 256x256 piksel (RGB).
* **Jumlah Kelas:** 3, dengan distribusi sebagai berikut:
    * **cats/**: 1.000 gambar kucing dari berbagai ras dan lingkungan.
    * **dogs/**: 1.000 gambar anjing dari berbagai ras dan aktivitas.
    * **snakes/**: 1.000 gambar ular dari berbagai spesies di habitat alaminya.

Dataset ini kemudian diatur ke dalam direktori `train/`, `val/`, dan `test/` untuk proses pelatihan dan evaluasi model.

## ⚙️ Alur Kerja Proyek

1.  **Persiapan Data:** Memuat dataset gambar dari direktori dan membaginya menjadi set pelatihan, validasi, dan pengujian.
2.  **Augmentasi Data:** `ImageDataGenerator` digunakan pada data pelatihan untuk menerapkan teknik augmentasi seperti rotasi, pergeseran, dan zoom, guna meningkatkan ketahanan model terhadap variasi gambar.
3.  **Membangun Model (Transfer Learning):**
    * Menggunakan **MobileNetV2** sebagai *base model* dengan *weights* dari ImageNet. Lapisan teratas (klasifikasi) dari MobileNetV2 dibekukan (*frozen*) untuk menjaga pengetahuan yang sudah ada.
    * Menambahkan lapisan kustom di atas *base model* yang terdiri dari `GlobalAveragePooling2D` dan `Dense` dengan aktivasi **Softmax** untuk 3 kelas.
4.  **Pelatihan Model (Fine-Tuning):**
    * Model dilatih dengan *optimizer* **Adam**.
    * Menggunakan *loss function* **`CategoricalCrossentropy`**.
    * Callback seperti `EarlyStopping` dan `ReduceLROnPlateau` diimplementasikan untuk mendapatkan hasil optimal dan mencegah *overfitting*.
5.  **Evaluasi:** Performa model dievaluasi pada set data pengujian untuk mengukur akurasi dan loss pada data yang belum pernah dilihat sebelumnya.
6.  **Konversi Model:** Model akhir dikonversi ke format `.tflite` dan `tfjs` agar siap untuk diimplementasikan di berbagai platform.

## 🚀 Instalasi & Penggunaan

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/reisyajunita/Dicoding_Klasifikasi-Gambar-Pengembangan-ML.git](https://github.com/reisyajunita/Dicoding_Klasifikasi-Gambar-Pengembangan-ML.git)
    ```
2.  **Instal Dependensi:**
    Pastikan Anda telah menginstal semua dependensi yang tercantum dalam file `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
3.  **Jalankan Notebook:**
    Buka dan jalankan `images_classification.ipynb` untuk melihat seluruh alur kerja, mulai dari pra-pemrosesan data hingga konversi model.
    ```bash
    jupyter notebook images_classification.ipynb
    ```

## 🧑‍💻 Author

* **Nama:** Reisya Junita
* **GitHub:** [@reisyajunita](https://github.com/reisyajunita)
