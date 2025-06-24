# Laporan Proyek Analisis Data - Bike Sharing 


## 1. Domain Proyek (Bike Sharing Data Analysis)

Landasan proyek ini adalah analisis data berbagi sepeda untuk memahami pola penggunaan dan faktor-faktor yang mempengaruhinya. Dalam era digital ini, ketersediaan data historis tentang penggunaan sepeda telah menjadi aset berharga. Proyek ini bertujuan untuk menggali wawasan dari dataset tersebut, mengidentifikasi tren, dan memprediksi pola penggunaan di masa depan.

Masalah analisis data berbagi sepeda menjadi krusial karena dampaknya langsung terhadap efisiensi operasional, perencanaan armada, dan kepuasan pengguna. Tingkat pemanfaatan yang rendah pada jam-jam tertentu atau di lokasi tertentu dapat menyebabkan pemborosan sumber daya, sementara lonjakan permintaan yang tidak terantisipasi dapat menyebabkan kekurangan sepeda dan hilangnya potensi pendapatan. Dengan persaingan yang semakin ketat di sektor mobilitas perkotaan, kemampuan untuk memahami dan mengoptimalkan penggunaan sepeda menjadi keunggulan kompetitif yang signifikan.

Untuk mengatasi tantangan ini, proyek ini akan fokus pada:
- **Identifikasi Faktor Kunci**: Menentukan variabel cuaca, waktu, atau musiman yang paling berpengaruh terhadap jumlah sewa sepeda.
- **Visualisasi Data Interaktif**: Menyajikan wawasan melalui dashboard interaktif yang mudah dipahami oleh berbagai pemangku kepentingan, dari operator hingga perencana kota.

## 2. Struktur Proyek

Proyek ini memiliki struktur direktori sebagai berikut:

```
bike-sharing-dashboard/
├── Bike-sharing-dataset/
│   ├── day.csv                 # Dataset harian
│   └── hour.csv                # Dataset per jam
├── dashboard/
│   └── dashboard.py            # Script utama untuk dashboard
├── notebook.ipynb             # Notebook eksplorasi data (Jupyter Notebook)
├── notebook.py                # Versi Python script dari notebook
├── requirements.txt           # Daftar dependensi proyek
├── README.md                  # Dokumentasi proyek
└── url.txt                    # URL Dashboard
```

## 3. Fitur Utama

-   **Data Loading & Preprocessing**: Memuat dan membersihkan dataset `day.csv` dan `hour.csv`.
-   **Analisis Eksplorasi Data (EDA)**: Menjelajahi dataset untuk memahami distribusi, korelasi antar variabel, dan pola-pola awal.
-   **Visualisasi Interaktif**: Dashboard yang dibangun dengan Streamlit untuk visualisasi data yang dinamis.
    -   **Tren Harian/Jam**: Menampilkan bagaimana jumlah sewa sepeda berubah sepanjang hari, minggu, atau musim.
    -   **Pengaruh Cuaca**: Memvisualisasikan dampak kondisi cuaca (suhu, kelembaban, kecepatan angin) terhadap permintaan sepeda.
    -   **Distribusi Pengguna**: Analisis pola sewa oleh pengguna terdaftar dan pengguna biasa.
    -   **Filter Interaktif**: Memungkinkan pengguna untuk memfilter data berdasarkan berbagai kriteria (misalnya, musim, hari kerja, situasi cuaca).

## 4. Cara Menjalankan Proyek

Untuk menjalankan proyek ini di lingkungan lokal Anda, ikuti langkah-langkah berikut:

### Prasyarat

Pastikan Anda memiliki Python (versi 3.8 atau lebih baru disarankan) dan `pip` terinstal di sistem Anda.

### Instalasi

1.  **Clone repositori**:
    ```bash
    git clone [https://github.com/reisyajunita/bike-sharing-dashboard.git](https://github.com/reisyajunita/bike-sharing-dashboard.git)
    cd bike-sharing-dashboard
    ```

2.  **Buat Virtual Environment (disarankan)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows: `venv\Scripts\activate`
    ```

3.  **Instal Dependensi**:
    ```bash
    pip install -r requirements.txt
    ```

### Menjalankan Jupyter Notebook

Untuk melihat analisis data terperinci dan eksplorasi, buka `notebook.ipynb`:

```bash
jupyter notebook notebook.ipynb

```

### Menjalankan Dashboard
```bash
cd dashboard
streamlit run dashboard.py
```
Setelah perintah dijalankan, dashboard akan terbuka secara otomatis di browser web default Anda.

## 5. Insight Utama (atau Hasil Analisis Singkat)
Berdasarkan analisis data:

* Pola Musiman: Penyewaan sepeda umumnya lebih tinggi pada musim Fall dan musim Summer, dan menurun selama musim Winter dan Spring.
* Pengaruh Cuaca: Suhu yang moderat dan cuaca cerah berkorelasi positif dengan jumlah sewa sepeda. Hujan atau salju cenderung menurunkan jumlah sewa.
* Jam Sibuk: Permintaan sepeda memuncak pada jam-jam sibuk pagi (sekitar 07:00-09:00) dan sore (sekitar 16:00-18:00) pada hari kerja, mencerminkan penggunaan untuk komuter.
* Pengaruh Hari Kerja/Akhir Pekan: Hari kerja menunjukkan pola sewa yang berbeda dibandingkan akhir pekan, dengan puncak yang lebih jelas pada jam sibuk di hari kerja.

## 6. Kesimpulan (dan Langkah Selanjutnya)
Proyek dashboard berbagi sepeda ini berhasil menyajikan wawasan berharga dari data historis, memungkinkan pemahaman yang lebih baik tentang faktor-faktor yang memengaruhi permintaan sepeda. Visualisasi interaktif memudahkan identifikasi tren dan pola, yang dapat digunakan oleh operator atau perencana kota untuk pengambilan keputusan.

### Langkah Selanjutnya

* Integrasi data real-time untuk pemantauan dashboard yang lebih dinamis.
* Peningkatan fitur prediksi permintaan menggunakan model Machine Learning yang lebih canggih.
* Penambahan analisis geospasial untuk memahami pola penggunaan berdasarkan lokasi.

### Rekomendasi Tindak Lanjut
* Mengoptimalkan layanan pada hari kerja dengan menyediakan lebih banyak unit sepeda.
* Meningkatkan strategi promosi untuk menarik lebih banyak pelanggan di hari libur.
* Menggunakan strategi penyesuaian stok sepeda sesuai dengan pola jam sibuk penyewaan.
* Mengadakan promosi musiman pada musim gugur dan musim panas untuk meningkatkan jumlah pelanggan.
* Menyediakan solusi bagi pengguna di musim dingin, seperti rute yang lebih aman atau sepeda dengan fitur khusus untuk cuaca dingin.
* Memanfaatkan data cuaca untuk memberikan rekomendasi atau penawaran spesial kepada pelanggan, seperti diskon pada hari-hari dengan cuaca yang optimal untuk bersepeda.
  
Secara keseluruhan, wawasan ini dapat digunakan untuk mengembangkan strategi bisnis yang lebih efektif, meningkatkan kepuasan pengguna, serta mengoptimalkan operasional layanan bike-sharing agar lebih sesuai dengan kebutuhan pelanggan.

