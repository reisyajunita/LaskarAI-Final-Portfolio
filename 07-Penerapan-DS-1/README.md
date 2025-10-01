# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

Proyek ini dibuat sebagai submission untuk kelas **"Penerapa Data Science"** dari [Dicoding Indonesia](https://www.dicoding.com/).

Tujuan dari proyek ini adalah untuk mengidentifikasi faktor-faktor utama yang memengaruhi tingginya tingkat atrisi (*attrition rate*) di perusahaan fiktif "Jaya Jaya Maju" dan memberikan rekomendasi berbasis data untuk mengatasi masalah tersebut melalui analisis data eksplorasi dan dashboard interaktif.

## Business Understanding

Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. 

Walaupun telah menjadi menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%.

Untuk mencegah hal ini semakin parah, manajer departemen HR ingin meminta bantuan Anda mengidentifikasi berbagai faktor yang mempengaruhi tingginya attrition rate tersebut. Selain itu, ia juga meminta Anda untuk membuat business dashboard untuk membantunya memonitori berbagai faktor tersebut.

**Tujuan**:  Mengidentifikasi faktor-faktor utama yang mempengaruhi tingginya attrition rate di Jaya Jaya Maju dan memberikan rekomendasi berbasis data.

### Permasalahan Bisnis

Perusahaan Jaya Jaya Maju mengalami tantangan serius terkait manajemen sumber daya manusia, terutama tingginya tingkat attrition (keluar atau berhentinya karyawan dari perusahaan). Meskipun perusahaan telah berdiri lama dan memiliki jumlah karyawan yang besar, attrition rate yang melebihi 10% menandakan adanya potensi masalah struktural atau kepuasan kerja di dalam organisasi.

Tingginya angka attrition ini berisiko menurunkan produktivitas, meningkatkan biaya rekrutmen dan pelatihan, serta mengganggu kontinuitas proyek dan budaya kerja. Manajer HR ingin mengetahui faktor-faktor apa saja yang paling berpengaruh terhadap keputusan karyawan untuk keluar, sehingga perusahaan dapat mengambil tindakan preventif yang tepat sasaran.

### Cakupan Proyek

Untuk menjawab permasalahan di atas, proyek ini memiliki cakupan sebagai berikut:

* Eksplorasi dan Pembersihan Data

    Melakukan pemahaman terhadap dataset karyawan yang disediakan, serta membersihkan data dari nilai-nilai yang hilang atau tidak valid.

* Exploratory Data Analysis (EDA)

    Menggali pola-pola dan hubungan antara fitur (seperti usia, departemen, gaji, lama bekerja, dll.) dengan status attrition.

* Feature Engineering dan Pra-pemrosesan

    Melakukan encoding terhadap variabel kategorik, normalisasi fitur numerik, dan penanganan class imbalance menggunakan metode seperti SMOTE.

* Pembangunan Model Prediktif

    Menggunakan algoritma klasifikasi (Random Forest) untuk memprediksi kemungkinan seorang karyawan akan keluar.

* Evaluasi Model

    Mengukur performa model menggunakan metrik seperti akurasi, precision, recall, F1-score, dan ROC-AUC.

* Pembuatan Dashboard Bisnis

    Mendesain dashboard interaktif yang membantu tim HR memantau faktor-faktor yang mempengaruhi attrition secara real-time.

* Kesimpulan & Rekomendasi Bisnis

    Menyusun saran berbasis data untuk membantu perusahaan menekan angka attrition di masa depan.

### Persiapan

Sumber data: [employee_dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

Setup environment:

1. **Akses ke Dashboard Tableau Public**
- Dashboard Tableau Public yang telah dibuat dapat diakses langsung melalui link berikut: [Dashboard](https://public.tableau.com/views/HRDashboard_17469559098280/HRAnalyticsDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link).
- Pastikan perangkat Anda terhubung ke internet untuk dapat mengakses link tersebut melalui browser.
- Tidak ada instalasi perangkat lunak tambahan yang diperlukan untuk melihat dashboard ini.

2. **Instalasi Python dan Library Pendukung**
- Pastikan Python 3.8 atau lebih baru telah terinstal di komputer Anda.

    Jika belum, unduh dari [https://www.python.org/downloads/](https://www.python.org/downloads/).

- Instal library Python yang dibutuhkan.

    Disarankan untuk menggunakan file requirements.txt yang telah disediakan:
    `pip install -r requirements.txt`

## Business Dashboard

Dashboard ini menyajikan data utama terkait karyawan perusahaan Jaya Jaya Maju:

<p align="center">
<img src="Reisya Junita_Dashboard.png" alt="alt text" />
</p>
<p align="center">Human Resources Dashboard</p>

**Gambaran Umum (KPI)**

- Total Karyawan: 1.058
- Total Attrisi: 179
- Tingkat Attrisi: 16,92%
- Karyawan Aktif: 879
- Rata-rata Usia: 37

**Insight**

Berikut rincian insight yang lebih detail, dikategorikan untuk kemudahan pemahaman:
1. Analisis Attrisi

- Tingkat Attrisi Tinggi: Tingkat atrisi sebesar 16,92% tergolong tinggi dan perlu diselidiki lebih lanjut. Ini menunjukkan bahwa sebagian besar karyawan meninggalkan perusahaan.

- Perbedaan Antar Departemen:

    - Research & Development memiliki atrisi tertinggi dengan selisih yang besar (36,87%). Ini adalah area yang sangat mengkhawatirkan.

    - Sales juga memiliki tingkat atrisi yang signifikan (59,78%).

    - Human Resources memiliki tingkat atrisi yang sangat rendah (3,35%).

- Dampak Education Field:

    Life Sciences dan Medical menunjukkan atrisi tertinggi.

- Marital Status dan Travel:

    - Karyawan Lajang (single) yang Sering Melakukan Perjalanan Dinas (Travel Frequently) memiliki tingkat **atrisi tertinggi (31,84%)**. Ini adalah temuan yang sangat spesifik dan penting.

    - Karyawan Cerai (Divorced) memiliki tingkat atrisi terendah, terlepas dari frekuensi perjalanan dinas.

    - Tidak Melakukan Perjalanan Dinas (Non-Travel) menunjukkan atrisi terendah di semua status pernikahan.

2. **Demografi Karyawan (Employee)**

- Gender : Lebih banyak karyawan pria (108) yang keluar daripada karyawan wanita (71).
- Distribusi Usia (Age):

    - Jumlah karyawan terbanyak berada dalam kelompok usia 30-33 tahun.

    - Atrisi tampaknya terkonsentrasi pada kelompok usia yang lebih muda (21-39 tahun).

- Job Satisfaction:
    - Sales Executives menunjukkan jumlah yang tinggi di semua peringkat kepuasan kerja, tetapi juga memiliki tingkat atrisi yang tinggi, mengindikasikan bahwa kepuasan mungkin bukan satu-satunya pendorong karyawan untuk keluar.

    - Laboratory Technicians juga memiliki jumlah karyawan yang signifikan di semua peringkat kepuasan.

    - Human Resources memiliki jumlah keseluruhan yang rendah, yang sejalan dengan atrisi mereka yang rendah.


3. **Area Masalah Potensial**

- Atrisi R&D dan Sales: Atrisi yang tinggi di departemen ini memerlukan perhatian segera. Wawancara keluar, survei, dan tinjauan manajemen harus dilakukan untuk memahami akar penyebabnya.

- Karyawan Muda (young), Lajang (single), dan Sering Melakukan Perjalanan Dinas: Demografi ini berisiko tinggi mengalami atrisi. Keseimbangan kerja-hidup, kebijakan perjalanan, dan sistem pendukung untuk kelompok ini harus dievaluasi.

- Strategi Retensi: Tingkat atrisi keseluruhan yang tinggi menunjukkan perlunya peningkatan strategi retensi di seluruh organisasi.


## Conclusion

Proyek ini bertujuan untuk menganalisis data karyawan untuk mengidentifikasi faktor-faktor yang berkontribusi terhadap atrisi dan menyajikan temuan dalam dashboard interaktif. Analisis ini dilakukan menggunakan Python, dengan library seperti Pandas, NumPy, Scikit-learn, dan Seaborn, yang diinstal sesuai dengan petunjuk pada bagian "Instalasi Python dan Library Pendukung". Pemodelan data memungkinkan pemahaman yang lebih mendalam tentang pola atrisi.

Dashboard yang dikembangkan (menggunakan Tableau Public) menyajikan visualisasi kunci yang merangkum tren atrisi, demografi karyawan, dan faktor-faktor terkait kepuasan kerja. Dashboard ini memungkinkan pemangku kepentingan untuk dengan cepat memahami area masalah utama, seperti tingginya tingkat atrisi di departemen Research & Development dan di antara karyawan lajang yang sering bepergian.

Kesimpulan dari analisis menunjukkan bahwa tingkat atrisi keseluruhan perusahaan adalah 16,92%, dengan variasi yang signifikan antar departemen dan kelompok karyawan. Temuan ini menggarisbawahi pentingnya menerapkan strategi retensi yang ditargetkan untuk mengatasi masalah atrisi yang spesifik, seperti meningkatkan keseimbangan kehidupan kerja bagi karyawan yang sering bepergian dan menyelidiki penyebab tingginya atrisi di Research & Development.

Dengan menggabungkan analisis data berbasis Python dan visualisasi dashboard yang efektif, proyek ini memberikan wawasan yang dapat ditindaklanjuti untuk membantu perusahaan mengurangi atrisi dan meningkatkan retensi karyawan.

### Rekomendasi Action Items (Optional)

Berdasarkan insight ini, tindakan berikut dapat dipertimbangkan:

- Selidiki R&D dan Sales: Lakukan analisis mendalam untuk memahami mengapa karyawan meninggalkan departemen ini.

- Tinjau Kebijakan Perjalanan Dinas: Nilai dampak perjalanan dinas yang sering terhadap kepuasan karyawan dan pertimbangkan pengaturan alternatif.

- Tingkatkan Dukungan untuk Karyawan Muda (Young Employees): Berikan pendampingan, peluang pengembangan karir, dan inisiatif keseimbangan kerja-hidup untuk karyawan yang lebih muda.

- Lakukan Wawancara Keluar: Kumpulkan umpan balik dari karyawan yang keluar untuk mengidentifikasi area yang perlu ditingkatkan.


## 🧑‍💻 Author

* **Nama:** Reisya Junita
* **GitHub:** [@reisyajunita](https://github.com/reisyajunita)
