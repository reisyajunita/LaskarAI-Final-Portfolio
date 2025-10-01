# Laporan Proyek Machine Learning - Reisya Junita Putri

## Project Overview
Industri perfilman Indonesia terus berkembang, menghasilkan ribuan film setiap tahunnya. Namun, banyaknya pilihan justru membuat penonton kesulitan menemukan film yang sesuai dengan preferensi mereka. Sistem rekomendasi film menjadi solusi penting untuk membantu penonton menemukan film yang relevan, meningkatkan kepuasan pengguna, dan mendorong penemuan film-film baru maupun kurang populer.

Menurut Aggarwal (2016), sistem rekomendasi telah menjadi komponen utama pada banyak platform digital, membantu pengguna menavigasi katalog besar dengan efisien. Di era digital, sistem ini juga mendukung industri hiburan dalam meningkatkan engagement dan loyalitas pengguna[1]. Penelitian oleh Zhang, Y., & Zhang, L. (2013) juga menunjukkan bahwa sistem rekomendasi film telah terbukti efektif dalam meningkatkan pengalaman pengguna dan penemuan konten[2].

Sistem rekomendasi film berbasis data sangat penting untuk:
- Membantu pengguna menemukan film yang sesuai dengan minat.

- Meningkatkan waktu tonton dan kepuasan pengguna di platform streaming.

- Memberikan peluang bagi film-film baru atau kurang populer untuk ditemukan.

Dalam proyek ini, pendekatan utama yang akan digunakan adalah content-based filtering dengan memanfaatkan teknik TF-IDF (Term Frequency-Inverse Document Frequency) dan cosine similarity untuk menganalisis kemiripan antar film berdasarkan fitur-fitur seperti genre, aktor, dan deskripsi film. Pendekatan ini dipilih karena kemampuannya dalam memberikan rekomendasi yang relevan bahkan tanpa data interaksi pengguna yang ekstensif.

## Business Understanding
### Problem Statements
1. Bagaimana membangun sistem rekomendasi film Indonesia yang dapat memberikan saran film relevan berdasarkan preferensi pengguna?

2. Bagaimana mengukur relevansi rekomendasi film yang dihasilkan secara objektif?

### Goals
1. Mengembangkan sistem rekomendasi film Indonesia berbasis content-based filtering.

2. Menghasilkan rekomendasi film yang relevan dan terukur secara metrik evaluasi (precision@3 berbasis genre).

### Solution Approach
1. Content-Based-Filtering
- Menganalisis fitur film seperti genre, aktor, sutradara, dan deskripsi.

- Menggunakan TF-IDF dan cosine similarity untuk menemukan film yang mirip dengan film yang disukai pengguna.

## Data Understanding
Dataset yang digunakan pada proyek sistem rekomendasi ini adalah [**IMDb Indonesia Movies**](https://www.kaggle.com/datasets/dionisiusdh/imdb-indonesian-movies/data) dari kaggle.

**Informasi Dataset**
- Jumlah Data: 1272 film Indonesia dan 11 kolom.
- Rentang Tahun: Dari tahun 1926 sampai dengan 2020.
- Jenis Tipe Data: Tipe data kategori (9 kolom) dan tipe data numerik (2 kolom).

### Fitur/variabel pada dataset:
| Fitur/Variabel   | Deskripsi                                        | Tipe Data   |
|------------------|--------------------------------------------------|-------------|
| title            | Judul film                                       | object      |
| year             | Tahun rilis film                                 | int64       |
| description      | Deskripsi singkat film                           | object      |
| genre            | Genre film (16 kategori)          | object      |
| rating           | Rating usia penonton (12 kategori)               | object      |
| users_rating     | Rating pengguna IMDb (skala 1–10)                | float64     |
| votes            | Jumlah vote IMDb                                 | object      |
| languages        | Bahasa yang digunakan dalam film                 | object      |
| directors        | Nama sutradara film                              | object      |
| actors           | Daftar aktor yang membintangi film              | object      |
| runtime          | Durasi film                                      | object      |

### Kondisi Data:
- Kolom **description, genre, rating, directors, dan runtime** memiliki *missing values*.
- Genre terbanyak: **Drama** (456 film).
- Mayoritas film berbahasa Indonesia.

## Exploratory Data Analysis
Tahap *Exploratory Data Analysis* (EDA) bertujuan untuk memahami karakteristik utama dataset IMDb Indonesian Movies sebelum dilakukan pemodelan.

### Statistik Deskriptif
#### **📊Statistik Deskriptif Kolom Numerik (year dan users_rating)**

<div align="center">

| Statistik | year      | users_rating |
|-----------|-----------|--------------|
| Count     | 1272      | 1272         |
| Mean      | 2007.02   | 6.14         |
| Std       | 12.97     | 1.39         |
| Min       | 1926      | 1.2          |
| 25%       | 2006      | 5.3          |
| 50%       | 2011      | 6.4          |
| 75%       | 2016      | 7.1          |
| Max       | 2020      | 9.4          |
<p align="center">Tabel 1. Statistik Deskriptif (Kolom Numerik)</p>

</div>

Pada Tabel 3.1. Statistik Deskriptif (Kolom Numerik), terdapat 2 kolom numerik, berikut penjelasannya:

**🎬 Tahun Rilis (`year`)**
- Jumlah data untuk tahun rilis adalah **1272 film**.
- Rata-rata tahun rilis film adalah sekitar **2007**.
- Sebagian besar film dalam dataset ini dirilis antara tahun **2006** (kuartil pertama) dan **2016** (kuartil ketiga).
- Rentang tahun rilis cukup lebar, dari tahun **1926 hingga 2020**, menunjukkan adanya film-film klasik hingga film yang relatif baru.
- Median tahun rilis adalah **2011**, yang sedikit lebih tinggi dari rata-rata, mengindikasikan kemungkinan adanya lebih banyak film yang dirilis pada paruh kedua periode waktu dalam dataset.

**⭐ Rating Pengguna (`users_rating`)**
- Jumlah data rating pengguna adalah **1272**, namun sebelumnya disebutkan 6144, kemungkinan terjadi kesalahan atau merujuk pada total skor atau vote dari sumber eksternal (perlu klarifikasi).
- Rata-rata rating pengguna adalah sekitar **6.14**.
- Sebagian besar rating pengguna berada di antara **5.3** (kuartil pertama) dan **7.1** (kuartil ketiga).
- Rentang rating cukup luas, dari **1.2 hingga 9.4**, menunjukkan variasi preferensi pengguna yang signifikan.
- Median rating adalah **6.4**, yang sedikit lebih tinggi dari rata-rata, mengindikasikan distribusi rating yang mungkin sedikit condong ke nilai yang lebih tinggi.

#### **🧾 Statistik Deskriptif Kolom Kategorikal**
<div align="center">

| Kolom       | Count | Unique | Top                 | Freq |
|-------------|--------|--------|----------------------|------|
| title       | 1272   | 1262   | Kuntilanak 2         | 2    |
| description | 840    | 840    | It tells the story of an Indonesian revolution...                 | 1    |
| genre       | 1236   | 15     | Drama                | 456  |
| rating      | 376    | 11     | 13+                  | 161  |
| votes       | 1272   | 312    | 58                   | -    |
| languages   | 1272   | 8      | Indonesian           | 1241 |
| directors   | 1265   | 377    | Nayato Fio Nuala     | 61   |
| actors      | 1272   | 1266   | [nan, ..., nan]      | 4    |
| runtime     | 869    | 85     | 90 min               | 109  |
<p align="center">Tabel 2. Statistik Deskriptif (Kolom Kategori)</p>
</div>

Pada Tabel 3.2. Statistik Deskriptif (Kolom Kategori), terdapat 9 kolom kategori, berikut penjelasannya:

**🎞️ Judul (`title`)**
- Terdapat **1272** judul film dalam dataset.
- Terdapat **1262** judul yang unik.
- Judul **"Kuntilanak 2"** muncul paling sering (**frekuensi 2**), yang bisa mengindikasikan adanya sekuel atau kemungkinan duplikasi/kesalahan data.

**📝 Deskripsi (`description`)**
- Terdapat **840** deskripsi yang tersedia, semuanya bersifat **unik**.
- Deskripsi yang muncul pertama secara statistik adalah:  
  _"It tells the story of an Indonesian revolution..."_

**🎭 Genre (`genre`)**
- Dataset mencakup **15** genre yang berbeda.
- Genre paling dominan adalah **"Drama"** dengan **frekuensi 456**.

**🔞 Rating Usia (`rating`)**
- Terdapat **11** kategori rating usia yang berbeda.
- Rating **"13+"** adalah yang paling sering muncul, sebanyak **161** kali.

**🌐 Bahasa (`languages`)**
- Film dalam dataset menggunakan **8** bahasa yang berbeda.
- Bahasa **"Indonesian"** adalah yang paling umum digunakan, muncul pada **1241** film.

**🎬 Sutradara (`directors`)**
- Terdapat **377** nama sutradara yang berbeda.
- Sutradara dengan jumlah film terbanyak adalah **Nayato Fio Nuala** (**frekuensi 61**).

**👥 Aktor (`actors`)**
- Terdapat **1272** data aktor, namun ada **nilai-nilai tidak lengkap atau tidak valid**.
- Kombinasi aktor **"[nan, nan, nan, nan, nan]"** muncul paling sering (**frekuensi 4**), menunjukkan kemungkinan masalah dalam pengisian data.

**⏱️ Durasi (`runtime`)**
- Terdapat **85** nilai durasi yang unik.
- Durasi yang paling umum adalah **"90 min"**, muncul sebanyak **109** kali.

### Univariate Data Analysis
#### **🎞️ Distribusi Genre Film**
<p align="center">
<img src="asset\genre.png" alt="alt text" />
<p align="center">Gambar 1. Distribusi Genre</p>

**🎭 Dominasi Genre Drama**
- Genre **"Drama"** memiliki jumlah sampel terbanyak secara signifikan, yaitu **456 film**, yang mencakup sekitar **36.9%** dari total data genre yang tersedia.
- Hal ini menunjukkan bahwa **drama merupakan genre yang sangat dominan** dalam dataset ini.

**😂😱 Popularitas Comedy dan Horror**
- Genre **"Comedy"** dan **"Horror"** juga cukup populer:
  - **Comedy**: 287 film (**23.2%**)
  - **Horror**: 231 film (**18.7%**)
- Kedua genre ini menempati posisi setelah Drama dan jauh lebih tinggi dibandingkan genre-genre lainnya.

**🔫 Action sebagai Genre Signifikan**
- Genre **"Action"** berada di posisi keempat dengan **132 film** (**10.7%**).
- Ini menunjukkan bahwa genre aksi juga cukup banyak diwakili dalam dataset.

**🌍 Genre dengan Representasi Sedang**
- Genre dengan jumlah sampel menengah:
  - **Adventure**: 49 film (**4.0%**)
  - **Biography**: 28 film (**2.3%**)
- Meskipun tidak sebanyak empat genre teratas, keduanya memiliki representasi yang lebih tinggi dibandingkan genre-genre minor.

**⚠️ Minimnya Representasi Beberapa Genre**
- Beberapa genre memiliki representasi yang sangat kecil dalam dataset:
  - **Thriller**, **Romance**, **Fantasy**, **Crime**, **Animation**, **Family**, **Sci-Fi**, **War**, dan **History** semuanya berada di bawah atau sedikit di atas **1%** dari total data genre.
- Hal ini menunjukkan bahwa genre-genre tersebut **kurang dominan** atau **jarang muncul** dalam koleksi film pada dataset ini.

#### **🗓️ Distribusi Tahun Rilis Film**
<p align="center">
<img src="asset\year.png" alt="alt text" />
<p align="center">Gambar 2. Distribusi Tahun Rilis Film</p>

**📈 Konsentrasi Rilis Film Terbaru**
- Sebagian besar film dalam dataset dirilis dalam beberapa tahun terakhir.
- Tahun dengan jumlah rilis terbanyak:
  - **2019**: 111 film (**8.7%**)
  - **2018**: 97 film (**7.6%**)
  - **2009**: 79 film (**6.2%**)
  - **2011**: 78 film (**6.1%**)
  - **2008**: 77 film (**6.1%**)

**🔼 Peningkatan Jumlah Rilis Seiring Waktu**
- Visualisasi histogram menunjukkan adanya tren **peningkatan jumlah film** yang dirilis dari waktu ke waktu.
- **Lonjakan signifikan** terjadi setelah tahun **2000**, menandakan era kebangkitan produksi film yang lebih intensif.

**🏔️ Puncak Rilis di Era Modern**
- Distribusi memuncak di sekitar tahun **2010-an**, sesuai dengan frekuensi data rilis tertinggi dalam tabel.
- Ini menunjukkan bahwa **dekade 2010-an adalah era paling produktif** dalam dataset ini.

**📉 Rilis Film Awal yang Jarang**
- Film-film yang dirilis pada paruh pertama abad ke-20 sangat sedikit.
- Beberapa tahun seperti **1953**, **1951**, **1950**, **1928**, dan **1926** hanya memiliki **1 film** masing-masing (**0.1%** dari total data), menunjukkan keterbatasan arsip atau cakupan data pada era tersebut.

**⚖️ Distribusi yang Tidak Merata**
- Secara umum, **distribusi tahun rilis sangat tidak merata**.
- Konsentrasi rilis **tertinggi** terjadi dalam **dua dekade terakhir**, dengan jumlah film jauh lebih banyak dibandingkan periode sebelumnya.

**🧭 Adanya Periode dengan Jumlah Rilis Sedang**
- Sekitar tahun **1980-an dan 1990-an** terdapat **peningkatan sedang** dalam jumlah film, lebih tinggi dibandingkan era sebelum 1970-an, namun masih rendah dibandingkan era modern.

**🧩 Kurva KDE Mengkonfirmasi Tren**
- Kurva **Kernel Density Estimate (KDE)** memperhalus distribusi histogram dan menegaskan **tren peningkatan** menuju tahun-tahun yang lebih baru.
- **Puncak kepadatan** distribusi terlihat jelas di sekitar **tahun 2010-an**, memperkuat analisis peningkatan rilis film dalam dekade tersebut.

#### **⭐ 5 Rating Film Teratas Berdasarkan Jumlah Pengguna**
<p align="center">
<img src="asset\rating.png" alt="alt text" />
<p align="center">Gambar 3. 5 Rating Film Teratas berdasarkan Jumlah Pengguna</p>

**🥇 Rating 7.2 Paling Banyak Dipilih**
- **Rating 7.2** merupakan rating yang **paling banyak diberikan oleh pengguna** dalam dataset ini.
- Terdapat **54 film** (sekitar **4.2%** dari total) yang memiliki rating ini.

**🥈 Rating 6.2 dan 6.6 Cukup Populer**
- **Rating 6.2** diberikan pada **52 film** (**4.1%**).
- **Rating 6.6** diberikan pada **50 film** (**3.9%**).
- Kedua rating ini sangat dekat frekuensinya dengan rating 7.2, menunjukkan **popularitas yang tinggi**.

**🥉 Rating 7.0 dan 6.8 Mengikuti**
- **Rating 7.0** muncul pada **48 film** (**3.8%**).
- **Rating 6.8** muncul pada **45 film** (**3.5%**).
- Kedua rating ini melengkapi **lima besar** rating terbanyak dalam dataset.

**📊 Perbedaan Jumlah Pengguna Tidak Terlalu Besar**
- Perbedaan jumlah film dalam 5 rating teratas **hanya berkisar antara 45 hingga 54 film**.
- Ini menunjukkan bahwa **beberapa nilai rating cukup populer dan sering digunakan**, dengan **distribusi frekuensi yang relatif merata** di antara rating-rating favorit pengguna.

## Data Preparation
Tahap ini melibatkan pembersihan, transformasi, dan ekstraksi fitur dari dataset film untuk mempersiapkannya dalam membangun sistem rekomendasi berbasis konten.

### **📦 Penghapusan Kolom yang Tidak Relevan**
- **Tujuan**: Mengurangi noise dan kompleksitas data dengan menghapus kolom yang tidak memberikan kontribusi signifikan terhadap sistem rekomendasi.

Kolom `rating`, `votes`, `languages`, dan `runtime` dihapus karena alasan berikut:

* **Banyak Missing Values:** `rating` dan `runtime` memiliki banyak data yang hilang.
* **Kurang Relevan untuk Konten:** Kolom-kolom ini (terutama `votes`, `languages`, `rating` usia) dianggap kurang penting untuk mengukur *kemiripan konten* film dibandingkan judul, deskripsi, genre, dan aktor.
* **Simplifikasi:** Menghapus kolom ini menyederhanakan dataset dan fokus pada fitur inti untuk model rekomendasi.

### **🚫 Penanganan Missing Values**
**Tujuan**: Menangani nilai yang hilang (missing values) untuk meningkatkan kualitas dan keandalan data.

**Metode:**

- Kolom `'title'`, `'description'`, dan `'actors'`: Diisi dengan string kosong (`''`)
- Kolom `'genre'`: Diisi dengan string `'Tidak Diketahui'`
- Kolom `'users_rating'`: Diisi dengan nilai **median** kolom tersebut (lebih tahan terhadap outlier dibanding mean)

### **🛠️ Ekstraksi dan Transformasi Fitur**
**Tujuan**: Menyiapkan representasi fitur yang relevan untuk perhitungan kesamaan antar film.

**Langkah-langkah:**

- **Penggabungan Teks**:
  - Kolom `'genre'`, `'actors'`, dan `'description'` digabung menjadi satu kolom baru bernama `'text_data'`
  - Tujuannya adalah menciptakan representasi tekstual komprehensif untuk setiap film

- **Vectorisasi Teks**:
  - Menggunakan **TF-IDF Vectorizer** untuk mengubah `'text_data'` menjadi matriks TF-IDF
  - **Stop words Bahasa Inggris** dihapus untuk mengurangi noise

- **Scaling Fitur Numerik**:
  - Kolom `'year'` dan `'users_rating'` di-scale menggunakan **MinMaxScaler**
  - Scaling ini memastikan semua fitur berada pada rentang yang sama dan mencegah dominasi fitur dengan skala lebih besar

### **🔍 Perhitungan Matriks Kesamaan**
**Tujuan**: Menghitung kesamaan antar film berdasarkan representasi fitur gabungan.

**Metode:**

- **Penggabungan Fitur**:
  - Matriks TF-IDF digabung dengan fitur numerik yang telah di-scale menggunakan `hstack`

- **Cosine Similarity**:
  - Digunakan untuk menghitung **kesamaan antar film**
  - Cocok digunakan karena robust terhadap perbedaan panjang dokumen

## Modeling
Tahapan ini membahas model sistem rekomendasi yang dibuat untuk memberikan rekomendasi film kepada pengguna. Sistem ini menggunakan pendekatan Content-Based Filtering untuk merekomendasikan film yang mirip dengan film yang telah disukai atau ditonton sebelumnya.

### **📌 Algoritma Rekomendasi: Content-Based Filtering**
**Deskripsi**:

Sistem rekomendasi ini menggunakan pendekatan **Content-Based Filtering**, di mana rekomendasi didasarkan pada **atribut atau karakteristik konten** dari film itu sendiri. Sistem menganalisis informasi seperti **deskripsi, genre, aktor, dan sutradara** untuk menemukan film lain yang serupa.

**Metode:**

1. **Ekstraksi Fitur Tekstual**:
   - Fitur seperti `genre`, `actors`, dan `description` digabung menjadi satu kolom `text_data`.
   - Kolom ini kemudian diubah menjadi representasi numerik menggunakan **TF-IDF Vectorizer**.
   - TF-IDF memberikan bobot yang lebih tinggi untuk kata-kata yang unik dalam satu film namun jarang di seluruh dataset.

2. **Scaling Fitur Numerik**:
   - Kolom numerik seperti `year` dan `users_rating` di-*scale* menggunakan **MinMaxScaler** agar berada dalam rentang yang sama.

3. **Penggabungan Fitur**:
   - Matriks TF-IDF digabung dengan fitur numerik yang telah di-*scale* menggunakan `hstack`.

4. **Perhitungan Similarity**:
   - Digunakan **Cosine Similarity** untuk menghitung kesamaan antar film berdasarkan gabungan fitur tersebut.

5. **Fungsi Rekomendasi**:  
   `get_movie_recommendations(movie_title, df, sim_matrix, n_recommendations=5)`

   - Menerima: judul film, DataFrame, matriks kesamaan, dan jumlah rekomendasi
   - Mencari indeks film dari judul
   - Menghitung skor similarity film tersebut dengan seluruh film lain
   - Mengurutkan film berdasarkan skor similarity secara menurun
   - Mengembalikan **n rekomendasi teratas** beserta skor similarity-nya

### **📤 Output Rekomendasi**
Sistem akan menghasilkan daftar **n film yang paling mirip** dengan film input, berdasarkan skor kesamaan (similarity score).  
Setiap rekomendasi berisi informasi berikut:

- Judul film  
- Deskripsi  
- Genre  
- Aktor  
- Tahun rilis  
- Rating pengguna  
- Skor similarity

| Index | Title              | Description                                      | Genre     | Actors                                                     | Year | Users Rating | Similarity Score |
|-------|--------------------|--------------------------------------------------|-----------|-------------------------------------------------------------|------|---------------|------------------|
| 11    | Milea              | Milea made the decision to part with Dilan as...| Drama     | Iqbaal Dhiafakhri Ramadhan, Vanesha Prescilla             | 2020 | 6.1           | 0.761454         |
| 160   | Dilan 1990         | Milea (Vanesha Prescilla) met with Dilan (Iqba...| Drama     | Iqbaal Dhiafakhri Ramadhan, Vanesha Prescilla             | 2018 | 7.3           | 0.740067         |
| 131   | #FriendButMarried  | Ayudia (Vanesha Prescilla) and Ditto (Adipati...| Biography | Adipati Dolken, Vanesha Prescilla, Refal Hady             | 2018 | 6.9           | 0.641758         |
| 9     | Mariposa           | Iqbal (Angga Yunanda) is like a Mariposa butte...| Drama     | Angga Yunanda, Adhisty Zara, Dannia Salsabilla            | 2020 | 8.5           | 0.633936         |
| 174   | Keluarga Cemara    | After the bankruptcy, Abah loses his house and...| Drama     | Nirina Zubir, Ringgo Agus Rahman, Adhisty Zara            | 2018 | 7.9           | 0.631335         |
<p align="center">Tabel 3. Rekomendasi Film</p>

Output ini menyajikan hasil konkret dari sistem rekomendasi untuk film **'Dilan 1991'**.

**Temuan dari Rekomendasi:**

1.  **Relevansi Tinggi:** Rekomendasi teratas adalah **'Milea'** dan **'Dilan 1990'**, yang merupakan sekuel dan prekuel langsung dari film input. Skor kemiripan yang tinggi (0.76 dan 0.74) menunjukkan model berhasil mengidentifikasi hubungan konten yang sangat kuat (kemungkinan besar dari deskripsi, aktor, judul, dan genre yang sama).
2.  **Kemiripan Lainnya:** Film seperti '#FriendButMarried', 'Mariposa', dan 'Keluarga Cemara' muncul berikutnya. Kemiripan mereka (skor ~0.63-0.64) mungkin berasal dari kombinasi genre (Drama), aktor yang sama (Vanesha Prescilla, Adhisty Zara), tema dalam deskripsi, atau kedekatan dalam rating pengguna.
3.  **Skor Menurun:** Skor kemiripan menurun secara wajar seiring peringkat rekomendasi, yang diharapkan.

### **✅❌ Kelebihan dan Kekurangan Content-Based Filtering**

- ✅ Dapat merekomendasikan item baru meskipun belum memiliki rating dari pengguna lain
- ✅ Tidak tergantung pada riwayat pengguna (tidak terkena masalah *cold start* untuk pengguna baru)
- ✅ Penjelasan rekomendasi transparan karena berbasis pada kesamaan fitur konten

**Kekurangan**:

- ❌ Bergantung sepenuhnya pada kualitas dan kelengkapan fitur yang tersedia
- ❌ Cenderung memberikan rekomendasi yang sangat mirip dengan film yang telah disukai (kurang variasi atau *serendipity*)


## Evaluation
Bagian ini menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja sistem rekomendasi dan hasil evaluasinya.

### **Evaluation Metric: Precision@k**
**Deskripsi**:  
Precision@k mengukur seberapa relevan *k* rekomendasi teratas. Dalam konteks ini, relevansi ditentukan berdasarkan kesamaan genre antara film yang direkomendasikan dengan film input. Metrik ini menghitung proporsi dari *k* rekomendasi teratas yang memiliki genre yang dianggap relevan dengan film input.

**Formula**:

    Precision@k = (Jumlah rekomendasi relevan dalam top-k) / k

Di mana "relevan" berarti film yang direkomendasikan memiliki setidaknya satu genre yang termasuk dalam daftar `relevant_genres` dari film input.

**Fungsi: `evaluate_genre_precision_at_k(recommended_movies, relevant_genres, k=3)`**:
- Menerima:
  - `recommended_movies`: DataFrame hasil rekomendasi.
  - `relevant_genres`: Daftar genre yang relevan dengan film input.
  - `k`: Jumlah rekomendasi teratas yang dievaluasi.
- Langkah:
  - Memilih *k* rekomendasi teratas berdasarkan skor kesamaan.
  - Menghitung jumlah rekomendasi dalam top-*k* yang memiliki minimal satu genre dalam `relevant_genres`.
  - Mengembalikan nilai Precision@k.

### **Precision@3**
Pada proyek ini akan menggunakan k=3

- **Deskripsi**:

    Precision@3 mengukur proporsi film pada 3 rekomendasi teratas yang memiliki genre sama dengan film acuan.

- **Formula**:

    Precision@3 = (Jumlah rekomendasi relevan dalam top-3) / 3
- **Cara Kerja**:
    
    Untuk setiap film input, dicek apakah genre pada 3 rekomendasi teratas sama dengan genre film input.

### **Evaluation Result**
Sebagai contoh, evaluasi Precision@3 untuk film **"Dilan 1991"** dengan genre relevan `['Drama', 'Romance']` menghasilkan nilai **0.67**. Ini berarti bahwa dari 3 film teratas yang direkomendasikan, 2 di antaranya memiliki genre **Drama** atau **Romance**, sehingga dianggap relevan.

Evaluasi sistem secara keseluruhan perlu mempertimbangkan beberapa film input dan genre relevan yang berbeda. Dengan menghitung Precision@k untuk masing-masing kasus, kemudian mengambil rata-rata, kita dapat memperoleh gambaran umum mengenai kinerja sistem rekomendasi.

## Apakah Problem Statements Sudah Terselesaikan?
- Problem Statement 1 (membangun sistem rekomendasi film relevan) telah terselesaikan dengan baik melalui pengembangan model content-based filtering yang menggunakan fitur gabungan (genre, aktor, deskripsi) dan cosine similarity. Sistem mampu memberikan rekomendasi film yang mirip secara konten dengan film input, seperti terlihat pada contoh rekomendasi film "Dilan 1991".

- Problem Statement 2 (mengukur relevansi rekomendasi secara objektif) juga telah terselesaikan dengan penggunaan metrik Precision@3 (Genre). Nilai precision@3 sebesar 0.67 menunjukkan bahwa dua dari tiga rekomendasi teratas memiliki genre yang sama dengan film input, menandakan relevansi rekomendasi yang cukup baik.

## Referensi
[1] Aggarwal, C. C. (2016). Recommender systems: The textbook. Springer. https://doi.org/10.1007/978-3-319-29659-3. ISBN 978-3-319-29657-9
[2] Zhang, Y., & Zhang, L. (2013). A review of hybrid recommender systems: Concepts, methodologies, and applications. Expert Systems with Applications, 40(4), 1069–1079. https://doi.org/10.1016/j.eswa.2012.08.011

**---Ini adalah bagian akhir laporan---**
