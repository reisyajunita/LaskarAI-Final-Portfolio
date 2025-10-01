#!/usr/bin/env python
# coding: utf-8

# # Proyek Analisis Data: Bike Sharing
# - **Nama:** Reisya Junita Putri
# - **Email:** reisyajunita@gmail.com
# - **ID Dicoding:** reisyajunita

# # Menentukan Pertanyaan Bisnis

# 1. Bagaimana perbandingan jumlah penyewaan sepeda pada workingday dan holiday?
# 2. Jam berapa pengguna paling banyak menggunakan rental bike sharing?
# 3. Bagaimana distribusi penyewaan sepeda antara 2011 dan 2012?
# 4. Musim apa yang penggunanya paling banyak menggunakan bike sharing?
# 5. Adakah hubungan antara temp, atemp, hum, dan windspeed terhadap jumlah penyewaan sepeda?

# # Import Packages/Library yang Digunakan

# In[233]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


# # Data Wrangling

# ## Gathering Data

# In[234]:


# Load tabel day
day_df = pd.read_csv("Bike-sharing-dataset/day.csv")
day_df.head()


# In[235]:


# Load tabel hour
hour_df = pd.read_csv("Bike-sharing-dataset/hour.csv")
hour_df.head()


# **Dataset Characteristics**
# 
# Kedua dataset yaitu day.csv dan hour.csv memiliki bidang berikut, kecuali hr (hour) yang tidak tersedia di day.csv.
# 
# - instant: index catatan
# - dteday: tanggal
# - season: musim (1: musim semi, 2: musim panas, 3: musim gugur, 4:musim dingin)
# - yr: tahun (0:2011, 1:2012)
# - mnth: bulan (1 sampai 12)
# - hr: jam (0 sampai 23) "hanya ada di dataset hour.csv"
# - holiday: hari cuaca hari libur atau tidak
# - weekday: hari dalam seminggu (0: minggu, 6:sabtu)
# - workingday: hari kerja (0: bukan hari kerja, 1: hari kerja)
# + weathersit:
#     - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# - temp: suhu yang dinormalisasi dalam Celsius. Nilai dibagi menjadi 41 (maks)
# - atemp: suhu perasaan yang dinormalisasi dalam Celsius. Nilai dibagi menjadi 50 (maks)
# - hum: kelembapan yang dinormalisasi. Nilai dibagi menjadi 100 (maks)
# - windspeed: kecepatan angin yang dinormalisasi. Nilai dibagi menjadi 67 (maks)
# - casual: jumlah pengguna biasa tanpa langganan
# - registered: jumlah pengguna terdaftar berlangganan
# - cnt: jumlah total sepeda sewaan termasuk casual dan registered

# ## Assessing Data

# ### Data day_df

# In[236]:


# Menilai data day_df
day_df.info()


# **Insight:**
# 
# Tidak ada masalah pada jumlah data seluruh kolom akan tetapi terdapat 2 tipe data yang akan diubah untuk mempermudah analisis data:
# 1. Pada kolom **dteday** yang seharusnya tipe data **datetime**, bukan **object**.
# 2. Pada kolom **season** kita ganti dari  **int** menjadi **category**.

# In[237]:


# View duplicate data
print("Jumlah duplikasi data: ", day_df.duplicated().sum())


# In[238]:


# Memeriksa paramater statistik
day_df.describe()


# **Insight:**
# 
# Jika diperhatikan, tidak terdapat keanehan pada parameter statistik di atas.

# In[239]:


# Rename columns
day_df.rename(columns={
    "dteday": "date",
    "yr": "year",
    "mnth": "month",
    "cnt": "total"
}, inplace=True)


# ### Data hour_df

# In[240]:


# View data hour_df
hour_df.info()


# **Insight:**
# 
# Tidak ada masalah pada jumlah data seluruh kolom akan tetapi terdapat 2 tipe data yang akan diubah untuk mempermudah analisis data:
# 1. Pada kolom **dteday** yang seharusnya tipe data **datetime**, bukan **object**.
# 2. Pada kolom **season** kita ganti dari  **int** menjadi **category**.

# In[241]:


# View duplicate data
print("Jumlah duplikasi: ", hour_df.duplicated().sum())


# In[242]:


# Memeriksa parameter statistik
hour_df.describe()


# **Insight:**
# 
# Jika diperhatikan, tidak terdapat keanehan pada parameter statistik di atas.

# In[243]:


# Rename columns
hour_df.rename(columns={
    "dteday": "date",
    "yr": "year",
    "mnth": "month",
    "hr": "hour",
    "cnt": "total"
}, inplace=True)


# ## Cleaning Data

# ### Data day_df

# In[244]:


# Mengganti tipe data dteday pada tabel day_df
day_df["date"] = pd.to_datetime(day_df["date"])

# Mengganti tipe data season pada tabel day_df
day_df['season'] = day_df['season'].astype('category')
day_df['season'] = day_df['season'].cat.set_categories([1, 2, 3, 4], ordered=True)

day_df.info()


# In[245]:


# Menghapus kolom "weathersit" karena tidak diperlukan dalam analisis
day_df = day_df.drop("weathersit", axis=1)


# ### Data hour_df

# In[246]:


# Menganti tipe data dteday pada tabel hour_df
hour_df["date"] = pd.to_datetime(hour_df["date"])

# Menganti tipe data season pada tabel hour_df
hour_df["season"] = hour_df["season"].astype("category")

hour_df.info()


# In[247]:


# Menghapus kolom "weathersit" karena tidak diperlukan dalam analisis
hour_df = hour_df.drop("weathersit", axis=1)


# # Exploratory Data Analysis

# ## Eksplorasi Data day_df

# ### Ringkasan Parameter Statistik

# In[248]:


# Melihat parameter statistik
day_df.describe(include="all")


# **Insight**
# 
# Pada rangkuman parameter statistik di atas, semua kolom memiliki 731 nilai, artinya tidak ada data yang hilang. Rata-rata total penyewaan sepeda **4504.35 sepeda per hari** dengan jumlah total penyewa sepeda memiliki standar deviasi **1937.21**, menunjukkan variasi yang besar. Dalam satu hari, jumlah minimum penyewa sepeda adalah **22** dan maksimum **8714**.
# 
# Pada kuartil pertama jumlah total penyewa sepeda **3152** sepeda, dengan median sebesar **4548** sepeda, dan kuartil ketiga sebesar **5956** sepeda. Ini menunjukkan bahwa separuh data berada antara **3152 dan 5956** penyewa sepeda per hari.

# ### Melihat Korelasi day_df

# In[249]:


day_df.corr()


# **Insight**
# 
# Pada table korelasi di atas, dapat dilihat bahwa jumlah total pengguna (count) **sangat berkolerasi terhadap registered (0.95)**, artinya jumlah penyewa registered lebih banyak dibandingkan casual. jumlah total pengguna (count) **cukup memiliki korelasi terhadap temp (0.63)**, artinya semakin hangat, semakin banyak orang yang menyewa sepeda. jumlah penyewa casual memiliki **korelasi negatif dengan workingday (-0.52)**, artinya pengguna casual lebih banyak menyewa pada hari libur.
# 
# Untuk holiday dengan total penyewa (count) memiliki nilai korelasi (-0.068), artinya hari libur tidak terlalu memengaruhi jumlah penyewaan sepeda. dan untuk windspeed dengan count (-0.23), artinya kecepatan angin sedikit mengurangi jumlah pengguna tapi tidak signifikan.
# 
# **Kesimpulan**
# 
# - Cuaca lebih hangat meningkatkan jumlah penyewaan sepeda.
# - Hari kerja lebih banyak digunakan oleh pengguna registered, sementara pengguna casual lebih aktif di hari libur.
# - Cuaca buruk dan angin kencang mengurangi jumlah penyewaan sepeda.

# In[270]:


# Membuat figure
plt.figure(figsize=(10, 6))

# Membuat histogram untuk hari kerja
sns.histplot(day_df[day_df["workingday"] == 1]["total"], bins=24, kde=True, color="blue", label="Hari Kerja", alpha=0.6)

# Membuat histogram untuk akhir pekan/libur
sns.histplot(day_df[day_df["workingday"] == 0]["total"], bins=24, kde=True, color="red", label="Akhir Pekan/Libur", alpha=0.6)

# Menambahkan judul dan label
plt.title("Distribusi Penyewaan Sepeda Berdasarkan Hari Kerja & Akhir Pekan", fontsize=14)
plt.xlabel("Jumlah Peminjaman", fontsize=12)
plt.ylabel("Frekuensi", fontsize=12)
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()


# **Insight**
# 
# Pada histogram di atas, dapat dilihat bahwa jumlah penyewaan sepeda lebih tinggi pada hari kerja dibanding akhir pekan/libur. Puncak jumlah penyewaan di hari kerja berada di sekitar 4000 - 6000 penyewa, sementara pada akhir pekan lebih merata dan tidak memiliki puncak yang jelas. 
# 
# Penyewaan sepeda tetap terjadi di akhir pekan, tetapi lebih sedikit. walaupun lebih rendah dibanding hari kerja, penyewaan akhir pekan tetap ada dalam jumlah yang signifikan.

# ### Pivot Table Jumlah Penyewa Casual dan Registered berdasarkan Weekday

# In[ ]:


# pivot table jumlah penyewa casual dan registered berdasarkan weekday
weekday_users = day_df.groupby(by="weekday").agg({
    "casual": "sum",
    "registered": "sum"
})

print(weekday_users)


# **Insight:**
# 
# Berdasarkan pivot table di atas, dapat dilihat bahwa jumlah penyewa pada weekday cukup banyak, ini terjadi pada penyewa dari **registered** sebanyak 423.935 users yaitu pada hari ke-4 dan untuk hari lainnya juga cukup banyak, sedangkan untuk pada user casual paling banyak penyewa terjadi pada hari ke-6 dan 0 sebanyak 153.852 dan 140521. 

# ### Pivot Table temp, atemp, hum, dan windspeed berdasarkan Bulan (month)

# In[ ]:


# pivot table jumlah pelanggan casual dan registered berdasarkan month
season_users = day_df.groupby(by="month").agg({
    "temp": "mean",
    "atemp": "mean",
    "hum": "mean",
    "windspeed": "mean"
})

print(season_users)


# **Insight**
# 
# Berdasarkan pivot table di atas, menunjukkan 

# ## Eksplorasi Data hour_df

# ### Ringkasan Parameter Statistik

# In[253]:


# Melihat parameter statistik
hour_df.describe(include="all")


# **Insight**
# 
# Pada rangkuman parameter statistik di atas, semua kolom memiliki **17379** nilai, menunjukkan bahwa tidak ada data yang hilang. Pada kolom rata-rata (mean) jam "hour" penyewaan sepeda terjadi pada **jam 11.5 atau sekitar jam 11 - 12 siang**. Nilai mediannya adalah 12, karena median ≈ mean, **distribusi data mendekati simetris**.
# 
# Pada nilai interquartile range, menunjukkan 50% tengah data berada antara **jam 6 pagi hingga jam 18 sore**. Data cenderung simetris karena mean dan mediannya hampir sama.

# In[254]:


# Membuat histogram
plt.figure(figsize=(10, 6))
sns.histplot(hour_df["hour"], bins=24, kde=True)
plt.title('Distribusi Kolom hour')
plt.xlabel('Hour')
plt.ylabel('Frekuensi')
plt.show()


# In[255]:


# Membuat boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=hour_df['hour'])
plt.title('Boxplot Kolom hour')
plt.xlabel('Hour')
plt.show()


# **Insight**
# 
# Pada visualisasi data di atas, histogram menunjukkan bahwa frekuensi tiap jam hampir sama, kecuali di awal (jam 0) dan akhir (jam 23) yang sedikit lebih rendah. Distribusi ini lebih menyerupai distribusi seragam (uniform distribution).
# 
# Sementara itu, pada boxplot terlihat bahwa median berada tepat di tengah box, yang menunjukkan bahwa distribusi tidak condong (skewed). Selain itu, panjang whisker yang seimbang menunjukkan bahwa sebaran data cukup simetris. Boxplot ini juga mengonfirmasi bahwa distribusi hampir seragam, sesuai dengan histogram, serta tidak terdapat outlier dalam data.

# ### Melihat Korelasi hour_df

# In[256]:


hour_df.corr()


# **Insight**
# 
# Pada tabel korelasi di atas, jumlah total penyewa sepeda **(count) berkorelasi tinggi dengan registered (0.97)**, artinya pengguna registered lebih banyak dibandingkan casual. Jumlah total penyewa sepeda **(count) berkolerasi cukup tinggi dengan hour (0.39)**, artinya ada jam-jam tertentu di mana lebih banyak orang yang menyewa sepeda.
# 
# **count juga berkorelasi positif dengan temp (0.40) dan atemp (0.40)**, artinya cuaca yang lebih hangat meningkatkan jumlah penyewaan sepeda. **Count berkorelasi negatif dengan hum (-0.32) dan windspeed (-0.09)**, artinya kelembaban tinggi dan angin kencang sedikit mengurangi jumlah penyewa.
# 
# Faktor yang mempengaruhi pengguna casual, untuk **hour memiliki korelasi tinggi dengan casual (0.30)**, artinya ada jam tertentu yang lebih populer untuk pengguna casual, pada workingday memiliki korelasi negatif dengan casual (-0.30), artinya lebih banyak pengguna casual di hari libur.
# 
# Faktor yang mempengaruhi pengguna registered, untuk total penyewa sepeda (count) memiliki korelasi tinggi dengan registered (0.97), artinya sebagian besar pengguna berasal dari kategori registered. Dan untuk hour berkolerasi cukup tinggi dengan registered (0.37), artinya ada pola penggunaan tertentu sepanjang hari.
# 
# **Kesimpulan**
# 
# - Cuaca lebih hangat meningkatkan jumlah penyewaan sepeda
# - Hari kerja lebih banyak digunakan oleh pengguna registered, sementara pengguna casual lebih aktif di hari libur.
# - waktu (jam) berpengaruh signifikan terhadap jumlah penyewa sepeda
# - Kelembaban tinggi dan angin kencang sedikit mengurangi jumlah penyewaan sepeda.

# ### Pivot Table Jumlah Penyewa Casual dan Registered berdasarkan Season

# In[257]:


# Pivot table jumlah penyewa pada setiap season berdasarkan casual dan registered
season_users = hour_df.groupby(by="season", observed=False).agg({
    "casual": "sum",
    "registered": "sum"
}).sort_values(by=["casual", "registered"], ascending=False)

print(season_users)


# ### Pivot Table Melihat Season Mana yang Paling Banyak Penyewanya berdasarkan Temp, Atemp, Hum, dan Windspeed

# In[258]:


# Pivot table melihat season mana yang paling banyak penyewanya berdasarkan temp, atemp, hum, dan windspeed
hour_df.groupby(by="season", observed=False).agg({
    "temp": "mean",
    "atemp": "mean",
    "hum": "mean",
    "windspeed": "mean",
    "total": "sum"
}).sort_values(by="total", ascending=False)


# In[259]:


# Pivot table melihat jam yang memiliki penyewaan terbanyak
hour_df.groupby(by="hour").agg({
    "total": "sum"
}).sort_values(by="total", ascending=False)


# **Insight:**
# 
# Berdasarkan pivot table di atas, dapat dilihat pada jam 17 dan 18 adalah jam-jam di mana terjadinya penyewaan sepeda terbanyak. Selain pada sore hari, di pagi hari juga terjadi penyewaan sepeda yang cukup banyak pada jam 8. 

# # Visualization & Explanatory Data Analysis

# ### Pertanyaan 1. Bagaimana perbandingan jumlah penyewaan sepeda pada workingday dan holiday?

# In[260]:


workingday_total = day_df[day_df["workingday"] == 1]["total"].sum()
holiday_total = day_df[day_df['holiday'] == 1]['total'].sum()

# Visualisasi Data
colors = ["#72bcd4", "#d3d3d3"]
plt.pie( [workingday_total, holiday_total], labels=["Workingday", "Holiday"],colors=colors, autopct="%1.1f%%", shadow=True)
plt.title("Perbandingan Total Penyewaan Sepeda pada Hari Kerja dan Hari Libur", fontsize=15, fontweight='bold')
plt.axis("equal")
plt.show()


# **Insight**
# 
# Pada pie chart di atas, total pengguna menyewa sepeda jauh lebih tinggi pada hari kerja **(96.7%)** dibandingkan hari libur **(3.3%)**, ini menunjukkan bahwa layanan lebih sering digunakan untuk keperluan rutin seperti bekerja atau sekolah. Walaupun masih ada pengguna pada hari libur, jumlahnya jauh lebih rendah dibandingkan hari kerja, kemungkinan karena orang cenderung beristirahat atau menggunakan transportasi yang lain. 
# 
# **Tindak lanjut**
# - Jika layanan ingin menarik lebih banyak pelanggan di hari libur, bisa dibuat promo atau diskon khusus.
# - Jika ingin mengoptimalkan layanan pada hari kerja, bisa dilakukan penyesuaian jumlah unit yang tersedia.

# ### Pertanyaan 2. Jam berapa pengguna paling banyak menggunakan rental bike sharing?

# In[261]:


# Melihat Pola Waktu Pengguna Menyewa Bike Sharing
# Menghitung mode dari kolom 'hr'
mode_hour = hour_df['hour'].mode()[0]
# Menghitung median dari kolom 'hr'
median_hour = hour_df['hour'].median()

# Membuat histogram untuk visualisasi
sns.histplot(data=hour_df, x="hour", bins=24)
plt.title("Distribusi Jam Penyewaan Sepeda", fontweight='bold')
plt.xlabel(None)
plt.ylabel(None)

# Menandai mode dan median pada histogram
plt.axvline(mode_hour, color='red', linestyle='--', label='Mode')
plt.axvline(median_hour, color='green', linestyle='-', label='Median')
plt.legend()
plt.show()


# **Insight**
# 
# Berdasarkan Histogram di atas:
# 1. Pada garis merah putus-putus (mode) menunjukkan bahwa sekitar jam 16 atau 4 sore adalah waktu yang paling sering terjadinya penyewaan sepeda. Kemungkinan penyewa menggunakan rental ini untuk aktivitas terkait pulang kerja, sekolah, atau aktivitas lainnya yang dilakukan pada sore hari.
# 2. Pada garis hijau (median) menunjukkan bahwa setengah dari total penyewaan terjadi sebelum jam 12 atau siang hari dan setengahnya lagi setelahnya.

# In[262]:


# Melihat Jam berapa Pengguna Casual dan Registered melakukan penyewaan
plt.figure(figsize=(10,5))

# Histogram untuk casual users
sns.histplot(hour_df, x="hour", bins=24, weights=hour_df["casual"], color="red", label="Casual", kde=True)

# Histogram untuk registered users
sns.histplot(hour_df, x="hour", bins=24, weights=hour_df["registered"], color="green", label="Registered", kde=True)

plt.title("Distribusi Penyewaan Sepeda Berdasarkan Jam (Casual vs Registered)", fontweight='bold')
plt.xlabel(None)
plt.ylabel(None)
plt.legend()
plt.show()


# **Insight**
# 
# Pada histogram di atas, kita bisa lihat bahwa pengguna casual cenderung lebih aktif di siang dan sore hari, kemungkinan karena mereka melakukan penyewaan sepeda untuk berolahraga atau keperluan rekreasi. Pada pengguna registered, kemungkinan besar memiliki pola lebih tetap, seperti jam sibuk di pagi hari (berangkat kerja/sekolah) dan sore hari (pulang kerja/sekolah).

# ### Pertanyaan 3. Bagaimana distribusi penyewaan sepeda antara 2011 dan 2012?

# In[263]:


# Menghitung total penyewaan per tahun
yearly_counts = day_df.groupby("year")["total"].sum().reset_index()

# Membuat barplot
plt.figure(figsize=(8,5))
sns.barplot(data=yearly_counts, x="year", y="total", hue="year", palette="viridis")

# Menyesuaikan tampilan
plt.title("Distribusi Penyewaan Sepeda Berdasarkan Tahun", fontsize=15, fontweight='bold')
plt.xlabel(None)
plt.ylabel(None)
plt.legend(title="Year")
plt.xticks(ticks=[0, 1], labels=["2011", "2012"])
plt.show()


# In[264]:


# Melihat pola data penyewaan sepeda per bulan di tahun 2011 dan 2012
plt.figure(figsize=(10, 6))
sns.lineplot(data=day_df, x="month", y="total", hue="year", marker="o", palette="viridis")

# Menyesuaikan tampilan
plt.title("Tren Penyewaan Sepeda Per Bulan (2011 vs 2012)", fontsize=15, fontweight='bold')
plt.xlabel("Bulan")
plt.ylabel("Total Penyewaan Sepeda")
plt.xticks(ticks=range(1,13), labels=["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
plt.legend(title="Tahun", labels=["2011", "2012"])
plt.show()


# **Insight**
# 
# Pada line chart di atas, menunjukkan pola kenaikan dan penurunan yang serupa. Penyewaan sepeda meningkat dari **Januari hingga Agustus**, mencapai puncak di sekitar **Juli hingga Agustus**, setelah itu jumlah penyewaan mulai menurun hingga **Desember**.
# 
# Garis hijau (2012) selalu di atas garis biru (2011), menunjukkan peningkatan jumlah pengguna pada tahun 2012. ini bisa disebabkan oleh faktor seperti peningkatan jumlah pelanggan, promosi, atau cuaca yang lebih mendukung.
# 
# Variasi data cukup stabil, tetapi ada beberapa fluktuasi terutama di bulan-bulan dengan jumlah penyewaan tinggi.

# ### Pertanyaan 4. Musim apa yang penggunanya paling banyak menggunakan bike sharing?

# In[265]:


# Buat kategori dengan urutan yang benar
season_order = [1, 2, 3, 4]
season_labels = ["Spring", "Summer", "Fall", "Winter"]

# Buat DataFrame baru yang hanya berisi total penyewaan per musim
season_rentals = day_df.groupby('season', observed=False)['total'].sum().reset_index()

# Pastikan musim ditampilkan sesuai urutan
season_rentals['season'] = season_rentals['season'].astype('category')
season_rentals['season'] = season_rentals['season'].cat.set_categories(season_order, ordered=True)

# Visualisasi dengan seaborn
sns.barplot(data=season_rentals, x="season", y="total", hue="season", palette="viridis")
plt.xticks(ticks=[0, 1, 2, 3], labels=season_labels)
plt.xlabel(None)
plt.ylabel(None)
plt.title("Jumlah Penyewaan Bike Sharing per Musim", fontweight='bold')
plt.show()


# **Insight**
# 
# Berdasarkan bar plot di atas, musim yang paling tinggi penyewaan sepeda terjadi pada musim gugur (Fall) dan diikuti oleh musim panas (Summer). Ini menunjukkan bahwa pengguna lebih sering menggunakan jasa bike sharing saat cuaca cenderung sejuk.
# 
# Musim dingin (Winter), merupakan musim yang memiliki jumlah penyewaan yang paling rendah. Dan untuk musim semi (Spring) mulai meningkat tetapi belum setinggi Summer/Fall.
# 
# **Kesimpulan & Strategi**
# 
# - Musim Fall dan Summer adalah periode terbaik untuk dilakukannya promosi layanan Bike Sharing (misalnya diskon atau langganan murah).
# - Winter bisa jadi tantangan, sehingga bisa dipertimbangkan strateginya, seperti:
#     1. Menyediakan rute yang lebih aman atau menawarkan insentif bagi pengguna musim dingin.
#     2. Menyewakan sepeda khusus untuk musim dingin (misalnya dengan ban anti selip)
# - Spring bisa jadi momen untuk meningkatkan awareness agar pengguna lebih siap menghadapi peak season di Summer & Fall

# # Analisis Lanjutan (Uji Korelasi Pearson)

# ### Pertanyaan 5. Adakah hubungan antara temp, atemp, hum, windspeed terhadap jumlah penyewaan sepeda?

# In[266]:


# Hitung korelasi
correlation_matrix = day_df[['temp', 'atemp', 'hum', 'windspeed', 'total']].corr()

# Visualisasi heatmap korelasi
plt.figure(figsize=(8,5))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap Korelasi Faktor Cuaca terhadap Penyewaan Sepeda", fontweight='bold')
plt.show()


# **Insight**
# 
# 1. temp dan atemp memiliki korelasi positif kuat terhadap jumlah penyewaan sepeda (+0.63). Ini menunjukkan **semakin tinggi suhu, semakin banyak sepeda yang disewa.**
# 2. hum memiliki korelasi negatif lemah terhadap jumlah penyewaan sepeda (-0.10), ini menunjukkan bahwa **kelembaban udara tidak terlalu berpengaruh terhadap jumlah penyewaan sepeda**, tetapi ada kecenderungan sedikit menurun saat kelembaban tinggi.
# 3. windspeed memiliki korelasi negatif sedang terhadap penyewaan sepeda (-0.23). Ini menunjukkan semakin tinggi kecepatan angin, semakin sedikit jumlah penyewaan sepeda, ini masuk akal karena angin yang kencang bisa membuat orang kurang nyaman untuk bersepeda.
# 
# **Kesimpulan**
# - Suhu (temp & atemp) adalah faktor yang paling berpengaruh terhadap penyewaan sepeda (korelasi kuat positif).
# - Kecepatan angin memiliki pengaruh sedang dan negatif, artinya angin kencang bisa mengurangi jumlah penyewaan sepeda.
# - Kelembaban memiliki pengaruh yang sangat kecil terhadap penyewaan sepeda (korelasi lemah).
# 
# **Tindakan yang bisa diambil**
# * Jika ingin meningkatkan jumlah penyewaan sepeda, promosi bisa lebih difokuskan pada hari-hari dengan suhu yang nyaman.
# * Bisa dibuat layanan rekomendasi ke pengguna, misalnya memberikan diskon saat kecepatan angin rendah agar orang lebih tertarik menyewa sepeda.
# * Untuk hari-hari dengan angin kencang, bisa ada peningkatan layanan keamanan seperti jalur sepeda yang lebih terlindungi dari angin.

# In[267]:


# Scatter plot hubungan temp, atemp, hum, windspeed dengan total rental
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Temperature vs Total Rentals
sns.scatterplot(ax=axes[0,0], data=day_df, x='temp', y='total', alpha=0.5, color="red")
axes[0,0].set_title("Temp vs Total Rentals", fontweight='bold')

# Feels-like Temperature vs Total Rentals
sns.scatterplot(ax=axes[0,1], data=day_df, x='atemp', y='total', alpha=0.5, color="orange")
axes[0,1].set_title("Atemp vs Total Rentals", fontweight='bold')

# Humidity vs Total Rentals
sns.scatterplot(ax=axes[1,0], data=day_df, x='hum', y='total', alpha=0.5, color="blue")
axes[1,0].set_title("Humidity vs Total Rentals", fontweight='bold')

# Windspeed vs Total Rentals
sns.scatterplot(ax=axes[1,1], data=day_df, x='windspeed', y='total', alpha=0.5, color="green")
axes[1,1].set_title("Windspeed vs Total Rentals", fontweight='bold')

plt.tight_layout()
plt.show()


# In[268]:


# Membuat kategori suhu
def categorize_temp(temp):
    if temp < 10:
        return "Dingin"
    elif 10 <= temp <= 25:
        return "Sedang"
    else:
        return "Panas"

day_df['temp_category'] = day_df['temp'].apply(categorize_temp)

# Boxplot jumlah penyewaan berdasarkan kategori suhu
plt.figure(figsize=(8,5))
sns.boxplot(data=day_df, x='temp_category', y='total', hue="temp_category", palette="viridis")
plt.title("Jumlah Penyewaan Berdasarkan Kategori Suhu", fontweight='bold')
plt.xlabel("Kategori Suhu")
plt.ylabel("Total Rentals")
plt.show()


# In[269]:


# Uji korelasi Pearson dengan p-value
corr_temp, p_temp = pearsonr(day_df['temp'], day_df['total'])
corr_atemp, p_atemp = pearsonr(day_df['atemp'], day_df['total'])
corr_hum, p_hum = pearsonr(day_df['hum'], day_df['total'])
corr_wind, p_wind = pearsonr(day_df['windspeed'], day_df['total'])

print(f"Korelasi Temp vs Total Rentals: {corr_temp:.2f} (p-value: {p_temp:.3f})")
print(f"Korelasi Atemp vs Total Rentals: {corr_atemp:.2f} (p-value: {p_atemp:.3f})")
print(f"Korelasi Hum vs Total Rentals: {corr_hum:.2f} (p-value: {p_hum:.3f})")
print(f"Korelasi Windspeed vs Total Rentals: {corr_wind:.2f} (p-value: {p_wind:.3f})")


# # Conclusions
