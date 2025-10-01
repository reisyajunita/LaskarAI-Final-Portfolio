{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "65613923",
   "metadata": {},
   "source": [
    "# Movie Recommendation System (IMDb Indonesian Movies)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cff229f8",
   "metadata": {},
   "source": [
    "## Project Overview"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6b510411",
   "metadata": {},
   "source": [
    "Dalam proyek ini, pendekatan utama yang akan digunakan adalah content-based filtering dengan memanfaatkan teknik TF-IDF (Term Frequency-Inverse Document Frequency) dan cosine similarity untuk menganalisis kemiripan antar film berdasarkan fitur-fitur seperti genre, aktor, dan deskripsi film. Pendekatan ini dipilih karena kemampuannya dalam memberikan rekomendasi yang relevan bahkan tanpa data interaksi pengguna yang ekstensif.\n",
    "\n",
    "| Fitur/Variabel   | Deskripsi                                        | Tipe Data   |\n",
    "|------------------|--------------------------------------------------|-------------|\n",
    "| title            | Judul film                                       | object      |\n",
    "| year             | Tahun rilis film                                 | int64       |\n",
    "| description      | Deskripsi singkat film                           | object      |\n",
    "| genre            | Genre film (16 kategori)          | object      |\n",
    "| rating           | Rating usia penonton (12 kategori)               | object      |\n",
    "| users_rating     | Rating pengguna IMDb (skala 1–10)                | float64     |\n",
    "| votes            | Jumlah vote IMDb                                 | object      |\n",
    "| languages        | Bahasa yang digunakan dalam film                 | object      |\n",
    "| directors        | Nama sutradara film                              | object      |\n",
    "| actors           | Daftar aktor yang membintangi film              | object      |\n",
    "| runtime          | Durasi film                                      | object      |\n",
    "\n",
    "**Sumber Data:**\n",
    "[**IMDb Indonesian Movies**](https://www.kaggle.com/datasets/dionisiusdh/imdb-indonesian-movies/data) (Kaggle)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1043b9e0",
   "metadata": {},
   "source": [
    "## Import Library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "a15407cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from scipy.sparse import hstack\n",
    "\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings ('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "29c35971",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>year</th>\n",
       "      <th>description</th>\n",
       "      <th>genre</th>\n",
       "      <th>rating</th>\n",
       "      <th>users_rating</th>\n",
       "      <th>votes</th>\n",
       "      <th>languages</th>\n",
       "      <th>directors</th>\n",
       "      <th>actors</th>\n",
       "      <th>runtime</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>#FriendButMarried 2</td>\n",
       "      <td>2020</td>\n",
       "      <td>Ayudia (Mawar De Jongh) is not satisfied enoug...</td>\n",
       "      <td>Biography</td>\n",
       "      <td>13+</td>\n",
       "      <td>6.5</td>\n",
       "      <td>120</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Rako Prijanto</td>\n",
       "      <td>['Adipati Dolken', 'Mawar Eva de Jongh', 'Vonn...</td>\n",
       "      <td>100 min</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>4 Mantan</td>\n",
       "      <td>2020</td>\n",
       "      <td>Sara, Airin, Rachel, and Amara were accidental...</td>\n",
       "      <td>Thriller</td>\n",
       "      <td>17+</td>\n",
       "      <td>6.4</td>\n",
       "      <td>8</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Hanny Saputra</td>\n",
       "      <td>['Ranty Maria', 'Jeff Smith', 'Melanie Berentz...</td>\n",
       "      <td>80 min</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Aku Tahu Kapan Kamu Mati</td>\n",
       "      <td>2020</td>\n",
       "      <td>After apparent death, Siena is able to see sig...</td>\n",
       "      <td>Horror</td>\n",
       "      <td>13+</td>\n",
       "      <td>5.4</td>\n",
       "      <td>17</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Hadrah Daeng Ratu</td>\n",
       "      <td>['Natasha Wilona', 'Ria Ricis', 'Al Ghazali', ...</td>\n",
       "      <td>92 min</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Anak Garuda</td>\n",
       "      <td>2020</td>\n",
       "      <td>Good Morning Indonesia, a school for poor orph...</td>\n",
       "      <td>Adventure</td>\n",
       "      <td>13+</td>\n",
       "      <td>9.1</td>\n",
       "      <td>27</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Faozan Rizal</td>\n",
       "      <td>['Tissa Biani Azzahra', 'Violla Georgie', 'Aji...</td>\n",
       "      <td>129 min</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Dignitate</td>\n",
       "      <td>2020</td>\n",
       "      <td>Alfi (Al Ghazali) meets Alana (Caitlin Halderm...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>17+</td>\n",
       "      <td>7.6</td>\n",
       "      <td>33</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Fajar Nugros</td>\n",
       "      <td>['Al Ghazali', 'Caitlin Halderman', 'Giorgino ...</td>\n",
       "      <td>109 min</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                      title  year  \\\n",
       "0       #FriendButMarried 2  2020   \n",
       "1                  4 Mantan  2020   \n",
       "2  Aku Tahu Kapan Kamu Mati  2020   \n",
       "3               Anak Garuda  2020   \n",
       "4                 Dignitate  2020   \n",
       "\n",
       "                                         description      genre rating  \\\n",
       "0  Ayudia (Mawar De Jongh) is not satisfied enoug...  Biography    13+   \n",
       "1  Sara, Airin, Rachel, and Amara were accidental...   Thriller    17+   \n",
       "2  After apparent death, Siena is able to see sig...     Horror    13+   \n",
       "3  Good Morning Indonesia, a school for poor orph...  Adventure    13+   \n",
       "4  Alfi (Al Ghazali) meets Alana (Caitlin Halderm...      Drama    17+   \n",
       "\n",
       "   users_rating votes   languages          directors  \\\n",
       "0           6.5   120  Indonesian      Rako Prijanto   \n",
       "1           6.4     8  Indonesian      Hanny Saputra   \n",
       "2           5.4    17  Indonesian  Hadrah Daeng Ratu   \n",
       "3           9.1    27  Indonesian       Faozan Rizal   \n",
       "4           7.6    33  Indonesian       Fajar Nugros   \n",
       "\n",
       "                                              actors  runtime  \n",
       "0  ['Adipati Dolken', 'Mawar Eva de Jongh', 'Vonn...  100 min  \n",
       "1  ['Ranty Maria', 'Jeff Smith', 'Melanie Berentz...   80 min  \n",
       "2  ['Natasha Wilona', 'Ria Ricis', 'Al Ghazali', ...   92 min  \n",
       "3  ['Tissa Biani Azzahra', 'Violla Georgie', 'Aji...  129 min  \n",
       "4  ['Al Ghazali', 'Caitlin Halderman', 'Giorgino ...  109 min  "
      ]
     },
     "execution_count": 101,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Memuat Dataset\n",
    "df = pd.read_csv('indonesian_movies.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8f3e8c9a",
   "metadata": {},
   "source": [
    "## Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1be0014f",
   "metadata": {},
   "source": [
    "### Deskripsi Variabel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "7148e8f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Data Shape: (1272, 11)\n",
      "\n",
      "Informasi Movies Dataset:\n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1272 entries, 0 to 1271\n",
      "Data columns (total 11 columns):\n",
      " #   Column        Non-Null Count  Dtype  \n",
      "---  ------        --------------  -----  \n",
      " 0   title         1272 non-null   object \n",
      " 1   year          1272 non-null   int64  \n",
      " 2   description   840 non-null    object \n",
      " 3   genre         1236 non-null   object \n",
      " 4   rating        376 non-null    object \n",
      " 5   users_rating  1272 non-null   float64\n",
      " 6   votes         1272 non-null   object \n",
      " 7   languages     1272 non-null   object \n",
      " 8   directors     1265 non-null   object \n",
      " 9   actors        1272 non-null   object \n",
      " 10  runtime       869 non-null    object \n",
      "dtypes: float64(1), int64(1), object(9)\n",
      "memory usage: 109.4+ KB\n",
      "\n",
      "Statistik Deskriptif (Kolom Numerik) Movies Dataset:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year</th>\n",
       "      <th>users_rating</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1272.000000</td>\n",
       "      <td>1272.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>2007.023585</td>\n",
       "      <td>6.144418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>12.968560</td>\n",
       "      <td>1.389315</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1926.000000</td>\n",
       "      <td>1.200000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>2006.000000</td>\n",
       "      <td>5.300000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>2011.000000</td>\n",
       "      <td>6.400000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>2016.000000</td>\n",
       "      <td>7.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>2020.000000</td>\n",
       "      <td>9.400000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              year  users_rating\n",
       "count  1272.000000   1272.000000\n",
       "mean   2007.023585      6.144418\n",
       "std      12.968560      1.389315\n",
       "min    1926.000000      1.200000\n",
       "25%    2006.000000      5.300000\n",
       "50%    2011.000000      6.400000\n",
       "75%    2016.000000      7.100000\n",
       "max    2020.000000      9.400000"
      ]
     },
     "execution_count": 102,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Melihat Bentuk Data (Shape)\n",
    "print('Data Shape:', df.shape)\n",
    "\n",
    "# Melihat Informasi Dataset\n",
    "print('\\nInformasi Movies Dataset:')\n",
    "df.info()\n",
    "\n",
    "# Melihat Statistik Deskriptif Dataset\n",
    "print('\\nStatistik Deskriptif (Kolom Numerik) Movies Dataset:')\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ad7797a9",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "- Terdapat 1272 Data dan 11 Kolom.\n",
    "- Terdapat nilang yang hilang atau *missing values*\n",
    "\n",
    "**🎬 Tahun Rilis (`year`)**\n",
    "- Jumlah data untuk tahun rilis adalah **1272 film**.\n",
    "- Rata-rata tahun rilis film adalah sekitar **2007**.\n",
    "- Sebagian besar film dalam dataset ini dirilis antara tahun **2006** (kuartil pertama) dan **2016** (kuartil ketiga).\n",
    "- Rentang tahun rilis cukup lebar, dari tahun **1926 hingga 2020**, menunjukkan adanya film-film klasik hingga film yang relatif baru.\n",
    "- Median tahun rilis adalah **2011**, yang sedikit lebih tinggi dari rata-rata, mengindikasikan kemungkinan adanya lebih banyak film yang dirilis pada paruh kedua periode waktu dalam dataset.\n",
    "\n",
    "**⭐ Rating Pengguna (`users_rating`)**\n",
    "- Jumlah data rating pengguna adalah **1272**, namun sebelumnya disebutkan 6144, kemungkinan terjadi kesalahan atau merujuk pada total skor atau vote dari sumber eksternal (perlu klarifikasi).\n",
    "- Rata-rata rating pengguna adalah sekitar **6.14**.\n",
    "- Sebagian besar rating pengguna berada di antara **5.3** (kuartil pertama) dan **7.1** (kuartil ketiga).\n",
    "- Rentang rating cukup luas, dari **1.2 hingga 9.4**, menunjukkan variasi preferensi pengguna yang signifikan.\n",
    "- Median rating adalah **6.4**, yang sedikit lebih tinggi dari rata-rata, mengindikasikan distribusi rating yang mungkin sedikit condong ke nilai yang lebih tinggi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "fc26b91e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Statistik Deskriptif (Kolom Kategorik) Movies Dataset:\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>description</th>\n",
       "      <th>genre</th>\n",
       "      <th>rating</th>\n",
       "      <th>votes</th>\n",
       "      <th>languages</th>\n",
       "      <th>directors</th>\n",
       "      <th>actors</th>\n",
       "      <th>runtime</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1272</td>\n",
       "      <td>840</td>\n",
       "      <td>1236</td>\n",
       "      <td>376</td>\n",
       "      <td>1272</td>\n",
       "      <td>1272</td>\n",
       "      <td>1265</td>\n",
       "      <td>1272</td>\n",
       "      <td>869</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>1262</td>\n",
       "      <td>840</td>\n",
       "      <td>15</td>\n",
       "      <td>11</td>\n",
       "      <td>312</td>\n",
       "      <td>8</td>\n",
       "      <td>377</td>\n",
       "      <td>1266</td>\n",
       "      <td>85</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>Kuntilanak 2</td>\n",
       "      <td>It tells the story of an Indonesian revolution...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>13+</td>\n",
       "      <td>6</td>\n",
       "      <td>Indonesian</td>\n",
       "      <td>Nayato Fio Nuala</td>\n",
       "      <td>[nan, nan, nan, nan, nan, nan, nan, nan, nan, ...</td>\n",
       "      <td>90 min</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>456</td>\n",
       "      <td>161</td>\n",
       "      <td>58</td>\n",
       "      <td>1241</td>\n",
       "      <td>61</td>\n",
       "      <td>4</td>\n",
       "      <td>109</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               title                                        description  \\\n",
       "count           1272                                                840   \n",
       "unique          1262                                                840   \n",
       "top     Kuntilanak 2  It tells the story of an Indonesian revolution...   \n",
       "freq               2                                                  1   \n",
       "\n",
       "        genre rating votes   languages         directors  \\\n",
       "count    1236    376  1272        1272              1265   \n",
       "unique     15     11   312           8               377   \n",
       "top     Drama    13+     6  Indonesian  Nayato Fio Nuala   \n",
       "freq      456    161    58        1241                61   \n",
       "\n",
       "                                                   actors runtime  \n",
       "count                                                1272     869  \n",
       "unique                                               1266      85  \n",
       "top     [nan, nan, nan, nan, nan, nan, nan, nan, nan, ...  90 min  \n",
       "freq                                                    4     109  "
      ]
     },
     "execution_count": 103,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Statistik Deskriptif untuk Kolom Kategorikal\n",
    "print('\\nStatistik Deskriptif (Kolom Kategorik) Movies Dataset:')\n",
    "df.describe(include=['object'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "17ccba0a",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "**🎞️ Judul (`title`)**\n",
    "- Terdapat **1272** judul film dalam dataset.\n",
    "- Terdapat **1262** judul yang unik.\n",
    "- Judul **\"Kuntilanak 2\"** muncul paling sering (**frekuensi 2**), yang bisa mengindikasikan adanya sekuel atau kemungkinan duplikasi/kesalahan data.\n",
    "\n",
    "**📝 Deskripsi (`description`)**\n",
    "- Terdapat **840** deskripsi yang tersedia, semuanya bersifat **unik**.\n",
    "- Deskripsi yang muncul pertama secara statistik adalah:  \n",
    "  _\"It tells the story of an Indonesian revolution...\"_\n",
    "\n",
    "**🎭 Genre (`genre`)**\n",
    "- Dataset mencakup **15** genre yang berbeda.\n",
    "- Genre paling dominan adalah **\"Drama\"** dengan **frekuensi 456**.\n",
    "\n",
    "**🔞 Rating Usia (`rating`)**\n",
    "- Terdapat **11** kategori rating usia yang berbeda.\n",
    "- Rating **\"13+\"** adalah yang paling sering muncul, sebanyak **161** kali.\n",
    "\n",
    "**🌐 Bahasa (`languages`)**\n",
    "- Film dalam dataset menggunakan **8** bahasa yang berbeda.\n",
    "- Bahasa **\"Indonesian\"** adalah yang paling umum digunakan, muncul pada **1241** film.\n",
    "\n",
    "**🎬 Sutradara (`directors`)**\n",
    "- Terdapat **377** nama sutradara yang berbeda.\n",
    "- Sutradara dengan jumlah film terbanyak adalah **Nayato Fio Nuala** (**frekuensi 61**).\n",
    "\n",
    "**👥 Aktor (`actors`)**\n",
    "- Terdapat **1272** data aktor, namun ada **nilai-nilai tidak lengkap atau tidak valid**.\n",
    "- Kombinasi aktor **\"[nan, nan, nan, nan, nan]\"** muncul paling sering (**frekuensi 4**), menunjukkan kemungkinan masalah dalam pengisian data.\n",
    "\n",
    "**⏱️ Durasi (`runtime`)**\n",
    "- Terdapat **85** nilai durasi yang unik.\n",
    "- Durasi yang paling umum adalah **\"90 min\"**, muncul sebanyak **109** kali."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "d4a379ad",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jumlah Duplikasi: 0\n"
     ]
    }
   ],
   "source": [
    "# Mengecek Duplikasi Data\n",
    "print('Jumlah Duplikasi:', df.duplicated().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1cc1bbb3",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "Tidak terdapat Nilai atau Data yang Terduplikasi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "e43b3704",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Jumlah nilai hilang per kolom:\n",
      "title             0\n",
      "year              0\n",
      "description     432\n",
      "genre            36\n",
      "rating          896\n",
      "users_rating      0\n",
      "votes             0\n",
      "languages         0\n",
      "directors         7\n",
      "actors            0\n",
      "runtime         403\n",
      "dtype: int64\n",
      "\n",
      "Persentase nilai hilang per kolom:\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "title            0.000000\n",
       "year             0.000000\n",
       "description     33.962264\n",
       "genre            2.830189\n",
       "rating          70.440252\n",
       "users_rating     0.000000\n",
       "votes            0.000000\n",
       "languages        0.000000\n",
       "directors        0.550314\n",
       "actors           0.000000\n",
       "runtime         31.682390\n",
       "dtype: float64"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Mengecek kolom apa saja yang nilainya hilang\n",
    "missing_values = df.isnull().sum()\n",
    "print(\"\\nJumlah nilai hilang per kolom:\")\n",
    "print(missing_values)\n",
    "\n",
    "# Untuk presetanse missing valuenya\n",
    "total_rows = df.shape[0]\n",
    "missing_percentage = (missing_values / total_rows) * 100\n",
    "print(\"\\nPersentase nilai hilang per kolom:\")\n",
    "missing_percentage"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19a4652f",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "Kolom **description, genre, rating, directors, dan runtime** memiliki *missing values*."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "77ff9c02",
   "metadata": {},
   "source": [
    "### Univariate Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b10455f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Distribusi Genre:\n",
      "           jumlah sampel  persentase\n",
      "genre                               \n",
      "Drama                456        36.9\n",
      "Comedy               287        23.2\n",
      "Horror               231        18.7\n",
      "Action               132        10.7\n",
      "Adventure             49         4.0\n",
      "Biography             28         2.3\n",
      "Thriller              12         1.0\n",
      "Romance               11         0.9\n",
      "Fantasy               10         0.8\n",
      "Crime                  9         0.7\n",
      "Animation              4         0.3\n",
      "Family                 2         0.2\n",
      "Sci-Fi                 2         0.2\n",
      "War                    2         0.2\n",
      "History                1         0.1\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Distribus Genre\n",
    "feature = 'genre'\n",
    "genre_counts = df[feature].value_counts()\n",
    "genre_percent = 100 * df[feature].value_counts(normalize=True)\n",
    "df_genre = pd.DataFrame({'jumlah sampel': genre_counts, 'persentase': genre_percent.round(1)})\n",
    "\n",
    "print(\"Distribusi Genre:\")\n",
    "print(df_genre)\n",
    "\n",
    "# Visualisasi\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, y=feature, order=df[feature].value_counts().index, palette=\"viridis\")\n",
    "plt.title(f'Distribusi {feature.capitalize()}') \n",
    "plt.ylabel(feature.capitalize())\n",
    "plt.xlabel('Jumlah')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4d15792d",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "**🎭 Dominasi Genre Drama**\n",
    "- Genre **\"Drama\"** memiliki jumlah sampel terbanyak secara signifikan, yaitu **456 film**, yang mencakup sekitar **36.9%** dari total data genre yang tersedia.\n",
    "- Hal ini menunjukkan bahwa **drama merupakan genre yang sangat dominan** dalam dataset ini.\n",
    "\n",
    "**😂😱 Popularitas Comedy dan Horror**\n",
    "- Genre **\"Comedy\"** dan **\"Horror\"** juga cukup populer:\n",
    "  - **Comedy**: 287 film (**23.2%**)\n",
    "  - **Horror**: 231 film (**18.7%**)\n",
    "- Kedua genre ini menempati posisi setelah Drama dan jauh lebih tinggi dibandingkan genre-genre lainnya.\n",
    "\n",
    "**🔫 Action sebagai Genre Signifikan**\n",
    "- Genre **\"Action\"** berada di posisi keempat dengan **132 film** (**10.7%**).\n",
    "- Ini menunjukkan bahwa genre aksi juga cukup banyak diwakili dalam dataset.\n",
    "\n",
    "**🌍 Genre dengan Representasi Sedang**\n",
    "- Genre dengan jumlah sampel menengah:\n",
    "  - **Adventure**: 49 film (**4.0%**)\n",
    "  - **Biography**: 28 film (**2.3%**)\n",
    "- Meskipun tidak sebanyak empat genre teratas, keduanya memiliki representasi yang lebih tinggi dibandingkan genre-genre minor.\n",
    "\n",
    "**⚠️ Minimnya Representasi Beberapa Genre**\n",
    "- Beberapa genre memiliki representasi yang sangat kecil dalam dataset:\n",
    "  - **Thriller**, **Romance**, **Fantasy**, **Crime**, **Animation**, **Family**, **Sci-Fi**, **War**, dan **History** semuanya berada di bawah atau sedikit di atas **1%** dari total data genre.\n",
    "- Hal ini menunjukkan bahwa genre-genre tersebut **kurang dominan** atau **jarang muncul** dalam koleksi film pada dataset ini."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "e2c6678f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Distribusi Tahun Rilis Film:\n",
      "      jumlah sampel  persentase\n",
      "year                           \n",
      "2019            111         8.7\n",
      "2018             97         7.6\n",
      "2009             79         6.2\n",
      "2011             78         6.1\n",
      "2008             77         6.1\n",
      "...             ...         ...\n",
      "1953              1         0.1\n",
      "1951              1         0.1\n",
      "1950              1         0.1\n",
      "1928              1         0.1\n",
      "1926              1         0.1\n",
      "\n",
      "[62 rows x 2 columns]\n"
     ]
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Distribusi Tahun Rilis Film\n",
    "feature = 'year'\n",
    "year_counts = df[feature].value_counts() \n",
    "year_percent = 100 * df[feature].value_counts(normalize=True) \n",
    "df_year_sorted = pd.DataFrame({'jumlah sampel': year_counts, 'persentase': year_percent.round(1)}).sort_values(by='jumlah sampel', ascending=False)\n",
    "\n",
    "print(\"Distribusi Tahun Rilis Film:\")\n",
    "print(df_year_sorted)\n",
    "\n",
    "# Visualisasi\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(data=df, x=feature, bins=30, kde=True)\n",
    "plt.title(f'Distribusi {feature.capitalize()} Rilis Film')\n",
    "plt.xlabel(feature.capitalize())\n",
    "plt.ylabel('Jumlah Film')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4dfc5218",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "**📈 Konsentrasi Rilis Film Terbaru**\n",
    "- Sebagian besar film dalam dataset dirilis dalam beberapa tahun terakhir.\n",
    "- Tahun dengan jumlah rilis terbanyak:\n",
    "  - **2019**: 111 film (**8.7%**)\n",
    "  - **2018**: 97 film (**7.6%**)\n",
    "  - **2009**: 79 film (**6.2%**)\n",
    "  - **2011**: 78 film (**6.1%**)\n",
    "  - **2008**: 77 film (**6.1%**)\n",
    "\n",
    "**🔼 Peningkatan Jumlah Rilis Seiring Waktu**\n",
    "- Visualisasi histogram menunjukkan adanya tren **peningkatan jumlah film** yang dirilis dari waktu ke waktu.\n",
    "- **Lonjakan signifikan** terjadi setelah tahun **2000**, menandakan era kebangkitan produksi film yang lebih intensif.\n",
    "\n",
    "**🏔️ Puncak Rilis di Era Modern**\n",
    "- Distribusi memuncak di sekitar tahun **2010-an**, sesuai dengan frekuensi data rilis tertinggi dalam tabel.\n",
    "- Ini menunjukkan bahwa **dekade 2010-an adalah era paling produktif** dalam dataset ini.\n",
    "\n",
    "**📉 Rilis Film Awal yang Jarang**\n",
    "- Film-film yang dirilis pada paruh pertama abad ke-20 sangat sedikit.\n",
    "- Beberapa tahun seperti **1953**, **1951**, **1950**, **1928**, dan **1926** hanya memiliki **1 film** masing-masing (**0.1%** dari total data), menunjukkan keterbatasan arsip atau cakupan data pada era tersebut.\n",
    "\n",
    "**⚖️ Distribusi yang Tidak Merata**\n",
    "- Secara umum, **distribusi tahun rilis sangat tidak merata**.\n",
    "- Konsentrasi rilis **tertinggi** terjadi dalam **dua dekade terakhir**, dengan jumlah film jauh lebih banyak dibandingkan periode sebelumnya.\n",
    "\n",
    "**🧭 Adanya Periode dengan Jumlah Rilis Sedang**\n",
    "- Sekitar tahun **1980-an dan 1990-an** terdapat **peningkatan sedang** dalam jumlah film, lebih tinggi dibandingkan era sebelum 1970-an, namun masih rendah dibandingkan era modern."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "c7e34da6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5 Rating Film Teratas:\n",
      "              jumlah sampel  persentase\n",
      "users_rating                           \n",
      "7.2                      54         4.2\n",
      "6.2                      52         4.1\n",
      "6.6                      50         3.9\n",
      "7.0                      48         3.8\n",
      "6.8                      45         3.5\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA90AAAJOCAYAAACqS2TfAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAABDiklEQVR4nO3dCbhd0/0//pXIRPRGEmOIoaVFQyhRQ9UUVDVFUTU0IVRpFFFFqJpaKaWGnxpCGrSGmqKoKeYpKqbv16xKJYixMqCC5P6fz/o+5/7vvbmZbu7KTe55vZ7nPLlnn73PWWfvdU7Oe69ht6utra1NAAAAQItr3/JPCQAAAAShGwAAAAoRugEAAKAQoRsAAAAKEboBAACgEKEbAAAAChG6AQAAoBChGwAAAAoRugEAAKAQoRtgEbblllvmG9XtxBNPTO3atUsLq3//+9+5fGeccUZrF2WhEPsijtm8uvTSS/O2jz/+eJFyAVCG0A1Qz3333Zd/1DZ1e/TRR+e4/b777ttgm86dO6evfvWr6de//nX69NNPm1Wm559/Pv9Aj+CyMAW8Od0W1MmA1tw/jY93hw4dUu/evdOPfvSjXC4WTnHcllxyyVRtGn92l1hiibT22munX/3qV2nKlCmtXTyANqtDaxcAYGF06KGHpn79+jVYtvrqq8/VthG0L7nkkvz35MmT09/+9rd0yimnpH/961/piiuumOeyRHg76aSTcohdddVVGzx25513pgXtBz/4QYN98dFHH6WDDz447bLLLvmxiuWWW26BlGd2+2dBqH+8v/jii3ycL7zwwnT77bfnsvXq1WuBlwlm54ILLsgnHeKzG98hv/3tb9M999yTHn744YW6xwTAokroBmjC5ptvnnbbbbdmbRutnfvss0/d/Z/97Gdp0003TVdddVX6wx/+0KJhtFOnTmlBW3fddfOt4v3338+hO5bVf9/N9fHHH6euXbumRUXj4x023njj9L3vfS/9/e9/Tz/5yU/m+zUizM+YMaNVjvf8iDJ/9tlnrV0MGonvtqWXXjr/fdBBB6Vdd9013XDDDbk3zyabbNLaxQNoc3QvB5iFqVOn5rAzv6Ll6Fvf+laqra1Nr776at3y119/PQfyr33ta2nxxRdPPXv2TLvvvnuDbtIxhjOWha222qquW2h0g29qTHele/w111yTW69WWmml1KVLl7TNNtukV155Zaay/fGPf0xf/vKX8+tvtNFG6cEHH2yxceIvvvhi/nHfo0ePXIYNN9ww3XTTTU2OUb3//vvzvlh22WVzmVtq/0Qvgx133DG3NkeL9Fe+8pXc62D69OkNyvHPf/4zB4/ll18+lzXKEF3Eo6dCc8TzVAJ5fZMmTUqHH3547oIe5YkeA6eddloOp02Nfz777LNzmWPdSnf1hx56KPfCiHLGYxdddFGTZRg1alTaeuut8z6N7aMbcbRwNhbjg7fffvscwmI/r7baamnw4MEN1omyxImjOAaxzgYbbJCuu+66mZ4ryn3IIYfkHh1f//rX8+tGi39T4vNw4IEH5hMJEfjC//7v/+au31En4/3FfoyyfPDBB012k446HesvtdRSqVu3bmm//fZLn3zySWrJcdbReyJeo3GdjeMQPWKWWWaZ/Po//elP8wmGOMYDBw5M3bt3z7ejjjoqv9fZmZu6Xt+0adPSEUcckV87TlBFL5P33nsvNVfUk/Daa6/lf6M+Rt2LYxjHIU4Uxvv78MMPZ9o3cXIp9kV8f8S6cewuv/zymV4jju0WW2yR3198vn7zm9/kOhr7sv77jNeO4xCf2ej+Hp/rqPuNj8Os5jGoHJ/6zzm35fzPf/6TjjzyyLTOOuvkngA1NTVphx12SP/zP//T7H0LELR0AzQhfrxH18vFFlsst3r//ve/z6GxuSo/AONHeMW4cePSI488ksNd/AiNdSIUReCNH5nxg/Pb3/52/mF/7rnnpmOPPTattdZaedvKv7Pyu9/9LrVv3z7/gIzgePrpp6e99947/eMf/6hbJ14rAlK8v6FDh+bX33nnnXMZK8G3uZ577rm02WabpRVXXDEdc8wxORjEiYB4/uuvvz6HhPoicESAiLHv0dLdUvsnfoDHj+cIKPFvdKGN14jxq3FMQwSlCJ0RZH7+85/noPfmm2+mW265JQeoCHNzEq39IcJ8nFg5+uijc3CKH/oVEQYjdMRzR4BZeeWV8/sbNmxYmjhxYg459UUgiXkAIphGeI2TF88880zabrvt8r6K0BEnhU444YQme0/EvorQ9P3vfz+H/5tvvjnv5wg1Q4YMyeu8++67dc8XxynCY+znSgiuOOecc/LzRB2K/XX11VfnUBj7KE5q1Bf7OI511K0I8k11+Y/9FGH6r3/9axo9enTdc4wZMybvv/j8xXGIejRixIj8b7TCNg5ZP/zhD/NJguHDh6cnn3wyd/OPkwxxIqO0Sl2JoQ1Rtihn7L84pnFsTz311HTrrbfmetanT58cxGdlbup649eOz2kc+1g36k7s79ifzRFDIkLU2RD1Mz47cRzi8xVh/LzzzktPPfVU7oLesWPHum3jxEecXNt///3ToEGD0p/+9KccjuPETNS/EHW+clIs6nt8H8SxinrdWDwe31cDBgzIn8sIvPFvc+fEmJdyRt278cYbc92OevXOO+/kk1rxuTVUBJgvtQDUefjhh2t33XXX2pEjR9b+7W9/qx0+fHhtz549a7t06VL75JNPznH7QYMG1Xbt2rX2vffey7dXXnml9owzzqht165dbZ8+fWpnzJhRt+4nn3wy0/Zjx46NJrHayy+/vG7Ztddem5fde++9M62/xRZb5FtFrBPrrrXWWrXTpk2rW37OOefk5c8880y+H4/F++rXr1/t559/XrfepZdemter/5xzEu8ztjnhhBPqlm2zzTa166yzTu2nn35atyze+6abblq7xhpr1C0bNWpU3vZb3/pW7RdffNHgeVti/zT1HD/96U9rl1hiibqyPfXUU3n7eJ55Fcc7tm18W3HFFWufeOKJBuuecsopuW68/PLLDZYfc8wxtYsttljt+PHj8/3XXnstP0dNTU3tu+++22DdnXfeOdfF119/vW7Z888/n7dv/F96U+99++23r/3yl79cd3/06NF5u3Hjxs32fTZ+rs8++yzX56233rrB8niu9u3b1z733HMNllfe0+9///tc3/bYY4/axRdfvPaOO+6YY5mvuuqqvO0DDzxQtyzqWiwbPHhwg3V32WWXXK/n9nPauOz163DFKqusktdvXGdjX9b/PG+yySb5c37QQQfVLYs6vdJKK830eWr8WnNb1yuv3b9//wavPXTo0FwHJk2aNNv3XdlvL730Uv7cxnG56KKLajt37ly73HLL1X788ce1Dz74YF7niiuuaLDt7bffPtPy2DeNj03U2Xi+X/ziF3XLfv7zn+d9E5+1ig8++KC2R48eefsoR3j77bdrO3TokOt5fSeeeGJer/5xqLyXxir7qPKc81LO+E6YPn16g+eL54n1Tj755NnuW4DZ0b0coJ7oQhvdZqMVLlr2ovWv0sIWLTBzI1pqo+UwbtF9OFqbo9U3ujrXb6mLbpYVn3/+ee5C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f/jhh2mzzTZLHTt2TLfddlt6/vnn05lnnpm6d+8+y20eeOCBHLpvvfXW9MQTT6StttoqDRgwID311FMLtOwAAAAwJx1SKzrttNNS796906hRo+qWrbbaarPd5uyzz25w/9RTT01/+9vf0s0335zWX3/9YmUFAACARaql+6abbkobbrhh2n333dOyyy6bQ/PFF188T88xY8aMNHXq1NSjR49i5QQAAIBFLnS/+uqr6YILLkhrrLFGHpd98MEHp0MPPTRddtllc/0cZ5xxRvroo4/SD3/4w1muM23atDRlypQGNwAAAGjT3cujlTpauqOLeIiW7meffTZdeOGFadCgQXPc/sorr0wnnXRS7l4eLeWzMnz48LweAAAAVE1L9worrJDWXnvtBsvWWmutNH78+Dlue/XVV6cDDjggXXPNNal///6zXXfYsGFp8uTJdbcJEybMd9kBAABgoW7pjpnLX3rppQbLXn755bTKKqvMdrurrroqDR48OAfvHXfccY6v07lz53wDAACAqmnpHjp0aHr00Udz9/JXXnkldxcfMWJEGjJkSINW6oEDB9bdj3Xiflxa7Jvf/GZ6++238y1asAEAAGBh0qqhu1+/fmn06NG55bpPnz7plFNOyZcE23vvvevWmThxYoPu5hHKv/jiixzMo3t65XbYYYe10rsAAACAprWrra2tTVUmZi/v1q1b2njfY1OHTl1auzgAAAALhQcvOr61i7DI5crodV1TU7NwtnQDAABAWyZ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhHVIVu+Oco1NNTU1rFwMAAIA2Sks3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFNIhVbHNTh+eFuvSubWLAQAAkJ7+1YmtXQQWltC9yy67pHbt2s20PJZ16dIlrb766mmvvfZKX/va11qijAAAAFA93cu7deuW7rnnnvTkk0/moB23p556Ki/74osv0l//+tfUt2/f9PDDD7d8iQEAAKAtt3Qvv/zyuSX7vPPOS+3b/19unzFjRjrssMPSl770pXT11Vengw46KB199NHpoYceaukyAwAAQNtt6R45cmQ6/PDD6wJ3fqL27dPPf/7zNGLEiNzyfcghh6Rnn322JcsKAAAAbT90RxfyF198cablsWz69On57xjb3dS4bwAAAKgWzepe/uMf/zjtv//+6dhjj039+vXLy8aNG5dOPfXUNHDgwHz//vvvT1//+tdbtrQAAADQ1kP3WWedlZZbbrl0+umnp3feeScvi/tDhw7N47jDdtttl77zne+0bGkBAACgrYfuxRZbLB133HH5NmXKlLyspqamwTorr7xyy5QQAAAAqil019c4bAMAAADzMZFadCmPcd29evVKHTp0yC3f9W8AAABAM1u699133zR+/Ph0/PHHpxVWWMEs5QAAANBSofuhhx5KDz74YFpvvfWaszkAAABUhWZ1L+/du3eqra1t+dIAAABAtYfus88+Ox1zzDHp3//+d8uXCAAAAKq5e/kee+yRPvnkk/SVr3wlLbHEEqljx44NHv/Pf/7TUuUDAACA6grd0dINAAAAFAjdgwYNas5mAAAAUFXmOnRPmTIl1dTU1P09O5X1AAAAoJrNdeju3r17mjhxYlp22WXTUkst1eS1uWNG81g+ffr0li4nAAAAtN3Qfc8996QePXrkv++9996SZQIAAIDqCt1bbLFF3d+rrbZavlZ349buaOmeMGFCy5YQAAAAquk63RG633vvvZmWx6XC4jEAAACgmaG7Mna7sY8++ih16dKlJcoFAAAA1XXJsCOOOCL/G4H7+OOPT0sssUTdYzF52j/+8Y+03nrrtXwpAQAAoK2H7qeeeqqupfuZZ55JnTp1qnss/u7bt2868sgjW76UAAAA0NZDd2XW8v322y+dc845rscNAAAALRW6K0aNGtWczQAAAKCqNCt0h8cffzxdc801afz48emzzz5r8NgNN9zQEmUDAACA6pu9/Oqrr06bbrppeuGFF9Lo0aPT559/np577rl0zz33pG7durV8KQEAAKBaQvepp56azjrrrHTzzTfnCdRifPeLL76YfvjDH6aVV1655UsJAAAA1RK6//Wvf6Udd9wx/x2h++OPP86XERs6dGgaMWJES5cRAAAAqid0d+/ePU2dOjX/veKKK6Znn302/z1p0qT0ySeftGwJAQAAoJomUvv2t7+dxowZk9ZZZ520++67p8MOOyyP545l22yzTcuXEgAAAKoldJ933nnp008/zX8fd9xxqWPHjumRRx5Ju+66a/rVr37V0mUEAACA6gjdX3zxRbrlllvS9ttvn++3b98+HXPMMSXKBgAAANU1prtDhw7poIMOqmvpnl9vvvlm2meffVLPnj3T4osvnrusxzXAZ2fatGm5hX2VVVZJnTt3Tquuumr605/+1CLlAQAAgFbtXr7RRhulp59+Oofe+fHhhx+mzTbbLG211VbptttuS8sss0z65z//mSdqm524NNk777yTRo4cmVZfffU0ceLENGPGjPkqCwAAACwUoftnP/tZOuKII9KECRPSBhtskLp27drg8XXXXXeunue0005LvXv3TqNGjapbttpqq812m9tvvz3df//96dVXX009evTIy6KlGwAAABY27Wpra2vndaMYxz3TE7Vrl+Kp4t/p06fP1fOsvfbaeWz4G2+8kYN0XH4sAv1PfvKTWW4Tj7/88stpww03TH/+859z4P/+97+fTjnllNw9fW5MmTIldevWLfU57pi0WJfOc7UNAABASU//6sTWLgLzoJIrJ0+enGpqalq2pfu1115LLSFaqy+44ILcan7sscemcePGpUMPPTR16tQpDRo0aJbbPPTQQ6lLly5p9OjR6f33389B/IMPPmjQYt54DHjc6u8cAAAAKK1ZoXtux3LvuOOO6ZJLLkkrrLBCk4/HOOxosT711FPz/fXXXz89++yz6cILL5xl6I5tojX9iiuuyGcVwh/+8Ie02267pfPPP7/J1u7hw4enk046aR7eIQAAALTC7OXz4oEHHkj//e9/Z/l4hPHoYl7fWmutlcaPHz/bbaIbeiVwV7aJru3RTb0pw4YNy03+lVuMRQcAAIBFOnTPScxc/tJLLzVYFuO1Z9eSHtu89dZb6aOPPmqwTYwzX2mllZrcJi4rFn3s698AAACgTYfuoUOHpkcffTR3L3/llVfSlVdemUaMGJGGDBnSoJV64MCBdff32muvfE3v/fbbLz3//PO5Nf2Xv/xlGjx48FxPpAYAAABtPnT369cvT4Z21VVXpT59+uQZyM8+++y09957160T1+Cu3918ySWXTGPGjEmTJk3K48Fj3QEDBqRzzz23ld4FAAAAtOBEai3pe9/7Xr7NyqWXXjrTsjXXXDMHbwAAAFiYtWpLNwAAALRlRUN3XHu7R48eJV8CAAAA2lbovuyyy9Lf//73uvtHHXVUWmqppdKmm26aXn/99QaToMVyAAAAqEbNCt0x23hlpvCxY8emP/7xj+n0009PSy+9dJ6RHAAAAGjmRGoTJkxIq6++ev77xhtvTLvuums68MAD8zW0t9xyy5YuIwAAAFRPS3dctuuDDz7If995551p2223zX936dIl/fe//23ZEgIAAEA1tXRHyD7ggAPS+uuvn15++eX03e9+Ny9/7rnn0qqrrtrSZQQAAIDqaemOMdwxadp7772Xrr/++tSzZ8+8/Iknnkh77rlnS5cRAAAAqqOl+4svvkjnnntuOvroo9NKK63U4LGTTjqpJcsGAAAA1dXS3aFDhzxTeYRvAAAAoIW7l2+zzTbp/vvvb86mAAAAUDWaNZHaDjvskI455pj0zDPPpA022CB17dq1wePf//73W6p8AAAAUF2h+2c/+1n+9w9/+MNMj7Vr1y5Nnz59/ksGAAAA1Ri6Z8yY0fIlAQAAgDamWWO66/v0009bpiQAAADQxjQrdEf38VNOOSWtuOKKackll0yvvvpqXn788cenkSNHtnQZAQAAoHpC929/+9t06aWX5kuHderUqW55nz590iWXXNKS5QMAAIDqCt2XX355GjFiRNp7773TYostVre8b9++6cUXX2zJ8gEAAEB1he4333wzrb766k1OsPb555+3RLkAAACgOkP32muvnR588MGZll933XVp/fXXb4lyAQAAQHVeMuzXv/51GjRoUG7xjtbtG264Ib300ku52/ktt9zS8qUEAACAamnp3mmnndLNN9+c7rrrrtS1a9ccwl944YW8bNttt235UgIAAEC1tHSHzTffPI0ZM6ZlSwMAAADV3tI9YcKE9MYbb9Tdf+yxx9Lhhx+eZzQHAAAA5iN077XXXunee+/Nf7/99tupf//+OXgfd9xx6eSTT27OUwIAAECb06zQ/eyzz6aNNtoo/33NNdekddZZ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bgAAAChE6AYAAIBChG4AAAAoROgGAACAQoRuAAAAKEToBgAAgEKEbgAAAChE6AYAAIBChG4AAAAoROgGAACAQoRuAAAAKEToBgAAgEKEbgAAAChE6AYAAIBChG4AAAAoROgGAACAQoRuAAAAKEToBgAAgLYaut988820zz77pJ49e6bFF188rbPOOunxxx+f7TZXXHFF6tu3b1piiSXSCiuskAYPHpw++OCDBVZmAAAAWOhD94cffpg222yz1LFjx3Tbbbel559/Pp155pmpe/fus9zm4YcfTgMHDkz7779/eu6559K1116bHnvssfSTn/xkgZYdAAAA5qRDakWnnXZa6t27dxo1alTdstVWW22224wdOzatuuqq6dBDD61b/6c//Wl+LgAAAFiYtGpL90033ZQ23HDDtPvuu6dll102rb/++uniiy+e7TabbLJJmjBhQrr11ltTbW1teuedd9J1112Xvvvd785ym2nTpqUpU6Y0uAEAAECbDt2vvvpquuCCC9Iaa6yR7rjjjnTwwQfnFuzLLrtslttEd/QY073HHnukTp06peWXXz5169Yt/fGPf5zlNsOHD8/rVG7Rug4AAACltauN5uJWEqE5WrofeeSRumURuseNG5e7kTclxn33798/DR06NG2//fZp4sSJ6Ze//GXq169fGjly5CxbuuNWES3dEbyH37FL6tK1Y4F3BgAA1eXwzf7a2kWABSpyZTTqTp48OdXU1CycY7pj5vG11167wbK11lorXX/99bNttY7W7gjaYd11101du3ZNm2++efrNb36Tn7Oxzp075xsAAABUTffyCM8vvfRSg2Uvv/xyWmWVVWa5zSeffJLat29Y7MUWWyz/24qN9gAAALBwhe7oIv7oo4+mU089Nb3yyivpyiuvTCNGjEhDhgypW2fYsGH5EmEVAwYMSDfccEMeCx5jwuMSYtElfaONNkq9evVqpXcCAAAAC1n38hiHPXr06BysTz755Hz5r7PPPjvtvffedevEmO3x48fX3d93333T1KlT03nnnZd+8YtfpKWWWiptvfXWLhkGAADAQqdVJ1Jr7QHvJlIDAICWYSI1qs2UuZxIrVW7lwMAAEBbJnQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCFCNwAAABQidAMAAEAhQjcAAAAUInQDAABAIUI3AAAAFCJ0AwAAQCEdUhX72caXppqamtYuBgAAAG2Ulm4AAAAoROgGAACAQoRuAAAAKEToBgAAgEKEbgAAAChE6AYAAIBChG4AAAAoROgGAACAQjqkKlRbW5v/nTJlSmsXBQAAgEVQJU9W8uWsVGXo/uCDD/K/vXv3bu2iAAAAsAibOnVq6tat2ywfr8rQ3aNHj/zv+PHjZ7tzoPSZsTjxM2HChFRTU9PaxaFKqYe0NnWQhYF6yMJAPVz0RAt3BO5evXrNdr2qDN3t2//fUPYI3Co0rS3qoHpIa1MPaW3qIAsD9ZCFgXq4aJmbRlwTqQEAAEAhQjcAAAAUUpWhu3PnzumEE07I/0JrUQ9ZGKiHtDZ1kIWBesjCQD1su9rVzml+cwAAAKBZqrKlGwAAABYEoRsAAAAKEboBAACgkKoL3X/84x/Tqquumrp06ZK++c1vpscee6y1i0Qb9sADD6QBAwakXr16pXbt2qUbb7yxweMxpcKvf/3rtMIKK6TFF1889e/fP/3zn/9stfLSNg0fPjz169cvfelLX0rLLrts2nnnndNLL73UYJ1PP/00DRkyJPXs2TMtueSSadddd03vvPNOq5WZtueCCy5I6667bt31ZzfZZJN022231T2uDrKg/e53v8v/Nx9++OF1y9RDFoQTTzwx1736tzXXXLPucfWw7amq0P3Xv/41HXHEEXlWwCeffDL17ds3bb/99undd99t7aLRRn388ce5nsXJnqacfvrp6dxzz00XXnhh+sc//pG6du2a62R82UJLuf/++/N/3o8++mgaM2ZM+vzzz9N2222X62fF0KFD080335yuvfbavP5bb72VfvCDH7RquWlbVlpppRxynnjiifT444+nrbfeOu20007pueeey4+rgyxI48aNSxdddFE+EVSfesiC8vWvfz1NnDix7vbQQw/VPaYetkG1VWSjjTaqHTJkSN396dOn1/bq1at2+PDhrVouqkN83EaPHl13f8aMGbXLL7987e9///u6ZZMmTart3Llz7VVXXdVKpaQavPvuu7k+3n///XX1rmPHjrXXXntt3TovvPBCXmfs2LGtWFLauu7du9decskl6iAL1NSpU2vXWGON2jFjxtRuscUWtYcddlherh6yoJxwwgm1ffv2bfIx9bBtqpqW7s8++yyfXY/uuxXt27fP98eOHduqZaM6vfbaa+ntt99uUCe7deuWhz2ok5Q0efLk/G+PHj3yv/HdGK3f9etidHNbeeWV1UWKmD59err66qtzb4voZq4OsiBFz58dd9yxQX0L6iELUgwnjOGHX/7yl9Pee++dxo8fn5erh21Th1Ql3n///fyf/HLLLddgedx/8cUXW61cVK8I3KGpOll5DFrajBkz8vjFzTbbLPXp0ycvi/rWqVOntNRSSzVYV12kpT3zzDM5ZMcQmhinOHr06LT22munp59+Wh1kgYiTPTHEMLqXN+a7kAUlGlguvfTS9LWvfS13LT/ppJPS5ptvnp599ln1sI2qmtANwP+18MR/6vXHjsGCEj8wI2BHb4vrrrsuDRo0KI9XhAVhwoQJ6bDDDstzW8SEutBadthhh7q/Y16BCOGrrLJKuuaaa/LEurQ9VdO9fOmll06LLbbYTDP/xf3ll1++1cpF9arUO3WSBeWQQw5Jt9xyS7r33nvzpFYVUd9iCM6kSZMarK8u0tKi9Wb11VdPG2ywQZ5VPyaaPOecc9RBFojothuT537jG99IHTp0yLc46RMTmsbf0ZKoHtIaolX7q1/9anrllVd8H7ZR7avpP/r4T/7uu+9u0M0y7kdXN1jQVltttfzlWb9OTpkyJc9irk7SkmIevwjc0ZX3nnvuyXWvvvhu7NixY4O6GJcUi/Fl6iIlxf/D06ZNUwdZILbZZps8xCF6W1RuG264YR5PW/lbPaQ1fPTRR+lf//pXvoSs78O2qaq6l8flwqIrW3ypbrTRRunss8/Ok7jst99+rV002vCXaJy1rD95WvzHHhNYxYQYMbb2N7/5TVpjjTVyEDr++OPzpBpxHWVoyS7lV155Zfrb3/6Wr9VdGRMWE/dFN7b4d//998/fkVE34xrKP//5z/N/7htvvHFrF582YtiwYblLZXz3TZ06NdfJ++67L91xxx3qIAtEfP9V5rKoiEt1xrWQK8vVQxaEI488Mg0YMCB3KY/LgcXljKNH7p577un7sI2qqtC9xx57pPfeey/9+te/zj8611tvvXT77bfPNJEVtJS4Fu1WW21Vdz++QEOc/IkJNI466qh84ufAAw/M3Yi+9a1v5TpprBkt6YILLsj/brnllg2Wjxo1Ku27777577POOitf0WHXXXfNLY9xvfjzzz+/VcpL2xTdegcOHJgnDYoflTGOMQL3tttumx9XB1kYqIcsCG+88UYO2B988EFaZpll8u+/Rx99NP8d1MO2p11cN6y1CwEAAABtUdWM6QYAAIAFTegGAACAQoRuAAAAKEToBgAAgEKEbgAAAChE6AYAAIBChG4AAAAoROgGAACAQoRuAFjInHjiiWm99dabp23atWuXbrzxxmJlAgCaR+gGgHmw7777pp133jm1BVtuuWUO63Hr0qVLWnvttdP555/f2sUCgDZF6AaAKvaTn/wkTZw4MT3//PPphz/8YRoyZEi66qqrWrtYANBmCN0A0EyrrrpqOvvssxssi27h0T28IlqRL7roovS9730vLbHEEmmttdZKY8eOTa+88kpuae7atWvadNNN07/+9a9Zvs64cePStttum5ZeeunUrVu3tMUWW6Qnn3xypvXef//9tMsuu+TXWWONNdJNN900x/cQ6y6//PLpy1/+ci53/e0mTZqUDjjggLTMMsukmpqatPXWW6f/+Z//makb/J///Oe8L6JsP/rRj9LUqVPr1om/99577/w+V1hhhXTWWWfl93344YfXrROhf8cdd0yLL754Wm211dKVV17ZYN/++9//zvvx6aefrtsmyhbL7rvvvnw//o37d999d9pwww3z+4r9+tJLL9VtE/t4p512Ssstt1xacsklU79+/dJdd901x30EAPND6AaAwk455ZQ0cODAHBrXXHPNtNdee6Wf/vSnadiwYenxxx9PtbW16ZBDDpnl9hFcBw0alB566KH06KOP5mD83e9+t0G4DSeddFJurf7f//3f/HiE3f/85z/zVNYIvp999ln+e/fdd0/vvvtuuu2229ITTzyRvvGNb6RtttmmwXNGkI2x5Lfccku+3X///el3v/td3eNHHHFEevjhh3OQHzNmTHrwwQdnOmEQ++att97Kwfn6669PI0aMyK/bHMcdd1w688wz837t0KFDGjx4cN1jH330Ud4vEcyfeuqp9J3vfCcNGDAgjR8/vlmvBQBzQ+gGgML222+/HIa/+tWvpqOPPjq33EYg3n777XPL92GHHVbXYtuUaGHeZ599cmCP9SOUfvLJJzngNh5vvueee6bVV189nXrqqTlkPvbYY3NVxunTp6e//OUvObDH60XAj22vvfba3HIcQf+MM85ISy21VLruuuvqtpsxY0a69NJLU58+fdLmm2+efvzjH+dQG+KkwGWXXZa3i7Ae64waNSq/VsWLL76YW5svvvji9M1vfjMH+0suuST997//bcaeTum3v/1t7gkQ49OPOeaY9Mgjj6RPP/00P9a3b998siPKEe8nToZ85StfmaseAQDQXEI3ABS27rrr1v0dXZvDOuus02BZBMMpU6Y0uf0777yTx15HUIwu3NHVOwJ14xba+q8T3bljvTm1GMfEadHVOlq44zWGDh2aDj744NyNPF6jZ8+e+fHK7bXXXmvQFT66gX/pS1+qux9dyCuv+eqrr6bPP/88bbTRRnWPR/m/9rWv1d2P7t/RIh1huyJOGnTv3j01R/19EGUJlfLE+znyyCPziYs4eRDv54UXXtDSDUBRHco+PQC0Xe3bt89dw+uLkNlYx44d6/6OccezWhatxk2JruUffPBBOuecc9Iqq6ySOnfunDbZZJO6buBNvU7leWf1nBXR4h5dsiN0R0iN91QJqHG/qRb4CKzz85rzqlKm+vu6qf08p/0agTu6uEfLewT7eM+77bbbTPsRAFqS0A0AzRQTjMUkYBXRUh0twS0txkRHi3SMRw4TJkzIk6a1hGh5jgDaWLQ8v/3227kVOlqzmyMmZ4sQHBPBrbzyynnZ5MmT08svv5y+/e1v5/vR6v3FF1/kMdYbbLBBXhaTzH344YcN9nOIfb3++uvnv+tPqjYv+zG64Mdkc5UTC9HVHwBK0r0cAJopxj7HzN0xOdgzzzyTW6QXW2yxFn+d6FYerxNdof/xj3/k1ulopS2pf//+uTU9rkl+55135nAa46OjVTwmKZsb0e089skvf/nLdO+996bnnnsu7b///rnlutIKHePU47UOPPDAPIY8wnf8He+vsk78vfHGG+cJ2mIfxFj2X/3qV83ajzfccEMO7NF9Pia0a+lWeQBoTOgGgHkQIS1af0PMPh6TdsXlwOKSVxFQY2KuljZy5Mjc8hutzzFR2aGHHpqWXXbZVFIE3ltvvTW3SMdEcDEJXFwO7PXXX68blz43/vCHP+TwHvsowvVmm22Wx1R36dKlbp3LL788P2e8VrRCx9jyCOz11/nTn/6UW8SjNTwuN/ab3/xmnt9TlCXGiselxGLW8pjIrv5YcgAooV1t48FoAMAsxWWmojv2eeed19pFWSR9/PHHacUVV8yX9YpW76a88cYbqXfv3nlW85j1HAAWZcZ0A8BciJbmGBMcE4sddNBBrV2cRUZ0F4/LgsUM5jGe++STT87Ld9ppp7p17rnnnjy+OmZ0j3HbRx11VB5HXhn3DQCLMqEbAObC4MGD84Rgv/jFLxoERuYsZguPS4N16tQpdw+PMfBLL710g5nIjz322HyJsehWHt2/r7jiiplmRgeARZHu5QAAAFCIidQAAACgEKEbAAAAChG6AQAAoBChGwAAAAoRugEAAKAQoRsAAAAKEboBAACgEKEbAAAAChG6AQAAIJXx/wHUbOkZyuEBqQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Distribusi User Rating (5 Teratas)\n",
    "feature = 'users_rating'\n",
    "rating_counts = df[feature].value_counts()\n",
    "top_5_ratings = rating_counts.nlargest(5)\n",
    "\n",
    "df_top_5_ratings = pd.DataFrame({'jumlah sampel': top_5_ratings, 'persentase': (100 * top_5_ratings / len(df)).round(1)})\n",
    "\n",
    "print(\"5 Rating Film Teratas:\")\n",
    "print(df_top_5_ratings)\n",
    "\n",
    "# Visualisasi 5 Rating Teratas\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=df, y=feature, order=top_5_ratings.index, palette=\"viridis\")\n",
    "plt.title(f'5 Rating Teratas Berdasarkan Jumlah Pengguna')\n",
    "plt.ylabel(feature.capitalize())\n",
    "plt.xlabel('Jumlah Pengguna')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f93ca407",
   "metadata": {},
   "source": [
    "**Insight**\n",
    "\n",
    "**🥇 Rating 7.2 Paling Banyak Dipilih**\n",
    "- **Rating 7.2** merupakan rating yang **paling banyak diberikan oleh pengguna** dalam dataset ini.\n",
    "- Terdapat **54 film** (sekitar **4.2%** dari total) yang memiliki rating ini.\n",
    "\n",
    "**🥈 Rating 6.2 dan 6.6 Cukup Populer**\n",
    "- **Rating 6.2** diberikan pada **52 film** (**4.1%**).\n",
    "- **Rating 6.6** diberikan pada **50 film** (**3.9%**).\n",
    "- Kedua rating ini sangat dekat frekuensinya dengan rating 7.2, menunjukkan **popularitas yang tinggi**.\n",
    "\n",
    "**🥉 Rating 7.0 dan 6.8 Mengikuti**\n",
    "- **Rating 7.0** muncul pada **48 film** (**3.8%**).\n",
    "- **Rating 6.8** muncul pada **45 film** (**3.5%**).\n",
    "- Kedua rating ini melengkapi **lima besar** rating terbanyak dalam dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6903e80",
   "metadata": {},
   "source": [
    "## Data Preparation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b66c4f5d",
   "metadata": {},
   "source": [
    "- Menerapkan dan menyebutkan teknik data preparation yang dilakukan.\n",
    "- Teknik yang digunakan pada notebook dan laporan harus berurutan.\n",
    "- Menjelaskan proses data preparation yang dilakukan.\n",
    "- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8fca234",
   "metadata": {},
   "source": [
    "### Drop Kolom"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "725c2ee8",
   "metadata": {},
   "source": [
    "Tahap ini bertujuan untuk mengurangi noise dan kompleksitas data dengan menghapus kolom yang tidak memberikan kontribusi signifikan terhadap sistem rekomendasi."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 109,
   "id": "19c920d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop Kolom yang Tidak Relevan\n",
    "df.drop(columns=['rating', 'votes', 'languages', 'runtime'], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5e43b1b",
   "metadata": {},
   "source": [
    "### Handling Missing Values"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5401ab4",
   "metadata": {},
   "source": [
    "Tahap ini bertujuan untuk mengisi nilai yang hilang (NaN) pada kolom-kolom yang akan digunakan dalam pemodelan, untuk meningkatkan kualitas dan keandalan data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 110,
   "id": "64ee05f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Menangani Missing Value\n",
    "df['title'].fillna('', inplace=True)\n",
    "df['description'].fillna('', inplace=True)\n",
    "df['actors'].fillna('', inplace=True)\n",
    "df['genre'].fillna('Tidak Diketahui', inplace=True)\n",
    "df['users_rating'].fillna(df['users_rating'].median(), inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a0ab144",
   "metadata": {},
   "source": [
    "### Preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "09126b78",
   "metadata": {},
   "source": [
    "Tahap ini bertujuan untuk menyiapkan representasi fitur yang relevan untuk perhitungan kesamaan antar film dan menghitung kesamaan antar film berdasarkan representasi fitur gabungan."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 111,
   "id": "91f0f80c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def prepare_features(df):\n",
    "    # Gabungkan kolom teks\n",
    "    text_data = df['genre'] + ' ' + df['actors'] + ' ' + df['description']\n",
    "\n",
    "    # 1. TF-IDF Vectorization\n",
    "    tfidf_vectorizer = TfidfVectorizer(stop_words='english')  \n",
    "    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)\n",
    "\n",
    "    # 2. Scaling numerik: year dan users_rating\n",
    "    scaler = MinMaxScaler()\n",
    "    year_scaled = scaler.fit_transform(df[['year']])\n",
    "    rating_scaled = scaler.fit_transform(df[['users_rating']])\n",
    "\n",
    "    # 3. Gabungkan semua ke dalam satu matrix\n",
    "    combined_matrix = hstack((tfidf_matrix, year_scaled, rating_scaled))\n",
    "\n",
    "    return combined_matrix, tfidf_vectorizer\n",
    "combined_matrix, tfidf_vectorizer = prepare_features(df)\n",
    "\n",
    "# Menghitung Similarity\n",
    "sim_matrix = cosine_similarity(combined_matrix, combined_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b063fc50",
   "metadata": {},
   "source": [
    "## Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8469b057",
   "metadata": {},
   "source": [
    "Tahapan ini membahas model sistem rekomendasi yang dibuat untuk memberikan rekomendasi film kepada pengguna. Sistem ini menggunakan pendekatan Content-Based Filtering untuk merekomendasikan film yang mirip dengan film yang telah disukai atau ditonton sebelumnya."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b8cb49d9",
   "metadata": {},
   "source": [
    "### Content-Based-Filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "id": "34fa5e3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fungsi Rekomendasi Film\n",
    "def get_movie_recommendations(movie_title, df, sim_matrix, n_recommendations=5):\n",
    "    try:\n",
    "        idx = df[df['title'].str.lower() == movie_title.lower()].index[0]\n",
    "    except IndexError:\n",
    "        print(f\"Film dengan judul '{movie_title}' tidak ditemukan dalam dataset.\")\n",
    "        return pd.DataFrame()\n",
    "    else:\n",
    "        sim_scores = list(enumerate(sim_matrix[idx]))\n",
    "        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "        movie_indices = [i[0] for i in sim_scores[1:n_recommendations+1]]\n",
    "        scores = [i[1] for i in sim_scores[1:n_recommendations+1]]\n",
    "        \n",
    "        recommendations = df.iloc[movie_indices][['title', 'description', 'genre', 'actors', 'year', 'users_rating']].copy()\n",
    "        recommendations['similarity_score'] = scores\n",
    "        return recommendations"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3724ef9",
   "metadata": {},
   "source": [
    "## Evaluation"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "35eff075",
   "metadata": {},
   "source": [
    "Bagian ini menjelaskan metrik evaluasi yang digunakan untuk mengukur kinerja sistem rekomendasi dan hasil evaluasinya."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 113,
   "id": "f248c292",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fungsi Evaluasi Metrik\n",
    "def evaluate_genre_precision_at_k(recommended_movies, relevant_genres, k=3):\n",
    "    if recommended_movies.empty:\n",
    "        return 0.0\n",
    "\n",
    "    top_k_recommendations = recommended_movies.head(k)\n",
    "    relevant_and_recommended_count = 0\n",
    "\n",
    "    for genre_str in top_k_recommendations['genre']:\n",
    "        genre_list = [g.strip() for g in genre_str.split(',')]\n",
    "        if any(g in relevant_genres for g in genre_list):\n",
    "            relevant_and_recommended_count += 1\n",
    "\n",
    "    precision_at_k = relevant_and_recommended_count / k if k > 0 else 0.0\n",
    "    return precision_at_k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 114,
   "id": "6b4333bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Rekomendasi Film: Dilan 1991\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>description</th>\n",
       "      <th>genre</th>\n",
       "      <th>actors</th>\n",
       "      <th>year</th>\n",
       "      <th>users_rating</th>\n",
       "      <th>similarity_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Milea</td>\n",
       "      <td>Milea made the decision to part with Dilan as ...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>['Iqbaal Dhiafakhri Ramadhan', 'Vanesha Presci...</td>\n",
       "      <td>2020</td>\n",
       "      <td>6.1</td>\n",
       "      <td>0.761454</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>160</th>\n",
       "      <td>Dilan 1990</td>\n",
       "      <td>Milea (Vanesha Prescilla) met with Dilan (Iqba...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>['Iqbaal Dhiafakhri Ramadhan', 'Vanesha Presci...</td>\n",
       "      <td>2018</td>\n",
       "      <td>7.3</td>\n",
       "      <td>0.740067</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>131</th>\n",
       "      <td>#FriendButMarried</td>\n",
       "      <td>Ayudia (Vanesha Prescilla) and Ditto (Adipati ...</td>\n",
       "      <td>Biography</td>\n",
       "      <td>['Adipati Dolken', 'Vanesha Prescilla', 'Refal...</td>\n",
       "      <td>2018</td>\n",
       "      <td>6.9</td>\n",
       "      <td>0.641758</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Mariposa</td>\n",
       "      <td>Iqbal (Angga Yunanda) is like a Mariposa butte...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>['Angga Yunanda', 'Adhisty Zara', 'Dannia Sals...</td>\n",
       "      <td>2020</td>\n",
       "      <td>8.5</td>\n",
       "      <td>0.633936</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>174</th>\n",
       "      <td>Keluarga Cemara</td>\n",
       "      <td>After the bankruptcy, Abah loses his house and...</td>\n",
       "      <td>Drama</td>\n",
       "      <td>['Nirina Zubir', 'Ringgo Agus Rahman', 'Adhist...</td>\n",
       "      <td>2018</td>\n",
       "      <td>7.9</td>\n",
       "      <td>0.631335</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 title                                        description  \\\n",
       "11               Milea  Milea made the decision to part with Dilan as ...   \n",
       "160         Dilan 1990  Milea (Vanesha Prescilla) met with Dilan (Iqba...   \n",
       "131  #FriendButMarried  Ayudia (Vanesha Prescilla) and Ditto (Adipati ...   \n",
       "9             Mariposa  Iqbal (Angga Yunanda) is like a Mariposa butte...   \n",
       "174    Keluarga Cemara  After the bankruptcy, Abah loses his house and...   \n",
       "\n",
       "         genre                                             actors  year  \\\n",
       "11       Drama  ['Iqbaal Dhiafakhri Ramadhan', 'Vanesha Presci...  2020   \n",
       "160      Drama  ['Iqbaal Dhiafakhri Ramadhan', 'Vanesha Presci...  2018   \n",
       "131  Biography  ['Adipati Dolken', 'Vanesha Prescilla', 'Refal...  2018   \n",
       "9        Drama  ['Angga Yunanda', 'Adhisty Zara', 'Dannia Sals...  2020   \n",
       "174      Drama  ['Nirina Zubir', 'Ringgo Agus Rahman', 'Adhist...  2018   \n",
       "\n",
       "     users_rating  similarity_score  \n",
       "11            6.1          0.761454  \n",
       "160           7.3          0.740067  \n",
       "131           6.9          0.641758  \n",
       "9             8.5          0.633936  \n",
       "174           7.9          0.631335  "
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Precision@3 (Genre): 0.67\n"
     ]
    }
   ],
   "source": [
    "# Contoh Penerapan Sistem Rekomendasi Film\n",
    "movie_title = 'Dilan 1991'\n",
    "recommendations = get_movie_recommendations(movie_title, df, sim_matrix)\n",
    "print(\"Rekomendasi Film:\", movie_title)\n",
    "display(recommendations)\n",
    "\n",
    "relevant_genres = ['Drama', 'Romance']\n",
    "precision_at_k = evaluate_genre_precision_at_k(recommendations, relevant_genres, k=3)\n",
    "print(f'\\nPrecision@3 (Genre): {precision_at_k:.2f}')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "90233fd3",
   "metadata": {},
   "source": [
    "### Insight: Hasil Rekomendasi dan Evaluasi Genre\n",
    "\n",
    "Output ini menyajikan hasil konkret dari sistem rekomendasi untuk film **'Dilan 1991'** dan evaluasi kinerjanya menggunakan metrik Precision@k berdasarkan genre.\n",
    "\n",
    "**Temuan dari Rekomendasi:**\n",
    "\n",
    "1.  **Relevansi Tinggi:** Rekomendasi teratas adalah **'Milea'** dan **'Dilan 1990'**, yang merupakan sekuel dan prekuel langsung dari film input. Skor kemiripan yang tinggi (0.76 dan 0.74) menunjukkan model berhasil mengidentifikasi hubungan konten yang sangat kuat (kemungkinan besar dari deskripsi, aktor, judul, dan genre yang sama).\n",
    "2.  **Kemiripan Lainnya:** Film seperti '#FriendButMarried', 'Mariposa', dan 'Keluarga Cemara' muncul berikutnya. Kemiripan mereka (skor ~0.63-0.64) mungkin berasal dari kombinasi genre (Drama), aktor yang sama (Vanesha Prescilla, Adhisty Zara), tema dalam deskripsi, atau kedekatan dalam rating pengguna.\n",
    "3.  **Skor Menurun:** Skor kemiripan menurun secara wajar seiring peringkat rekomendasi, yang diharapkan.\n",
    "\n",
    "**Temuan dari Evaluasi (Precision@3 Genre):**\n",
    "\n",
    "1.  **Konteks:** Evaluasi ini mengukur seberapa banyak dari **3 rekomendasi teratas** yang termasuk dalam genre yang dianggap relevan (dalam kasus ini, **'Drama'** atau **'Romance'**).\n",
    "2.  **Perhitungan:**\n",
    "    * Top 3 Rekomendasi: 'Milea' (Genre: Drama), 'Dilan 1990' (Genre: Drama), '#FriendButMarried' (Genre: Biography).\n",
    "    * Genre yang Relevan di Top 3: 'Drama', 'Drama' (2 film).\n",
    "    * Precision@3 = (Jumlah Rekomendasi Relevan di Top 3) / 3 = 2 / 3 ≈ 0.67.\n",
    "3.  **Makna:** Skor **Precision@3 ≈ 0.67** menunjukkan bahwa **dua pertiga** dari rekomendasi teratas untuk 'Dilan 1991' sesuai dengan kriteria genre yang diinginkan ('Drama' atau 'Romance').\n",
    "\n",
    "**Insight Keseluruhan:**\n",
    "\n",
    "Kombinasi output rekomendasi dan evaluasi ini memberikan gambaran:\n",
    "\n",
    "* **Efektivitas Model:** Model berbasis konten ini efektif dalam menemukan film yang sangat mirip (seperti dalam satu seri).\n",
    "* **Relevansi Genre:** Model menunjukkan kemampuan yang cukup baik (presisi 67% di 3 teratas) dalam merekomendasikan film dengan genre yang dianggap relevan ('Drama'/'Romance') untuk contoh spesifik ini. Ini mengindikasikan bahwa fitur yang digunakan (teks + rating) secara tidak langsung juga menangkap preferensi genre."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
