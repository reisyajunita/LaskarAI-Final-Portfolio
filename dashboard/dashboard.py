import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import timedelta, datetime

# Set page config
st.set_page_config(page_title="Bike Sharing Dashboard", layout="wide")

# Helper Functions
@st.cache_data
def load_data():
    day_df = pd.read_csv("Bike-sharing-dataset/day.csv")
    hour_df = pd.read_csv("Bike-sharing-dataset/hour.csv")
    day_df["dteday"] = pd.to_datetime(day_df["dteday"])
    return day_df, hour_df

def create_daily_user_df(day_df):
    daily_user_df = day_df.resample(rule="D", on="dteday").agg({
        "casual": "sum",
        "registered": "sum",
        "cnt": "sum"
    })
    daily_user_df = daily_user_df.reset_index()
    daily_user_df.rename(columns={"casual": "Total Casual", "registered": "Total Registered", "cnt": "Total User"}, inplace=True)
    return daily_user_df

def format_with_commas(number):
    return f"{number:,}"

def create_metric_chart(df, column, color, chart_type, height=150):
    chart_data = df[[column]]
    if chart_type=='Bar':
        st.bar_chart(chart_data, y=column, color=color, height=height)
    elif chart_type=='Area':
        st.area_chart(chart_data, y=column, color=color, height=height)

def calculate_delta(df, column):
    if len(df) < 2:
        return 0, 0
    current_value = df[column].iloc[-1]
    previous_value = df[column].iloc[-2]
    delta = current_value - previous_value
    delta_percent = (delta / previous_value) * 100 if previous_value != 0 else 0
    return delta, delta_percent

def display_metric(col, title, value, df, column, color, chart_selection):
    with col:
        with st.container(border=True):
            delta, delta_percent = calculate_delta(df, column)
            delta_str = f"{delta:+,.0f} ({delta_percent:+.2f}%)"

            # Format nilai agar sesuai satuan
            if column == 'temp' or column == 'atemp':
                unit = "°C"
            elif column == 'hum':
                unit = "%"
            elif column == 'windspeed':
                unit = " km/h"
            else:
                unit = ""

            formatted_value = f"{format_with_commas(int(value))}{unit}"  # Format tanpa desimal

            st.metric(title, formatted_value, delta=delta_str)
            create_metric_chart(df, column, color, chart_selection)

# Helper func no.1: Total penyewaan pada hari kerja dan libur
def workingday_vs_holiday(day_df):
    workingday_total = day_df[day_df["workingday"] == 1]["cnt"].sum()
    holiday_total = day_df[day_df["holiday"] == 1]["cnt"].sum()
    return (workingday_total, holiday_total)

# Helper func no.2: Mode dan Median jam penyewaan
def mode_hour(hour_df):
    mode_hour = hour_df["hr"].mode()[0]  # Mode adalah nilai yang paling sering muncul
    median_hour = hour_df["hr"].median()  # Median adalah nilai tengah dari distribusi
    return mode_hour, median_hour

# Helper func no.3: Penyewaan per tahun
def year_by_year(day_df):
    yearly_counts = day_df.groupby("yr")["cnt"].sum().reset_index()
    yearly_counts["yr"] = yearly_counts["yr"].map({0: 2011, 1: 2012})  # Ubah 0/1 menjadi tahun asli
    return yearly_counts

# Helper func no.4: Penyewaan berdasarkan musim
def season_(day_df):
    season_labels = ["Spring", "Summer", "Fall", "Winter"]
    season_rentals = day_df.groupby("season")["cnt"].sum().reset_index()
    season_rentals["season"] = season_rentals["season"].map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})
    return season_labels, season_rentals

# Helper func no.5: Hubungan penyewaan berdasarkan temp, atemp, hum, dan windspeed
def corr_count(day_df):
    correlation_matrix = day_df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
    return correlation_matrix

# Load dataset
day_df, hour_df = load_data()

# Dapatkan data untuk visualisasi
workingday_total, holiday_total = workingday_vs_holiday(day_df)
mode_hour_val, median_hour_val = mode_hour(hour_df)
yearly_counts = year_by_year(day_df)
season_labels, season_rentals = season_(day_df)
correlation_matrix = corr_count(day_df)
daily_user_df = create_daily_user_df(day_df)

# Sidebar settings
st.sidebar.title("Bike Sharing Dashboard")

date_min = day_df['dteday'].min().date()
date_max = day_df['dteday'].max().date()
start_date = st.sidebar.date_input("Tanggal Mulai", date_min, date_min, date_max)
end_date = st.sidebar.date_input("Tanggal Akhir", date_max, date_min, date_max)
chart_selection = st.sidebar.selectbox("Pilih Tipe Grafik", ("Bar", "Area"))

# Filter DataFrame berdasarkan rentang tanggal
mask = (day_df['dteday'].dt.date >= start_date) & (day_df['dteday'].dt.date <= end_date)
df_filtered = day_df.loc[mask].set_index('dteday')

# Dashboard Header
st.header("📊 Bike Sharing Dashboard")

st.subheader("Daily Users")

# Filter daily_user_df berdasarkan tanggal yang dipilih
filtered_daily_user_df = daily_user_df[
    (daily_user_df["dteday"] >= pd.Timestamp(start_date)) &
    (daily_user_df["dteday"] <= pd.Timestamp(end_date))
]

# Hitung total setelah filter
total_casual = filtered_daily_user_df["Total Casual"].sum()
total_registered = filtered_daily_user_df["Total Registered"].sum()
total_users = filtered_daily_user_df["Total User"].sum()

# Mengatur kolom untuk layout
col1, col2, col3 = st.columns(3)

# kolom 1: Total Casual
with col1:
    st.metric("Total Casual", value=f"{total_casual:,}")

# kolom 2: Total Registered 
with col2:
    st.metric("Total Registered", value=f"{total_registered:,}")

# kolom 3: Total Users
with col3:
    st.metric("Total Keseluruhan", value=f"{total_users:,}")

st.subheader("Statistik Keseluruhan")

metrics = [
    ("Suhu Rata-rata", "temp", '#29b5e8'),
    ("Suhu Terasa Rata-rata", "atemp", '#FF9F36'),
    ("Kelembapan Rata-rata", "hum", '#D45B90'),
    ("Kecepatan Angin", "windspeed", '#7D44CF')
]

cols = st.columns(4)
for col, (title, column, color) in zip(cols, metrics):
    # Konversi satuan dan agregasi di sini
    if column in ['temp', 'atemp', 'hum', 'windspeed']:
        # Konversi satuan dan hitung rata-rata
        total_value = (df_filtered[column] * (41 if column == 'temp' else 50 if column == 'atemp'
        else 100 if column == 'hum' else 67 if column == 'windspeed' else 1)).mean()
    else:
        total_value = df_filtered[column].sum()

    display_metric(col, title, total_value, df_filtered, column, color, chart_selection)

# Streamlit UI
st.title("Visualisasi Data Bike Sharing")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Workingday vs Holiday", "Rush Hour", "2011 vs 2012", "Favorite Season", "Correlation Weather"])

# --- Tab 1: Pie Chart ---
with tab1:
    st.header("Perbandingan Total Penyewaan Sepeda pada Hari Kerja dan Hari Libur")
    
    colors = ["#29b5e8", "#d3d3d3"]
    fig, ax = plt.subplots()
    ax.pie(
        [workingday_total, holiday_total],
        labels=["Workingday", "Holiday"],
        colors=colors,
        autopct="%1.1f%%",
        shadow=True
    )
    st.pyplot(fig)

# --- Tab 2: Histogram Penyewaan Sepeda ---
with tab2:
    st.header("Distribusi Jam Penyewaan Sepeda")

    fig, ax = plt.subplots()
    sns.histplot(hour_df, x="hr", bins=24, color="#FF9F36", ax=ax)

    # Menandai mode dan median pada histogram
    ax.axvline(mode_hour_val, color='red', linestyle='--', label='Mode')
    ax.axvline(median_hour_val, color='green', linestyle='-', label='Median')
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)

    # Histogram Casual vs Registered
    st.header("Distribusi Jam Penyewaan Sepeda Berdasarkan Casual dan Registered")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(hour_df, x="hr", weights=hour_df["casual"], bins=24, color="red", label="Casual", kde=True, ax=ax)
    sns.histplot(hour_df, x="hr", weights=hour_df["registered"], bins=24, color="#7D44CF", label="Registered", kde=True, ax=ax)
    ax.legend()
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)

# --- Tab 3: Bar Chart Penyewaan Berdasarkan Tahun ---
with tab3:
    st.header("Distribusi Penyewaan Sepeda Berdasarkan Tahun")
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=yearly_counts, 
        x="yr", 
        y="cnt", 
        hue="yr", 
        palette=["#FF9F36", "#7D44CF"],
        ax=ax
    )
    ax.set_title("Distribusi Penyewaan Sepeda Berdasarkan Tahun", fontsize=15, fontweight='bold', color="#333333")
    ax.set_xticklabels(["2011", "2012"], fontsize=12, fontweight='bold', color="#444444")
    ax.set_yticklabels(ax.get_yticks(), fontsize=10, color="#666666")
    ax.legend(title="Tahun", loc="upper left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)


# Tab 4: Penyewaan Berdasarkan Musim
with tab4:
    st.header("Distribusi Penyewaan Sepeda Berdasarkan Musim")
    colors = ['#29b5e8','#FF9F36','#D45B90','#7D44CF']
    fig, ax = plt.subplots()
    sns.barplot(data=season_rentals, x="season", y="cnt", hue="season", palette=colors, ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)

# Tab 5: Korelasi Cuaca
with tab5:
    st.header("Heatmap Korelasi Faktor Cuaca terhadap Penyewaan Sepeda")
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)
    
    st.subheader("Scatter Plot Hubungan Faktor Cuaca terhadap Penyewaan Sepeda")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.scatterplot(ax=axes[0,0], data=day_df, x='temp', y='cnt', alpha=0.5, color="red").set_title("Temp vs Rentals")
    sns.scatterplot(ax=axes[0,1], data=day_df, x='atemp', y='cnt', alpha=0.5, color="orange").set_title("Atemp vs Rentals")
    sns.scatterplot(ax=axes[1,0], data=day_df, x='hum', y='cnt', alpha=0.5, color="blue").set_title("Humidity vs Rentals")
    sns.scatterplot(ax=axes[1,1], data=day_df, x='windspeed', y='cnt', alpha=0.5, color="green").set_title("Windspeed vs Rentals")
    ax.set_xlabel("")
    ax.set_ylabel("")
    st.pyplot(fig)
    
    
with st.expander(' Lihat DataFrame (Rentang Waktu Terpilih)'):
    st.dataframe(df_filtered)