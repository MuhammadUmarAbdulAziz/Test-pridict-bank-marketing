import streamlit as st
import pandas as pd
import joblib
import os
import sys

# -------------------- CONFIGURASI HALAMAN --------------------
st.set_page_config(
    page_title="Prediksi Nasabah Bank",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            font-family: 'Segoe UI', sans-serif;
            background-color: #f0f2f6;
        }
        h1, h2, h3 {
            color: #002060;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_pipeline_model.pkl")

model = load_model()

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank-additional-full_cleaned_Fix_4.csv")
    return df

data = load_data()

# -------------------- INPUT FORM --------------------
st.title("🏦 Prediksi Ketertarikan Nasabah terhadap Produk Bank")

st.sidebar.subheader("📋 Input Data Nasabah")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 18, 100, 30)
    job = st.selectbox("Pekerjaan", sorted(data['job'].unique()))
    marital = st.selectbox("Status Pernikahan", sorted(data['marital'].unique()))
    education = st.selectbox("Pendidikan", sorted(data['education'].unique()))
    default = st.selectbox("Memiliki Kredit Macet?", sorted(data['default'].unique()))
    housing = st.selectbox("Memiliki Pinjaman Rumah?", sorted(data['housing'].unique()))
    loan = st.selectbox("Memiliki Pinjaman Pribadi?", sorted(data['loan'].unique()))
    contact = st.selectbox("Jenis Kontak", sorted(data['contact'].unique()))

with col2:
    month = st.selectbox("Bulan Kontak Terakhir", sorted(data['month'].unique()))
    day_of_week = st.selectbox("Hari Kontak Terakhir", sorted(data['day_of_week'].unique()))
    duration = st.number_input("Durasi Kontak (detik)", 0, 5000, 100)
    campaign = st.number_input("Jumlah Kontak dalam Kampanye Ini", 1, 50, 1)
    previous = st.number_input("Jumlah Kontak Sebelumnya", 0, 100, 0)
    poutcome = st.selectbox("Hasil Kontak Sebelumnya", sorted(data['poutcome'].unique()))
    emp_var_rate = st.number_input("Variasi Tingkat Kerja", -3.0, 3.0, 1.1)
    cons_price_idx = st.number_input("Indeks Harga Konsumen", 90.0, 100.0, 93.2)
    cons_conf_idx = st.number_input("Indeks Kepercayaan Konsumen", -50.0, 0.0, -36.4)
    euribor3m = st.number_input("Tingkat Euribor 3 Bulan", 0.0, 6.0, 4.8)
    nr_employed = st.number_input("Jumlah Pegawai", 4000.0, 5500.0, 5191.0)

# -------------------- PREDIKSI --------------------
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "duration": duration,
        "campaign": campaign,
        "previous": previous,
        "poutcome": poutcome,
        "emp_var_rate": emp_var_rate,
        "cons_price_idx": cons_price_idx,
        "cons_conf_idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr_employed": nr_employed
    }])

    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100
        label = "Ya" if pred == 1 else "Tidak"
        st.success(f"Hasil Prediksi: {label} ({prob:.2f}%)")

        # Simpan hasil prediksi
        result_df = input_df.copy()
        result_df["Prediksi"] = label
        result_df["Probabilitas (%)"] = round(prob, 2)

        st.subheader("📄 Hasil Prediksi")
        st.dataframe(result_df)

        # Download button
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Unduh Hasil Prediksi",
            data=csv,
            file_name="hasil_prediksi.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Gagal memproses prediksi: {e}")

# -------------------- PREVIEW DATA & STATISTIK --------------------
st.markdown("---")
st.subheader("📊 Data Nasabah Asli")
st.dataframe(data.head(10), use_container_width=True)

if st.checkbox("📈 Tampilkan Statistik Deskriptif"):
    st.write(data.describe(include="all").T)
