import streamlit as st
import pandas as pd
import joblib
import os

st.write(\"Current Directory:\", os.getcwd())
st.write(\"Files in Dir:\", os.listdir())


# --- CONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Nasabah Bank",
    page_icon="🏦",
    layout="centered"
)

st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #002060;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return joblib.load("best_xgb_model.pkl")

model = load_model()

st.title("🏦 Prediksi Ketertarikan Nasabah terhadap Produk Bank")
st.markdown("Silakan isi data calon nasabah di bawah ini:")

# --- INPUT FITUR ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", 18, 100, 30)
    job = st.selectbox("Pekerjaan", ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"])
    marital = st.selectbox("Status Pernikahan", ["single", "married", "divorced", "unknown"])
    education = st.selectbox("Pendidikan", ["illiterate", "basic.4y", "basic.6y", "basic.9y", "high.school", "professional.course", "university.degree"])
    default = st.selectbox("Memiliki Kredit Macet?", ["yes", "no", "unknown"])
    housing = st.selectbox("Memiliki Pinjaman Rumah?", ["yes", "no", "unknown"])
    loan = st.selectbox("Memiliki Pinjaman Pribadi?", ["yes", "no", "unknown"])
    contact = st.selectbox("Jenis Kontak", ["cellular", "telephone"])

with col2:
    month = st.selectbox("Bulan Kontak Terakhir", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.selectbox("Hari Kontak Terakhir", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.number_input("Durasi Kontak (detik)", 0, 5000, 100)
    campaign = st.number_input("Jumlah Kontak dalam Kampanye Ini", 1, 50, 1)
    pdays = st.number_input("Hari Sejak Kontak Sebelumnya", 0, 999, 999)
    previous = st.number_input("Jumlah Kontak Sebelumnya", 0, 100, 0)
    poutcome = st.selectbox("Hasil Kontak Sebelumnya", ["failure", "nonexistent", "success"])
    emp_var_rate = st.number_input("Variasi Tingkat Kerja", -3.0, 3.0, 1.1)
    cons_price_idx = st.number_input("Indeks Harga Konsumen", 90.0, 100.0, 93.2)
    cons_conf_idx = st.number_input("Indeks Kepercayaan Konsumen", -50.0, 0.0, -36.4)
    euribor3m = st.number_input("Tingkat Euribor 3 Bulan", 0.0, 6.0, 4.8)
    nr_employed = st.number_input("Jumlah Pegawai", 4000.0, 5500.0, 5191.0)

# --- PREDIKSI ---
if st.button("Prediksi" ):
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
        "pdays": pdays,
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
        st.success(f"Prediksi: {label} ({prob:.2f}%)")
    except Exception as e:
        st.error(f"Gagal memproses prediksi: {e}")
