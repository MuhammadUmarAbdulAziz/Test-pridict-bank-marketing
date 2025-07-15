import streamlit as st
import pandas as pd
import xgboost as xgb

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Nasabah Bank",
    page_icon="🏦",
    layout="centered"
)

# --- CSS untuk Tema ---
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

# --- Fungsi Encoding Manual ---
def encode_input(df):
    mapping = {
        "job": {
            "admin.": 0, "blue-collar": 1, "entrepreneur": 2, "housemaid": 3,
            "management": 4, "retired": 5, "self-employed": 6, "services": 7,
            "student": 8, "technician": 9, "unemployed": 10, "unknown": 11
        },
        "marital": {"single": 0, "married": 1, "divorced": 2, "unknown": 3},
        "education": {
            "illiterate": 0, "basic.4y": 1, "basic.6y": 2, "basic.9y": 3,
            "high.school": 4, "professional.course": 5, "university.degree": 6
        },
        "default": {"yes": 1, "no": 0, "unknown": -1},
        "housing": {"yes": 1, "no": 0, "unknown": -1},
        "loan": {"yes": 1, "no": 0, "unknown": -1},
        "contact": {"cellular": 0, "telephone": 1},
        "month": {
            "jan": 0, "feb": 1, "mar": 2, "apr": 3, "may": 4, "jun": 5,
            "jul": 6, "aug": 7, "sep": 8, "oct": 9, "nov": 10, "dec": 11
        },
        "day_of_week": {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4},
        "poutcome": {"failure": 0, "nonexistent": 1, "success": 2}
    }

    for col, col_map in mapping.items():
        df[col] = df[col].map(col_map)

    return df

# --- Load Model ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("model_xgb.json")
    return model

model = load_model()

# --- Judul Aplikasi ---
st.title("🏦 Prediksi Ketertarikan Nasabah terhadap Produk Bank")
st.markdown("Silakan isi data calon nasabah di bawah ini:")

# --- Input User ---
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
    previous = st.number_input("Jumlah Kontak Sebelumnya", 0, 100, 0)
    poutcome = st.selectbox("Hasil Kontak Sebelumnya", ["failure", "nonexistent", "success"])
    emp_var_rate = st.number_input("Variasi Tingkat Kerja", -3.0, 3.0, 1.1)
    cons_price_idx = st.number_input("Indeks Harga Konsumen", 90.0, 100.0, 93.2)
    cons_conf_idx = st.number_input("Indeks Kepercayaan Konsumen", -50.0, 0.0, -36.4)
    euribor3m = st.number_input("Tingkat Euribor 3 Bulan", 0.0, 6.0, 4.8)
    nr_employed = st.number_input("Jumlah Pegawai", 4000.0, 5500.0, 5191.0)

# --- Proses Prediksi ---
if st.button("Prediksi"):
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
        encoded_df = encode_input(input_df)
        pred = model.predict(encoded_df)[0]
        prob = model.predict_proba(encoded_df)[0][1] * 100
        label = "Ya" if pred == 1 else "Tidak"
        st.success(f"Prediksi: {label} ({prob:.2f}%)")
    except Exception as e:
        st.error(f"Gagal memproses prediksi: {e}")
