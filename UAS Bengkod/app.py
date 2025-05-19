# Simpan kode berikut dalam file app.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Memuat model dan preprocessing objects
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# Fungsi untuk memproses input
def preprocess_input(data):
    # Encoding untuk kolom biner
    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        data[col] = data[col].map({'Tidak': 0, 'Ya': 1})
    
    # Encoding untuk kolom kategorikal
    data['Gender'] = label_encoders['Gender'].transform([data['Gender']])[0]
    data['CAEC'] = label_encoders['CAEC'].transform([data['CAEC']])[0]
    data['CALC'] = label_encoders['CALC'].transform([data['CALC']])[0]
    data['MTRANS'] = label_encoders['MTRANS'].transform([data['MTRANS']])[0]
    
    # Konversi ke dataframe
    df = pd.DataFrame([data])
    
    # Standarisasi
    df_scaled = scaler.transform(df)
    
    return df_scaled

# Fungsi untuk memetakan prediksi ke label
def get_prediction_label(pred):
    return label_encoders['NObeyesdad'].inverse_transform([pred])[0]

# Antarmuka Streamlit
st.title('Prediksi Tingkat Obesitas')
st.write('Aplikasi ini memprediksi tingkat obesitas berdasarkan kebiasaan dan kondisi fisik.')

# Input form
with st.form("input_form"):
    st.header("Informasi Demografis")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox('Jenis Kelamin', ['Pria', 'Wanita'])
        age = st.number_input('Usia', min_value=10, max_value=100, value=25)
        height = st.number_input('Tinggi Badan (m)', min_value=1.3, max_value=2.5, value=1.7, step=0.01)
        weight = st.number_input('Berat Badan (kg)', min_value=30, max_value=200, value=70)
    with col2:
        family_history = st.selectbox('Riwayat Keluarga Obesitas', ['Tidak', 'Ya'])
        favc = st.selectbox('Sering Makan Makanan Tinggi Kalori', ['Tidak', 'Ya'])
    
    st.header("Kebiasaan Makan")
    col1, col2 = st.columns(2)
    with col1:
        fcvc = st.slider('Frekuensi Makan Sayur (1-3)', 1, 3, 2)
        ncp = st.slider('Jumlah Makan Besar per Hari (1-4)', 1, 4, 3)
        caec = st.selectbox('Makan Camilan', ['Tidak', 'Kadang', 'Sering', 'Selalu'])
    with col2:
        smoke = st.selectbox('Merokok', ['Tidak', 'Ya'])
        ch2o = st.slider('Konsumsi Air per Hari (gelas)', 1, 10, 2)
        scc = st.selectbox('Memantau Asupan Kalori', ['Tidak', 'Ya'])
    
    st.header("Aktivitas Fisik dan Gaya Hidup")
    col1, col2 = st.columns(2)
    with col1:
        faf = st.slider('Frekuensi Aktivitas Fisik (0-3)', 0, 3, 1)
        tue = st.slider('Waktu Penggunaan Perangkat Elektronik (jam)', 0, 10, 2)
    with col2:
        calc = st.selectbox('Konsumsi Alkohol', ['Tidak', 'Kadang', 'Sering'])
        mtrans = st.selectbox('Transportasi yang Digunakan', ['Mobil', 'Sepeda Motor', 'Sepeda', 'Angkutan Umum', 'Berjalan'])
    
    submitted = st.form_submit_button("Prediksi")

if submitted:
    # Membuat dictionary dari input
    input_data = {
        'Gender': 'Female' if gender == 'Wanita' else 'Male',
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }
    
    # Preprocessing input
    processed_input = preprocess_input(input_data)
    
    # Prediksi
    prediction = model.predict(processed_input)[0]
    prediction_label = get_prediction_label(prediction)
    
    # Mapping label ke bahasa Indonesia
    label_mapping = {
        'Insufficient_Weight': 'Berat Badan Kurang',
        'Normal_Weight': 'Berat Badan Normal',
        'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
        'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
        'Obesity_Type_I': 'Obesitas Tipe I',
        'Obesity_Type_II': 'Obesitas Tipe II',
        'Obesity_Type_III': 'Obesitas Tipe III'
    }
    
    # Menampilkan hasil
    st.success(f"Hasil Prediksi: {label_mapping[prediction_label]}")
    
    # Menampilkan penjelasan
    st.subheader("Penjelasan Hasil:")
    if prediction_label == 'Insufficient_Weight':
        st.write("Anda memiliki berat badan kurang. Disarankan untuk meningkatkan asupan nutrisi dan berkonsultasi dengan ahli gizi.")
    elif prediction_label == 'Normal_Weight':
        st.write("Anda memiliki berat badan normal. Pertahankan gaya hidup sehat dan pola makan seimbang.")
    elif prediction_label in ['Overweight_Level_I', 'Overweight_Level_II']:
        st.write("Anda mengalami kelebihan berat badan. Disarankan untuk meningkatkan aktivitas fisik dan mengurangi asupan kalori.")
    else:
        st.write("Anda mengalami obesitas. Sangat disarankan untuk berkonsultasi dengan dokter atau ahli gizi untuk rencana penurunan berat badan yang sehat.")
    
    # Menampilkan BMI
    bmi = weight / (height ** 2)
    st.subheader("Indeks Massa Tubuh (BMI):")
    st.write(f"BMI Anda: {bmi:.1f}")
    
    # Interpretasi BMI
    if bmi < 18.5:
        st.write("Kategori: Kurang berat badan")
    elif 18.5 <= bmi < 25:
        st.write("Kategori: Normal")
    elif 25 <= bmi < 30:
        st.write("Kategori: Kelebihan berat badan")
    else:
        st.write("Kategori: Obesitas")