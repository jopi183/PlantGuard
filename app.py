import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
import base64

# Konfigurasi halaman
st.set_page_config(
    page_title="PlantGuard",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #388E3C;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #E8F5E8 0%, #A5D6A7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

MODEL_FILE = "mobilenetv2_plantvillage.h5"
MODEL_URL = "https://drive.google.com/file/d/1WcYTyVSdMIiebFvQif4RjjdnJaM_hqd3/view?usp=sharing"  # ganti dengan link model

@st.cache_resource
def load_model():
    try:
        # Cari model berdasarkan lokasi file .py
        model_path = os.path.join(os.path.dirname(__file__), MODEL_FILE)

        # Kalau file belum ada, coba download
        if not os.path.exists(model_path):
            st.warning("📥 Mengunduh model, harap tunggu...")
            gdown.download(MODEL_URL, model_path, quiet=False)

        # Load model
        model = tf.keras.models.load_model(model_path)
        return model

    except Exception as e:
        st.error(f"⚠️ Model gagal dimuat: {str(e)}")
        return None
        
# Fungsi untuk memuat label
@st.cache_data
def load_labels():
    labels = [
        "Pepper Bell - Bacterial Spot",
        "Pepper Bell - Healthy",
        "Potato - Early Blight",
        "Potato - Late Blight", 
        "Potato - Healthy",
        "Tomato - Bacterial Spot",
        "Tomato - Early Blight",
        "Tomato - Late Blight",
        "Tomato - Leaf Mold",
        "Tomato - Septoria Leaf Spot",
        "Tomato - Spider Mites",
        "Tomato - Target Spot",
        "Tomato - Yellow Leaf Curl Virus",
        "Tomato - Mosaic Virus",
        "Tomato - Healthy"
    ]
    return labels

# Fungsi untuk preprocessing gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Fungsi untuk prediksi
def predict_disease(model, image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class])
    return predicted_class, confidence

# Fungsi untuk memberikan saran berdasarkan penyakit
def get_disease_info(disease_name):
    disease_info = {
        "Pepper Bell - Bacterial Spot": {
            "severity": "warning",
            "description": "Bercak bakteri pada paprika yang disebabkan oleh Xanthomonas campestris.",
            "symptoms": "• Bercak kecil berwarna coklat tua\n• Daun menguning dan rontok\n• Buah memiliki bercak kasar",
            "treatment": "• Gunakan fungisida berbasis tembaga\n• Hindari penyiraman dari atas\n• Buang bagian tanaman yang terinfeksi\n• Rotasi tanaman",
            "prevention": "• Tanam dengan jarak yang cukup\n• Pastikan drainase baik\n• Hindari kelembaban berlebih"
        },
        "Pepper Bell - Healthy": {
            "severity": "success",
            "description": "Tanaman paprika dalam kondisi sehat!",
            "symptoms": "• Daun hijau segar\n• Pertumbuhan normal\n• Tidak ada tanda penyakit",
            "treatment": "• Lanjutkan perawatan rutin\n• Berikan pupuk seimbang\n• Monitor kondisi tanaman",
            "prevention": "• Jaga kelembaban optimal\n• Berikan sirkulasi udara baik\n• Pemupukan teratur"
        },
        "Potato - Early Blight": {
            "severity": "warning",
            "description": "Penyakit hawar awal kentang yang disebabkan oleh Alternaria solani.",
            "symptoms": "• Bercak coklat dengan lingkaran konsentris\n• Dimulai dari daun tua\n• Daun menguning dan layu",
            "treatment": "• Aplikasi fungisida preventif\n• Buang daun yang terinfeksi\n• Perbaiki sirkulasi udara\n• Kurangi kelembaban",
            "prevention": "• Rotasi tanaman 3-4 tahun\n• Tanam varietas tahan\n• Jaga kebersihan lahan"
        },
        "Potato - Late Blight": {
            "severity": "danger",
            "description": "Penyakit hawar akhir kentang yang sangat berbahaya, disebabkan oleh Phytophthora infestans.",
            "symptoms": "• Bercak basah berwarna hijau gelap\n• Lapisan putih di bawah daun\n• Umbi membusuk\n• Bau tidak sedap",
            "treatment": "• Segera aplikasi fungisida sistemik\n• Buang seluruh bagian terinfeksi\n• Perbaiki drainase\n• Kurangi irigasi",
            "prevention": "• Gunakan bibit bebas penyakit\n• Hindari penanaman saat musim hujan\n• Monitor cuaca"
        },
        "Potato - Healthy": {
            "severity": "success",
            "description": "Tanaman kentang dalam kondisi sehat!",
            "symptoms": "• Daun hijau segar\n• Pertumbuhan vigorous\n• Tidak ada gejala penyakit",
            "treatment": "• Lanjutkan program pemupukan\n• Monitor perkembangan\n• Jaga kelembaban tanah",
            "prevention": "• Rotasi tanaman teratur\n• Pemupukan berimbang\n• Pengendalian hama rutin"
        },
        "Tomato - Bacterial Spot": {
            "severity": "warning",
            "description": "Bercak bakteri pada tomat yang disebabkan oleh Xanthomonas spp.",
            "symptoms": "• Bercak kecil berwarna coklat\n• Halo kuning di sekitar bercak\n• Buah berbintik-bintik",
            "treatment": "• Semprot dengan bakterisida\n• Buang bagian terinfeksi\n• Perbaiki sanitasi\n• Kurangi kelembaban daun",
            "prevention": "• Gunakan benih bebas patogen\n• Hindari penyiraman overhead\n• Sterilisasi alat"
        },
        "Tomato - Early Blight": {
            "severity": "warning",
            "description": "Hawar awal tomat yang disebabkan oleh Alternaria solani.",
            "symptoms": "• Bercak coklat dengan target spot\n• Dimulai dari daun bawah\n• Daun menguning",
            "treatment": "• Aplikasi fungisida berbasis tembaga\n• Pruning daun terinfeksi\n• Mulching untuk mencegah percikan\n• Perbaiki sirkulasi udara",
            "prevention": "• Rotasi tanaman\n• Jarak tanam optimal\n• Pemupukan kalium cukup"
        },
        "Tomato - Late Blight": {
            "severity": "danger",
            "description": "Hawar akhir tomat, penyakit yang sangat merusak disebabkan oleh Phytophthora infestans.",
            "symptoms": "• Bercak basah hijau gelap\n• Lapisan jamur putih\n• Buah membusuk cepat\n• Bau busuk",
            "treatment": "• Segera aplikasi fungisida sistemik\n• Buang tanaman terinfeksi\n• Perbaiki drainase\n• Hindari penyiraman berlebih",
            "prevention": "• Tanam varietas tahan\n• Monitor kelembaban\n• Aplikasi preventif saat cuaca lembab"
        },
        "Tomato - Leaf Mold": {
            "severity": "warning",
            "description": "Jamur daun tomat yang disebabkan oleh Passalora fulva.",
            "symptoms": "• Bercak kuning di permukaan atas daun\n• Lapisan jamur abu-abu di bawah daun\n• Daun mengering",
            "treatment": "• Tingkatkan sirkulasi udara\n• Kurangi kelembaban\n• Aplikasi fungisida\n• Pruning daun terinfeksi",
            "prevention": "• Tanam dengan jarak cukup\n• Ventilasi greenhouse baik\n• Hindari penyiraman daun"
        },
        "Tomato - Septoria Leaf Spot": {
            "severity": "warning",
            "description": "Bercak daun septoria yang disebabkan oleh Septoria lycopersici.",
            "symptoms": "• Bercak kecil bulat berwarna abu-abu\n• Tepi bercak coklat tua\n• Titik hitam di tengah bercak",
            "treatment": "• Aplikasi fungisida protektif\n• Buang daun terinfeksi\n• Mulching\n• Avoid overhead watering",
            "prevention": "• Rotasi tanaman\n• Sanitasi kebun\n• Drainase baik"
        },
        "Tomato - Spider Mites": {
            "severity": "warning",
            "description": "Serangan tungau laba-laba yang menyebabkan kerusakan daun.",
            "symptoms": "• Titik-titik kuning pada daun\n• Jaring halus di daun\n• Daun keperakan dan kering",
            "treatment": "• Semprot dengan mitisida\n• Tingkatkan kelembaban udara\n• Gunakan predator alami\n• Cuci daun dengan air",
            "prevention": "• Jaga kelembaban optimal\n• Hindari stress kekeringan\n• Monitor rutin"
        },
        "Tomato - Target Spot": {
            "severity": "warning",
            "description": "Bercak target pada tomat yang disebabkan oleh Corynespora cassiicola.",
            "symptoms": "• Bercak bulat dengan lingkaran konsentris\n• Warna coklat muda hingga tua\n• Daun menguning dan rontok",
            "treatment": "• Aplikasi fungisida sistemik\n• Buang debris tanaman\n• Perbaiki sirkulasi udara\n• Mulching",
            "prevention": "• Rotasi tanaman\n• Varietas tahan\n• Sanitasi lahan"
        },
        "Tomato - Yellow Leaf Curl Virus": {
            "severity": "danger",
            "description": "Virus keriting kuning tomat yang ditularkan oleh kutu kebul.",
            "symptoms": "• Daun mengkerut dan menguning\n• Pertumbuhan terhambat\n• Buah kecil dan sedikit\n• Tanaman kerdil",
            "treatment": "• Cabut dan musnahkan tanaman terinfeksi\n• Kendalikan vektor kutu kebul\n• Gunakan mulsa reflektif\n• Tanam varietas tahan",
            "prevention": "• Gunakan jaring serangga\n• Eliminasi gulma inang\n• Tanam varietas tahan virus"
        },
        "Tomato - Mosaic Virus": {
            "severity": "danger",
            "description": "Virus mosaik tomat yang menyebabkan pola mozaik pada daun.",
            "symptoms": "• Pola mozaik hijau muda-tua\n• Daun kerut dan cacat\n• Buah belang-belang\n• Pertumbuhan terhambat",
            "treatment": "• Cabut tanaman terinfeksi\n• Sterilisasi alat\n• Gunakan varietas tahan\n• Eliminasi gulma inang",
            "prevention": "• Gunakan benih bebas virus\n• Sanitasi alat dan tangan\n• Isolasi tanaman sakit"
        },
        "Tomato - Healthy": {
            "severity": "success",
            "description": "Tanaman tomat dalam kondisi sehat dan optimal!",
            "symptoms": "• Daun hijau segar dan normal\n• Pertumbuhan vigorous\n• Tidak ada gejala penyakit\n• Produksi buah baik",
            "treatment": "• Lanjutkan perawatan rutin\n• Monitor perkembangan\n• Pemupukan seimbang\n• Penyiraman teratur",
            "prevention": "• Rotasi tanaman teratur\n• Sanitasi kebun\n• Pemupukan berimbang\n• Pengendalian hama preventif"
        }
    }
    
    return disease_info.get(disease_name, {
        "severity": "info",
        "description": "Informasi tidak tersedia untuk penyakit ini.",
        "symptoms": "Gejala tidak terdefinisi",
        "treatment": "Konsultasi dengan ahli pertanian",
        "prevention": "Lakukan praktik pertanian yang baik"
    })

# Header aplikasi
st.markdown('<h1 class="main-header">🌱 PlantGuard: Plant Disease Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Deteksi Penyakit Tanaman dengan Computer Vision</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">By: Joshua Pinem</p>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.header("🔧 Pengaturan")
    
    st.markdown("""
    <div class="info-box">
        <h4>📋 Panduan Penggunaan:</h4>
        <ol>
            <li>Pilih metode input (Upload/Kamera)</li>
            <li>Pastikan gambar daun jelas</li>
            <li>Klik "Analisis Gambar"</li>
            <li>Lihat hasil dan rekomendasi</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="info-box">
        <h4>🌿 Tanaman yang Didukung:</h4>
        • Paprika (Pepper Bell)<br>
        • Kentang (Potato)<br>
        • Tomat (Tomato)
    </div>
    """, unsafe_allow_html=True)

# Load model dan label
model = load_model()
labels = load_labels()

if model is None:
    st.stop()

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Input Gambar")
    
    # Tab untuk upload dan kamera
    tab1, tab2 = st.tabs(["📁 Upload File", "📷 Kamera"])
    
    image = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Pilih gambar daun tanaman",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar dengan format PNG, JPG, atau JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    
    with tab2:
        camera_image = st.camera_input("Ambil foto daun tanaman")
        
        if camera_image is not None:
            image = Image.open(camera_image)
    
    if image is not None:
        st.image(image, caption="Gambar yang akan dianalisis", use_column_width=True)
        
        # Tombol analisis
        if st.button("🔍 Analisis Gambar", type="primary", use_container_width=True):
            with st.spinner("Menganalisis gambar..."):
                try:
                    predicted_class, confidence = predict_disease(model, image)
                    disease_name = labels[predicted_class]
                    disease_info = get_disease_info(disease_name)
                    
                    # Simpan hasil di session state
                    st.session_state.prediction_result = {
                        'disease_name': disease_name,
                        'confidence': confidence,
                        'disease_info': disease_info
                    }
                    
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat analisis: {str(e)}")

with col2:
    st.subheader("📊 Hasil Analisis")
    
    if hasattr(st.session_state, 'prediction_result'):
        result = st.session_state.prediction_result
        disease_name = result['disease_name']
        confidence = result['confidence']
        disease_info = result['disease_info']
        
        # Tampilkan confidence score
        st.metric(
            label="Tingkat Kepercayaan",
            value=f"{confidence*100:.1f}%",
            delta=f"{'Tinggi' if confidence > 0.8 else 'Sedang' if confidence > 0.6 else 'Rendah'}"
        )
        
        # Box hasil berdasarkan severity
        if disease_info['severity'] == 'success':
            st.markdown(f"""
            <div class="success-box">
                <h3>✅ {disease_name}</h3>
                <p>{disease_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif disease_info['severity'] == 'warning':
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ {disease_name}</h3>
                <p>{disease_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif disease_info['severity'] == 'danger':
            st.markdown(f"""
            <div class="danger-box">
                <h3>🚨 {disease_name}</h3>
                <p>{disease_info['description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detail informasi
        with st.expander("📋 Detail Gejala"):
            st.markdown(disease_info['symptoms'])
        
        with st.expander("💊 Rekomendasi Pengobatan"):
            st.markdown(disease_info['treatment'])
        
        with st.expander("🛡️ Pencegahan"):
            st.markdown(disease_info['prevention'])
            
    else:
        st.markdown("""
        <div class="info-box">
            <h4>👆 Silakan upload gambar atau ambil foto</h4>
            <p>Sistem akan menganalisis gambar dan memberikan diagnosis serta rekomendasi perawatan yang sesuai.</p>
        </div>
        """, unsafe_allow_html=True)

# Informasi tambahan
st.markdown("---")

col3, col4, col5 = st.columns(3)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h4>🎯 Akurasi Model</h4>
        <h3>95%+</h3>
        <p>Tingkat akurasi tinggi</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h4>🌿 Kelas Penyakit</h4>
        <h3>15</h3>
        <p>Jenis penyakit yang dapat dideteksi</p>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div class="metric-card">
        <h4>⚡ Kecepatan</h4>
        <h3>< 3s</h3>
        <p>Waktu analisis rata-rata</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>🔬 Dikembangkan menggunakan TensorFlow & MobileNetV2 | 
    📱 Kompatibel dengan perangkat mobile | 
    🌱 Untuk kemajuan pertanian berkelanjutan</p>
</div>
""", unsafe_allow_html=True)

# Informasi teknis (tersembunyi)
with st.expander("ℹ️ Informasi Teknis"):
    st.write("**Model**: MobileNetV2 dengan Transfer Learning")
    st.write("**Dataset**: PlantVillage Dataset")
    st.write("**Input Size**: 224x224 pixels")
    st.write("**Framework**: TensorFlow/Keras")
    st.write("**Deployment**: Streamlit")
