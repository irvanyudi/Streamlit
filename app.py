import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
import torch.nn as nn
import io
import matplotlib.pyplot as plt
import numpy as np


# --- Model Setup ---
model_path = "model_vit.pth"
class_names = ["cataract", "diabetic", "glaucoma", "normal"]

model = vit_b_16()
model.heads = nn.Linear(model.heads.head.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Edukasi Penyakit ---
edukasi_penyakit = {
    "normal": """### 📜 Deskripsi
Citra fundus menunjukkan struktur retina yang normal, tidak terdapat tanda-tanda katarak, glaukoma, atau retinopati diabetik.

### 🩺 Rekomendasi Edukatif
- Pertahankan gaya hidup sehat, termasuk asupan vitamin A.
- Lakukan pemeriksaan mata rutin setiap 6–12 bulan.
- Gunakan pelindung mata saat di luar ruangan.
- Jaga kadar gula & tekanan darah stabil.
""",
    "cataract": """### 📜 Deskripsi
Katarak adalah kekeruhan pada lensa mata yang menyebabkan penglihatan kabur.

### 🩺 Rekomendasi Edukatif
- Periksakan ke dokter mata untuk evaluasi lebih lanjut.
- Operasi bisa diperlukan bila gejala berat.
- Hindari merokok dan kontrol gula darah.
- Gunakan kacamata hitam saat keluar rumah.
""",
    "glaucoma": """### 📜 Deskripsi
Glaukoma adalah kerusakan saraf optik akibat tekanan bola mata yang tinggi.

### 🩺 Rekomendasi Edukatif
- Glaukoma bersifat permanen, deteksi dini penting.
- Periksakan tekanan bola mata & lapang pandang.
- Terapi bisa berupa tetes mata, laser, atau operasi.
- Pemeriksaan rutin untuk riwayat keluarga glaukoma.
""",
    "diabetic": """### 📜 Deskripsi
Retinopati diabetik adalah kerusakan retina akibat diabetes, berpotensi menyebabkan kebutaan.

### 🩺 Rekomendasi Edukatif
- Jaga kadar gula darah secara ketat (HbA1c < 7%).
- Pemeriksaan retina rutin setiap tahun.
- Terapi: laser, injeksi anti-VEGF, atau vitrektomi.
- Konsultasi ke dokter retina sangat disarankan.
"""
}

# --- Navigasi Sidebar ---
st.set_page_config(page_title="Deteksi Penyakit Mata", layout="wide", page_icon="👁️")
menu = st.sidebar.selectbox("Navigasi", ["Home", "Deteksi", "Tentang Aplikasi"])

# --- Footer ---
def show_footer():
    st.markdown(
        """
        <hr style="margin-top: 50px; margin-bottom: 10px">
        <div style='text-align: center; color: grey; font-size: 0.9em;'>
            © 2025 | Dibuat oleh <b>Irvan Yudistiansyah</b> | Untuk keperluan edukasi & skripsi
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Home Page ---
if menu == "Home":
    st.title("👁️ Deteksi Penyakit Mata Menggunakan Citra Fundus Retina")


    st.markdown("""
    Selamat datang di aplikasi **Deteksi Penyakit Mata**, yang menggunakan teknologi machine learning
    untuk menganalisis gambar fundus retina dan memberikan prediksi awal terhadap penyakit mata
    seperti **Glaukoma**, **Retinopati Diabetik**
    
    ### 🔍 Fitur Utama:
    - Upload gambar fundus retina.
    - Deteksi penyakit secara otomatis.
    - Hasil diagnosis disertai edukasi kesehatan mata.

    ---
    """)
    show_footer()

# --- Tentang Page ---
elif menu == "Tentang Aplikasi":
    st.title("ℹ️ Tentang Aplikasi")

    st.markdown("""
    Aplikasi ini merupakan bagian dari proyek skripsi yang bertujuan untuk membantu deteksi dini penyakit mata
    berbasis citra retina. Dengan memanfaatkan teknologi deep learning, aplikasi ini dapat mengklasifikasikan
    gambar fundus mata ke dalam beberapa kategori penyakit.

    ### 👨‍💻 Pembuat
    - Nama: **Irvan Yudistiansyah**
    - NIM: 1234567890
    - Universitas: Universitas Nusaputra Sukabumi
    - Program Studi: Teknik Informatika

    ### 📌 Tujuan
    Membantu pengguna, terutama tenaga medis dan pasien, dalam memberikan deteksi awal penyakit mata
    secara cepat dan non-invasif.

    ### 💡 Teknologi yang Digunakan
    - Python
    - Streamlit
    - Vision Transformer (ViT)
    - PIL, Matplotlib

    ---
    """)
    show_footer()

# --- Deteksi Page ---
elif menu == "Deteksi":
    st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>👁️ Deteksi Penyakit Mata </h1>", unsafe_allow_html=True)
    st.markdown("### 📄 Unggah gambar fundus mata untuk mendapatkan hasil diagnosa dan edukasi.")

    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            pred_idx = torch.argmax(probabilities).item()
            pred_label = class_names[pred_idx]
            confidence = probabilities[pred_idx].item()

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.image(image, caption="🖼️ Gambar Fundus", use_container_width=True)
            st.markdown("##### ℹ️ Informasi Gambar")
            st.markdown(f"- 📁 Nama File: `{uploaded_file.name}`")
            st.markdown(f"- 🧪 Format: `{image.format or uploaded_file.type}`")
            st.markdown(f"- 🔍 Resolusi: `{image.size[0]}x{image.size[1]} px`")

        with col2:
            st.markdown(f"<div style='background-color:#e6f4ea;padding:12px;border-radius:10px'><b>✅ Hasil Deteksi:</b> <span style='color:#2e7d32'>{pred_label.capitalize()} ({confidence*100:.2f}%)</span></div>", unsafe_allow_html=True)

            st.markdown("#### 📊 Probabilitas Klasifikasi")
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(class_names, probabilities.numpy() * 100, color="#1976D2")
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilitas (%)")
            st.pyplot(fig)

            st.markdown("---")
            st.markdown(edukasi_penyakit[pred_label])

            laporan_text = f"""
📄 Hasil Deteksi Fundus Mata

🖼️ File: {uploaded_file.name}
🧠 Prediksi: {pred_label.capitalize()}
📊 Confidence: {confidence*100:.2f}%

--- Edukasi ---
{edukasi_penyakit[pred_label]}
            """
            laporan_bytes = io.BytesIO()
            laporan_bytes.write(laporan_text.encode('utf-8'))
            laporan_bytes.seek(0)

            st.download_button(
                "📅 Unduh Laporan Diagnosa",
                laporan_bytes,
                file_name=f"laporan_{pred_label}.txt",
                mime="text/plain"
            )
    else:
        st.info("Silakan unggah gambar fundus mata terlebih dahulu untuk memulai diagnosa.")
    show_footer()
