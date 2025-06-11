# -*- coding: utf-8 -*-
import streamlit as st
import torch
import torchaudio  # Menggantikan librosa
import io          # Diperlukan untuk memproses file di memori
import numpy as np
import pandas as pd
from transformers import pipeline

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Perbandingan Model Analisis Emosi 🗣️",
    page_icon="🤖",
    layout="wide"
)

# --- FUNGSI CACHE UNTUK MEMUAT SEMUA MODEL ---

@st.cache_resource
def load_audeering_model():
    """Memuat model emosi dimensional (Audeering)."""
    classifier = pipeline("audio-classification", model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim", framework="pt")
    return classifier

@st.cache_resource
def load_stt_model():
    """Memuat model Speech-to-Text (Whisper)."""
    stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", framework="pt")
    return stt_pipe

@st.cache_resource
def load_superb_model():
    """Memuat model emosi kategorikal (SUPERB)."""
    classifier = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", framework="pt")
    return classifier

# Panggil fungsi untuk memuat ketiga model
with st.spinner("Memuat semua model AI, ini mungkin memakan waktu beberapa menit..."):
    model_audeering = load_audeering_model()
    model_stt = load_stt_model()
    model_superb = load_superb_model()

st.success("✅ Semua model berhasil dimuat!")


# --- FUNGSI BANTU UNTUK MENAMPILKAN HASIL (TIDAK PERLU DIUBAH) ---

def display_audeering_results(predictions):
    st.subheader("Model 1: Audeering (Dimensional)")
    if not predictions:
        st.warning("Tidak ada prediksi dari model Audeering.")
        return
    label_mapping = {'arousal': 'Energi (Arousal)', 'dominance': 'Dominasi (Dominance)', 'valence': 'Positif/Negatif (Valence)'}
    for p in predictions:
        p['label'] = label_mapping.get(p['label'], p['label'])
    df = pd.DataFrame(predictions)
    df = df.rename(columns={'label': 'Dimensi Emosi', 'score': 'Skor'})
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index('Dimensi Emosi'))

def display_superb_results(predictions):
    st.subheader("Model 2: SUPERB (Kategorikal)")
    if not predictions:
        st.warning("Tidak ada prediksi dari model SUPERB.")
        return
    top_emotion = max(predictions, key=lambda x: x['score'])
    st.success(f"Prediksi Emosi Utama: **{top_emotion['label'].capitalize()}**")
    df = pd.DataFrame(predictions).sort_values(by='score', ascending=False).reset_index(drop=True)
    df = df.rename(columns={'label': 'Emosi', 'score': 'Skor'})
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df.set_index('Emosi'))

# --- FUNGSI ANALISIS UTAMA (TIDAK PERLU DIUBAH) ---
def perform_analysis(audio_data):
    """Fungsi untuk menjalankan semua analisis pada data audio."""
    st.header("🗣️ Hasil Transkripsi Ucapan")
    transcription_result = model_stt(audio_data)
    transcribed_text = transcription_result.get("text", "Tidak ada teks yang terdeteksi.")
    if not transcribed_text.strip():
        transcribed_text = "Tidak dapat mendeteksi ucapan yang jelas."
    st.info(transcribed_text.upper())
    st.markdown("---")

    st.header("⚖️ Perbandingan Hasil Analisis Emosi")
    col1, col2 = st.columns(2)

    with col1:
        predictions_audeering = model_audeering(audio_data, top_k=None)
        display_audeering_results(predictions_audeering)

    with col2:
        predictions_superb = model_superb(audio_data, top_k=None)
        display_superb_results(predictions_superb)

# --- TAMPILAN UTAMA ---
st.title("🤖 Perbandingan Model Analisis Emosi Audio")
st.markdown("Analisis audio dari file yang sama menggunakan dua model emosi berbeda secara berdampingan.")
st.markdown("---")

if not all([model_audeering, model_stt, model_superb]):
    st.error("Gagal memuat satu atau lebih model AI. Harap refresh halaman.")
    st.stop()

# --- Fitur Unggah File ---
st.header("Analisis dari File Audio atau Video")
uploaded_file = st.file_uploader(
    "Pilih file (.wav, .mp3, .flac, .mp4)...",
    type=["wav", "mp3", "flac", "mp4"]
)

if uploaded_file is not None:
    # Cek tipe file untuk menampilkan pemutar yang sesuai
    if uploaded_file.type.startswith('audio'):
        st.audio(uploaded_file, "Pilih file (.wav, .mp3, .flac, .mp4)...")
    elif uploaded_file.type.startswith('video'):
        st.video(uploaded_file)

    if st.button("Analisis File", key="file_analysis"):
        with st.spinner("Mengekstrak audio dan menganalisis dengan semua model..."):
            
            # ===================================================================
            # ===== BLOK KODE YANG DIUBAH DARI LIBROSA KE TORCHAUDIO ==========
            # ===================================================================
            try:
                # Baca file yang diunggah ke dalam buffer di memori
                buffer = io.BytesIO(uploaded_file.read())
                
                # Gunakan torchaudio untuk memuat audio, bahkan dari video (jika FFmpeg terinstal)
                waveform, original_sample_rate = torchaudio.load(buffer)
                
                target_sample_rate = 16000 # Sample rate yang dibutuhkan model

                # Ubah sample rate jika perlu
                if original_sample_rate != target_sample_rate:
                    resampler = torchaudio.transforms.Resample(original_sample_rate, target_sample_rate)
                    waveform = resampler(waveform)

                # Pastikan audio mono (satu channel)
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Pipeline transformers menerima raw numpy array
                audio_input = waveform.squeeze().numpy()
                
                # Panggil fungsi analisis utama dengan data audio yang sudah siap
                perform_analysis(audio_input)

            except Exception as e:
                st.error(f"Gagal memproses file audio/video: {e}")
                st.info("Pastikan FFmpeg sudah terinstal di sistem Anda untuk dapat memproses file MP3 atau MP4.")
            # ===================================================================
            # ================= AKHIR BLOK KODE YANG DIUBAH ====================
            # ===================================================================

st.markdown("---")
st.markdown("Dibuat untuk perbandingan model.")